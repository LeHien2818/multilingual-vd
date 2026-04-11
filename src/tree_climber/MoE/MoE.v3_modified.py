import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import copy
import logging
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm.auto import tqdm
from config_hien import BATCH_SIZE, EPOCHS, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, LANG_CLUSTER, F1_BEST_STATE, MODEL_SAVE_DIR
import os

# =====================================================================
# LOGGING SETUP - Write to both file and console
# =====================================================================
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Also capture print statements
class PrintLogger:
    def __init__(self, log_func):
        self.log_func = log_func
    
    def write(self, message):
        if message.strip():
            self.log_func(message.strip())
    
    def flush(self):
        pass

# Redirect print to logging
original_print = print
def print(*args, **kwargs):
    message = ' '.join(str(arg) for arg in args)
    log.info(message)

# =====================================================================
# THÀNH PHẦN 1: MẠNG CHUYÊN GIA (EXPERT NETWORKS) - IMPROVED
# =====================================================================
class CWE_Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(CWE_Expert, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# =====================================================================
# MulVulExpert - Language-Specific Expert with Parameter Pools (VulPy)
# =====================================================================
class MulVulExpert(nn.Module):
    """Expert VulPy: Uses language-specific learnable parameter pools with attention.
    
    For vulnerabilities spanning multiple languages (Python & CCPP),
    this expert learns language-specific adaptations via prepended pool tokens.
    """
    def __init__(self, pretrained_model, input_dim=768, hidden_dim=256, num_langs=2, pool_length=5, dropout_rate=0.1, temperature=0.1):
        super(MulVulExpert, self).__init__()
        self.backbone = pretrained_model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pool_length = pool_length
        self.temperature = temperature
        self.num_langs = num_langs
        
        # Language-specific learnable parameter pools
        self.parameter_pool = nn.Parameter(
            torch.randn(num_langs, pool_length, input_dim) * 0.02
        )
        # Language keys for attention-based pool selection
        self.keys = nn.Parameter(
            torch.randn(num_langs, input_dim) * 0.02
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, language_ids=None):
        """Forward pass with language-specific parameter pool routing.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            language_ids: Optional language IDs for routing (0=Python, 1=CCPP)
                         If None, learns routing from input
        
        Returns:
            logits: Vulnerability predictions (batch_size, 1)
            aux_loss: Language alignment auxiliary loss (scalar)
        """
        if self.backbone is None:
            raise ValueError("MulVulExpert requires a valid pretrained backbone")

        batch_size = input_ids.size(0)
        raw_embeds = self.backbone.embeddings(input_ids)
        cls_query = raw_embeds[:, 0, :]
        
        if language_ids is not None:
            # Use provided language IDs for pool selection
            pool = self.parameter_pool[language_ids]  # (batch_size, pool_length, input_dim)
            
            # Compute auxiliary loss: alignment with language key
            query_norm = F.normalize(cls_query, p=2, dim=1)  # (batch_size, input_dim)
            keys_norm = F.normalize(self.keys, p=2, dim=1)  # (num_langs, input_dim)
            selected_keys = keys_norm[language_ids]  # (batch_size, input_dim)
            cosine_sim = torch.sum(query_norm * selected_keys, dim=1)  # (batch_size,)
            aux_loss = (1.0 - cosine_sim).mean()
        else:
            # Learn routing via attention over language keys
            query_norm = F.normalize(cls_query, p=2, dim=1)
            keys_norm = F.normalize(self.keys, p=2, dim=1)
            scores = torch.matmul(query_norm, keys_norm.transpose(0, 1)) / self.temperature  # (batch_size, num_langs)
            lang_ids = torch.argmax(scores, dim=1)  # (batch_size,)
            pool = self.parameter_pool[lang_ids]  # (batch_size, pool_length, input_dim)
            aux_loss = torch.tensor(0.0, device=cls_query.device)
        
        keep_len = raw_embeds.size(1) - self.pool_length
        raw_embeds = raw_embeds[:, :keep_len, :]
        attention_mask_truncated = attention_mask[:, :keep_len]

        new_embeds = torch.cat([pool, raw_embeds], dim=1)

        pool_mask = torch.ones(batch_size, self.pool_length,
                               dtype=attention_mask.dtype,
                               device=attention_mask.device)
        new_mask = torch.cat([pool_mask, attention_mask_truncated], dim=1)

        outputs = self.backbone(inputs_embeds=new_embeds, attention_mask=new_mask)
        hidden_states = outputs.last_hidden_state

        # Mean pooling trên pool tokens
        pool_hidden = hidden_states[:, 0:self.pool_length, :]
        final_repr = pool_hidden.mean(dim=1)

        logits = self.classifier(final_repr)
        
        return logits, aux_loss

# =====================================================================
# THÀNH PHẦN 2: BỘ ĐỊNH TUYẾN ĐA NHIỆM (GATING NETWORK / ROUTER) - IMPROVED
# =====================================================================
class TopKRouter(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=1):
        super(TopKRouter, self).__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.norm = nn.LayerNorm(input_dim)
        self.routing_layer = nn.Linear(input_dim, num_experts)
        nn.init.kaiming_normal_(self.routing_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.routing_layer.bias)

    def forward(self, x):
        x_norm = self.norm(x)
        logits = self.routing_layer(x_norm)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        sparse_weights = torch.zeros_like(logits).scatter(-1, top_k_indices, top_k_weights)
        return sparse_weights, top_k_indices, logits

# =====================================================================
# THÀNH PHẦN 3: KIẾN TRÚC TỔNG THỂ MOE-VD KẾT HỢP - IMPROVED WITH FINE-TUNING
# =====================================================================
class MoE_VulnerabilityDetector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_experts=4, top_k=1, dropout_rate=0.1, 
                 encoder_name="microsoft/codebert-base", freeze_encoder=False):
        super(MoE_VulnerabilityDetector, self).__init__()
        self.num_experts = num_experts
        self.expert_ccpp_idx = 0
        self.expert_python_idx = 1
        self.expert_vulpy_idx = 2
        
        # Add CodeBERT encoder to the model (like baseline)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
            self.encoder = AutoModel.from_pretrained(encoder_name)
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
            print(f"CodeBERT encoder loaded: {encoder_name} (freeze={freeze_encoder})")
        except Exception as e:
            print(f"Warning: Could not load CodeBERT: {e}")
            self.encoder = None
            self.tokenizer = None
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.router = TopKRouter(input_dim, num_experts, top_k)
        
        # Initialize experts: CCPP (0), Python (1), VulPy (2) + others
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if i == 2:  # Expert VulPy with MulVulExpert technique
                self.experts.append(MulVulExpert(pretrained_model=self.encoder, input_dim=input_dim, hidden_dim=hidden_dim,
                                                 num_langs=2, pool_length=5, dropout_rate=dropout_rate))
            else:
                self.experts.append(CWE_Expert(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate))

    def _apply_special_routing(self, routing_weights, languages=None, vuln_labels=None, cwe_labels=None, cwe_dpy_set=None):
        """Apply dual-routing constraints from instruction.md."""
        if languages is None:
            return routing_weights

        updated_weights = routing_weights.clone()
        for i, language in enumerate(languages):
            language_lower = str(language).strip().lower()

            if language_lower == "python":
                updated_weights[i].zero_()
                updated_weights[i, self.expert_python_idx] = 0.5
                updated_weights[i, self.expert_vulpy_idx] = 0.5
                continue

            if language_lower == "ccpp":
                if vuln_labels is None or cwe_labels is None or cwe_dpy_set is None:
                    continue

                is_vulnerable = float(vuln_labels[i].item()) >= 0.5
                cwe_id = int(cwe_labels[i].item())
                if is_vulnerable and cwe_id in cwe_dpy_set:
                    updated_weights[i].zero_()
                    updated_weights[i, self.expert_ccpp_idx] = 0.5
                    updated_weights[i, self.expert_vulpy_idx] = 0.5

        return updated_weights

    def get_expert_logits(self, x, languages=None, raw_input_ids=None, raw_attention_mask=None):
        """Return per-expert logits for language-based inference policy."""
        x_norm = self.input_norm(x)
        expert_outputs = []
        language_id_map = {"python": 0, "ccpp": 1}
        
        for i, expert in enumerate(self.experts):
            if i == self.expert_vulpy_idx and isinstance(expert, MulVulExpert):
                lang_ids = None
                if languages is not None:
                    lang_ids = torch.tensor(
                        [language_id_map.get(str(languages[j]).strip().lower(), 0) for j in range(len(languages))],
                        device=x_norm.device, dtype=torch.long
                    )
                if raw_input_ids is None or raw_attention_mask is None:
                    raise ValueError("raw_input_ids and raw_attention_mask are required for MulVulExpert")
                logit, _ = expert(raw_input_ids, raw_attention_mask, language_ids=lang_ids)  # unpack aux_loss
                expert_outputs.append(logit)
            else:
                expert_outputs.append(expert(x_norm))
        
        return torch.cat(expert_outputs, dim=1)

    def encode_code(self, code_strings, device):
        """Encode code strings to embeddings using CodeBERT (on-the-fly, trainable)"""
        if self.encoder is None:
            raise ValueError("Encoder not loaded")
        
        inputs = self.tokenizer(
            code_strings,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.encoder(**inputs)
        # Mean pooling with attention mask
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.unsqueeze(-1).sum(dim=1)
        embeddings = sum_hidden / sum_mask
        
        return embeddings, inputs['input_ids'], attention_mask

    def forward(self, x, languages=None, vuln_labels=None, cwe_labels=None, cwe_dpy_set=None,
                apply_special_routing=False, raw_input_ids=None, raw_attention_mask=None):
        x = self.input_norm(x)
        
        routing_weights, top_k_indices, router_logits = self.router(x)
        if apply_special_routing:
            routing_weights = self._apply_special_routing(
                routing_weights,
                languages=languages,
                vuln_labels=vuln_labels,
                cwe_labels=cwe_labels,
                cwe_dpy_set=cwe_dpy_set,
            )
        batch_size = x.size(0)
        final_output = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        total_aux_loss = torch.tensor(0.0, device=x.device)

        routing_probs = F.softmax(router_logits, dim=-1)
        fraction_routed = routing_weights.gt(0).float().mean(dim=0)
        prob_per_expert = routing_probs.mean(dim=0)

        # Map language strings to language IDs for MulVulExpert
        language_id_map = {"python": 0, "ccpp": 1}
        
        # Vectorized expert processing
        for i, expert in enumerate(self.experts):
            expert_mask = (routing_weights[:, i] > 0)
            if expert_mask.any():
                expert_inputs = x[expert_mask]
                weights = routing_weights[expert_mask, i].unsqueeze(1)
                
                # Special handling for VulPy expert with MulVulExpert
                if i == self.expert_vulpy_idx and isinstance(expert, MulVulExpert):
                    lang_ids = None
                    if languages is not None:
                        lang_ids = torch.tensor(
                            [language_id_map.get(str(languages[j]).strip().lower(), 0) 
                             for j in range(len(languages)) if expert_mask[j]],
                            device=x.device, dtype=torch.long
                        )
                    if raw_input_ids is None or raw_attention_mask is None:
                        raise ValueError("raw_input_ids and raw_attention_mask are required for MulVulExpert")
                    expert_outputs, aux_loss = expert(raw_input_ids[expert_mask], raw_attention_mask[expert_mask], language_ids=lang_ids)
                    total_aux_loss = total_aux_loss + aux_loss
                else:
                    expert_outputs = expert(expert_inputs)
                
                final_output[expert_mask] += expert_outputs * weights

        return final_output, fraction_routed, prob_per_expert, router_logits, total_aux_loss


def get_language_policy_logits(model, emb, languages, raw_input_ids=None, raw_attention_mask=None):
    """Inference policy:
    - CCPP -> Expert CCPP (0)
    - Python -> Expert Python (1), with Expert VulPy (2) as reference output
    - Other languages -> fallback to routed output
    """
    routed_logits, _, _, _, _ = model(
        emb,
        raw_input_ids=raw_input_ids,
        raw_attention_mask=raw_attention_mask,
    )  # unpack aux_loss
    expert_logits = model.get_expert_logits(
        emb,
        languages=languages,
        raw_input_ids=raw_input_ids,
        raw_attention_mask=raw_attention_mask,
    )  # pass languages for MulVulExpert

    final_logits = routed_logits.clone()
    vulpy_logits = torch.full_like(final_logits, float('nan'))

    for i, language in enumerate(languages):
        language_lower = str(language).strip().lower()
        if language_lower == "ccpp":
            final_logits[i, 0] = expert_logits[i, model.expert_ccpp_idx]
        elif language_lower == "python":
            final_logits[i, 0] = expert_logits[i, model.expert_python_idx]
            vulpy_logits[i, 0] = expert_logits[i, model.expert_vulpy_idx]

    return final_logits, vulpy_logits

# =====================================================================
# THÀNH PHẦN 4: HÀM MẤT MÁT ĐA NHIỆM VÀ ĐÁNH GIÁ (LOSS & METRICS) - IMPROVED
# =====================================================================
def focal_loss(logits, targets, alpha=0.25, gamma=2.0, label_smoothing=0.0, pos_weight=None):
    """Focal Loss with label smoothing and class weighting for handling class imbalance"""
    # Apply label smoothing
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
    
    # Apply class weighting if provided
    if pos_weight is not None:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pos_weight)
    else:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    probs = torch.sigmoid(logits)
    pt = targets * probs + (1 - targets) * (1 - probs)
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * bce_loss
    return loss.mean()

def moe_multitask_loss(binary_logits, binary_targets, router_logits, cwe_labels,
                       fraction_routed, prob_per_expert, num_experts, 
                       pos_weight=None, use_focal=True, label_smoothing=0.05, alpha=0.005, beta=0.2, expert_aux_loss=None):
    """Improved loss function with class weighting, focal loss, and label smoothing
    
    Changes from baseline to match performance:
    - Reduced label_smoothing: 0.1 -> 0.05 (less aggressive)
    - Reduced alpha (aux loss weight): 0.01 -> 0.005 (focus more on main task)
    - Reduced beta (CWE loss weight): 0.3 -> 0.2 (focus more on binary classification)
    - Pass pos_weight to focal_loss for class balance
    - Incorporate expert_aux_loss (from MulVulExpert language alignment)
    """
    if use_focal:
        # Pass pos_weight to focal loss for better class balance (like baseline)
        bce_loss = focal_loss(binary_logits, binary_targets, alpha=0.25, gamma=2.0, 
                             label_smoothing=label_smoothing, pos_weight=pos_weight)
    else:
        if pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_targets, pos_weight=pos_weight)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_targets)
    
    # Less aggressive label smoothing for CWE classification
    ce_loss_cwe = F.cross_entropy(router_logits, cwe_labels, label_smoothing=0.02)
    
    # Load balancing auxiliary loss (reduced weight)
    aux_loss = num_experts * torch.sum(fraction_routed * prob_per_expert)
    
    # Expert auxiliary loss (MulVulExpert language alignment)
    if expert_aux_loss is None:
        expert_aux_loss = torch.tensor(0.0, device=binary_logits.device)
    
    total_loss = bce_loss + beta * ce_loss_cwe + alpha * aux_loss + 0.5 * expert_aux_loss
    return total_loss, bce_loss, ce_loss_cwe, aux_loss

def calculate_accuracy(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == targets).float().sum()
    return correct / targets.numel()

def calculate_f1_score(preds, targets):
    """Calculate F1 score"""
    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1.item(), precision.item(), recall.item()

def find_optimal_threshold(model, data_loader, device):
    """Find optimal classification threshold using F1 score on validation set"""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for code_batch, vuln, cwe, cluster_type, languages in data_loader:
            emb, input_ids, attention_mask = model.encode_code(code_batch, device)
            logits, _ = get_language_policy_logits(
                model,
                emb,
                languages,
                raw_input_ids=input_ids,
                raw_attention_mask=attention_mask,
            )
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(vuln.numpy())
    
    all_probs = np.array(all_probs).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Debug: Print probability distribution
    print(f"\nProbability distribution:")
    print(f"  Min: {all_probs.min():.4f}, Max: {all_probs.max():.4f}")
    print(f"  Mean: {all_probs.mean():.4f}, Std: {all_probs.std():.4f}")
    print(f"  Median: {np.median(all_probs):.4f}")
    print(f"  Q1: {np.percentile(all_probs, 25):.4f}, Q3: {np.percentile(all_probs, 75):.4f}")
    
    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = {}
    
    # Search with finer granularity
    thresholds_to_try = np.arange(0.05, 0.95, 0.02)
    
    print(f"\nSearching optimal threshold...")
    for threshold in thresholds_to_try:
        preds = (all_probs > threshold).astype(float)
        preds_tensor = torch.tensor(preds)
        targets_tensor = torch.tensor(all_targets)
        
        f1, precision, recall = calculate_f1_score(preds_tensor, targets_tensor)
        acc = (preds == all_targets).mean()
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': acc
            }
    
    print(f"\nBest threshold search results:")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  F1 Score: {best_metrics['f1']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    
    return best_threshold, best_metrics

def get_num_classes_cwe(data_list, label_field='cwe'):
    """Get number of classes needed based on max index in selected label field.
    Supports both positive IDs and special negative values (-1, -2, etc.).
    """
    cwe_ids = set()
    for item in data_list:
        if label_field in item:
            if isinstance(item[label_field], list):
                cwe_ids.update(item[label_field])
            else:
                cwe_ids.add(item[label_field])
    
    if not cwe_ids:
        return 0, [], set()
    
    # Separate positive and negative CWE IDs
    positive_cwes = {c for c in cwe_ids if c >= 0}
    negative_cwes = {c for c in cwe_ids if c < 0}
    
    # For CrossEntropy, we need num_classes based only on positive indices
    if positive_cwes:
        max_cwe_index = max(positive_cwes)
        min_positive_cwe = min(positive_cwes)
        num_classes = max_cwe_index + 1
    else:
        min_positive_cwe = float('inf')
        num_classes = 0
    
    min_cwe_index = min(cwe_ids)
    
    print(f"Unique CWE indices found: {len(cwe_ids)}")
    print(f"  Positive CWE IDs: {len(positive_cwes)}")
    print(f"  Negative CWE IDs (special values): {len(negative_cwes)} {sorted(list(negative_cwes))}")
    print(f"CWE index range: [{min_cwe_index}, {max_cwe_index if positive_cwes else 'N/A'}]")
    print(f"Number of CWE classes (for positive, max+1): {num_classes}")
    
    return num_classes, sorted(list(cwe_ids)), negative_cwes

def build_auto_cwe_clusters(data_list, target_experts=None, min_experts=4, max_experts=32, min_samples_threshold=20, label_field='cwe'):
    """Automatically cluster label IDs into fewer balanced groups for router training.
    Supports both positive CWE IDs and special negative CWE values (-1, -2, etc.).
    
    Args:
        data_list: full dataset containing labels for clustering
        target_experts: target number of clusters (None = sqrt(num_unique))
        min_experts: minimum number of clusters to maintain
        max_experts: maximum number of clusters
        min_samples_threshold: CWEs with freq <= this are grouped into "rare" cluster
        label_field: name of field used to build clusters (default: cwe)
    """
    cwes = [int(item['cwe']) for item in data_list]
    cwe_values = [int(item[label_field]) for item in data_list if label_field in item]
    if not cwe_values:
        raise ValueError(f"No values found in '{label_field}' for auto clustering")

    cwe_counter = Counter(cwe_values)
    
    # Separate positive and negative CWE IDs
    positive_cwes = {cwe_id: freq for cwe_id, freq in cwe_counter.items() if cwe_id >= 0}
    negative_cwes = {cwe_id: freq for cwe_id, freq in cwe_counter.items() if cwe_id < 0}
    
    unique_cwes = sorted(cwe_counter.keys())
    num_unique = len(unique_cwes)
    num_positive = len(positive_cwes)
    num_negative = len(negative_cwes)

    # For positive CWEs: Separate rare and frequent
    rare_cwes = {cwe_id: freq for cwe_id, freq in positive_cwes.items() if freq <= min_samples_threshold}
    frequent_cwes = {cwe_id: freq for cwe_id, freq in positive_cwes.items() if freq > min_samples_threshold}
    
    num_rare = len(rare_cwes)
    num_frequent = len(frequent_cwes)
    
    print("\nAuto CWE clustering enabled")
    print(f"  Total unique CWE classes: {num_unique}")
    print(f"  Positive CWE IDs: {num_positive}")
    print(f"  Negative CWE IDs (special): {num_negative} {sorted(list(negative_cwes.keys()))}")
    print(f"  Positive CWEs with > {min_samples_threshold} samples: {num_frequent}")
    print(f"  Positive CWEs with <= {min_samples_threshold} samples (will be merged): {num_rare}")
    
    # Determine target experts for frequent CWEs only
    if target_experts is None:
        # Scale experts by sqrt of frequent classes (not rare)
        target_experts = max(1, int(round(np.sqrt(num_frequent)))) if num_frequent > 0 else 1

    # Always keep at least `min_experts` total clusters so MoE does not collapse
    frequent_target = int(target_experts)
    num_special_clusters = (1 if num_rare > 0 else 0) + (1 if num_negative > 0 else 0)
    
    if num_special_clusters > 0:
        frequent_target = max(1, frequent_target)
        frequent_target = max(min_experts - num_special_clusters, frequent_target)
    else:
        frequent_target = max(min_experts, frequent_target)

    frequent_target = min(max_experts - num_special_clusters, frequent_target)
    frequent_target = max(1, frequent_target)
    num_experts = frequent_target + num_special_clusters
    
    # Assign positive CWEs to clusters with load balancing
    cluster_loads = [0 for _ in range(num_experts)]
    
    cwe_to_cluster = {}
    
    sorted_by_freq = sorted(frequent_cwes.items(), key=lambda x: x[1], reverse=True)
    rare_cluster_idx = -1
    negative_cluster_idx = -1
    
    if num_negative > 0:
        negative_cluster_idx = num_experts - 1
    if num_rare > 0:
        rare_cluster_idx = num_experts - 1 if num_negative == 0 else num_experts - 2
    
    # Assign frequent positive CWEs first
    num_assignment_clusters = num_experts - num_special_clusters
    for cwe_id, freq in sorted_by_freq:
        # Find cluster with minimum load (excluding special clusters)
        best_cluster = min(
            range(num_assignment_clusters),
            key=lambda idx: cluster_loads[idx]
        )
        cwe_to_cluster[cwe_id] = best_cluster
        cluster_loads[best_cluster] += freq
    
    # Assign all rare positive CWEs to rare cluster
    if num_rare > 0:
        rare_cluster_total = 0
        for cwe_id, freq in rare_cwes.items():
            cwe_to_cluster[cwe_id] = rare_cluster_idx
            rare_cluster_total += freq
        cluster_loads[rare_cluster_idx] = rare_cluster_total
    
    # Assign all negative CWEs to negative cluster
    if num_negative > 0:
        negative_cluster_total = 0
        for cwe_id, freq in negative_cwes.items():
            cwe_to_cluster[cwe_id] = negative_cluster_idx
            negative_cluster_total += freq
        cluster_loads[negative_cluster_idx] = negative_cluster_total
    
    print(f"  Router experts/clusters: {num_experts}")
    print(f"  Cluster loads (samples): {cluster_loads}")
    if num_rare > 0:
        print(f"  Rare cluster (idx {rare_cluster_idx}): {num_rare} CWE types, {cluster_loads[rare_cluster_idx]} samples")
    if num_negative > 0:
        print(f"  Negative CWE cluster (idx {negative_cluster_idx}): {num_negative} CWE types {sorted(list(negative_cwes.keys()))}, {cluster_loads[negative_cluster_idx]} samples")

    return num_experts, cwe_to_cluster, cwe_counter, cluster_loads



def calculate_class_weights(data_list):
    """Calculate class weights for imbalanced data"""
    vuln_count = sum(1 for item in data_list if item['vuln'] == 1)
    safe_count = len(data_list) - vuln_count
    
    if vuln_count == 0 or safe_count == 0:
        return None
    
    # Weight for positive class (vulnerable)
    pos_weight = safe_count / vuln_count
    print(f"Class distribution - Vulnerable: {vuln_count}, Safe: {safe_count}")
    print(f"Positive class weight: {pos_weight:.2f}")
    
    return torch.tensor([pos_weight])


def _update_group_metrics(group_metrics, group_name, pred, target):
    if group_name not in group_metrics:
        group_metrics[group_name] = {
            "tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0
        }

    if pred == 1 and target == 1:
        group_metrics[group_name]["tp"] += 1
    elif pred == 1 and target == 0:
        group_metrics[group_name]["fp"] += 1
    elif pred == 0 and target == 0:
        group_metrics[group_name]["tn"] += 1
    else:
        group_metrics[group_name]["fn"] += 1

    group_metrics[group_name]["total"] += 1


def _finalize_group_metrics(group_metrics):
    for group_name in group_metrics:
        metrics = group_metrics[group_name]
        metrics["accuracy"] = (metrics["tp"] + metrics["tn"]) / metrics["total"] if metrics["total"] > 0 else 0
        metrics["precision"] = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if (metrics["tp"] + metrics["fp"]) > 0 else 0
        metrics["recall"] = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) > 0 else 0
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0

    return group_metrics


def calculate_overall_metrics_from_groups(group_metrics):
    total_samples = sum(metrics["total"] for metrics in group_metrics.values())
    if total_samples == 0:
        return {
            "total": 0,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    overall_tp = sum(metrics["tp"] for metrics in group_metrics.values())
    overall_fp = sum(metrics["fp"] for metrics in group_metrics.values())
    overall_fn = sum(metrics["fn"] for metrics in group_metrics.values())
    overall_tn = sum(metrics["tn"] for metrics in group_metrics.values())
    overall_accuracy = (overall_tp + overall_tn) / total_samples
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    return {
        "total": total_samples,
        "accuracy": overall_accuracy,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "tp": overall_tp,
        "fp": overall_fp,
        "fn": overall_fn,
        "tn": overall_tn,
    }


def evaluate_by_group(model, data_loader, device, group_getter, threshold=0.5):
    model.eval()
    group_metrics = {}

    with torch.no_grad():
        for code_batch, vuln, cwe, cluster_type, languages in data_loader:
            emb, input_ids, attention_mask = model.encode_code(code_batch, device)
            vuln, cwe = vuln.to(device), cwe.to(device)
            logits, _ = get_language_policy_logits(
                model,
                emb,
                languages,
                raw_input_ids=input_ids,
                raw_attention_mask=attention_mask,
            )

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            for i in range(len(cwe)):
                group_name = group_getter(int(cwe[i].item()), str(languages[i]))
                pred = preds[i].item()
                target = vuln[i].item()
                _update_group_metrics(group_metrics, group_name, pred, target)

    return _finalize_group_metrics(group_metrics)

def evaluate_by_cwe(model, data_loader, device, threshold=0.5):
    """Evaluate model performance grouped by CWE ID with detailed metrics"""
    return evaluate_by_group(
        model,
        data_loader,
        device,
        group_getter=lambda cwe_id, language: cwe_id,
        threshold=threshold,
    )


def evaluate_by_language(model, data_loader, device, threshold=0.5):
    """Evaluate model performance grouped by language with detailed metrics"""
    return evaluate_by_group(
        model,
        data_loader,
        device,
        group_getter=lambda cwe_id, language: language,
        threshold=threshold,
    )


def evaluate_by_language_cwe(model, data_loader, device, threshold=0.5):
    """Evaluate model performance by combined key: language + CWE"""
    return evaluate_by_group(
        model,
        data_loader,
        device,
        group_getter=lambda cwe_id, language: f"{language} | CWE-{cwe_id}",
        threshold=threshold,
    )

def precompute_embeddings(data_list, encoder, device, batch_size=BATCH_SIZE):
    """Pre-compute all embeddings to avoid redundant encoding during training"""
    print("Pre-computing embeddings...")
    embeddings = []
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        batch_embeddings = []
        
        for item in batch:
            emb = encoder.encode(item["code"])
            batch_embeddings.append(emb)
        
        embeddings.extend(batch_embeddings)
        print(f"Pre-computed {min(i+batch_size, len(data_list))}/{len(data_list)} embeddings")
    
    return embeddings


def build_tqdm(iterable, desc, unit, leave=False):
    """Create a tqdm progress bar that behaves well on both TTY and non-TTY terminals."""
    is_tty = sys.stdout.isatty() or sys.stderr.isatty()
    return tqdm(
        iterable,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=is_tty,
        ascii=not is_tty,
        mininterval=1.0,
        smoothing=0.1,
    )

# =====================================================================
# THÀNH PHẦN 5: LOAD DỮ LIỆU TỪ FILE
# =====================================================================

def load_raw_data(file_path):
    """Load C code data from JSONL file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append(item)
        print(f"Loaded {len(data)} samples from {file_path}")
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}.")
        return None
    return data


def get_language_label(item):
    language = str(item.get("language", "Unknown")).strip()
    return language if language else "Unknown"


def remap_type_index(train_data, val_data, test_data, enable_lang_cluster=False):
    """Build per-item cluster_type used for router clustering.

    If LANG_CLUSTER is False: cluster_type = cwe.
    If LANG_CLUSTER is True:  cluster_type = language_to_index[language].
    """
    all_data = train_data + val_data + test_data

    if not enable_lang_cluster:
        for item in all_data:
            item["cluster_type"] = int(item["cwe"])
        print("LANG_CLUSTER=False -> cluster_type is copied from cwe")
        return train_data, val_data, test_data, None

    languages = sorted({get_language_label(item) for item in all_data})
    language_to_index = {language: idx for idx, language in enumerate(languages)}

    for item in all_data:
        language = get_language_label(item)
        item["cluster_type"] = language_to_index[language]

    print("LANG_CLUSTER=True -> cluster_type is remapped from language index")
    print(f"  Unique languages: {len(languages)}")
    print(f"  Language to index map: {language_to_index}")

    return train_data, val_data, test_data, language_to_index


class VulnerabilityDataset(Dataset):
    def __init__(self, data_list, encoder=None, embeddings=None):
        self.data = data_list
        self.encoder = encoder
        self.embeddings = embeddings
        self.use_cache = embeddings is not None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Return raw code strings (not embeddings) for on-the-fly encoding
        code_str = item["code"]
        vuln_label = torch.tensor([item["vuln"]], dtype=torch.float32)
        # print(f"Item {idx} - CWE: {item['cwe']}, type: {type(item['cwe'])}")
        cwe_label = torch.tensor(item["cwe"], dtype=torch.long)
        cluster_type_label = torch.tensor(item.get("cluster_type", item["cwe"]), dtype=torch.long)
        language_label = get_language_label(item)
        return code_str, vuln_label, cwe_label, cluster_type_label, language_label

# =====================================================================
# STRATIFIED DATA SPLIT - Maintain class balance across train/val/test
# =====================================================================
def build_stratified_data_split(data_list, seed=42, train_ratio=0.7, val_ratio=0.15):
    """Split data into stratified train/val/test subsets maintaining class balance.
    
    Args:
        data_list: List of data dictionaries with 'vuln' key
        seed: Random seed for reproducibility
        train_ratio: Proportion of data for training (default 0.7 = 70%)
        val_ratio: Proportion of data for validation (default 0.15 = 15%)
        test_ratio: Proportion of data for testing (default 0.15 = 15%)
    
    Returns:
        train_data, val_data, test_data: Stratified subsets of data
    """
    labels = [int(item['vuln']) for item in data_list]
    indices = np.arange(len(labels))
    
    # First split: 70% train, 30% temp
    train_idx, temp_idx, train_y, temp_y = train_test_split(
        indices,
        labels,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=labels
    )
    
    # Second split: 50/50 of temp for val/test (15/15 of total)
    val_ratio_in_temp = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio_in_temp),
        random_state=seed,
        stratify=temp_y
    )
    
    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    test_data = [data_list[i] for i in test_idx]
    
    # Log class distribution
    train_vuln = sum(1 for item in train_data if item['vuln'] == 1)
    val_vuln = sum(1 for item in val_data if item['vuln'] == 1)
    test_vuln = sum(1 for item in test_data if item['vuln'] == 1)
    
    print(f"\nStratified Data Split (seed={seed}):")
    print(f"  Train: {len(train_data)} samples ({train_vuln} vulnerable, {len(train_data)-train_vuln} safe)")
    print(f"  Val:   {len(val_data)} samples ({val_vuln} vulnerable, {len(val_data)-val_vuln} safe)")
    print(f"  Test:  {len(test_data)} samples ({test_vuln} vulnerable, {len(test_data)-test_vuln} safe)")
    print(f"  Train vuln%: {100*train_vuln/len(train_data):.1f}% | Val vuln%: {100*val_vuln/len(val_data):.1f}% | Test vuln%: {100*test_vuln/len(test_data):.1f}%")
    
    return train_data, val_data, test_data

# =====================================================================
# THÀNH PHẦN 6: QUY TRÌNH TRAIN - VALIDATE - TEST (OPTIMIZED)
# =====================================================================
def run_pipeline():
    train_data = load_raw_data(TRAIN_DATA_PATH)
    val_data = load_raw_data(VAL_DATA_PATH)
    test_data = load_raw_data(TEST_DATA_PATH)

    if not train_data or not val_data or not test_data:
        print("Error: Failed to load one or more dataset files from config paths.")
        print(f"  TRAIN_DATA_PATH: {TRAIN_DATA_PATH}")
        print(f"  VAL_DATA_PATH: {VAL_DATA_PATH}")
        print(f"  TEST_DATA_PATH: {TEST_DATA_PATH}")
        return

    train_data, val_data, test_data, language_to_index = remap_type_index(
        train_data,
        val_data,
        test_data,
        enable_lang_cluster=LANG_CLUSTER,
    )

    # 1. Khởi tạo thiết bị (GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"TRAINING DEVICE: {device}")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Training samples: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # 2. Analyze raw CWE space and build automatic clusters (from full data)
    print("\nAnalyzing cluster_type distribution...")
    all_data_for_cwe = train_data + val_data + test_data
    raw_num_classes, cwe_ids, negative_cwes_set = get_num_classes_cwe(all_data_for_cwe, label_field='cluster_type')
    num_experts, cwe_to_cluster, cwe_counter, cluster_loads = build_auto_cwe_clusters(
        all_data_for_cwe,
        target_experts=None,
        min_experts=4,
        max_experts=32,
        min_samples_threshold=20,  # Merge low-frequency labels into rare cluster
        label_field='cluster_type'
    )
    num_experts = max(num_experts, 3)
    print(f"Setting num_experts (clustered) = {num_experts}")

    cwe_dpy_set = {
        int(item["cwe"]) for item in all_data_for_cwe
        if get_language_label(item).strip().lower() == "python"
    }
    print(f"Python CWE set size (cweDpy): {len(cwe_dpy_set)}")
    
    
    
    # Validate cluster_type values and cluster mapping coverage
    print("\nValidating cluster_type indices and clusters...")
    all_cwes = []
    for item in train_data + val_data + test_data:
        all_cwes.append(item['cluster_type'])
    
    all_cwes_set = set(all_cwes)
    positive_cwes = {c for c in all_cwes_set if c >= 0}
    negative_cwes_in_data = {c for c in all_cwes_set if c < 0}
    
    min_cwe = min(all_cwes)
    max_cwe = max(all_cwes)
    
    # Validate positive CWEs are in valid range
    invalid_positive_cwes = [c for c in positive_cwes if c >= raw_num_classes] if raw_num_classes > 0 else []
    # Negative CWEs are always valid (special values)
    missing_mapping_cwes = sorted(set(c for c in all_cwes if c not in cwe_to_cluster))
    all_cluster_labels = [cwe_to_cluster[c] for c in all_cwes if c in cwe_to_cluster]
    invalid_cluster_labels = [cl for cl in all_cluster_labels if cl < 0 or cl >= num_experts]
    
    print(f"  CWE range in data: [{min_cwe}, {max_cwe}]")
    print(f"  Positive CWE range: [0, {max(positive_cwes) if positive_cwes else 'N/A'}]")
    print(f"  Raw CWE range required: [0, {raw_num_classes-1}] (positive only)")
    print(f"  Negative CWE values (special): {sorted(list(negative_cwes_in_data))}")
    print(f"  Cluster label range required: [0, {num_experts-1}]")
    print(f"  Invalid positive CWEs found: {len(invalid_positive_cwes)}")
    print(f"  Missing cluster mappings: {len(missing_mapping_cwes)}")
    print(f"  Invalid cluster labels found: {len(invalid_cluster_labels)}")
    
    if invalid_positive_cwes:
        print(f"    ERROR: Found {len(invalid_positive_cwes)} positive CWE values outside valid range!")
        print(f"    Invalid CWE IDs: {sorted(set(invalid_positive_cwes))}")
        raise ValueError(f"Positive CWE indices must be in [0, {raw_num_classes}). Found invalid: {invalid_positive_cwes}")

    if missing_mapping_cwes:
        print(f"    ERROR: Missing cluster mapping for {len(missing_mapping_cwes)} CWE values!")
        print(f"    Missing CWE IDs: {missing_mapping_cwes[:10]}")
        raise ValueError("CWE-to-cluster mapping is incomplete")

    if invalid_cluster_labels:
        print(f"    ERROR: Found invalid cluster labels!")
        print(f"    Sample invalid cluster labels: {sorted(set(invalid_cluster_labels))[:10]}")
        raise ValueError(f"Cluster labels must be in [0, {num_experts})")
    
    print(f"    All CWE indices are valid (including {len(negative_cwes_in_data)} special negative CWE values)!")
    print("")
    
    # 3. Calculate class weights for imbalanced data
    pos_weight = calculate_class_weights(train_data)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    
    # 4. No precomputation - we'll encode on-the-fly for fine-tuning
    print("\n" + "="*60)
    print("Using on-the-fly encoding (fine-tuning enabled)")
    print("="*60)
    
    # 5. Tạo DataLoaders WITHOUT cached embeddings
    train_dataset = VulnerabilityDataset(train_data)
    val_dataset = VulnerabilityDataset(val_data)
    test_dataset = VulnerabilityDataset(test_data)
    
    # Create WeightedRandomSampler for class balance (like baseline)
    train_labels = [int(item['vuln']) for item in train_data]
    train_class_counts = np.bincount(train_labels, minlength=2)
    class_weights_sampling = 1.0 / np.maximum(train_class_counts, 1)
    sample_weights = [class_weights_sampling[label] for label in train_labels]
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"\nWeighted Sampling (like baseline):")
    print(f"  Train class counts (safe/vuln): {train_class_counts[0]}/{train_class_counts[1]}")
    print(f"  Class weights for sampling: {class_weights_sampling}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 6. Khởi tạo Model với encoder (fine-tunable)
    model = MoE_VulnerabilityDetector(
        input_dim=768, 
        hidden_dim=256, 
        num_experts=num_experts, 
        top_k=1, 
        dropout_rate=0.1,
        encoder_name="microsoft/codebert-base",
        freeze_encoder=False  # CRITICAL: Allow fine-tuning
    ).to(device)
    
    # CRITICAL FIX: Lower learning rate like baseline (2e-5 instead of 2e-3)
    # High lr (2e-3) was causing instability and poor performance
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
    
    # Adjust epochs and scheduler from config
    epochs = EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0
    best_model_state = None
    monitor_name = "F1" if F1_BEST_STATE else "Accuracy"
    best_monitor_value = 0.0
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} (trainable: {trainable_params:,})")
    print(f"Learning rate: 2e-5 (same as baseline)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {epochs}")
    print(f"Patience: {patience}")
    print(f"Best state criterion: Validation {monitor_name}")
    
    print(f"\n{'='*60}")
    print(f"BẮT ĐẦU HUẤN LUYỆN KIẾN TRÚC MoE (OPTIMIZED)")
    print(f"{'='*60}")
    
    for epoch in build_tqdm(range(epochs), desc="Training epochs", unit="epoch", leave=True):
        # --- TRAIN LOOP ---
        model.train()
        train_loss, train_acc = 0.0, 0.0
        train_bce, train_ce, train_aux = 0.0, 0.0, 0.0

        train_pbar = build_tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            unit="batch",
            leave=False,
        )
        for batch_idx, (code_batch, vuln, cwe, cluster_type, languages) in enumerate(train_pbar, start=1):
            # Encode code on-the-fly (allows gradient flow to encoder)
            emb, input_ids, attention_mask = model.encode_code(code_batch, device)
            vuln, cwe, cluster_type = vuln.to(device), cwe.to(device), cluster_type.to(device)
            cwe_cluster = torch.tensor(
                [cwe_to_cluster[int(c.item())] for c in cluster_type],
                dtype=torch.long,
                device=device
            )
            optimizer.zero_grad()

            logits, frac_routed, prob_exp, router_logits, aux_loss_expert = model(
                emb,
                languages=languages,
                vuln_labels=vuln,
                cwe_labels=cwe,
                cwe_dpy_set=cwe_dpy_set,
                apply_special_routing=True,
                raw_input_ids=input_ids,
                raw_attention_mask=attention_mask,
            )
            loss, bce, ce, aux = moe_multitask_loss(logits, vuln, router_logits, cwe_cluster,
                                                     frac_routed, prob_exp, num_experts=num_experts, 
                                                     pos_weight=pos_weight, use_focal=True,
                                                     label_smoothing=0.05,
                                                     alpha=0.005, beta=0.2, expert_aux_loss=aux_loss_expert)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_bce += bce.item()
            train_ce += ce.item()
            train_aux += aux.item()
            batch_acc = calculate_accuracy(logits, vuln).item()
            train_acc += batch_acc

            train_pbar.set_postfix({
                "loss": f"{train_loss / batch_idx:.4f}",
                "acc": f"{train_acc / batch_idx:.4f}",
            })

            # Heartbeat log for screen/ssh sessions where tqdm refresh can look frozen.
            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} - batch {batch_idx}/{len(train_loader)} "
                    f"| loss={train_loss / batch_idx:.4f} | acc={train_acc / batch_idx:.4f}"
                )

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        
        with torch.no_grad():
            for code_batch, vuln, cwe, cluster_type, languages in val_loader:
                # Encode code on-the-fly (no grad in eval)
                emb, input_ids, attention_mask = model.encode_code(code_batch, device)
                vuln, cwe, cluster_type = vuln.to(device), cwe.to(device), cluster_type.to(device)
                cwe_cluster = torch.tensor(
                    [cwe_to_cluster[int(c.item())] for c in cluster_type],
                    dtype=torch.long,
                    device=device
                )
                logits, frac_routed, prob_exp, router_logits, aux_loss_expert = model(
                    emb,
                    languages=languages,
                    vuln_labels=vuln,
                    cwe_labels=cwe,
                    cwe_dpy_set=cwe_dpy_set,
                    apply_special_routing=True,
                    raw_input_ids=input_ids,
                    raw_attention_mask=attention_mask,
                )
                loss, _, _, _ = moe_multitask_loss(logits, vuln, router_logits, cwe_cluster,
                                                    frac_routed, prob_exp, num_experts=num_experts,
                                                    pos_weight=pos_weight, use_focal=True,
                                                    label_smoothing=0.05,
                                                    alpha=0.005, beta=0.2, expert_aux_loss=aux_loss_expert)

                val_loss += loss.item()
                val_acc += calculate_accuracy(logits, vuln).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # Calculate validation F1 for better model selection (like baseline)
        model.eval()
        val_preds_all = []
        val_labels_all = []
        with torch.no_grad():
            for code_batch, vuln, cwe, cluster_type, languages in val_loader:
                emb, input_ids, attention_mask = model.encode_code(code_batch, device)
                logits, _ = get_language_policy_logits(
                    model,
                    emb,
                    languages,
                    raw_input_ids=input_ids,
                    raw_attention_mask=attention_mask,
                )
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_preds_all.extend(preds.cpu().numpy().flatten())
                val_labels_all.extend(vuln.numpy().flatten())
        
        val_preds_tensor = torch.tensor(val_preds_all)
        val_labels_tensor = torch.tensor(val_labels_all)
        val_f1, val_precision, val_recall = calculate_f1_score(val_preds_tensor, val_labels_tensor)
        
        # Learning Rate Schedule
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:3d}/{epochs} | LR: {current_lr:.6f} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f} F1: {val_f1:.4f}")
        
        # Select monitoring metric for best checkpoint and early stopping
        monitor_value = val_f1 if F1_BEST_STATE else avg_val_acc

        if monitor_value > best_monitor_value:
            best_monitor_value = monitor_value
            best_val_f1 = val_f1
            best_val_acc = avg_val_acc
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            model_save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_epoch_{epoch+1}.pt")
            torch.save(best_model_state, model_save_path)
            print(
                f"  ✓ New best {monitor_name}: {monitor_value:.4f} "
                f"(Val F1: {val_f1:.4f}, Val Acc: {avg_val_acc:.4f}, "
                f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\nEarly stopping triggered! Best Val {monitor_name}: {best_monitor_value:.4f}, "
                    f"F1: {best_val_f1:.4f}, Acc: {best_val_acc:.4f}, Loss: {best_val_loss:.4f}"
                )
                model.load_state_dict(best_model_state)
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Find optimal threshold on validation set
    print("\n" + "="*60)
    print("Finding optimal classification threshold...")
    print("="*60)
    optimal_threshold, metrics = find_optimal_threshold(model, val_loader, device)
    print("="*60)

    # --- TEST LOOP ---
    print("\n" + "="*60)
    print("ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ (TEST SET)")
    print("="*60)
    model.eval()
    
    # Collect all predictions and probabilities with INDIVIDUAL EXPERT OUTPUTS
    test_probabilities = []  # Final policy predictions
    test_labels = []
    test_cwe_labels = []
    test_language_labels = []
    test_routed_clusters = []  
    test_codes = []  
    test_cluster_types = []
    test_vulpy_probabilities = []  # VulPy reference for Python
    test_ccpp_probabilities = []  # Expert CCPP for CCPP samples
    test_python_probabilities = []  # Expert Python for Python samples
    
    with torch.no_grad():
        for code_batch, vuln, cwe, cluster_type, languages in test_loader:
            emb, input_ids, attention_mask = model.encode_code(code_batch, device)
            vuln, cwe = vuln.to(device), cwe.to(device)
            
            # Get final policy logits (language-specific expert selection)
            logits, vulpy_logits = get_language_policy_logits(
                model,
                emb,
                languages,
                raw_input_ids=input_ids,
                raw_attention_mask=attention_mask,
            )
            
            # Get individual expert logits for expert-specific outputs
            expert_logits_all = model.get_expert_logits(
                emb,
                languages=languages,
                raw_input_ids=input_ids,
                raw_attention_mask=attention_mask,
            )
            # expert_logits_all has shape (batch_size, num_experts)
            # Expert indices: 0=CCPP, 1=Python, 2=VulPy
            
            _, _, _, router_logits, _ = model(
                emb,
                raw_input_ids=input_ids,
                raw_attention_mask=attention_mask,
            )  # unpack aux_loss
            probs = torch.sigmoid(logits)
            vulpy_probs = torch.sigmoid(vulpy_logits)
            ccpp_probs = torch.sigmoid(expert_logits_all[:, 0])  # Expert CCPP
            python_probs = torch.sigmoid(expert_logits_all[:, 1])  # Expert Python
            
            # Extract routing decisions: argmax of router_logits gives dominant cluster for each sample
            routed_clusters = torch.argmax(router_logits, dim=1).cpu().numpy().tolist()
            
            # Collect all data
            test_probabilities.extend(probs.cpu().numpy().flatten().tolist())
            test_labels.extend(vuln.cpu().numpy().flatten().tolist())
            test_cwe_labels.extend(cwe.cpu().numpy().tolist())
            test_language_labels.extend(list(languages))
            test_routed_clusters.extend(routed_clusters)
            test_codes.extend(code_batch)  
            test_cluster_types.extend(cluster_type.cpu().numpy().tolist())
            test_vulpy_probabilities.extend(vulpy_probs.cpu().numpy().flatten().tolist())
            test_ccpp_probabilities.extend(ccpp_probs.cpu().numpy().flatten().tolist())
            test_python_probabilities.extend(python_probs.cpu().numpy().flatten().tolist())
    


    # Convert to numpy for easier computation
    test_probs_np = np.array(test_probabilities)
    test_labels_np = np.array(test_labels)
    
    # Calculate metrics for BOTH thresholds
    # Threshold 0.5 (default)
    preds_default = (test_probs_np >= 0.5).astype(int)
    acc_default = (preds_default == test_labels_np).mean()
    
    # Calculate F1, precision, recall for default threshold
    preds_default_tensor = torch.tensor(preds_default, dtype=torch.float32)
    labels_tensor = torch.tensor(test_labels_np, dtype=torch.float32)
    f1_default, precision_default, recall_default = calculate_f1_score(preds_default_tensor, labels_tensor)
    
    # Optimal threshold
    preds_optimal = (test_probs_np >= optimal_threshold).astype(int)
    acc_optimal = (preds_optimal == test_labels_np).mean()
    
    # Calculate F1, precision, recall for optimal threshold
    preds_optimal_tensor = torch.tensor(preds_optimal, dtype=torch.float32)
    f1_optimal, precision_optimal, recall_optimal = calculate_f1_score(preds_optimal_tensor, labels_tensor)
    
    # Use optimal threshold predictions for detailed logging
    test_predictions = preds_optimal.tolist()
    
    # Log test predictions to file with timestamp
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    debug_info_file = results_dir / f"test_debug_info_{timestamp}.csv"
    cluster_map_file = results_dir / f"cwe_cluster_map_{timestamp}.json"
    
    debug_info = {
        "cluster_types": test_cluster_types,
        "routed_clusters": test_routed_clusters,
        "cwe_labels": test_cwe_labels,
        "language_labels": test_language_labels,
        "probabilities": test_probabilities,
        "codes": test_codes
    }
    debug_info_df = pd.DataFrame(debug_info)
    debug_info_df.to_csv(debug_info_file, index=False)
    print(f"Detailed test debug info saved to {debug_info_file}")

    # Identify rare CWEs
    rare_cwes_info = {str(cwe_id): {"freq": int(freq), "cluster": int(cwe_to_cluster[cwe_id])} 
                      for cwe_id, freq in cwe_counter.items() if freq <= 20}
    
    
    with open(cluster_map_file, "w") as f:
        json.dump(
            {
                "raw_num_classes": int(raw_num_classes),
                "num_experts": int(num_experts),
                "min_samples_threshold": 20,
                "cluster_loads": [int(x) for x in cluster_loads],
                "cwe_to_cluster": {str(k): int(v) for k, v in sorted(cwe_to_cluster.items())},
                "cwe_frequency": {str(k): int(v) for k, v in sorted(cwe_counter.items())},
                "rare_cwes_merged": rare_cwes_info,
                "num_rare_cwes": len(rare_cwes_info)
            },
            f,
            indent=2
        )
    print(f"✓ CWE cluster mapping saved to {cluster_map_file}")
    print(f"  (Merged {len(rare_cwes_info)} CWEs with <= 20 samples into rare cluster)")
    
    # LANGUAGE-SPECIFIC PREDICTION FILES
    # Separate CCPP and Python samples
    ccpp_indices = [i for i in range(len(test_language_labels)) if str(test_language_labels[i]).strip().lower() == "ccpp"]
    python_indices = [i for i in range(len(test_language_labels)) if str(test_language_labels[i]).strip().lower() == "python"]
    
    # Save CCPP predictions (using Expert CCPP only)
    pred_file_ccpp = results_dir / f"prediction_file_ccpp_{timestamp}.jsonl"
    print(f"\nSaving CCPP predictions (Expert CCPP) to {pred_file_ccpp}...")
    with open(pred_file_ccpp, "w") as f:
        for idx in ccpp_indices:
            ccpp_pred = (np.array(test_ccpp_probabilities[idx]) >= optimal_threshold).astype(int)
            pred_data = {
                "sample_id": idx,
                "predicted_label": int(ccpp_pred),
                "real_label": int(test_labels[idx]),
                "probability": float(test_ccpp_probabilities[idx]),
                "expert": "Expert CCPP",
                "cwe_id": int(test_cwe_labels[idx]),
                "language": "ccpp",
                "is_correct": int(ccpp_pred) == int(test_labels[idx])
            }
            f.write(json.dumps(pred_data) + "\n")
    print(f"✓ CCPP predictions (Expert CCPP) saved to {pred_file_ccpp}")
    
    # Save Python predictions (using Expert Python only)
    pred_file_py = results_dir / f"prediction_file_py_{timestamp}.jsonl"
    print(f"Saving Python predictions (Expert Python) to {pred_file_py}...")
    with open(pred_file_py, "w") as f:
        for idx in python_indices:
            python_pred = (np.array(test_python_probabilities[idx]) >= optimal_threshold).astype(int)
            pred_data = {
                "sample_id": idx,
                "predicted_label": int(python_pred),
                "real_label": int(test_labels[idx]),
                "probability": float(test_python_probabilities[idx]),
                "expert": "Expert Python",
                "cwe_id": int(test_cwe_labels[idx]),
                "language": "python",
                "is_correct": int(python_pred) == int(test_labels[idx])
            }
            f.write(json.dumps(pred_data) + "\n")
    print(f"✓ Python predictions (Expert Python) saved to {pred_file_py}")
    
    # Save VulPy reference predictions for Python (using Expert VulPy only)
    pred_file_vulpy = results_dir / f"prediction_file_vulpy_{timestamp}.jsonl"
    print(f"Saving VulPy reference predictions to {pred_file_vulpy}...")
    with open(pred_file_vulpy, "w") as f:
        for idx in python_indices:
            vulpy_pred = (np.array(test_vulpy_probabilities[idx]) >= optimal_threshold).astype(int)
            pred_data = {
                "sample_id": idx,
                "predicted_label": int(vulpy_pred),
                "real_label": int(test_labels[idx]),
                "probability": float(test_vulpy_probabilities[idx]),
                "expert": "Expert VulPy",
                "cwe_id": int(test_cwe_labels[idx]),
                "language": "python",
                "is_correct": int(vulpy_pred) == int(test_labels[idx])
            }
            f.write(json.dumps(pred_data) + "\n")
    print(f"✓ VulPy reference predictions saved to {pred_file_vulpy}")
    
    # Keep original combined prediction file for compatibility
    pred_file = results_dir / f"test_predictions_{timestamp}.jsonl"
    print(f"Saving combined test predictions to {pred_file}...")
    with open(pred_file, "w") as f:
        for i in range(len(test_predictions)):
            pred_data = {
                "sample_id": i,
                "predicted_label": int(test_predictions[i]),
                "real_label": int(test_labels[i]),
                "probability": float(test_probabilities[i]),
                "cwe_id": int(test_cwe_labels[i]),
                "language": str(test_language_labels[i]),
                "is_correct": int(test_predictions[i]) == int(test_labels[i])
            }
            f.write(json.dumps(pred_data) + "\n")
    print(f"✓ Combined test predictions saved to {pred_file}")
    
    # Save summary report
    report_file = results_dir / f"test_report_{timestamp}.txt"
    with open(report_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("TEST PREDICTIONS SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Total test samples: {len(test_predictions)}\n")
        f.write(f"Correct predictions: {sum(test_predictions[i] == test_labels[i] for i in range(len(test_predictions)))}\n")
        f.write(f"Accuracy: {sum(test_predictions[i] == test_labels[i] for i in range(len(test_predictions))) / len(test_predictions) * 100:.2f}%\n")
        f.write("\nFirst 10 predictions:\n")
        f.write(f"{'ID':<5} {'Pred':<6} {'Real':<6} {'Prob':<10} {'CWE':<5} {'Lang':<10} {'OK':<5}\n")
        f.write("-"*77 + "\n")
        for i in range(min(10, len(test_predictions))):
            is_correct = "✓" if test_predictions[i] == test_labels[i] else "✗"
            f.write(f"{i:<5} {int(test_predictions[i]):<6} {int(test_labels[i]):<6} "
                   f"{test_probabilities[i]:<10.4f} {int(test_cwe_labels[i]):<5} {str(test_language_labels[i]):<10} {is_correct:<5}\n")
    
    print(f"✓ Report saved to {report_file}\n")

    prediction_file = results_dir / "predictions.csv"
    prediction_df = pd.DataFrame(
        {
            "code": test_codes,
            "predicted_label": preds_optimal.astype(int),
            "true_label": test_labels_np.astype(int),
            "language": test_language_labels,
            "probability": test_probs_np,
        }
    )
    prediction_df.to_csv(prediction_file, index=False)
    print(f"✓ Instruction prediction file saved to {prediction_file}")

    python_mask = np.array([str(lang).strip().lower() == "python" for lang in test_language_labels])
    ccpp_mask = np.array([str(lang).strip().lower() == "ccpp" for lang in test_language_labels])

    # LANGUAGE-SPECIFIC RESULT FILES
    # Convert expert probabilities to numpy for easier filtering
    ccpp_probs_np = np.array(test_ccpp_probabilities)
    python_probs_np = np.array(test_python_probabilities)
    vulpy_probs_np = np.array(test_vulpy_probabilities)
    
    # Generate CCPP result file (Expert CCPP predictions only)
    result_file_ccpp = results_dir / f"result_file_ccpp_{timestamp}.txt"
    print(f"Generating CCPP result file (Expert CCPP) to {result_file_ccpp}...")
    if ccpp_indices:
        ccpp_labels = test_labels_np[ccpp_indices].astype(int)
        ccpp_preds = (ccpp_probs_np[ccpp_indices] >= optimal_threshold).astype(int)
        
        macro_precision_ccpp, macro_recall_ccpp, macro_f1_ccpp, _ = precision_recall_fscore_support(
            ccpp_labels, ccpp_preds, average="macro", zero_division=0)
        overall_accuracy_ccpp = accuracy_score(ccpp_labels, ccpp_preds)
        
        with open(result_file_ccpp, "w", encoding="utf-8") as f:
            f.write("MoE Inference Result Summary - CCPP Only (Expert CCPP)\n")
            f.write("=" * 60 + "\n")
            f.write(f"macro f1: {macro_f1_ccpp:.6f}\n")
            f.write(f"macro precision: {macro_precision_ccpp:.6f}\n")
            f.write(f"accuracy: {overall_accuracy_ccpp:.6f}\n")
            f.write(f"macro recall: {macro_recall_ccpp:.6f}\n\n")
            f.write("Classification report (CCPP Expert CCPP)\n")
            f.write(classification_report(ccpp_labels, ccpp_preds, digits=6, zero_division=0))
        print(f"✓ CCPP result file (Expert CCPP) saved to {result_file_ccpp}")
    else:
        print(f"⚠ No CCPP samples found, skipping CCPP result file")
    
    # Generate Python result file (Expert Python predictions only)
    result_file_py = results_dir / f"result_file_py_{timestamp}.txt"
    print(f"Generating Python result file (Expert Python) to {result_file_py}...")
    if python_indices:
        python_labels = test_labels_np[python_indices].astype(int)
        python_preds = (python_probs_np[python_indices] >= optimal_threshold).astype(int)
        
        macro_precision_py, macro_recall_py, macro_f1_py, _ = precision_recall_fscore_support(
            python_labels, python_preds, average="macro", zero_division=0)
        overall_accuracy_py = accuracy_score(python_labels, python_preds)
        
        with open(result_file_py, "w", encoding="utf-8") as f:
            f.write("MoE Inference Result Summary - Python Only (Expert Python)\n")
            f.write("=" * 60 + "\n")
            f.write(f"macro f1: {macro_f1_py:.6f}\n")
            f.write(f"macro precision: {macro_precision_py:.6f}\n")
            f.write(f"accuracy: {overall_accuracy_py:.6f}\n")
            f.write(f"macro recall: {macro_recall_py:.6f}\n\n")
            f.write("Classification report (Python Expert Python)\n")
            f.write(classification_report(python_labels, python_preds, digits=6, zero_division=0))
        print(f"✓ Python result file (Expert Python) saved to {result_file_py}")
    else:
        print(f"⚠ No Python samples found, skipping Python result file")
    
    # Generate VulPy reference result file (Expert VulPy predictions for Python only)
    result_file_vulpy = results_dir / f"result_file_vulpy_{timestamp}.txt"
    print(f"Generating VulPy reference result file to {result_file_vulpy}...")
    if python_indices:
        vulpy_labels = test_labels_np[python_indices].astype(int)
        vulpy_preds = (vulpy_probs_np[python_indices] >= optimal_threshold).astype(int)
        
        macro_precision_vulpy, macro_recall_vulpy, macro_f1_vulpy, _ = precision_recall_fscore_support(
            vulpy_labels, vulpy_preds, average="macro", zero_division=0)
        overall_accuracy_vulpy = accuracy_score(vulpy_labels, vulpy_preds)
        
        with open(result_file_vulpy, "w", encoding="utf-8") as f:
            f.write("MoE Inference Result Summary - Python Only (Expert VulPy Reference)\n")
            f.write("=" * 60 + "\n")
            f.write(f"macro f1: {macro_f1_vulpy:.6f}\n")
            f.write(f"macro precision: {macro_precision_vulpy:.6f}\n")
            f.write(f"accuracy: {overall_accuracy_vulpy:.6f}\n")
            f.write(f"macro recall: {macro_recall_vulpy:.6f}\n\n")
            f.write("Classification report (Python Expert VulPy Reference)\n")
            f.write(classification_report(vulpy_labels, vulpy_preds, digits=6, zero_division=0))
        print(f"✓ VulPy reference result file saved to {result_file_vulpy}")
    else:
        print(f"⚠ No Python samples found, skipping VulPy reference result file")
    
    # Keep original combined result file for compatibility
    result_file = results_dir / "test_result.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            test_labels_np.astype(int),
            preds_optimal.astype(int),
            average="macro",
            zero_division=0,
        )
        overall_accuracy = accuracy_score(test_labels_np.astype(int), preds_optimal.astype(int))

        f.write("MoE Inference Result Summary (Combined Language Policy)\n")
        f.write("=" * 60 + "\n")
        f.write(f"macro f1: {macro_f1:.6f}\n")
        f.write(f"macro precision: {macro_precision:.6f}\n")
        f.write(f"accuracy: {overall_accuracy:.6f}\n")
        f.write(f"macro recall: {macro_recall:.6f}\n\n")

        f.write("Classification report (final prediction policy)\n")
        f.write(classification_report(test_labels_np.astype(int), preds_optimal.astype(int), digits=6, zero_division=0))
        f.write("\n")

        if ccpp_mask.any():
            f.write("\nCCPP only (Expert CCPP as final)\n")
            f.write("-" * 60 + "\n")
            f.write(classification_report(
                test_labels_np[ccpp_mask].astype(int),
                preds_optimal[ccpp_mask].astype(int),
                digits=6,
                zero_division=0,
            ))

        if python_mask.any():
            f.write("\nPython only (Expert Python as final)\n")
            f.write("-" * 60 + "\n")
            f.write(classification_report(
                test_labels_np[python_mask].astype(int),
                preds_optimal[python_mask].astype(int),
                digits=6,
                zero_division=0,
            ))

            vulpy_probs_py = vulpy_probs_np[python_mask]
            valid_mask = ~np.isnan(vulpy_probs_py)
            if valid_mask.any():
                vulpy_preds = (vulpy_probs_py[valid_mask] >= optimal_threshold).astype(int)
                vulpy_labels = test_labels_np[python_mask][valid_mask].astype(int)
                f.write("\nPython only (Expert VulPy reference output)\n")
                f.write("-" * 60 + "\n")
                f.write(classification_report(vulpy_labels, vulpy_preds, digits=6, zero_division=0))

    print(f"✓ Combined result file saved to {result_file}")
    
    # Print sample predictions
    print("Sample Test Predictions (first 10):")
    print(f"{'Sample':<8} {'Predicted':<12} {'Real':<8} {'Probability':<12} {'CWE':<6} {'Lang':<10} {'Correct':<9}")
    print("-" * 77)
    for i in range(min(10, len(test_predictions))):
        is_correct = "✓" if test_predictions[i] == test_labels[i] else "✗"
        print(f"{i:<8} {int(test_predictions[i]):<12} {int(test_labels[i]):<8} "
              f"{test_probabilities[i]:<12.4f} {int(test_cwe_labels[i]):<6} {str(test_language_labels[i]):<10} {is_correct:<9}")

    # Print comprehensive metrics comparison
    print("\n" + "="*80)
    print("TEST METRICS COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<20} {'Threshold=0.5':<20} {'Threshold={:.3f}':<20} {'Diff':<15}".format(optimal_threshold))
    print("-"*80)
    
    acc_diff = (acc_optimal - acc_default) * 100
    precision_diff = (precision_optimal - precision_default) * 100
    recall_diff = (recall_optimal - recall_default) * 100
    f1_diff = (f1_optimal - f1_default) * 100
    
    print(f"{'Accuracy':<20} {acc_default*100:<19.2f}% {acc_optimal*100:<19.2f}% {acc_diff:+.2f}%")
    print(f"{'Precision':<20} {precision_default*100:<19.2f}% {precision_optimal*100:<19.2f}% {precision_diff:+.2f}%")
    print(f"{'Recall':<20} {recall_default*100:<19.2f}% {recall_optimal*100:<19.2f}% {recall_diff:+.2f}%")
    print(f"{'F1 Score':<20} {f1_default*100:<19.2f}% {f1_optimal*100:<19.2f}% {f1_diff:+.2f}%")
    print("-"*80)
    
    if f1_optimal > f1_default:
        print(f"✓ Optimal threshold IMPROVES F1 by {f1_diff:+.2f}%")
    elif f1_optimal < f1_default:
        print(f"✗ Optimal threshold DEGRADES F1 by {f1_diff:.2f}% (using default 0.5 is better)")
    else:
        print(f"= Optimal threshold has SAME F1 as default")
    print("="*80)
    
    # --- ROUTING ANALYSIS BY LANGUAGE ---
    print("\n" + "="*80)
    print("CLUSTER ROUTING DISTRIBUTION BY LANGUAGE (C++ and Python)")
    print("="*80)
    
    # Analyze routing for C++ (ccpp) and Python
    cluster_details = {cluster_id: {"ccpp": 0, "python": 0} for cluster_id in range(max(3, num_experts))}

    if len(test_language_labels) != len(test_routed_clusters):
        print("Error: Mismatch in lengths of language labels and routed clusters!")
        print(f"  test_language_labels length: {len(test_language_labels)}")
        print(f"  test_routed_clusters length: {len(test_routed_clusters)}")
    else:
        for i in range(len(test_language_labels)):
            lang = test_language_labels[i]
            cluster = test_routed_clusters[i]
            if cluster not in cluster_details:
                cluster_details[cluster] = {"ccpp": 0, "python": 0}
            if str(lang).lower() == "ccpp":
                cluster_details[cluster]["ccpp"] += 1
            elif str(lang).lower() == "python":
                cluster_details[cluster]["python"] += 1
            


    for cluster_id, counts in cluster_details.items():
        ccpp_count = counts["ccpp"]
        python_count = counts["python"]
        lang_total = ccpp_count + python_count
        if lang_total == 0:
            continue  # Skip clusters with no samples
        
        
        # Calculate percentages and print
        print(f"\nCluster {cluster_id} Language Routing Distribution (Total: {lang_total} samples):")
        print(f"{'Cluster':<12} {'Count':<12} {'Percentage':<15}")
        print("-" * 40)
        
        
        percentage_ccpp = (ccpp_count / lang_total) * 100
        percentage_python = (python_count / lang_total) * 100
        print(f"cpp: {ccpp_count:<12} {percentage_ccpp:>6.2f}%")
        print(f"python: {python_count:<12} {percentage_python:>6.2f}%")

        print("-" * 40)
        print(f"{'Total':<12} {lang_total:<12} {100.0:>6.2f}%")
    
    print("="*80)
    
    # --- PER-CWE EVALUATION ---
    print("\n" + "="*60)
    print("ĐÁNH GIÁ THEO TỪNG CWE ID (with optimal threshold)")
    print("="*60)
    cwe_metrics = evaluate_by_cwe(model, test_loader, device, threshold=optimal_threshold)
    
    print(f"\n{'CWE ID':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Samples':<10}")
    print("-" * 70)
    
    for cwe_id in sorted(cwe_metrics.keys()):
        metrics = cwe_metrics[cwe_id]
        print(f"{cwe_id:<10} {metrics['accuracy']*100:<11.2f}% {metrics['precision']*100:<11.2f}% "
              f"{metrics['recall']*100:<11.2f}% {metrics['f1']*100:<11.2f}% {metrics['total']:<10}")
    
    print("-" * 70)
    overall_metrics = calculate_overall_metrics_from_groups(cwe_metrics)
    total_samples = overall_metrics["total"]
    overall_acc = overall_metrics["accuracy"]
    overall_precision = overall_metrics["precision"]
    overall_recall = overall_metrics["recall"]
    overall_f1 = overall_metrics["f1"]
    
    print(f"{'Overall':<10} {overall_acc*100:<11.2f}% {overall_precision*100:<11.2f}% "
          f"{overall_recall*100:<11.2f}% {overall_f1*100:<11.2f}% {total_samples:<10}")

    print("\n" + "="*60)
    print("TEST RESULTS BY LANGUAGE (with optimal threshold)")
    print("="*60)
    language_metrics = evaluate_by_language(model, test_loader, device, threshold=optimal_threshold)

    print(f"{'Language':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Samples':<10}")
    print("-" * 75)
    for language in sorted(language_metrics.keys()):
        metrics = language_metrics[language]
        print(f"{language:<15} {metrics['accuracy']*100:<11.2f}% {metrics['precision']*100:<11.2f}% "
              f"{metrics['recall']*100:<11.2f}% {metrics['f1']*100:<11.2f}% {metrics['total']:<10}")

    print("-" * 75)
    language_overall = calculate_overall_metrics_from_groups(language_metrics)
    print(f"{'Overall':<15} {language_overall['accuracy']*100:<11.2f}% {language_overall['precision']*100:<11.2f}% "
          f"{language_overall['recall']*100:<11.2f}% {language_overall['f1']*100:<11.2f}% {language_overall['total']:<10}")

    print("\n" + "="*60)
    print("TEST RESULTS BY LANGUAGE + CWE (with optimal threshold)")
    print("="*60)
    language_cwe_metrics = evaluate_by_language_cwe(model, test_loader, device, threshold=optimal_threshold)

    print(f"{'Language | CWE':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Samples':<10}")
    print("-" * 85)
    for group_name in sorted(language_cwe_metrics.keys()):
        metrics = language_cwe_metrics[group_name]
        print(f"{group_name:<25} {metrics['accuracy']*100:<11.2f}% {metrics['precision']*100:<11.2f}% "
              f"{metrics['recall']*100:<11.2f}% {metrics['f1']*100:<11.2f}% {metrics['total']:<10}")

    print("-" * 85)
    language_cwe_overall = calculate_overall_metrics_from_groups(language_cwe_metrics)
    print(f"{'Overall':<25} {language_cwe_overall['accuracy']*100:<11.2f}% {language_cwe_overall['precision']*100:<11.2f}% "
          f"{language_cwe_overall['recall']*100:<11.2f}% {language_cwe_overall['f1']*100:<11.2f}% {language_cwe_overall['total']:<10}")
    
    # Save comprehensive final report
    final_report_file = results_dir / f"final_report_{timestamp}.txt"
    with open(final_report_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("MOE VULNERABILITY DETECTION - FINAL TRAINING REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log file: {log_file}\n")
        f.write(f"Predictions file: {pred_file}\n")
        f.write(f"CWE cluster map: {cluster_map_file}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("DATASET SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Training samples: {len(train_data)}\n")
        f.write(f"Validation samples: {len(val_data)}\n")
        f.write(f"Test samples: {len(test_data)}\n")
        f.write(f"Total samples: {len(train_data) + len(val_data) + len(test_data)}\n")
        f.write(f"Raw CWE classes (full data max+1): {raw_num_classes}\n")
        f.write(f"Router experts after auto-clustering: {num_experts}\n")
        f.write(f"Cluster loads: {cluster_loads}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("="*70 + "\n")
        f.write(f"Input dimension: 768\n")
        f.write(f"Hidden dimension: 256\n")
        f.write(f"Number of experts: {num_experts}\n")
        f.write(f"Auto-clustering: enabled (min_samples_threshold=20)\n")
        f.write(f"Rare cluster: {len(rare_cwes_info)} CWE types merged (total {sum(c for c_id, c in cwe_counter.items() if c <= 20)} samples)\n")
        f.write(f"Top-k routing: 1\n")
        f.write(f"Dropout rate: 0.1\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("TEST RESULTS - THRESHOLD COMPARISON\n")
        f.write("="*70 + "\n")
        f.write(f"{'Metric':<20} {'Threshold=0.5':<20} {'Optimal={:.3f}':<20} {'Diff':<15}\n".format(optimal_threshold))
        f.write("-"*70 + "\n")
        f.write(f"{'Accuracy':<20} {acc_default*100:<19.2f}% {acc_optimal*100:<19.2f}% {acc_diff:+.2f}%\n")
        f.write(f"{'Precision':<20} {precision_default*100:<19.2f}% {precision_optimal*100:<19.2f}% {precision_diff:+.2f}%\n")
        f.write(f"{'Recall':<20} {recall_default*100:<19.2f}% {recall_optimal*100:<19.2f}% {recall_diff:+.2f}%\n")
        f.write(f"{'F1 Score':<20} {f1_default*100:<19.2f}% {f1_optimal*100:<19.2f}% {f1_diff:+.2f}%\n")
        f.write("-"*70 + "\n")
        if f1_optimal > f1_default:
            f.write(f"✓ Optimal threshold IMPROVES F1 by {f1_diff:+.2f}%\n")
        elif f1_optimal < f1_default:
            f.write(f"✗ Optimal threshold DEGRADES F1 by {f1_diff:.2f}% (default is better)\n")
        else:
            f.write(f"= Optimal threshold has SAME F1 as default\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("="*70 + "\n")
        f.write(f"Overall Accuracy: {overall_acc*100:.2f}%\n")
        f.write(f"Overall Precision: {overall_precision*100:.2f}%\n")
        f.write(f"Overall Recall: {overall_recall*100:.2f}%\n")
        f.write(f"Overall F1 Score: {overall_f1*100:.2f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("PER-CWE EVALUATION RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"{'CWE ID':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Samples':<10}\n")
        f.write("-" * 70 + "\n")
        
        for cwe_id in sorted(cwe_metrics.keys()):
            metrics = cwe_metrics[cwe_id]
            f.write(f"{cwe_id:<10} {metrics['accuracy']*100:<11.2f}% {metrics['precision']*100:<11.2f}% "
                   f"{metrics['recall']*100:<11.2f}% {metrics['f1']*100:<11.2f}% {metrics['total']:<10}\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"{'Overall':<10} {overall_acc*100:<11.2f}% {overall_precision*100:<11.2f}% "
               f"{overall_recall*100:<11.2f}% {overall_f1*100:<11.2f}% {total_samples:<10}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("TEST RESULTS BY LANGUAGE\n")
        f.write("="*70 + "\n")
        f.write(f"{'Language':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Samples':<10}\n")
        f.write("-" * 75 + "\n")
        for language in sorted(language_metrics.keys()):
            metrics = language_metrics[language]
            f.write(f"{language:<15} {metrics['accuracy']*100:<11.2f}% {metrics['precision']*100:<11.2f}% "
                   f"{metrics['recall']*100:<11.2f}% {metrics['f1']*100:<11.2f}% {metrics['total']:<10}\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Overall':<15} {language_overall['accuracy']*100:<11.2f}% {language_overall['precision']*100:<11.2f}% "
               f"{language_overall['recall']*100:<11.2f}% {language_overall['f1']*100:<11.2f}% {language_overall['total']:<10}\n")

        # f.write("\n" + "="*70 + "\n")
        # f.write("TEST RESULTS BY LANGUAGE + CWE\n")
        # f.write("="*70 + "\n")
        # f.write(f"{'Language | CWE':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Samples':<10}\n")
        # f.write("-" * 85 + "\n")
        # for group_name in sorted(language_cwe_metrics.keys()):
        #     metrics = language_cwe_metrics[group_name]
        #     f.write(f"{group_name:<25} {metrics['accuracy']*100:<11.2f}% {metrics['precision']*100:<11.2f}% "
        #            f"{metrics['recall']*100:<11.2f}% {metrics['f1']*100:<11.2f}% {metrics['total']:<10}\n")
        # f.write("-" * 85 + "\n")
        # f.write(f"{'Overall':<25} {language_cwe_overall['accuracy']*100:<11.2f}% {language_cwe_overall['precision']*100:<11.2f}% "
        #        f"{language_cwe_overall['recall']*100:<11.2f}% {language_cwe_overall['f1']*100:<11.2f}% {language_cwe_overall['total']:<10}\n")
        
        # f.write("\n" + "="*70 + "\n")
        # f.write("TRAINING COMPLETED SUCCESSFULLY\n")
        # f.write("="*70 + "\n")
    
    print(f"✓ Final report saved to {final_report_file}")
    print("\n" + "="*60)
    print("Quy trình huấn luyện hoàn tất thành công!")
    print("="*60)
    print(f"Log file: {log_file}")
    print(f"Results directory: {results_dir}")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()