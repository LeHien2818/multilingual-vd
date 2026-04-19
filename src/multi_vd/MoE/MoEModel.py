import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# Expert
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

# MulVul Assistant
class MulVulAssistant(nn.Module):
    
    def __init__(self, pretrained_model, input_dim=768, hidden_dim=256, num_langs=2, pool_length=5, dropout_rate=0.1, temperature=0.1):
        super(MulVulAssistant, self).__init__()
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
            raise ValueError("MulVulAssistant requires a valid pretrained backbone")

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

# Router
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

# MoE Architecture
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
        
        # Initialize experts: CCPP (0), Python (1), Mulvul (2) 
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if i == 2:  # MulVulAssistant
                self.experts.append(MulVulAssistant(pretrained_model=self.encoder, input_dim=input_dim, hidden_dim=hidden_dim,
                                                 num_langs=2, pool_length=5, dropout_rate=dropout_rate))
            else:
                self.experts.append(CWE_Expert(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate))

    def _apply_special_routing(self, routing_weights, languages=None, vuln_labels=None, cwe_labels=None, cwe_dpy_set=None):
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
            if i == self.expert_vulpy_idx and isinstance(expert, MulVulAssistant):
                lang_ids = None
                if languages is not None:
                    lang_ids = torch.tensor(
                        [language_id_map.get(str(languages[j]).strip().lower(), 0) for j in range(len(languages))],
                        device=x_norm.device, dtype=torch.long
                    )
                if raw_input_ids is None or raw_attention_mask is None:
                    raise ValueError("raw_input_ids and raw_attention_mask are required for MulVulAssistant")
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

        # Map language strings to language IDs for MulVulAssistant
        language_id_map = {"python": 0, "ccpp": 1}
        
        # Vectorized expert processing
        for i, expert in enumerate(self.experts):
            expert_mask = (routing_weights[:, i] > 0)
            if expert_mask.any():
                expert_inputs = x[expert_mask]
                # weights = routing_weights[expert_mask, i].unsqueeze(1)
                
                # Special handling for VulPy expert with MulVulAssistant
                if i == self.expert_vulpy_idx and isinstance(expert, MulVulAssistant):
                    lang_ids = None
                    if languages is not None:
                        lang_ids = torch.tensor(
                            [language_id_map.get(str(languages[j]).strip().lower(), 0) 
                             for j in range(len(languages)) if expert_mask[j]],
                            device=x.device, dtype=torch.long
                        )
                    if raw_input_ids is None or raw_attention_mask is None:
                        raise ValueError("raw_input_ids and raw_attention_mask are required for MulVulAssistant")
                    expert_outputs, aux_loss = expert(raw_input_ids[expert_mask], raw_attention_mask[expert_mask], language_ids=lang_ids)
                    total_aux_loss = total_aux_loss + aux_loss
                else:
                    expert_outputs = expert(expert_inputs)
                
                final_output[expert_mask] += expert_outputs

        return final_output, fraction_routed, prob_per_expert, router_logits, total_aux_loss