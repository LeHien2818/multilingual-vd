import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class Language_Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(Language_Expert, self).__init__()
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

# ROUTER
class Router(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=1):
        super(Router, self).__init__()
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

# MOE-VD Architecture 
class MoE_VulnerabilityDetector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_experts=4, top_k=1, dropout_rate=0.1, 
                 encoder_name="microsoft/codebert-base", freeze_encoder=False):
        super(MoE_VulnerabilityDetector, self).__init__()
        self.num_experts = num_experts
        
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
        self.router = Router(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([Language_Expert(input_dim, hidden_dim, dropout_rate) for _ in range(num_experts)])

    def _apply_data_routing(self, routing_weights, languages=None):
        if languages is None:
            return routing_weights

        updated_weights = routing_weights.clone()
        for i, language in enumerate(languages):
            language_lower = str(language).strip().lower()
            if language_lower == "python":
                updated_weights[i].zero_()
                updated_weights[i, self.expert_python_idx] = 1.0        

            if language_lower == "ccpp":      
                updated_weights[i].zero_()
                updated_weights[i, self.expert_ccpp_idx] = 1.0

        return updated_weights

    def encode_code(self, code_strings, device):
       
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
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)
        embeddings = sum_hidden / sum_mask
        
        return embeddings
    
    def get_expert_logits(self, x, routed_expert_ids=None):
        """Return per-expert logits for language-based inference policy."""
        x_norm = self.input_norm(x)
        expert_outputs = []
        
        for i in range(len(x)):
            if routed_expert_ids is not None:
                expert_id = int(routed_expert_ids[i])
                expert_outputs.append(self.experts[expert_id](x_norm[i:i+1]))
            else:
                raise ValueError("routed_expert_ids must be provided for get_expert_logits")
        
        return torch.cat(expert_outputs, dim=1)

    def forward(self, x, languages=None):
        x_orig = x
        x = self.input_norm(x)
        
        routing_weights, top_k_indices, router_logits = self.router(x)
        batch_size = x.size(0)
        final_output = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        routing_weights = self._apply_data_routing(routing_weights, languages)

        routing_probs = F.softmax(router_logits, dim=-1)
        fraction_routed = routing_weights.gt(0).float().mean(dim=0)
        prob_per_expert = routing_probs.mean(dim=0)

        # Vectorized expert processing
        for i, expert in enumerate(self.experts):
            expert_mask = (routing_weights[:, i] > 0)
            if expert_mask.any():
                expert_inputs = x[expert_mask]
                weights = routing_weights[expert_mask, i].unsqueeze(1)
                expert_outputs = expert(expert_inputs)
                final_output[expert_mask] += expert_outputs * weights

        return final_output, fraction_routed, prob_per_expert, router_logits