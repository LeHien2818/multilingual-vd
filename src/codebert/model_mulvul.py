# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

    
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.backbone = getattr(encoder, "roberta", None)
        if self.backbone is None:
            self.backbone = getattr(encoder, "base_model", encoder)
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.pool_length = getattr(args, "pool_length", 5)
        self.temperature = getattr(args, "temperature", 0.1)
        self.num_langs = getattr(args, "num_langs", 2)

        hidden_size = getattr(self.config, "hidden_size", self.backbone.config.hidden_size)
        num_labels = getattr(self.config, "num_labels", 2)

        self.parameter_pool = nn.Parameter(
            torch.randn(self.num_langs, self.pool_length, hidden_size) * 0.02
        )
        self.keys = nn.Parameter(
            torch.randn(self.num_langs, hidden_size) * 0.02
        )
        self.classifier = nn.Linear(hidden_size, num_labels)
    
        
    def forward(self, input_ids=None,labels=None, language=None): 
        batch_size = input_ids.size(0)
        attention_mask = input_ids.ne(1)
        raw_embeds = self.backbone.embeddings(input_ids)
        cls_query = raw_embeds[:, 0, :]

        if language is not None:
            pool = self.parameter_pool[language]
            
            query_norm = F.normalize(cls_query, p=2, dim=1)
            keys_norm = F.normalize(self.keys, p=2, dim=1)
            selected_keys = keys_norm[language]
            cosine_sim = torch.sum(query_norm * selected_keys, dim=1)
            aux_loss = (1.0 - cosine_sim).mean()
        else:
            query_norm = F.normalize(cls_query, p=2, dim=1)
            keys_norm = F.normalize(self.keys, p=2, dim=1)
            scores = torch.matmul(query_norm, keys_norm.transpose(0, 1)) / self.temperature
            lang_ids = torch.argmax(scores, dim=1)
            pool = self.parameter_pool[lang_ids]
            aux_loss = torch.tensor(0.0, device=cls_query.device)

        # Truncate + concat pool tokens
        keep_len = max(raw_embeds.size(1) - self.pool_length, 0)
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
        prob = F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits,labels) + aux_loss
            return loss,prob
        else:
            return prob
      
        
 
