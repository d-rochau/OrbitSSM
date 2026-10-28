
from __future__ import annotations
import torch.nn as nn

class LlamaWrapper(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int | None=None, intermediate_size: int | None=None, max_position_embeddings: int=8192, num_key_value_heads: int | None=None, attention_dropout: float=0.0, rms_norm_eps: float=1e-05, tie_word_embeddings: bool=True):
        super().__init__()
        from transformers import LlamaConfig, LlamaModel
        if n_heads is None:
            if d_model >= 256:
                n_heads = 8
            elif d_model >= 128:
                n_heads = 4
            else:
                n_heads = 4
        if intermediate_size is None:
            intermediate_size = int(8 / 3 * d_model)
            intermediate_size = (intermediate_size + 15) // 16 * 16
        if num_key_value_heads is None:
            num_key_value_heads = n_heads
        config = LlamaConfig(vocab_size=vocab_size, hidden_size=d_model, intermediate_size=intermediate_size, num_hidden_layers=n_layers, num_attention_heads=n_heads, num_key_value_heads=num_key_value_heads, max_position_embeddings=max_position_embeddings, attention_dropout=attention_dropout, rms_norm_eps=rms_norm_eps, tie_word_embeddings=tie_word_embeddings, use_cache=False)
        self.config = config
        self.llama = LlamaModel(config)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.llama.embed_tokens.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.initializer_range)

    def forward(self, input_ids, position_ids=None, state=None, return_embeddings=False):
        out = self.llama(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        if return_embeddings:
            return out.last_hidden_state
        return self.lm_head(out.last_hidden_state)

    def state_size(self, sequence_length: int) -> int:
        cfg = self.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        kv_per_layer = 2 * cfg.num_key_value_heads * head_dim * sequence_length
        return cfg.num_hidden_layers * kv_per_layer
