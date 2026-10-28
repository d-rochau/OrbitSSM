
from __future__ import annotations
from dataclasses import dataclass
from .lm import MambaLMConfig
LLAMA31_VOCAB = 128256

@dataclass(frozen=True)
class SizeRecipe:
    name: str
    d_model: int
    n_layers: int
    mlp_dim: int
    gpt3_base_lr: float

    @property
    def peak_lr(self) -> float:
        return 5.0 * self.gpt3_base_lr

    def model_cfg(self, *, ssm: str | dict='official_mamba3') -> MambaLMConfig:
        return MambaLMConfig(vocab_size=LLAMA31_VOCAB, n_layers=self.n_layers, d_model=self.d_model, mlp_dim=self.mlp_dim, ssm=ssm)
SIZES: dict[str, SizeRecipe] = {'180M': SizeRecipe('180M', d_model=768, n_layers=12, mlp_dim=1500, gpt3_base_lr=0.0006), 'w1024': SizeRecipe('w1024', d_model=1024, n_layers=24, mlp_dim=2048, gpt3_base_lr=0.0003), '440M': SizeRecipe('440M', d_model=1024, n_layers=24, mlp_dim=2048, gpt3_base_lr=0.0003), '880M': SizeRecipe('880M', d_model=1536, n_layers=24, mlp_dim=3072, gpt3_base_lr=0.00025), '1.5B': SizeRecipe('1.5B', d_model=2048, n_layers=24, mlp_dim=4096, gpt3_base_lr=0.0002)}
RECIPE = dict(optimizer='AdamW', betas=(0.9, 0.95), weight_decay=0.1, grad_clip=1.0, dropout=0.0, lr_schedule='linear_warmup_cosine', lr_min=1e-05, lr_peak_multiplier=5.0, batch_tokens=500000, seq_len=2048, precision='bfloat16', norm='RMSNorm', bias=False, total_tokens=100000000000)
