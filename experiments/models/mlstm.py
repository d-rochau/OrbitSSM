
from __future__ import annotations
import torch.nn as nn
_KERNEL_MODES = {'fp32': dict(chunkwise_kernel='chunkwise--native_autograd', sequence_kernel='native_sequence__native', step_kernel='native', autocast_kernel_dtype='float32'), 'fast': dict(chunkwise_kernel='chunkwise--triton_xl_chunk', sequence_kernel='native_sequence__triton', step_kernel='triton', autocast_kernel_dtype='bfloat16')}

class XLSTMLargeWrapper(nn.Module):

    def __init__(self, *, vocab_size: int, d_model: int, n_layers: int, num_heads: int=4, kernel_mode: str='fp32', chunk_size: int=64, qk_dim_factor: float=0.5, v_dim_factor: float=1.0, ffn_proj_factor: float=2.6667, gate_soft_cap: float=15.0, output_logit_soft_cap: float=30.0, weight_mode: str='single', forget_gate_bias_init: tuple[float, float] | None=(3.0, 6.0)):
        super().__init__()
        from xlstm.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig
        if d_model % num_heads != 0:
            raise ValueError(f'd_model {d_model} not divisible by num_heads {num_heads}')
        if kernel_mode not in _KERNEL_MODES:
            raise ValueError(f'kernel_mode must be one of {sorted(_KERNEL_MODES)}; got {kernel_mode!r}')
        cfg = xLSTMLargeConfig(embedding_dim=d_model, num_heads=num_heads, num_blocks=n_layers, vocab_size=vocab_size, chunk_size=chunk_size, qk_dim_factor=qk_dim_factor, v_dim_factor=v_dim_factor, ffn_proj_factor=ffn_proj_factor, gate_soft_cap=gate_soft_cap, output_logit_soft_cap=output_logit_soft_cap, weight_mode=weight_mode, return_last_states=True, **_KERNEL_MODES[kernel_mode])
        self.config = cfg
        self.kernel_mode = kernel_mode
        self.backbone = xLSTMLarge(cfg)
        self.forget_gate_bias_init = forget_gate_bias_init
        if forget_gate_bias_init is not None:
            self._init_forget_gate_bias(*forget_gate_bias_init)

    def _init_forget_gate_bias(self, low: float, high: float) -> None:
        import torch
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith('fgate_preact.bias'):
                    param.copy_(torch.linspace(low, high, param.numel(), device=param.device))

    def forward(self, input_ids, position_ids=None, state=None, return_embeddings=False):
        if return_embeddings:
            raise NotImplementedError('XLSTMLargeWrapper does not expose pre-head embeddings; the mqmkvar loss/accuracy path uses return_embeddings=False.')
        out = self.backbone(input_ids)
        return out[0] if isinstance(out, tuple) else out
