"""Frozen ComfyUI YuE2/Apple Silicon compatibility, without ComfyUI imports."""
from dataclasses import dataclass, field
import math
from . import OfficialProfile


@dataclass(frozen=True)
class ComfyUIYuE2MPSProfile(OfficialProfile):
    name: str = field(default='comfyui-yue2-mps-v1', init=False)
    carry_penalty_history: bool = field(default=True, init=False)
    supports_continuation: bool = field(default=True, init=False)

    def stage_boundary(self, device):
        # Preserve the validated compatibility execution sequence. This is not
        # established as a general upstream correctness fix.
        if device.type == 'mps':
            import torch
            torch.mps.empty_cache()
            torch.mps.synchronize()

    def validate(self, device, *, backend, quantization, rng_device):
        import platform
        import os
        import subprocess
        import torch
        from importlib.metadata import version
        for key, expected in {'MTLFLASHATTN_KERNEL': 'auto', 'MTLFLASHATTN_V2_PREUSE': 'auto',
                              'MTLFLASHATTN_V2_FP32_MIN_SEQ': '2048', 'MTLFLASHATTN_TORCH_CHUNK': '2048'}.items():
            if os.environ.get(key, expected).lower() != expected:
                raise ValueError(f'{self.name} requires {key}={expected}')
        if device.type != 'mps' or backend not in ('torch', 'torch-eager') or quantization != 'none':
            raise ValueError(f'{self.name} requires MPS, torch/torch-eager and unquantized BF16 models')
        if rng_device not in (None, 'auto', 'cpu'):
            raise ValueError(f'{self.name} requires CPU sampling RNG')
        if platform.system() != 'Darwin' or platform.mac_ver()[0] != '26.6.2':
            raise RuntimeError(f'{self.name} is validated on macOS 26.6.2')
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        if cpu != 'Apple M5 Pro':
            raise RuntimeError(f'{self.name} is validated on Apple M5 Pro, found {cpu}')
        for name, expected in [('torch', '2.14.0'), ('mtlflashattn', '0.2.0')]:
            if version(name) != expected:
                raise RuntimeError(f'{self.name} requires {name}=={expected}')
        if torch.version.git_version != '08187d9e0fba026dc8217405802ab5381dc88d90':
            raise RuntimeError(f'{self.name} requires the validated Torch build')

    def sampling_rng_device(self, requested):
        return 'cpu'

    def decoder_output_padding(self, stride):
        return stride % 2

    def rms_norm(self, x, weight, eps):
        from .vendor import metal_norm
        if x.device.type != 'mps':
            raise RuntimeError(f'{self.name} normalization requires MPS')
        result = metal_norm.fused_rmsnorm_modulate(x.contiguous().view(-1, weight.numel()),
                                                 weight.contiguous().view(-1), eps)
        if metal_norm._last_backend != 'kernel':
            raise RuntimeError('ComfyUI parity requires the Metal RMSNorm kernel')
        return result.view_as(x)

    def metal_eligible(self, q, k, v, attn_mask, dropout_p, is_causal):
        # Gates preserved from mtlflashattn 0.2.0 sdpa._eligibility (MIT).
        # Thresholds belong to this frozen profile, never process-global env state.
        from metal_flash_attn import sdpa as mfa
        if q.device.type != 'mps' or attn_mask is not None or dropout_p:
            return False
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            return False
        if q.dtype not in mfa._SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
            return False
        d = q.shape[-1]
        if d > mfa.MAX_HEAD_DIM or k.shape[-1] != d or v.shape[-1] != d:
            return False
        hq, hkv = q.shape[1], k.shape[1]
        if hkv == 0 or hq % hkv or k.shape[2] != v.shape[2]:
            return False
        lq, lk = q.shape[2], k.shape[2]
        if is_causal and lq != lk:
            return False
        return (max(lq, lk) >= 4096 or
                (max(lq, lk) >= 1024 and mfa._select_tier(q, k, v) in mfa._FAST_TIERS) or
                q.shape[0] * hq * lq * lk * 2 >= 12 * 1024**3)

    def nar_attention(self, query, key, value, *, attn_mask=None, dropout_p=0.,
                      is_causal=False, scale=None, **kwargs):
        if self.metal_eligible(query, key, value, attn_mask, dropout_p, is_causal):
            from metal_flash_attn._kernel import flash_attn_forward
            return flash_attn_forward(query, key, value,
                                      scale=scale if scale is not None else 1. / math.sqrt(query.shape[-1]),
                                      causal=is_causal)
        return super().nar_attention(query, key, value, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)

    def ar_attention(self, query, key, value, *, attn_mask=None, is_causal=False):
        if attn_mask is None and not is_causal and self.metal_eligible(query, key, value, None, 0., False):
            return self.nar_attention(query, key, value)
        return super().ar_attention(query, key, value, attn_mask=attn_mask, is_causal=is_causal)

    def identity(self):
        import json
        from pathlib import Path
        return {**super().identity(), 'reference': json.loads(
            Path(__file__).with_name('comfyui_reference.json').read_text()),
            'attention_thresholds': {'min_seq': 4096, 'fast_min_seq': 1024, 'min_score_bytes': 12 * 1024**3}}
