"""Instance-owned numerical profiles; importing this module needs no Metal package."""
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class NumericalProfile(Protocol):
    """Immutable operations/policy. Weights, caches and RNG belong to the engine."""
    name: str
    carry_penalty_history: bool
    supports_continuation: bool

    def rms_norm(self, x, weight, eps): ...
    def ar_attention(self, query, key, value, **kwargs): ...
    def nar_attention(self, query, key, value, **kwargs): ...
    def decoder_output_padding(self, stride): ...
    def sampling_rng_device(self, requested): ...
    def stage_boundary(self, device): ...
    def validate_continuation(self, *, carry=None, chunk_seconds=0, overlap_seconds=0,
                              known_latents=None, blend_seconds=0): ...
    def validate(self, device, *, backend, quantization, rng_device): ...
    def identity(self): ...


@dataclass(frozen=True)
class OfficialProfile:
    """Released numerical behavior at the recorded engine revision."""
    name: str = field(default='official', init=False)
    carry_penalty_history: bool = field(default=False, init=False)
    supports_continuation: bool = field(default=False, init=False)

    def rms_norm(self, x, weight, eps):
        import torch
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype) * weight

    def ar_attention(self, query, key, value, **kwargs):
        from ..modeling_yue2 import sdpa
        return sdpa(query, key, value, **kwargs)

    def nar_attention(self, query, key, value, **kwargs):
        from torch.nn.functional import scaled_dot_product_attention
        return scaled_dot_product_attention(query, key, value, **kwargs)

    def decoder_output_padding(self, stride):
        return 0

    def sampling_rng_device(self, requested):
        if requested not in (None, 'auto'):
            raise ValueError(f'{self.name} requires rng_device=auto; select a profile for RNG overrides')
        return 'auto'

    def stage_boundary(self, device):
        """Upstream performs no extra allocator operations between stages."""

    def validate_continuation(self, *, carry=None, chunk_seconds=0, overlap_seconds=0,
                              known_latents=None, blend_seconds=0):
        if not self.supports_continuation and (
                (carry is not None and len(carry) > 0) or chunk_seconds or overlap_seconds
                or known_latents is not None or blend_seconds):
            raise ValueError(f'{self.name} does not support fork continuation/chunk overrides; '
                             'select a profile that supports continuation')

    def validate(self, device, *, backend, quantization, rng_device):
        self.sampling_rng_device(rng_device)

    def identity(self):
        return {'name': self.name, 'revision': 2,
                'upstream_revision': 'bd90e4ccae671d869b3ecaca6d7e893927d29442',
                'operations': {name: f'{getattr(type(self), name).__module__}.{getattr(type(self), name).__qualname__}'
                               for name in ('rms_norm', 'ar_attention', 'nar_attention',
                                            'decoder_output_padding', 'sampling_rng_device',
                                            'stage_boundary', 'validate_continuation')},
                'carry_penalty_history': self.carry_penalty_history,
                'supports_continuation': self.supports_continuation}


def resolve_profile(profile):
    if isinstance(profile, str):
        if profile == 'official':
            return OfficialProfile()
        if profile == 'comfyui-yue2-mps-v1':
            return ComfyUIYuE2MPSProfile()
        raise ValueError(f'Unknown numerical profile: {profile}')
    if not isinstance(profile, NumericalProfile):
        raise TypeError('profile must be a built-in name or implement NumericalProfile')
    return profile


from .comfyui import ComfyUIYuE2MPSProfile

__all__ = ['NumericalProfile', 'OfficialProfile', 'ComfyUIYuE2MPSProfile', 'resolve_profile']
