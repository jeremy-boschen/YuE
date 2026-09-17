"""Memory-bounded acoustic flow matching with one AR prefill per original chunk.

Only PyTorch is required. The reference 32-step midpoint solver, full-song CPU
FP32 noise draw, boundary positions, and original context chunks are preserved.
Attention query tiling changes temporary storage, never the visible key set.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .protocol import CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END, chunk_ranges


@dataclass
class Chunk:
    ar_tokens: list[int]
    noise: torch.Tensor
    nar_cond_end: int = 0
    lead: int = 0          # leading frames already solved, pinned during the ODE


def _integers(values, name):
    result = list(values)
    if not result or any(isinstance(v, bool) or not isinstance(v, Integral) for v in result):
        raise ValueError(f"{name} must be a nonempty sequence of integer token IDs")
    return [int(v) for v in result]


def song_chunks(prefix, codec, seed, context=CONTEXT, chunk_frames=None,
                overlap_frames=0, known_frames=0):
    """Draw the complete noise tensor once, then take views at historical cuts."""
    prefix = _integers(prefix, "prefix")
    codec = _integers(codec, "codec")
    if min(prefix) < 0 or min(codec) < 0 or max(codec) >= CODEC_SIZE:
        raise ValueError("Token IDs are outside their allowed vocabulary")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(context, bool) or not isinstance(context, Integral) or not 1 <= context <= CONTEXT:
        raise ValueError(f"context must be an integer in 1..{CONTEXT}")
    ranges = chunk_ranges(len(codec), len(prefix), int(context))
    if chunk_frames:
        # Pin the chunk length so a longer song is voiced exactly like a short
        # render of the same score: same prefix, same tokens, same noise view.
        size = int(chunk_frames)
        limit = ranges[0][1] - ranges[0][0] if len(ranges) == 1 else (int(context) - len(prefix) - 3) // 2
        if not 1 <= size <= max(limit, 1):
            raise ValueError(f"chunk_frames must be 1..{max(limit, 1)} for this prefix")
        ranges = [(a, min(a + size, len(codec))) for a in range(0, len(codec), size)]
    known = max(0, int(known_frames))
    lead = max(0, int(overlap_frames))
    if lead or known:
        # Widen every chunk after the first to the left. Those frames are already
        # solved, so they enter the ODE pinned to their known values and give the
        # new frames real audio to continue from instead of silence.
        room = (int(context) - len(prefix) - 3) // 2
        widened = []
        for index, (a, b) in enumerate(ranges):
            # Carried frames already sit at the head of the first range, so it is
            # pinned in place. Later chunks must reach back before their start.
            if index == 0:
                widened.append((a, b, min(known, b - a)))
            else:
                take = min(lead, a, max(0, room - (b - a)))
                widened.append((a - take, b, take))
        ranges = widened
    else:
        ranges = [(a, b, 0) for a, b in ranges]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=generator)
    return [Chunk(prefix + [value + CODEC_OFFSET for value in codec[a:b]] + [MUSIC_END], noise[a:b], 0, take)
            for a, b, take in ranges]


def attention(q, k, v, *, causal=False, backend="sdpa", query_chunk_size=None):
    """Attend [tokens, heads, dim] tensors without materializing a song mask.

    CPU/MPS bound the number of query rows for a potential math SDPA fallback.
    CUDA normally uses PyTorch's fused SDPA without an external flash package.
    """
    if backend not in {"sdpa", "math", "flash"}:
        raise ValueError("attention must be sdpa, math, or flash")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("Expected Q/K/V [tokens, heads, dim] with matching K/V")
    if min(q.shape) < 1 or min(k.shape) < 1 or q.shape[1] % k.shape[1]:
        raise ValueError("Invalid attention lengths or grouped-query head count")
    if causal and len(q) != len(k):
        raise ValueError("Causal prefill requires matching Q/K sequence lengths")
    if backend == "flash" and q.device.type != "cuda":
        raise ValueError("Explicit flash SDPA requires a CUDA device")
    if query_chunk_size is not None and (isinstance(query_chunk_size, bool) or
                                        not isinstance(query_chunk_size, Integral) or query_chunk_size < 1):
        raise ValueError("query_chunk_size must be a positive integer")
    block = query_chunk_size or (len(q) if q.device.type == "cuda" and backend != "math" else 256)
    query = q.transpose(0, 1).unsqueeze(0)
    key = k.transpose(0, 1).unsqueeze(0)
    value = v.transpose(0, 1).unsqueeze(0)
    grouped = query.shape[1] != key.shape[1]
    if grouped and q.device.type == "mps":
        groups = query.shape[1] // key.shape[1]
        key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
        grouped = False
    context = nullcontext()
    if backend != "sdpa":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        context = sdpa_kernel(SDPBackend.MATH if backend == "math" else SDPBackend.FLASH_ATTENTION)
    outputs = []
    with context:
        for start in range(0, len(q), block):
            end = min(start + block, len(q))
            used_key = key[..., :end, :] if causal else key
            used_value = value[..., :end, :] if causal else value
            # is_causal on a rectangular Q/K uses an upper-left triangle, so a
            # later query block needs its absolute query positions explicitly.
            mask = None
            if causal and start:
                mask = (torch.arange(end, device=q.device)[None, :] <=
                        torch.arange(start, end, device=q.device)[:, None])
            outputs.append(F.scaled_dot_product_attention(
                query[..., start:end, :], used_key, used_value,
                attn_mask=mask, is_causal=causal and start == 0, enable_gqa=grouped,
            ))
    return torch.cat(outputs, dim=-2)[0].transpose(0, 1)


class CachedNAR:
    """One original acoustic chunk; AR prefix KV is invariant during the ODE."""

    def __init__(self, model, chunk: Chunk, attention="sdpa", query_chunk_size=None, known=None):
        self.model, self.chunk = model, chunk
        self.backend, self.query_chunk_size = attention, query_chunk_size
        self.known = None
        weight = next(model.vae2llm.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        if chunk.noise.ndim != 2 or chunk.noise.shape[1] != 64 or len(chunk.noise) < 1:
            raise ValueError("Expected nonempty acoustic noise [frames,64]")
        if not torch.isfinite(chunk.noise).all():
            raise ValueError("Acoustic noise contains non-finite values")
        self.ar_length, self.nar_length = len(chunk.ar_tokens), len(chunk.noise) + 2
        if self.ar_length < 1 or min(chunk.ar_tokens) < 0 or max(chunk.ar_tokens) >= model.config.vocab_size:
            raise ValueError("AR prefix is empty or outside the model vocabulary")
        if self.ar_length + self.nar_length > model.config.max_position_embeddings:
            raise ValueError("Original acoustic chunk exceeds the model context")
        if chunk.nar_cond_end < 0:
            raise ValueError("nar_cond_end must be nonnegative")
        self.visible_length = min(chunk.nar_cond_end, self.ar_length) if chunk.nar_cond_end else self.ar_length
        positions = torch.arange(self.ar_length, self.ar_length + self.nar_length, device=self.device)[None]
        self.cos, self.sin = model.model.rotary_emb(positions)
        local = torch.arange(self.nar_length, device=self.device).clamp(max=model.config.max_latent_frames - 1)
        self.pos_emb = model.latent_pos_embed(local)[None]
        if known is not None:
            if known.ndim != 2 or known.shape[1] != 64 or not len(known) or len(known) > len(chunk.noise):
                raise ValueError("Leading context must be latents shaped [n,64] with n <= chunk frames")
            if not torch.isfinite(known).all():
                raise ValueError("Leading context contains non-finite values")
            self.known = known.to(device=self.device, dtype=self.dtype)
        self.cache = []
        self._prefill()

    def _attention(self, q, k, v, causal=False):
        return attention(q, k, v, causal=causal, backend=self.backend, query_chunk_size=self.query_chunk_size)

    @torch.inference_mode()
    def _prefill(self):
        backbone = self.model.model
        ids = torch.tensor([self.chunk.ar_tokens], dtype=torch.long, device=self.device)
        positions = torch.arange(self.ar_length, device=self.device)[None]
        cos, sin = backbone.rotary_emb(positions)
        x = backbone.embed_tokens(ids)
        for layer in backbone.layers:
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
            # Clone only for restricted visibility; a slice would retain the
            # storage of invisible codec tokens for every layer.
            cached = (k[0, :self.visible_length], v[0, :self.visible_length])
            if self.visible_length != self.ar_length:
                cached = tuple(t.clone() for t in cached)
            self.cache.append(cached)
            h = self._attention(q[0], k[0], v[0], causal=True)
            x = x + layer.self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.mlp(layer.post_attention_layernorm(x))

    @torch.inference_mode()
    def velocity(self, state, raw_t):
        model = self.model
        if tuple(state.shape) != tuple(self.chunk.noise.shape):
            raise ValueError("ODE state shape changed")
        x_nar = F.pad(state, (0, 0, 1, 1))
        shifted = model._shift_t_value(raw_t, self.device, self.dtype)
        x = model.vae2llm(x_nar[None])
        x = x + model.time_embedder(shifted.expand(self.nar_length))[None]
        x = x + self.pos_emb
        for layer, (ar_k, ar_v) in zip(model.model.layers, self.cache):
            q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), self.cos, self.sin)
            k, v = torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))
            h = self._attention(q[0], k, v)
            x = x + layer.nar_self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        return model.llm2vae(model.model.norm(x))[0, 1:-1]

    @torch.inference_mode()
    def solve(self, steps=32, cancelled: Callable[[], bool] | None = None,
              on_progress: Callable[[int, int], None] | None = None):
        """Solve a chunk, reporting each submitted midpoint step without syncing.

        CUDA work may still be executing when ``on_progress`` runs. The existing
        CPU result transfer completes that work before this method returns.
        Callback exceptions propagate to the caller.
        """
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("steps must be a positive integer")
        noise = self.chunk.noise.to(device=self.device, dtype=self.dtype)
        keep = len(self.known) if self.known is not None else 0

        def pin(x, t):
            """Hold the known frames on the flow path x_t = t*noise + (1-t)*data."""
            if not keep:
                return x
            x = x.clone()
            x[:keep] = t * noise[:keep] + (1.0 - t) * self.known
            return x

        state = pin(noise, 1.0) if keep else noise
        dt = 1.0 / steps
        for step in range(steps):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            t = 1.0 - step * dt
            raw = torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            first = self.velocity(state, raw)
            mid = pin(state - first * (dt / 2), t - dt / 2)
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            raw_mid = torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            state = pin(state - self.velocity(mid, raw_mid) * dt, max(0.0, t - dt))
            if on_progress is not None:
                on_progress(step + 1, int(steps))
        result = state.float().cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result

    def close(self):
        self.cache.clear()
        self.cos = self.sin = self.pos_emb = None


@contextmanager
def _offload_ar(model, enabled):
    """Temporarily move unused AR modules; this model cannot serve concurrently."""
    modules = [model.model.embed_tokens, model.lm_head]
    for layer in model.model.layers:
        modules.extend((layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp))
    moved = []
    try:
        if enabled:
            for module in modules:
                device = next(module.parameters()).device
                if device.type != "cpu":
                    module.to(device="cpu")
                    moved.append((module, device))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        yield
    finally:
        for module, device in moved:
            module.to(device=device)


@torch.inference_mode()
def synthesize(model, prefix: Sequence[int], codec: Sequence[int], seed: int,
               steps=32, context=CONTEXT, attention="sdpa", offload_ar=False,
               cancelled=None, query_chunk_size=None, chunk_frames=None, overlap_frames=0,
               known_latents=None, blend_frames=0,
               on_progress: Callable[[int, int], None] | None = None):
    """Return CPU FP32 [frames,64] latents, solving original chunks serially.

    Defaults preserve the release protocol, including the single full-song chunk
    layout: pass nothing and this behaves exactly as before. ``chunk_frames`` and
    ``overlap_frames`` pin the chunk length so a long song is voiced like a short
    render of the same score. ``known_latents`` are the leading frames of an
    earlier take, never re-solved: they enter the first chunk pinned, so new
    frames are composed against real audio, and ``blend_frames`` crossfades out of
    the carried take instead of cutting. Explicit steps/context overrides ``offload_ar`` is
    an optional memory tradeoff and requires exclusive access to ``model``.
    Progress counts submitted midpoint steps across all original chunks; it
    introduces no device synchronization. Callback exceptions propagate after
    the current chunk's cache is released and any offloaded weights restored.
    """
    if model.training:
        raise ValueError("synthesize requires model.eval()")
    if known_latents is not None:
        # This function returns CPU numpy, so numpy is what a caller has to hand
        # when continuing a take. Accept it, and make the carried frames a tensor
        # once here rather than leaving the first chunk holding whatever was
        # passed while later chunks hold a torch.cat result.
        known_latents = torch.as_tensor(known_latents)
        if known_latents.ndim != 2 or known_latents.shape[1] != 64:
            raise ValueError("known_latents must be shaped [frames,64]")
    carried = 0 if known_latents is None else len(known_latents)
    chunks = song_chunks(prefix, codec, seed, context, chunk_frames=chunk_frames,
                         overlap_frames=overlap_frames, known_frames=carried)
    output = []
    for chunk_index, chunk in enumerate(chunks):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before acoustic prefill")
        known = None
        blend = 0
        if chunk_index == 0 and carried:
            # Hard-pin everything but the last `blend` frames. Those are solved
            # freely so the model writes its own way out of the carried audio,
            # then the output crossfades from the take into that solution.
            blend = max(0, min(int(blend_frames), chunk.lead - 1))
            known = known_latents[:chunk.lead - blend]
        elif chunk.lead and output:
            known = torch.cat(output, dim=0)[-chunk.lead:]
            if len(known) != chunk.lead:
                known = None
        engine = CachedNAR(model, chunk, attention, query_chunk_size, known)
        # Drop the prefix cache before restoring AR weights, including on
        # cancellation/failure, to keep the restoration memory peak bounded.
        with _offload_ar(model, offload_ar):
            try:
                progress = None
                if on_progress is not None:
                    def progress(completed, total):
                        on_progress(chunk_index * total + completed, total * len(chunks))
                solved = engine.solve(steps, cancelled, on_progress=progress)
            finally:
                engine.close()
        del engine
        if chunk_index == 0 and carried:
            head = chunk.lead - blend
            output.append(known_latents[:head].to(solved.dtype))
            if blend:
                ramp = torch.linspace(1.0, 0.0, blend, dtype=solved.dtype).unsqueeze(1)
                tail = known_latents[head:chunk.lead].to(solved.dtype)
                output.append(ramp * tail + (1.0 - ramp) * solved[head:chunk.lead])
        output.append(solved[chunk.lead:] if known is not None else solved)
    return torch.cat(output, dim=0)
