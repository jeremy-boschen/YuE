# Numerical source attribution

`metal_norm.py` retains the computational implementation from
ComfyUI-AppleSilicon-FP8 `_patches/fused_norm_mps.py`, revision
`74734a108eb1640c24e131ee088b995ff962c47f`, working-tree file SHA256
`77b4f659181c2233c03567dfbaff61ff37b9caea6555611c2d2042cecb2cc0d0`.
Global installation and ComfyUI plumbing were omitted. Kernel arithmetic is unchanged.
Copyright (c) 2026 Paweł Mazurkiewicz; MIT, see APPLESILICON_FP8_LICENSE.txt.

`../comfyui.py:metal_eligible` adapts the gates from the installed mtlflashattn
0.2.0 `metal_flash_attn/sdpa.py:_eligibility`, using frozen instance policy instead
of global threshold variables. The package declares MIT licensing and identifies
Paweł Mazurkiewicz as its author. Its MIT notice is retained in
MTLFLASHATTN_LICENSE.txt. Repository: https://github.com/pawel-mazurkiewicz/mtlflashattn.
Attention computation calls the standalone package's original kernel.

The carried repetition-history change in the engine sampler follows
ComfyUI-FL-YuE2 `yue2/sampling.py`, revision
`16d0c0f8ec4f26a38a1e9a232307641672104e81`, source SHA256
`c7a5dca332a945fa79c4cd03e8756f8b860c0d6a213aca890b577be286025c24`.
Its notice credits the YuE2 authors, copyright 2026, Apache-2.0, adapted from
YuE2 commit `92a73cc7652fcc1f937855e4b765e0a0edd7ff2e`.
Generation, cancellation, progress and request-local RNG remain in the shared
engine sampler. No reference sampling loop is vendored or patched.
