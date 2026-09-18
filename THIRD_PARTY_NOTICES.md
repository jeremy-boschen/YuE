# Third-party code notices

The Oobleck VAE and SnakeBeta implementation in `modeling_vae.py` is derived
from stable-audio-tools commit `a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45`.
The module hierarchy, weight normalization and activation equations preserve
the checkpoint's original inference implementation.

- Oobleck / stable-audio-tools: Copyright (c) 2023 Stability AI, MIT.
  Full text: `licenses/stable-audio-tools-MIT.txt`.
- SnakeBeta / BigVGAN: Copyright (c) 2022 NVIDIA CORPORATION, MIT.
  Full text: `licenses/SnakeBeta-NVIDIA-MIT.txt`.

These notices cover the identified source code and retain its original licenses.
The YuE2 model checkpoint weights are separately licensed under CC BY-NC 4.0,
with additional permission for individual creators; see MODEL_LICENSE for
academic-use terms, scope and full terms. This does not relicense third-party code.

## ComfyUI YuE2 MPS profile

See `src/yue2/profiles/vendor/README.md` for source revisions and modifications,
and its bundled MIT notices for the standalone Metal RMSNorm and attention gates.
The continuation-history adaptation retains the YuE2 authors' Apache-2.0 attribution.
