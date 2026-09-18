# Numerical profiles

`YuE2Pipeline(..., profile="official")` preserves the released engine's numerical
behavior at upstream commit `bd90e4ccae671d869b3ecaca6d7e893927d29442`,
before any fork changes. `profile="comfyui-yue2-mps-v1"` selects the measured ComfyUI YuE2 stack.
Both are built in; neither imports ComfyUI. The equivalent objects are:

```python
from yue2 import YuE2Pipeline, OfficialProfile, ComfyUIYuE2MPSProfile
pipe = YuE2Pipeline(model_dir, vae_dir, profile=ComfyUIYuE2MPSProfile())
```

The CLI accepts `--profile`. A profile belongs to one pipeline and is immutable.
Construct a new pipeline to change it. There is no process-wide installation or
replacement of engine/Torch functions. Model constructors receive the profile;
normalization, AR attention, NAR attention and decoder construction call its
explicit operations. Checkpoint tensor names and shapes are unchanged.

The shared sampler accepts explicit `prior` model-token IDs for repetition history.
The official pipeline leaves that history empty; the compatibility profile feeds
carried tokens and uses a fresh CPU generator per request. Generation, caching,
callbacks, cancellation and token budgets remain in the shared engine loop.

## Official boundary

`official` performs no extra MPS stage-boundary cache drain and requires
`GenerationConfig(rng_device="auto")`, preserving upstream device selection.
Nonempty semantic carry, known acoustic latents, custom chunks, overlap and
blending require a profile with `supports_continuation=True`. The ComfyUI
profile enables those capabilities and retains the previously validated MPS
cache-drain sequence. The cache drain is a compatibility choice, not a proven
general upstream bug fix.

The shared implementations remain reusable; profile policy is enforced at the
pipeline boundary. Low-level sampling/NAR primitives are not standalone profile
contracts. Empty model-ready hooks and provenance recording remain neutral API
extensions. A registered model-mutating hook changes the supplied model, so its
weights/adapter identity must be recorded separately by the caller.

Profile identity revision 2 records the upstream baseline, stage policy and
continuation capability. Saved revision-1 pipelines are rejected on identity
mismatch; reconstruct them explicitly with the intended profile and re-export.
This metadata revision does not change ComfyUI numerical operations.

## Compatibility scope

The frozen contract targets Apple M5 Pro, macOS26.6.2, Torch2.14.0 at commit
`08187d9e0fba026dc8217405802ab5381dc88d90`, mtlflashattn0.2.0, BF16 music model
and FP32 VAE. See the bundled `profiles/comfyui_reference.json` for reference
source revisions and launch settings. Construction rejects unsupported runtime,
backend, quantization, RNG and kernel overrides instead of silently falling back.
Stock attention on the profile's explicitly defined small/masked paths is part
of the contract, not an error fallback. A selected Metal kernel failure is fatal.

The generic package retains the upstream dependency pins. This profile requires
an explicitly provisioned matching runtime; `audiogen-yue2/env/requirements.lock.txt`
is the validated hash lock. Install that lock, then this fork with `--no-deps`.
The optional `comfyui-mps` extra names the standalone attention dependency; it does
not replace the full compatibility runtime lock.

`v1` versions our behavior contract, not a ComfyUI release. A new intentional
numerical behavior requires a new profile version and its own exact fixtures.
Profiles do not promise identity across hardware or arbitrary package upgrades.
Song input formatting, output file encoding and album lineage belong to callers.

## Caller-supplied profiles

Pass an object implementing `yue2.profiles.NumericalProfile`. The protocol owns
operations, padding/sampling policy, `stage_boundary(device)`,
`supports_continuation`, `validate_continuation(...)`, runtime validation and a
JSON-serializable identity. Subclassing `OfficialProfile` supplies conservative
defaults; override only the policies your profile deliberately changes.
Use immutable objects and keep weights, caches and RNG state in the engine.
There is no global registry or discovery plugin. Implement validation for every
supported backend. Caller-supplied profiles use eager execution; vLLM and CUDA
graphs are reserved for the exact built-in official profile, so they cannot
silently bypass custom operations.
The two bundled profiles support official execution and the restricted MPS
compatibility execution respectively.

Take configs record profile identity, installed engine provenance and recursive
runtime source hashes. Pipeline exports retain the profile; reopening a custom
profile requires supplying the same object identity because arbitrary classes
are never deserialized. Raw HF model exports retain the checkpoint architecture;
pass the profile explicitly when loading them outside the pipeline.

## Validation

Tests load untouched source directly from the pinned upstream Git commit and
compare AR logits, sampled tokens, original single/multiple acoustic chunks and
even/odd decoder strides on CPU and available MPS. Additional tests verify interleaved profile isolation, NAR operation
injection, decoder padding, carried history and request-local RNG. Release checks
compare the original riff and all four recorded continuation stages, additional
seeds and a second fixture against frozen semantic arrays, latent bytes and
decoded PCM using audiogen-yue2's diagnostics. Similar-sounding output does not
pass. Numerical source attribution and licenses ship in `profiles/vendor/`.
