"""Compare against untouched Git sources, not the fork's profile=None path."""
import importlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from yue2 import modeling_yue2, modeling_vae, nar, sampling
from yue2.profiles import OfficialProfile, ComfyUIYuE2MPSProfile
from yue2.protocol import Sampling, VOCAB_SIZE, CODEC_OFFSET

UPSTREAM = 'bd90e4ccae671d869b3ecaca6d7e893927d29442'


@pytest.fixture(scope='module')
def upstream(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path_factory.mktemp('official-upstream')
    name = '_yue2_untouched_upstream'
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    for module in ('protocol', 'modeling_yue2', 'modeling_vae', 'nar', 'sampling'):
        source = subprocess.check_output(
            ['git', 'show', f'{UPSTREAM}:src/yue2/{module}.py'], cwd=root)
        (directory / f'{module}.py').write_bytes(source)
    yield SimpleNamespace(**{module: importlib.import_module(f'{name}.{module}')
                            for module in ('modeling_yue2', 'modeling_vae', 'nar', 'sampling')})
    for key in list(sys.modules):
        if key == name or key.startswith(name + '.'):
            del sys.modules[key]


DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
def test_official_ar_and_acoustic_match_untouched_upstream(upstream, device, monkeypatch):
    kwargs = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                  vocab_size=32, max_position_embeddings=128, max_latent_frames=128)
    torch.manual_seed(173)
    reference = upstream.modeling_yue2.YuE2ForCausalLM(upstream.modeling_yue2.YuE2Config(**kwargs)).eval()
    actual = modeling_yue2.YuE2ForCausalLM(modeling_yue2.YuE2Config(**kwargs), profile=OfficialProfile()).eval()
    actual.load_state_dict(reference.state_dict())
    dtype = torch.bfloat16 if device == 'mps' else torch.float32
    reference.to(device=device, dtype=dtype)
    actual.to(device=device, dtype=dtype)
    ids = torch.tensor([[1, 2, 3]], device=device)
    with torch.inference_mode():
        assert torch.equal(reference(ids, use_cache=False).logits, actual(ids, use_cache=False).logits)
    for module in (nar, upstream.nar):
        monkeypatch.setattr(module, 'CODEC_OFFSET', 8)
        monkeypatch.setattr(module, 'MUSIC_END', 7)
    for context in (15, 128):  # multiple original chunks and one original chunk
        expected = upstream.nar.synthesize(reference, [2, 3], [1] * 11, 42, steps=2, context=context)
        result = nar.synthesize(actual, [2, 3], [1] * 11, 42, steps=2, context=context)
        assert torch.equal(result, expected)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('stride', [4, 5])
def test_official_decoder_matches_upstream(upstream, device, stride):
    torch.manual_seed(77)
    expected = upstream.modeling_vae.DecoderBlock(8, 4, stride, 'elu').eval().to(device)
    actual = modeling_vae.DecoderBlock(8, 4, stride, 'elu', profile=OfficialProfile()).eval().to(device)
    actual.load_state_dict(expected.state_dict())
    x = torch.randn(1, 8, 7, device=device)
    with torch.inference_mode():
        assert torch.equal(actual(x), expected(x))


@pytest.mark.parametrize('device', DEVICES)
def test_official_sampler_matches_upstream(upstream, device):
    class Logits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, device=device))
            self.config = SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1, head_dim=1)

        def forward(self, ids, past_key_values=None, **kwargs):
            logits = torch.full((1, 1, VOCAB_SIZE), -100., device=device)
            logits[..., CODEC_OFFSET:CODEC_OFFSET + 4] = torch.tensor([5., 4., 3., 2.], device=device)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    args = (Logits(), [42], Sampling(min_tokens=20, max_tokens=20, repetition_penalty=1.2), 777, 'semantic')
    expected = upstream.sampling.generate_tokens(*args)
    actual = sampling.generate_tokens(*args, rng_device=OfficialProfile().sampling_rng_device('auto'))
    assert actual[0] == expected[0] and actual[2] == expected[2]


def test_stage_policy_keeps_allocator_operations_out_of_official(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, 'empty_cache', lambda: calls.append('empty'))
    monkeypatch.setattr(torch.mps, 'synchronize', lambda: calls.append('sync'))
    from yue2.pipeline import YuE2Pipeline
    pipe = object.__new__(YuE2Pipeline)
    pipe.device = torch.device('mps')
    pipe._profile = OfficialProfile()
    pipe._stage_boundary()
    assert calls == []
    pipe._profile = ComfyUIYuE2MPSProfile()
    pipe._stage_boundary()
    assert calls == ['empty', 'sync']


@pytest.mark.parametrize('kwargs', [dict(carry=[1]), dict(chunk_seconds=1),
    dict(overlap_seconds=1), dict(known_latents=torch.zeros(2, 64)), dict(blend_seconds=1)])
def test_continuation_is_an_explicit_profile_capability(kwargs):
    with pytest.raises(ValueError, match='does not support'):
        OfficialProfile().validate_continuation(**kwargs)
    ComfyUIYuE2MPSProfile().validate_continuation(**kwargs)


@pytest.mark.parametrize('requested', ['cpu', 'device'])
def test_official_rejects_rng_preferences(requested):
    with pytest.raises(ValueError, match='rng_device=auto'):
        OfficialProfile().validate(torch.device('cpu'), backend='torch', quantization='none', rng_device=requested)
    with pytest.raises(ValueError, match='rng_device=auto'):
        OfficialProfile().sampling_rng_device(requested)


def test_pipeline_rejects_extensions_before_touching_models():
    from yue2.pipeline import YuE2Pipeline, SymbolicPlan
    pipe = object.__new__(YuE2Pipeline)
    pipe._profile = OfficialProfile()
    plan = SymbolicPlan(None, None, [], [])
    with pytest.raises(ValueError, match='does not support'):
        pipe.generate_semantic(plan, carry=[1])
    with pytest.raises(ValueError, match='does not support'):
        pipe.synthesize(None, chunk_seconds=2)
