"""Profiles select instance-owned behavior without changing other models."""
from dataclasses import FrozenInstanceError, dataclass
import pytest
import torch
from yue2.profiles import (OfficialProfile, OfficialContinuationProfile, ComfyUIYuE2MPSProfile,
                           official_numerics, resolve_profile)
from yue2.modeling_yue2 import YuE2ForCausalLM, YuE2Config, RMSNorm, sdpa
from yue2.modeling_vae import DecoderBlock


def config():
    return YuE2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, intermediate_size=64,
                      vocab_size=71, max_position_embeddings=64, max_latent_frames=64)


@dataclass(frozen=True)
class ScaledProfile(OfficialProfile):
    name: str = 'test-scaled'

    def rms_norm(self, x, weight, eps):
        return super().rms_norm(x, weight, eps) * 0.5

    def ar_attention(self, q, k, v, **kwargs):
        return super().ar_attention(q, k, v, **kwargs) * 0.75


def test_official_matches_unprofiled_and_custom_models_are_isolated():
    torch.manual_seed(17)
    vanilla = YuE2ForCausalLM(config()).eval()
    official = YuE2ForCausalLM(config(), profile=OfficialProfile()).eval()
    custom = YuE2ForCausalLM(config(), profile=ScaledProfile()).eval()
    for model in (official, custom):
        model.load_state_dict(vanilla.state_dict())
    ids = torch.tensor([[1, 2, 3]])
    norm, attention = RMSNorm.forward, sdpa
    with torch.inference_mode():
        expected = vanilla(ids, use_cache=False).logits
        changed = custom(ids, use_cache=False).logits
        assert not torch.equal(changed, expected)
        assert torch.equal(official(ids, use_cache=False).logits, expected)
        assert torch.equal(vanilla(ids, use_cache=False).logits, expected)
    from yue2 import modeling_yue2
    assert modeling_yue2.sdpa is attention and RMSNorm.forward is norm
    assert set(official.state_dict()) == set(vanilla.state_dict())


def test_profile_resolves_and_is_immutable():
    assert resolve_profile('official') == OfficialProfile()
    p = resolve_profile('comfyui-yue2-mps-v1')
    assert isinstance(p, ComfyUIYuE2MPSProfile)
    assert resolve_profile(p) is p
    with pytest.raises(FrozenInstanceError):
        p.name = 'changed'
    with pytest.raises(ValueError, match='Unknown'):
        resolve_profile('missing')
    with pytest.raises(TypeError):
        resolve_profile(object())
    with pytest.raises(ValueError, match='requires MPS'):
        p.validate(torch.device('cpu'), backend='torch', quantization='none', rng_device='auto')


def test_decoder_padding_is_owned_by_each_instance():
    original = DecoderBlock(8, 4, 5, 'elu')
    compat = DecoderBlock(8, 4, 5, 'elu', profile=ComfyUIYuE2MPSProfile())
    later = DecoderBlock(8, 4, 5, 'elu', profile=OfficialProfile())
    assert original.layers[1].output_padding == later.layers[1].output_padding == (0,)
    assert compat.layers[1].output_padding == (1,)


def test_nar_uses_supplied_operation_without_patching_functional():
    from yue2.nar import attention
    operation = torch.nn.functional.scaled_dot_product_attention
    torch.manual_seed(2)
    q, k, v = (torch.randn(5, 2, 8) for _ in range(3))
    expected = attention(q, k, v)
    actual = attention(q, k, v, operation=lambda *a, **kw: operation(*a, **kw) * 0)
    assert torch.count_nonzero(actual) == 0
    assert torch.equal(attention(q, k, v), expected)
    assert torch.nn.functional.scaled_dot_product_attention is operation


def test_pipeline_profile_roundtrip_and_custom_identity(tmp_path, monkeypatch):
    from yue2 import pipeline
    from yue2.protocol import GenerationConfig

    class SmallPipeline(pipeline.YuE2Pipeline):
        def __init__(self, model_dir, vae_dir, *, profile='official', **kwargs):
            self._profile = resolve_profile(profile)
            self.model_dir, self.vae_dir = model_dir, vae_dir
            self.load_timing = {}
            self.generation_config = GenerationConfig()
            self.weights = {}

    source = tmp_path / 'source'
    source.mkdir()
    monkeypatch.setattr(pipeline, 'copy_model_files', lambda a, b: b.mkdir(parents=True))
    monkeypatch.setattr(pipeline, 'resolve_model', lambda p, **kw: p)
    for index, profile in enumerate((OfficialProfile(), OfficialContinuationProfile(), ComfyUIYuE2MPSProfile(),
                                     ScaledProfile())):
        pipe = SmallPipeline(source, source, profile=profile)
        destination = tmp_path / str(index)
        pipe.save_pretrained(destination)
        if isinstance(profile, ScaledProfile):
            with pytest.raises(ValueError, match='Unknown numerical profile'):
                SmallPipeline.from_pretrained(destination, progress=False)
            restored = SmallPipeline.from_pretrained(destination, progress=False, profile=profile)
        else:
            restored = SmallPipeline.from_pretrained(destination, progress=False)
        assert restored.profile.identity() == profile.identity()
        with pytest.raises(AttributeError):
            restored.profile = OfficialProfile()
        with pytest.raises(ValueError, match='differs'):
            SmallPipeline.from_pretrained(destination, progress=False, profile=ScaledProfile(name='other'))


def test_official_continuation_computes_as_official_and_only_its_policy_differs():
    torch.manual_seed(17)
    official = YuE2ForCausalLM(config(), profile=OfficialProfile()).eval()
    continuing = YuE2ForCausalLM(config(), profile=resolve_profile('official-continuation')).eval()
    continuing.load_state_dict(official.state_dict())
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        assert torch.equal(continuing(ids, use_cache=False).logits, official(ids, use_cache=False).logits)
    p, base = OfficialContinuationProfile(), OfficialProfile()
    assert resolve_profile('official-continuation') == p
    assert p.supports_continuation and p.carry_penalty_history
    assert official_numerics(p) and official_numerics(base)
    assert not official_numerics(ComfyUIYuE2MPSProfile()) and not official_numerics(ScaledProfile())
    identity, reference = p.identity(), base.identity()
    assert identity['name'] == 'official-continuation' and identity['numerics'] == 'official'
    assert identity['operations'] == reference['operations']
    assert identity['upstream_revision'] == reference['upstream_revision']
    assert identity != reference
    assert base.identity() == {**reference, 'carry_penalty_history': False, 'supports_continuation': False}
    p.validate_continuation(carry=[1], chunk_seconds=1, known_latents=torch.zeros(2, 64))
    p.validate(torch.device('cpu'), backend='torch', quantization='none', rng_device='auto')
    with pytest.raises(ValueError, match='rng_device=auto'):
        p.validate(torch.device('cpu'), backend='torch', quantization='none', rng_device='cpu')


@pytest.mark.parametrize('name, graph', [('official', True), ('official-continuation', True),
                                         ('comfyui-yue2-mps-v1', False)])
def test_cuda_graph_follows_official_numerics(monkeypatch, name, graph):
    from yue2 import pipeline
    from yue2.protocol import GenerationConfig
    seen = {}

    def generate_tokens(*args, use_cuda_graph, **kwargs):
        seen['graph'] = use_cuda_graph
        return [], {}, False
    monkeypatch.setattr(pipeline, 'generate_tokens', generate_tokens)
    pipe = object.__new__(pipeline.YuE2Pipeline)
    pipe._profile, pipe.backend, pipe.progress = resolve_profile(name), 'torch', False
    pipe.generation_config = GenerationConfig()
    pipe._load_model = lambda: None
    pipe._generate([1], None, 0, 'semantic')
    assert seen['graph'] is graph
