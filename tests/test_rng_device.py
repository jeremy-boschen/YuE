"""Forcing a standard generator so a seed names a take on every backend."""
import pytest
import torch

from yue2.protocol import GenerationConfig
from yue2.sampling import rng_device_for

CPU = torch.device("cpu")
CUDA = torch.device("cuda:0")
MPS = torch.device("mps")


class TestResolution:
    def test_auto_is_the_released_behaviour(self):
        # Compute device on CPU and CUDA, CPU everywhere else.
        assert rng_device_for(CPU, "auto") == CPU
        assert rng_device_for(CUDA, "auto") == CUDA
        assert rng_device_for(MPS, "auto") == CPU

    def test_auto_is_the_default_and_none_means_auto(self):
        assert GenerationConfig().rng_device == "auto"
        assert rng_device_for(MPS) == rng_device_for(MPS, None) == rng_device_for(MPS, "auto")

    def test_cpu_is_the_portable_choice(self):
        # The point of the knob: one stream no matter what computes the logits.
        assert all(rng_device_for(d, "cpu") == CPU for d in (CPU, CUDA, MPS))

    def test_device_always_follows_the_compute_device(self):
        assert rng_device_for(CUDA, "device") == CUDA
        assert rng_device_for(MPS, "device") == MPS

    def test_only_auto_moves_the_probabilities_on_mps(self):
        # generate_tokens copies to the RNG device exactly when they disagree.
        assert rng_device_for(MPS, "auto") != MPS
        assert rng_device_for(MPS, "device") == MPS
        assert rng_device_for(CUDA, "auto") == CUDA, "must not add a copy to the CUDA path"

    def test_unknown_choice_is_rejected(self):
        with pytest.raises(ValueError, match="rng_device"):
            rng_device_for(CPU, "gpu")


class TestConfig:
    def test_invalid_value_rejected_at_construction(self):
        with pytest.raises(ValueError, match="rng_device"):
            GenerationConfig(rng_device="gpu")

    @pytest.mark.parametrize("choice", ["auto", "cpu", "device"])
    def test_round_trips_so_a_run_records_it(self, choice):
        config = GenerationConfig(rng_device=choice)
        assert config.to_dict()["rng_device"] == choice
        assert GenerationConfig.from_dict(config.to_dict()).rng_device == choice

    def test_older_config_without_the_key_still_loads(self):
        data = GenerationConfig().to_dict()
        del data["rng_device"]
        assert GenerationConfig.from_dict(data).rng_device == "auto"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_streams_actually_differ_between_devices():
    """Why the knob exists: same seed, different device, different draws."""
    probabilities = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

    def draws(device):
        generator = torch.Generator(device=device).manual_seed(831001)
        p = probabilities.to(device)
        return [int(torch.multinomial(p, 1, generator=generator)) for _ in range(32)]

    assert draws(CPU) != draws(MPS), "if these ever match, the knob is redundant"
    assert draws(CPU) == draws(CPU), "the CPU stream must be stable to be worth forcing"
