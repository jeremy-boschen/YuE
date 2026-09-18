"""Carry affects penalties, not the returned tokens or request-local RNG lifetime."""
from types import SimpleNamespace

import torch

from yue2.sampling import generate_tokens
from yue2.protocol import CODEC_OFFSET, VOCAB_SIZE, Sampling
from yue2.sampling import generate_tokens as engine_generate


class FixedLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1, head_dim=1)

    def forward(self, ids, past_key_values=None, **kwargs):
        logits = torch.full((1, 1, VOCAB_SIZE), -100.)
        logits[..., CODEC_OFFSET] = 10.
        logits[..., CODEC_OFFSET + 1] = 9.
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def test_carried_history_changes_first_choice_and_expires():
    model = FixedLogits()
    sampling = Sampling(temperature=0, repetition_penalty=2, penalty_window=2,
                        min_tokens=4, max_tokens=4)
    prior = [CODEC_OFFSET, CODEC_OFFSET]
    prefix = [42] + prior
    old = engine_generate(model, prefix, sampling, 777, 'semantic')[0]
    empty = generate_tokens(model, prefix, sampling, 777, 'semantic')[0]
    carried = generate_tokens(model, prefix, sampling, 777, 'semantic', prior=prior)[0]
    assert old == empty == [CODEC_OFFSET, CODEC_OFFSET + 1, CODEC_OFFSET, CODEC_OFFSET]
    assert carried == [CODEC_OFFSET + 1, CODEC_OFFSET, CODEC_OFFSET, CODEC_OFFSET + 1]
    assert prior == [CODEC_OFFSET, CODEC_OFFSET]
    assert len(carried) == 4  # prior tokens are not returned as new generation


def test_continuation_seed_is_request_local():
    sampling = Sampling(min_tokens=8, max_tokens=8)
    model = FixedLogits()
    args = (model, [42, CODEC_OFFSET], sampling, 779, 'semantic')
    first = generate_tokens(*args, prior=[CODEC_OFFSET])[0]
    torch.rand(1000)  # unrelated process-global random draws must not perturb it
    assert generate_tokens(*args, prior=[CODEC_OFFSET])[0] == first
