"""The on_model_ready seam: adapters attach here rather than to a private attribute."""
from contextlib import nullcontext

import torch

from yue2.pipeline import YuE2Pipeline


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2))


def make_pipe(model=None):
    """A pipeline shell with only what _load_model touches, so no weights are needed."""
    pipe = object.__new__(YuE2Pipeline)
    pipe._model = model
    pipe.device = torch.device("cpu")
    pipe.quantization = None
    pipe.load_timing = {}
    pipe.on_model_ready = []
    pipe._status = lambda *a, **k: nullcontext()
    return pipe


def test_default_is_inert():
    pipe = make_pipe(Tiny())
    assert pipe.on_model_ready == []
    assert pipe._load_model() is pipe._model


def test_hook_fires_per_stage_with_stage_identity():
    pipe = make_pipe(Tiny())
    seen = []
    pipe.on_model_ready.append(lambda model, **kw: seen.append((model, kw)))

    pipe._load_model()
    pipe._load_model(for_nar=True)

    assert [kw["for_nar"] for _, kw in seen] == [False, True], "NAR stage must be distinguishable"
    assert all(model is pipe._model for model, _ in seen)
    # Model was already resident, so neither call constructed or moved it.
    assert [kw["fresh"] for _, kw in seen] == [False, False]
    assert [kw["loaded"] for _, kw in seen] == [False, False]


def test_hook_sees_a_move_onto_the_device():
    pipe = make_pipe(Tiny())
    pipe.device = torch.device("meta")  # forces the device-mismatch branch
    seen = []
    pipe.on_model_ready.append(lambda model, **kw: seen.append(kw))
    pipe._load_model()
    assert seen[0]["loaded"] is True
    assert seen[0]["fresh"] is False


def test_mutation_in_place_reaches_the_caller():
    pipe = make_pipe(Tiny())

    def bump(model, **kw):
        with torch.no_grad():
            model.w += 1.0

    pipe.on_model_ready.append(bump)
    model = pipe._load_model()
    assert torch.equal(model.w, torch.ones(2))


def test_hooks_run_in_registration_order():
    pipe = make_pipe(Tiny())
    order = []
    pipe.on_model_ready.extend([lambda m, **k: order.append("a"), lambda m, **k: order.append("b")])
    pipe._load_model()
    assert order == ["a", "b"]
