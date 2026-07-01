"""Property tests for the SA3 fast-load optimization (skip_param_init).

These use a tiny toy module rather than the real SA3 checkpoint, so they
run in CI without GPU or the 9 GB weights: what we need to guarantee is
the *contract* of skip_param_init — random inits are no-op'd during
construction, deterministic buffers are untouched, and a
construct-then-load is bit-identical to the un-skipped path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.init as init  # noqa: E402

from acestep.engine import sa3_helpers  # noqa: E402


class _Toy(nn.Module):
    """Stand-in for the SA3 model shape: random-init'd params (Linear/
    Embedding call nn.init in reset_parameters) plus a deterministic,
    arange-computed buffer (like a rotary ``inv_freq``)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 32)
        self.emb = nn.Embedding(16, 64)
        self.register_buffer(
            "inv_freq", 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
        )


def test_random_fills_are_noop_inside_context():
    t = torch.ones(10, 10)
    with sa3_helpers.skip_param_init():
        init.kaiming_uniform_(t)
        init.xavier_uniform_(t)
        init.normal_(t)
        init.uniform_(t)
        init.trunc_normal_(t)
        t.normal_()
        t.uniform_()
        assert torch.equal(t, torch.ones(10, 10)), "fills should be no-ops"


def test_fills_restored_after_context():
    with sa3_helpers.skip_param_init():
        pass
    t = torch.ones(10, 10)
    init.normal_(t)
    assert not torch.equal(t, torch.ones(10, 10)), "init must work again after"
    # in-place Tensor methods restored too
    assert torch.Tensor.normal_ is not (lambda self: self)
    u = torch.ones(10)
    u.uniform_()
    assert not torch.equal(u, torch.ones(10))


def test_fills_restored_on_exception():
    with pytest.raises(RuntimeError):
        with sa3_helpers.skip_param_init():
            raise RuntimeError("boom")
    t = torch.ones(8)
    init.normal_(t)
    assert not torch.equal(t, torch.ones(8)), "must restore even if body raises"


def test_construct_then_load_is_bit_identical():
    """The property that makes the optimization safe: skipping init and
    then loading the checkpoint yields exactly the un-skipped result,
    for every parameter AND buffer."""
    torch.manual_seed(0)
    reference = _Toy().eval()
    saved = {k: v.clone() for k, v in reference.state_dict().items()}

    # Mirror production: construct AND load inside the context.
    with sa3_helpers.skip_param_init():
        fast = _Toy()
        missing, unexpected = fast.load_state_dict(saved, strict=False)
    fast.eval()

    assert list(missing) == [], f"unexpected missing keys: {missing}"
    assert list(unexpected) == [], f"unexpected keys: {unexpected}"
    for name, ref_val in reference.state_dict().items():
        assert torch.equal(fast.state_dict()[name], ref_val), name


def test_computed_buffer_unaffected_by_skip():
    """A deterministic (arange-computed) buffer must be identical whether
    or not init is skipped — the no-ops only touch random fills."""
    torch.manual_seed(0)
    normal = _Toy()
    with sa3_helpers.skip_param_init():
        skipped = _Toy()
    assert torch.equal(normal.inv_freq, skipped.inv_freq)
