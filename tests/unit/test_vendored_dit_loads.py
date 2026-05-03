"""Smoke test for the vendored turbo DiT load path.

Catches the failure class where someone (a pod sync, a manual cp, etc.)
overwrites .py files in the checkpoint directory with versions that
import from a Python package layout DEMON does not provide. With the
vendored class loaded directly from `acestep.models`, the contents of
`<checkpoint>/configuration_acestep_v15.py` and
`<checkpoint>/modeling_acestep_v15_turbo.py` are no longer executed,
so this test asserts the load succeeds *and* the loaded class is the
vendored one.

Skips if the checkpoint isn't on disk so CI without weights stays green.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.paths import checkpoints_dir


CHECKPOINT_NAME = "acestep-v15-turbo"


def _checkpoint_path() -> Path:
    return checkpoints_dir() / CHECKPOINT_NAME


@pytest.fixture(scope="module")
def turbo_checkpoint() -> Path:
    path = _checkpoint_path()
    if not (path / "config.json").exists():
        pytest.skip(f"checkpoint not on disk: {path}")
    if not (path / "model.safetensors").exists():
        pytest.skip(f"weights not on disk: {path / 'model.safetensors'}")
    return path


def test_vendored_dit_loads(turbo_checkpoint: Path) -> None:
    from acestep.models.modeling_acestep_v15_turbo import (
        AceStepConditionGenerationModel,
    )

    model = AceStepConditionGenerationModel.from_pretrained(
        str(turbo_checkpoint),
        attn_implementation="eager",
        dtype="bfloat16",
    )

    assert isinstance(model, AceStepConditionGenerationModel)
    assert model.config.model_type == "acestep"
    assert type(model).__module__ == "acestep.models.modeling_acestep_v15_turbo"


def test_load_dit_uses_vendored_class(turbo_checkpoint: Path) -> None:
    """End-to-end check that ModelContext._load_dit returns the vendored class."""
    from acestep.engine.model_context import ModelContext
    from acestep.models.modeling_acestep_v15_turbo import (
        AceStepConditionGenerationModel,
    )

    ctx = ModelContext.__new__(ModelContext)
    model = ctx._load_dit(str(turbo_checkpoint), attn_impl="eager")

    assert isinstance(model, AceStepConditionGenerationModel)
    assert type(model).__module__ == "acestep.models.modeling_acestep_v15_turbo"
