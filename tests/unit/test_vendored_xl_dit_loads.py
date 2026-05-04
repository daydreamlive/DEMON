"""Smoke test for the vendored XL turbo DiT load path.

Mirrors test_vendored_dit_loads.py for the XL checkpoint. Asserts the
load succeeds via ModelContext._load_dit's auto_map dispatch and that
the returned class comes from acestep.models.modeling_acestep_v15_xl_turbo
(not the upstream .py file in the checkpoint directory).

Skips if the XL checkpoint is not on disk so CI without weights stays green.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.paths import checkpoints_dir


CHECKPOINT_NAME = "acestep-v15-xl-turbo"


def _checkpoint_path() -> Path:
    return checkpoints_dir() / CHECKPOINT_NAME


@pytest.fixture(scope="module")
def xl_turbo_checkpoint() -> Path:
    path = _checkpoint_path()
    if not (path / "config.json").exists():
        pytest.skip(f"checkpoint not on disk: {path}")
    # XL is sharded; the index file is the canonical "weights present" marker.
    if not (path / "model.safetensors.index.json").exists():
        pytest.skip(f"weights not on disk: {path / 'model.safetensors.index.json'}")
    return path


def test_vendored_xl_dit_loads(xl_turbo_checkpoint: Path) -> None:
    from acestep.models.modeling_acestep_v15_xl_turbo import (
        AceStepConditionGenerationModel,
    )

    model = AceStepConditionGenerationModel.from_pretrained(
        str(xl_turbo_checkpoint),
        attn_implementation="eager",
        dtype="bfloat16",
    )

    assert isinstance(model, AceStepConditionGenerationModel)
    assert model.config.model_type == "acestep"
    assert type(model).__module__ == "acestep.models.modeling_acestep_v15_xl_turbo"


def test_load_dit_dispatches_to_xl_class(xl_turbo_checkpoint: Path) -> None:
    """End-to-end check that ModelContext._load_dit's auto_map dispatch
    routes the XL checkpoint to the XL vendored class."""
    from acestep.engine.model_context import ModelContext
    from acestep.models.modeling_acestep_v15_xl_turbo import (
        AceStepConditionGenerationModel,
    )

    ctx = ModelContext.__new__(ModelContext)
    model = ctx._load_dit(str(xl_turbo_checkpoint), attn_impl="eager")

    assert isinstance(model, AceStepConditionGenerationModel)
    assert type(model).__module__ == "acestep.models.modeling_acestep_v15_xl_turbo"
