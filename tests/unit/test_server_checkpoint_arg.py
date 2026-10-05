"""``--checkpoint`` parsing in the realtime server: both the space and
the ``=`` form select the family; absence leaves the default."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.streaming.families import resolve_checkpoint
from demos.realtime_motion_graph_web.server import checkpoint_arg


def test_space_form():
    assert checkpoint_arg(["--accel", "eager", "--checkpoint", "xl"]) == "xl"


def test_equals_form():
    assert checkpoint_arg(["--checkpoint=sa3-small", "--accel", "eager"]) == "sa3-small"


def test_absent():
    assert checkpoint_arg(["--accel", "eager"]) is None


def test_sa3_small_alias_resolves_to_sa3_family():
    for args in (["--checkpoint", "sa3-small"], ["--checkpoint=sa3-small"]):
        assert resolve_checkpoint(checkpoint_arg(args)) == ("sa3", "small-music")
