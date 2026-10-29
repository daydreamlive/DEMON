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


# ---- comma-separated allow-list ----

import pytest

from demos.realtime_motion_graph_web.server import served_checkpoints_arg


def _pairs(args):
    return [(e.name, e.family, e.model_id) for e in served_checkpoints_arg(args)]


def test_single_value_is_one_entry_list():
    assert _pairs(["--checkpoint", "sa3-small"]) == [("sa3-small", "sa3", "small-music")]
    assert _pairs(["--checkpoint=acestep-v15-turbo"]) == [
        ("acestep-v15-turbo", "acestep", "acestep-v15-turbo"),
    ]


def test_absent_allow_list_is_none():
    assert served_checkpoints_arg(["--accel", "eager"]) is None


def test_comma_list_keeps_order():
    assert _pairs(["--checkpoint", "sa3-small,sa3-medium"]) == [
        ("sa3-small", "sa3", "small-music"),
        ("sa3-medium", "sa3", "medium"),
    ]


def test_comma_list_strips_drops_empties_and_dedupes():
    assert _pairs(["--checkpoint", " acestep-v15-turbo , ,xl,acestep-v15-turbo,"]) == [
        ("acestep-v15-turbo", "acestep", "acestep-v15-turbo"),
        ("xl", "acestep", "acestep-v15-xl-turbo"),
    ]


def test_same_model_twice_keeps_first_name():
    assert _pairs(["--checkpoint", "xl,acestep-v15-xl-turbo"]) == [
        ("xl", "acestep", "acestep-v15-xl-turbo"),
    ]


def test_mixed_family_list_exits_with_the_pair():
    with pytest.raises(SystemExit) as exc:
        served_checkpoints_arg(["--checkpoint", "sa3-medium,xl"])
    msg = str(exc.value)
    assert "'xl'" in msg and "'sa3-medium'" in msg


def test_empty_list_exits():
    with pytest.raises(SystemExit):
        served_checkpoints_arg(["--checkpoint", " , "])
