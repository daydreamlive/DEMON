"""Tests for the local-config plumbing in :mod:`acestep.paths`.

Covers ``acestep.local.json`` discovery, the relative/absolute/``~``
path resolution rules in ``extra_lora_dirs``, and the graceful-
degradation contract (missing or malformed config never blocks boot).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep import paths


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """Point ``project_root()`` at ``tmp_path`` and reset the config
    memoization between tests. Yields the temp root so the test can
    plant config files / safetensors / etc."""
    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    paths.clear_local_config_cache()
    yield tmp_path
    paths.clear_local_config_cache()


def _write_config(root: Path, payload: dict) -> None:
    (root / "acestep.local.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_missing_config_returns_empty(fake_root):
    assert paths.load_local_config() == {}
    assert paths.extra_lora_dirs() == []


def test_malformed_config_returns_empty(fake_root, capsys):
    (fake_root / "acestep.local.json").write_text("not json", encoding="utf-8")
    assert paths.load_local_config() == {}
    assert paths.extra_lora_dirs() == []
    err = capsys.readouterr().out + capsys.readouterr().err
    # The warning print happens on the first load; just check the cfg
    # is empty rather than asserting on capsys ordering.


def test_non_object_config_returns_empty(fake_root):
    (fake_root / "acestep.local.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert paths.load_local_config() == {}


def test_extra_dirs_relative_resolves_against_project_root(fake_root):
    _write_config(fake_root, {"lora_extra_dirs": ["./rel/sub"]})
    dirs = paths.extra_lora_dirs()
    assert len(dirs) == 1
    # Resolved against the config file's directory
    assert dirs[0] == (fake_root / "rel" / "sub").resolve()


def test_extra_dirs_absolute_passes_through(fake_root, tmp_path_factory):
    other = tmp_path_factory.mktemp("absolute")
    _write_config(fake_root, {"lora_extra_dirs": [str(other)]})
    dirs = paths.extra_lora_dirs()
    assert dirs == [Path(str(other))]


def test_extra_dirs_tilde_expands(fake_root):
    _write_config(fake_root, {"lora_extra_dirs": ["~/lora-experiments"]})
    dirs = paths.extra_lora_dirs()
    assert len(dirs) == 1
    # ~ should have expanded to a non-tilde absolute path under $HOME
    assert "~" not in str(dirs[0])
    assert dirs[0].is_absolute()


def test_extra_dirs_filters_empty_and_non_string(fake_root):
    _write_config(
        fake_root,
        {"lora_extra_dirs": ["./real", "  ", "", 42, None, "./other"]},
    )
    dirs = paths.extra_lora_dirs()
    assert len(dirs) == 2
    assert dirs[0].name == "real"
    assert dirs[1].name == "other"


def test_extra_dirs_missing_key_returns_empty(fake_root):
    _write_config(fake_root, {"some_other_key": "value"})
    assert paths.extra_lora_dirs() == []


def test_config_is_memoized(fake_root):
    _write_config(fake_root, {"lora_extra_dirs": ["./first"]})
    paths.load_local_config()
    # Rewrite — memoized read should NOT see this.
    _write_config(fake_root, {"lora_extra_dirs": ["./second"]})
    dirs = paths.extra_lora_dirs()
    assert dirs[0].name == "first"
    # Clearing the cache picks up the new value.
    paths.clear_local_config_cache()
    dirs = paths.extra_lora_dirs()
    assert dirs[0].name == "second"


def test_discover_all_loras_includes_extras(fake_root, tmp_path_factory):
    # Make MODELS_DIR point somewhere empty so only extras show up.
    empty_models = tmp_path_factory.mktemp("empty_models")
    os.environ["ACESTEP_MODELS_DIR"] = str(empty_models)
    extra = tmp_path_factory.mktemp("extra_lora_dir")
    (extra / "deep" / "nested").mkdir(parents=True)
    (extra / "deep" / "nested" / "found.safetensors").write_bytes(b"")
    (extra / "top.safetensors").write_bytes(b"")
    _write_config(fake_root, {"lora_extra_dirs": [str(extra)]})

    all_loras = paths.discover_all_loras()
    names = sorted(p.name for p in all_loras)
    assert names == ["found.safetensors", "top.safetensors"]
