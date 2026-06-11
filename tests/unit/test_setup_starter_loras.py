"""demon-setup's starter LoRA pack: manifest shape, idempotent fetch,
non-fatal failures. Network is monkeypatched out; the step's contract
is that it lands each repo under loras/<repo-name>/, skips repos that
already have a .safetensors, and never raises (the demo runs fine
without LoRAs, so the pack must not be able to block setup).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep import setup as demon_setup
from acestep.paths import loras_dir


@pytest.fixture
def tmp_models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    return tmp_path


def test_manifest_is_well_formed():
    repos = demon_setup.STARTER_LORA_REPOS
    assert len(repos) == 16
    assert all(len(r.split("/")) == 2 for r in repos)
    # Destination dirs are derived from the repo name; collisions would
    # silently merge two LoRAs into one directory.
    names = [r.rsplit("/", 1)[-1] for r in repos]
    assert len(set(names)) == len(names)


def test_downloads_each_repo_then_skips_existing(tmp_models_dir, monkeypatch):
    calls: list[str] = []

    def fake_snapshot(*, repo_id, local_dir, allow_patterns):
        calls.append(repo_id)
        assert "*.safetensors" in allow_patterns
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "fake.safetensors").write_bytes(b"")
        (d / "fake.metadata.json").write_text("{}", encoding="utf-8")
        return local_dir

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)

    demon_setup._download_starter_loras()
    assert calls == list(demon_setup.STARTER_LORA_REPOS)
    for repo in demon_setup.STARTER_LORA_REPOS:
        name = repo.rsplit("/", 1)[-1]
        assert (loras_dir() / name / "fake.safetensors").exists()

    # Idempotent re-run: every repo already has a .safetensors → no fetches.
    calls.clear()
    demon_setup._download_starter_loras()
    assert calls == []


def test_failures_never_block_setup(tmp_models_dir, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    # Must not raise; setup continues to the engine step regardless.
    demon_setup._download_starter_loras()
    assert not list(loras_dir().rglob("*.safetensors"))


def test_env_gate_for_managed_deployments(monkeypatch):
    # Pods curate their own LoRA library; DEMON_SKIP_STARTER_LORAS=1 in
    # the pod env must disable the pack without a CLI flag.
    monkeypatch.delenv("DEMON_SKIP_STARTER_LORAS", raising=False)
    assert demon_setup._env_skip_loras() is False
    for truthy in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("DEMON_SKIP_STARTER_LORAS", truthy)
        assert demon_setup._env_skip_loras() is True
    for falsy in ("0", "false", "no", ""):
        monkeypatch.setenv("DEMON_SKIP_STARTER_LORAS", falsy)
        assert demon_setup._env_skip_loras() is False
