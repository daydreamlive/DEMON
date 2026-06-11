"""The `--preset minimal` build matrix (dry-run, no GPU).

Runs `python -m acestep.engine.trt.build --preset minimal --dry-run` in a
subprocess against a temp ACESTEP_MODELS_DIR and asserts the printed
matrix is exactly the minimal engine set: the 60s profile (decoder + VAE
encode + VAE decode) plus the fixed 1s windowed VAE decode — no 120/240s
profiles. This is the set `demon-setup` builds.
"""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent


def _run_dry_run(tmp_path, *extra_args: str) -> str:
    env = dict(os.environ)
    env["ACESTEP_MODELS_DIR"] = str(tmp_path)
    # Repo root first so a sibling ACE-Step checkout can't shadow acestep.
    env["PYTHONPATH"] = str(_REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable, "-m", "acestep.engine.trt.build",
            "--preset", "minimal", "--dry-run", *extra_args,
        ],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"dry-run failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout


def test_minimal_preset_matrix(tmp_path):
    out = _run_dry_run(tmp_path)
    # Builds: the full 60s profile plus the fixed 1s windowed decode.
    assert "VAE encode 60s" in out
    assert "VAE decode 60s" in out
    assert "Decoder turbo 60s, refit" in out
    assert "VAE decode fixed 1s (25 fr)" in out
    # Skips: the larger profiles and opt-in extras.
    assert "120s" not in out
    assert "240s" not in out
    assert "DreamVAE" not in out


def test_minimal_preset_duration_override_widens(tmp_path):
    out = _run_dry_run(tmp_path, "--duration", "60", "120")
    assert "VAE encode 60s" in out
    assert "VAE decode 60s" in out
    assert "VAE encode 120s" in out
    assert "VAE decode 120s" in out
    assert "Decoder turbo 120s, refit" in out
