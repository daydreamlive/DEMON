"""DEMON one-command bootstrap.

``uv run demon-setup`` takes a fresh clone to a runnable realtime web
demo in one idempotent pass:

    1. Doctor   - GPU / CUDA / TensorRT / Node / disk checks, and an
                  explicit printout of where models live on this machine.
    2. Models   - downloads the ACE-Step v1.5 checkpoints (HuggingFace,
                  ModelScope fallback) via :mod:`acestep.model_downloader`.
    3. Engines  - builds the minimal TensorRT engine set
                  (``acestep.engine.trt.build --preset minimal``):
                  60 s decoder + 60 s VAE encode + fixed 1 s windowed
                  VAE decode. Existing engines are skipped.
    4. Summary  - what's on disk and the exact command to launch the demo.

Every step skips work that's already done, so re-running after a partial
failure (network drop mid-download, OOM mid-build) resumes where it
left off.

Engine builds target the default ``acestep-v15-turbo`` checkpoint. The
XL checkpoint needs an FP8 calibration artifact and is a power-user
path - see docs/INSTALL.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

# Reused by the demo server's preflight banner so the "how do I fix
# this" hint can't drift from what this script actually does.
SETUP_COMMAND = "uv run demon-setup"
DEMO_COMMAND = "uv run python -u -m demos.realtime_motion_graph_web.run"

_MIN_FREE_DISK_GB = 40.0   # ~18 GB checkpoints + ~10 GB ONNX/engines + slack
_ADVISORY_VRAM_GB = 16.0   # 60 s decoder build peaks ~13.5 GB workspace


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _header(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _doctor() -> bool:
    """Environment checks. Returns False on a hard failure (no usable
    GPU stack); advisory problems only warn."""
    from acestep.paths import models_dir

    _header("1/3  Environment check")

    hard_fail = False

    # Where everything lands on this machine. Printed loudly because
    # "models are in the repo's checkpoints/ dir" is the single most
    # common wrong assumption (humans and LLM helpers alike).
    md = models_dir()
    env_override = os.environ.get("ACESTEP_MODELS_DIR")
    if env_override:
        _ok(f"models dir: {md}  (from ACESTEP_MODELS_DIR)")
    else:
        _ok(f"models dir: {md}  (default; override with ACESTEP_MODELS_DIR)")

    # Python - uv pins ==3.11.* via pyproject, but a bare-venv install
    # can drift.
    v = sys.version_info
    if (v.major, v.minor) == (3, 11):
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _warn(
            f"Python {v.major}.{v.minor} - DEMON is pinned to 3.11; "
            f"use `uv sync` / `uv run` so the right interpreter is used"
        )

    # Torch + CUDA.
    try:
        import torch
    except Exception as exc:
        _fail(f"PyTorch import failed: {exc}. Run `uv sync` first.")
        return False
    if not torch.cuda.is_available():
        _fail(
            "No CUDA GPU visible to PyTorch. DEMON requires an NVIDIA "
            "GPU (tested on RTX 3090 / 4090 / 5090). If you have one, "
            "check the driver: `nvidia-smi` should list it."
        )
        hard_fail = True
    else:
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        _ok(f"GPU: {props.name} ({vram_gb:.0f} GB VRAM)")
        if vram_gb < _ADVISORY_VRAM_GB:
            _warn(
                f"{vram_gb:.0f} GB VRAM is below the tested floor "
                f"(~{_ADVISORY_VRAM_GB:.0f} GB). The 60 s decoder engine "
                f"build peaks around 13.5 GB of workspace; builds and "
                f"inference may OOM."
            )

    # TensorRT.
    try:
        import tensorrt
        _ok(f"TensorRT {tensorrt.__version__}")
    except Exception as exc:
        _fail(f"TensorRT import failed: {exc}. Run `uv sync` first.")
        hard_fail = True

    # Disk space where models will land.
    probe = md
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free_gb = shutil.disk_usage(probe).free / (1024 ** 3)
        if free_gb < _MIN_FREE_DISK_GB:
            _warn(
                f"{free_gb:.0f} GB free at {probe} - setup needs roughly "
                f"{_MIN_FREE_DISK_GB:.0f} GB (checkpoints + ONNX + engines)"
            )
        else:
            _ok(f"disk: {free_gb:.0f} GB free at {probe}")
    except OSError:
        _warn(f"could not stat free disk space at {probe}")

    # Node (web demo only - warn, never fail).
    node = shutil.which("node")
    if node is None:
        _warn(
            "Node.js not found on PATH. Only needed for the bundled web "
            "demo; install Node 20+ from https://nodejs.org before "
            f"running `{DEMO_COMMAND}`."
        )
    else:
        try:
            ver = subprocess.run(
                [node, "--version"], capture_output=True, text=True,
                timeout=10,
            ).stdout.strip()
            _ok(f"Node.js {ver}")
        except Exception:
            _ok(f"Node.js found at {node}")

    return not hard_fail


def _download_models() -> bool:
    from acestep.model_downloader import ensure_main_model
    from acestep.paths import checkpoints_dir

    _header("2/3  Model checkpoints (ACE-Step v1.5)")
    print(f"  destination: {checkpoints_dir()}")
    print("  source: huggingface.co/ACE-Step/Ace-Step1.5 "
          "(ModelScope fallback), ~18 GB on first run\n")

    success, msg = ensure_main_model()
    if success:
        _ok(msg)
        return True
    _fail(msg)
    _fail(
        "Model download failed. Re-run `demon-setup` to resume, or "
        "download manually: `uv run acestep-download`."
    )
    return False


def _build_engines(extra_args: list[str]) -> bool:
    from acestep.paths import trt_engines_dir

    _header("3/3  TensorRT engines (minimal preset)")
    print(f"  destination: {trt_engines_dir()}")
    print("  set: 60s decoder + 60s VAE encode + fixed 1s windowed VAE "
          "decode")
    print("  ONNX is fetched prebuilt from huggingface.co/daydreamlive/"
          "demon-onnx;")
    print("  expect roughly 10-30 minutes of build time on first run.\n")

    cmd = [
        sys.executable, "-m", "acestep.engine.trt.build",
        "--preset", "minimal", *extra_args,
    ]
    print(f"  running: {' '.join(cmd)}\n")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        _fail(
            f"engine build exited with {rc}. Re-run `{SETUP_COMMAND}` to "
            "resume (finished engines are skipped), or run the build "
            "command above directly for more control."
        )
        return False
    return True


def _summary(*, engines_skipped: bool) -> None:
    from acestep.paths import models_dir, trt_engines_dir

    _header("Setup complete")
    print(f"  models dir: {models_dir()}")
    trt_dir = trt_engines_dir()
    if trt_dir.is_dir():
        engines = sorted(
            d.name for d in trt_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
            and (d / f"{d.name}.engine").exists()
        )
        if engines:
            print("  engines:")
            for name in engines:
                print(f"    {name}")
    if engines_skipped:
        print("\n  Engines were skipped (--skip-engines). The demo's "
              "default TRT mode")
        print("  needs them; either build later with")
        print("    uv run python -m acestep.engine.trt.build --preset minimal")
        print(f"  or launch with `-- --accel compile` (slow warmup, no "
              f"engines needed).")
    print()
    print("  Launch the web demo:")
    print(f"    {DEMO_COMMAND}")
    print("  then open http://localhost:6660")
    print()


def main() -> int:
    # Piped stdout (CI, `| tee`, log wrappers) is block-buffered by
    # default, which holds phase banners back until kilobytes accumulate
    # and lets the subprocesses' unbuffered stderr interleave ahead of
    # them. Line-buffer it, and never die on a character the console's
    # code page can't encode (legacy Windows pipes are cp125x). stderr
    # needs neither: Python starts it line-buffered with
    # errors="backslashreplace", which already can't raise and keeps
    # more fidelity than "replace" would.
    try:
        sys.stdout.reconfigure(line_buffering=True, errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

    parser = argparse.ArgumentParser(
        prog="demon-setup",
        description=(
            "One-command DEMON bootstrap: environment check, model "
            "download, minimal TensorRT engine build. Idempotent - "
            "re-run any time; finished work is skipped."
        ),
    )
    parser.add_argument(
        "--skip-doctor", action="store_true",
        help="Skip the environment checks.",
    )
    parser.add_argument(
        "--skip-models", action="store_true",
        help="Skip the checkpoint download step.",
    )
    parser.add_argument(
        "--skip-engines", action="store_true",
        help="Skip the TensorRT engine build (e.g. when planning to run "
             "with --accel compile).",
    )
    parser.add_argument(
        "--duration", nargs="*", type=int, default=None,
        help="Extra engine durations to build beyond the minimal 60s "
             "(forwarded to the engine builder), e.g. --duration 60 120.",
    )
    args = parser.parse_args()

    print()
    print("DEMON setup - streaming music diffusion for ACE-Step v1.5")

    if not args.skip_doctor:
        if not _doctor():
            print()
            print("Environment check failed; fix the [FAIL] items above "
                  "and re-run.")
            print("(Use --skip-doctor to proceed anyway at your own risk.)")
            return 1

    if not args.skip_models:
        if not _download_models():
            return 1

    if not args.skip_engines:
        extra: list[str] = []
        if args.duration:
            extra.extend(["--duration", *[str(d) for d in args.duration]])
        if not _build_engines(extra):
            return 1

    _summary(engines_skipped=args.skip_engines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
