"""YuE2 runtime location: upstream source, weights, engines, identity.

YuE2's model code is not a DEMON dependency. It is imported from a
checkout of upstream YuE (multimodal-art-projection/YuE @
:data:`CODE_REVISION`) named by an environment variable, with
``tiktoken`` optionally supplied from an extra directory. Weights are
local directories only; nothing here downloads.

Environment:

* ``DEMON_YUE2_ROOT``: weights root holding ``YuE2-3B/`` and
  ``YuE2-Vae/`` (m-a-p/YuE2-3B @ :data:`MODEL_REVISION`, m-a-p/YuE2-Vae
  @ :data:`VAE_REVISION`), each with its ``weights_manifest.json``.
* ``DEMON_YUE2_YUE_SRC``: upstream ``src`` directory (the one that
  contains the ``yue2`` package).
* ``DEMON_YUE2_EXTRA_PATH``: optional extra import directories
  (``os.pathsep``-separated), e.g. one holding ``tiktoken``.
* ``DEMON_YUE2_TRT_DIR``: optional engine directory holding
  ``flexible_song/velocity.trt`` (acoustic NAR) and
  ``vae_fp32_t37.trt`` (VAE window); see ``acestep.engine.trt.yue2_build``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

CODE_REVISION = "0edaf2f4053ef4731334b8329834b107977f9637"
MODEL_REVISION = "29b3558dd46954a0cd9021dc76d5c91864a0f1c7"
VAE_REVISION = "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a"

ROOT_ENV = "DEMON_YUE2_ROOT"
YUE_SRC_ENV = "DEMON_YUE2_YUE_SRC"
EXTRA_PATH_ENV = "DEMON_YUE2_EXTRA_PATH"
TRT_DIR_ENV = "DEMON_YUE2_TRT_DIR"

MODEL_DIR = "YuE2-3B"
VAE_DIR = "YuE2-Vae"
_WEIGHT_FILES = ("config.json", "model.safetensors", "weights_manifest.json")


def weights_root() -> Optional[Path]:
    value = os.environ.get(ROOT_ENV, "").strip()
    return Path(value) if value else None


def trt_dir() -> Optional[Path]:
    value = os.environ.get(TRT_DIR_ENV, "").strip()
    return Path(value) if value else None


def import_paths() -> list:
    """Directories the upstream import needs, in sys.path order."""
    paths = []
    src = os.environ.get(YUE_SRC_ENV, "").strip()
    if src:
        paths.append(src)
    for extra in os.environ.get(EXTRA_PATH_ENV, "").split(os.pathsep):
        if extra.strip():
            paths.append(extra.strip())
    return paths


def ensure_import_paths() -> None:
    """Put the upstream source (and extras) on sys.path, idempotently."""
    for path in reversed(import_paths()):
        if path not in sys.path:
            sys.path.insert(0, path)


def missing_modules() -> list:
    """Upstream modules that cannot be imported after
    :func:`ensure_import_paths` (empty when everything resolves)."""
    ensure_import_paths()
    return [m for m in ("yue2", "tiktoken") if importlib.util.find_spec(m) is None]


def _manifest_entry(path: Path) -> dict:
    manifest = json.loads((path / "weights_manifest.json").read_text())
    return manifest["files"]["model.safetensors"]


def weights_status(root: Optional[Path]) -> Tuple[bool, str]:
    """Cheap offline check: both checkpoint dirs exist with their files,
    and each ``model.safetensors`` has the byte size its manifest
    records. The SHA256 check happens at load (:func:`load_verified`)."""
    if root is None:
        return False, f"{ROOT_ENV} is not set (weights root with {MODEL_DIR}/ and {VAE_DIR}/)"
    for name in (MODEL_DIR, VAE_DIR):
        path = root / name
        for file in _WEIGHT_FILES:
            if not (path / file).is_file():
                return False, f"missing {path / file}"
        try:
            expected = int(_manifest_entry(path)["bytes"])
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            return False, f"unreadable manifest in {path}: {exc}"
        actual = (path / "model.safetensors").stat().st_size
        if actual != expected:
            return False, f"{path / 'model.safetensors'} is {actual} bytes, manifest says {expected}"
    return True, f"weights at {root}"


def load_verified(path: Path, *, vae: bool = False, device: str = "cuda"):
    """Load one checkpoint dir after a SHA256 check against its manifest.

    Reads the file once into RAM (hash and load from the same bytes),
    builds the module on the meta device and assigns the state dict, so
    no random initialization is ever allocated. The VAE loads decoder
    only (the acoustic stage never encodes). Returns
    ``(module, identity)``; ``identity`` is the dict upstream's
    ``model_identity`` would return.
    """
    import torch
    from safetensors.torch import load

    ensure_import_paths()
    from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig
    from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM

    config_bytes = (path / "config.json").read_bytes()
    data = (path / "model.safetensors").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    expected = _manifest_entry(path)
    if digest != expected["sha256"] or len(data) != int(expected["bytes"]):
        raise ValueError(f"YuE2 checkpoint integrity failure: {path}")
    identity = {
        "files": {"model.safetensors": {"sha256": digest, "bytes": len(data)}},
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    state = load(data)
    del data
    config = json.loads(config_bytes)
    with torch.device("meta"):
        model = (
            YuE2VAE(YuE2VAEConfig(**config), decoder_only=True) if vae
            else YuE2ForCausalLM(YuE2Config(**config))
        )
    if vae:
        state = {k: v for k, v in state.items() if k.startswith("decoder.")}
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device=device, dtype=torch.float32 if vae else torch.bfloat16)
    model.eval().requires_grad_(False)
    return model, identity
