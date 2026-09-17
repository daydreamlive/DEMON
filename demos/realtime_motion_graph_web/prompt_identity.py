"""Version keys for local text inference; checkpoint paths are not identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def checkpoint_digest(directory: str) -> str:
    root = Path(directory)
    files = sorted(p for p in root.iterdir() if p.is_file() and (
        p.suffix in {".safetensors", ".bin"}
        or p.name in {"config.json", "generation_config.json", "tokenizer.json",
                      "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                      "spiece.model", "model.safetensors.index.json", "pytorch_model.bin.index.json"}
    ))
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        return ""
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def revision_for(checkpoint: str, sampler: str, guard: str, runtime: dict) -> str:
    if not checkpoint:
        return ""
    payload = [checkpoint, sampler, guard, runtime]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
