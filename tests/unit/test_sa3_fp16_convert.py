"""Tests for the SA3 fp16 checkpoint converter.

Uses tiny synthetic safetensors rather than the real 9 GB checkpoint, so
it runs in CI without GPU or weights. The contract we lock:
  * float32 tensors -> float16, values == the plain fp16 cast
  * bf16 / int / bool tensors preserved verbatim (bf16 gains no space)
  * safetensors __metadata__ and sidecar files preserved
  * idempotent (a second pass converts nothing)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "sa3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import sa3_convert_fp16 as conv  # noqa: E402


def _make_checkpoint(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    dit = {
        "w": torch.randn(8, 8, dtype=torch.float32),
        "b": torch.randn(8, dtype=torch.float32),
        "step": torch.tensor([3, 1, 4], dtype=torch.int64),  # non-float
    }
    save_file(dit, str(root / "model.safetensors"), metadata={"format": "pt", "tag": "x"})
    (root / "model_config.json").write_text('{"model": {}}', encoding="utf-8")
    enc = root / "t5gemma-b-b-ul2"
    enc.mkdir(exist_ok=True)
    # already-2-byte encoder weights: must be left untouched
    save_file(
        {"h": torch.randn(4, 4).to(torch.bfloat16)},
        str(enc / "model.safetensors"),
    )
    (enc / "tokenizer.json").write_text("{}", encoding="utf-8")
    return dit


def test_fp32_converted_bf16_and_int_preserved(tmp_path):
    src = tmp_path / "src"
    ref = _make_checkpoint(src)
    dst = tmp_path / "dst"

    conv.convert_checkpoint(src, dst)

    out = load_file(str(dst / "model.safetensors"))
    assert out["w"].dtype == torch.float16
    assert out["b"].dtype == torch.float16
    assert out["step"].dtype == torch.int64                     # non-float kept
    assert torch.equal(out["w"], ref["w"].to(torch.float16))    # faithful cast
    assert torch.equal(out["step"], ref["step"])

    enc = load_file(str(dst / "t5gemma-b-b-ul2" / "model.safetensors"))
    assert enc["h"].dtype == torch.bfloat16                     # bf16 untouched


def test_metadata_and_sidecars_preserved(tmp_path):
    src = tmp_path / "src"
    _make_checkpoint(src)
    dst = tmp_path / "dst"

    conv.convert_checkpoint(src, dst)

    with safe_open(str(dst / "model.safetensors"), framework="pt") as f:
        assert f.metadata().get("tag") == "x"
    assert (dst / "model_config.json").read_text(encoding="utf-8") == '{"model": {}}'
    assert (dst / "t5gemma-b-b-ul2" / "tokenizer.json").exists()


def test_bf16_only_file_is_copied_not_reserialized(tmp_path):
    src = tmp_path / "enc.safetensors"
    save_file({"h": torch.randn(4, 4).to(torch.bfloat16)}, str(src))
    dst = tmp_path / "enc_out.safetensors"

    converted, total = conv.convert_or_copy_safetensors(src, dst)

    assert (converted, total) == (0, 0)                         # copy path
    assert src.read_bytes() == dst.read_bytes()                 # byte-identical


def test_idempotent(tmp_path):
    src = tmp_path / "src"
    _make_checkpoint(src)
    dst1 = tmp_path / "dst1"
    conv.convert_checkpoint(src, dst1)
    # second pass over an already-fp16 dir converts nothing
    converted, total = conv.convert_or_copy_safetensors(
        dst1 / "model.safetensors", tmp_path / "again.safetensors"
    )
    assert converted == 0
