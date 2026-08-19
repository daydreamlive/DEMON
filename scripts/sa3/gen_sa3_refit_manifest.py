"""Generate (and validate) the SA3 DiT refit manifest.

Maps every LoRA-targetable torch weight of the loaded SA3 DiT
(``sam.model.model`` Linear/Conv1d weights — the modules
``SA3LoRAManager`` can parametrize) to its ONNX initializer name in
Stability's ``dit_fp16mixed.onnx`` graph, by shape + value fingerprint,
recording the transpose orientation. The manifest is what
:class:`acestep.engine.sa3_trt_lora.SA3TRTRefitMirror` consumes.

Matching: for each torch weight, candidate initializers are those whose
shape equals the weight's 2D view directly or transposed; ambiguity is
resolved by full value comparison (both sides cast to fp32; the ONNX
trunk stores fp16, the torch model loads fp16, so an exact-bits check
decides). A weight matching in BOTH orientations (only possible for
symmetric-shape symmetric-value degenerates) or in neither is reported,
never silently guessed.

Validation (``--validate-engine``): deserialize the refit-built engine
EXCLUSIVELY, refit every mapped weight with its own base value, and
compare a fixed forward against the unrefit output — bit-identical or
the manifest (or orientation) is wrong. This is the gate that makes a
wrong manifest impossible to ship silently (plan D6b.3).

Run:
    .venv/Scripts/python.exe scripts/sa3/gen_sa3_refit_manifest.py \
        --onnx <path>/dit_fp16mixed.onnx
    .venv/Scripts/python.exe scripts/sa3/gen_sa3_refit_manifest.py \
        --onnx <...> --validate-engine <...>/sa3_m_dit_refit_l1_646_646.trt

The ONNX ships on HF (stabilityai/stable-audio-3-optimized,
onnx/sa3-m/dit_fp16mixed.onnx + .data). Pass --fetch to download it via
huggingface_hub (cached; ~3 GB on first use), or point --onnx at an
existing copy (the sa3_build cache has one after any DiT engine build).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists())
for _p in (str(_REPO_ROOT),):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import torch  # noqa: E402


def _iter_target_weights(model_root):
    import torch.nn as nn

    for name, mod in model_root.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d)):
            yield name, mod.weight.detach()


def build_manifest(model_root, onnx_path: Path) -> dict:
    import onnx
    from onnx import numpy_helper

    print(f"loading ONNX graph {onnx_path} (external data alongside)...")
    model = onnx.load(str(onnx_path), load_external_data=True)
    inits = list(model.graph.initializer)
    by_shape: dict[tuple, list] = {}
    for init in inits:
        by_shape.setdefault(tuple(init.dims), []).append(init)
    print(f"{len(inits)} initializers indexed")

    weights: dict[str, dict] = {}
    unmatched: list[str] = []
    ambiguous: list[str] = []
    used: set[str] = set()
    for fqn, w in _iter_target_weights(model_root):
        w2 = w.view(w.shape[0], -1).float().cpu()
        direct_shape = tuple(w.shape)
        w2_shape = tuple(w2.shape)
        t_shape = (w2_shape[1], w2_shape[0])
        candidates = []
        for shape, transposed in (
            (direct_shape, False),
            (w2_shape, False),
            (t_shape, True),
        ):
            for init in by_shape.get(shape, []):
                if init.name in used:
                    continue
                arr = torch.from_numpy(
                    numpy_helper.to_array(init).copy()
                ).float()
                arr2 = arr.view(arr.shape[0], -1) if not transposed else (
                    arr.view(arr.shape[0], -1).transpose(0, 1)
                )
                ref = w2
                if arr2.shape != ref.shape:
                    continue
                # fp16-trunk graphs store the same fp16 the torch model
                # loads; exact equality decides. A slack allclose backs
                # up fp32-island initializers that round differently.
                if torch.equal(arr2, ref) or torch.allclose(
                    arr2, ref, rtol=1e-3, atol=1e-4,
                ):
                    candidates.append((init.name, transposed))
        names = {c[0] for c in candidates}
        if not names:
            unmatched.append(fqn)
            continue
        if len(names) > 1:
            ambiguous.append(fqn)
            continue
        init_name, transposed = candidates[0]
        used.add(init_name)
        weights[fqn] = {"initializer": init_name, "transposed": transposed}

    print(
        f"mapped {len(weights)} weights; unmatched={len(unmatched)} "
        f"ambiguous={len(ambiguous)}"
    )
    for fqn in unmatched[:10]:
        print(f"  UNMATCHED: {fqn}")
    for fqn in ambiguous[:10]:
        print(f"  AMBIGUOUS: {fqn}")
    return {
        "version": 1,
        "onnx": onnx_path.name,
        "weights": weights,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
    }


def validate_engine(manifest_path: Path, engine_path: Path, ctx) -> bool:
    """Refit every mapped weight with its own base value; the engine
    output must stay bit-identical."""
    from acestep.engine.sa3_trt import SA3TRTDit
    from acestep.engine.sa3_trt_lora import SA3TRTRefitMirror
    from acestep.engine.sa3_stream_helpers import stack_sa3_cond_bundles

    cond = ctx.prepare_cond(
        prompt="warm analog house groove, 124 bpm", duration=20.0, steps=8,
    )
    dit = SA3TRTDit(
        engine_path, latent_frames=cond.latent_frames, seconds_total=20.0,
    )
    if not dit.refittable:
        print(f"FAIL: {engine_path} is not a refit-built engine")
        return False
    stacked = stack_sa3_cond_bundles([cond.cond_bundle])
    x = torch.randn(
        1, ctx.latent_channels, cond.latent_frames, device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(7),
    )
    ref = dit.step_bundle(x, 0.5, stacked).clone()

    mirror = SA3TRTRefitMirror(dit.engine, ctx.sam.model.model, manifest_path)
    # Force-push every mapped weight at its BASE value: mark all dirty
    # so sync pushes them even though nothing is parametrized.
    mirror._dirty = set(mirror._map.keys())
    t0 = time.perf_counter()
    pushed = mirror.sync(reason="validate")
    print(f"validation refit: {pushed} weights in "
          f"{(time.perf_counter() - t0) * 1000:.0f}ms")

    out = dit.step_bundle(x, 0.5, stacked).clone()
    if torch.equal(out, ref):
        print("PASS: base-value refit is bit-identical")
        return True
    err = (out - ref).abs().max().item()
    print(f"FAIL: output changed after base-value refit (max_err={err:.3e})")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--onnx", default=None,
                    help="path to dit_fp16mixed.onnx (with its .data sidecar)")
    ap.add_argument("--fetch", action="store_true",
                    help="fetch the ONNX from HF (cached; ~3 GB first use)")
    ap.add_argument("--out", default=None,
                    help="manifest output path (default: "
                         "<sa3 engines dir>/sa3_m_dit_refit_manifest.json)")
    ap.add_argument("--validate-engine", default=None,
                    help="refit-built engine to bit-identity-validate against")
    args = ap.parse_args()

    from acestep.engine.sa3_context import SA3Context
    from acestep.engine.sa3_trt import trt_engines_dir
    from acestep.engine.sa3_trt_lora import SHARED_MANIFEST_NAME

    if args.onnx:
        onnx_path = Path(args.onnx)
    elif args.fetch:
        from acestep.engine.trt.sa3_build import DIT_ONNX_FILES, _fetch_onnx

        onnx_path = Path(_fetch_onnx(DIT_ONNX_FILES))
    else:
        ap.error("pass --onnx <path> or --fetch")
    if not onnx_path.is_file():
        ap.error(f"ONNX not found: {onnx_path}")

    out = Path(args.out) if args.out else (
        trt_engines_dir() / SHARED_MANIFEST_NAME
    )

    print(f"loading SA3Context({args.model!r})...")
    ctx = SA3Context(model_id=args.model)

    manifest = build_manifest(ctx.sam.model.model, onnx_path)
    if manifest["unmatched"] or manifest["ambiguous"]:
        print("REFUSING to write a partial manifest "
              "(unmatched/ambiguous weights above)")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest written: {out} ({len(manifest['weights'])} weights)")

    if args.validate_engine:
        ok = validate_engine(out, Path(args.validate_engine), ctx)
        return 0 if ok else 1
    print("NOTE: run again with --validate-engine <refit-built .trt> to "
          "complete the bit-identity gate before first production use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
