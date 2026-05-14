"""Extract per-layer tactic names from a built TRT engine.

Settles whether the production FP8 engine is actually using Blackwell
SM_120 FP8 tensor core tactics, or whether Myelin fell back to SM_80
BF16 GEMM kernels like the NVFP4 stub did.

This determines the headroom available from a custom plugin:
- SM_120 FP8 tactics already in use ⇒ plugin path gives modest gains
  (10-30% from removing overhead / per-channel scale handling)
- SM_80 BF16 tactics in use ⇒ plugin path can deliver near-2x by
  actually using Blackwell FP8 tensor cores.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import tensorrt as trt


def inspect(engine_path: Path) -> dict:
    print(f"[{engine_path.name}]")
    print(f"  size: {engine_path.stat().st_size / 1e6:.1f} MB")
    logger = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = rt.deserialize_cuda_engine(f.read())
    inspector = engine.create_engine_inspector()
    info_str = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    # Save full JSON for archeology.
    out_json = engine_path.with_suffix(".engine_info.json")
    out_json.write_text(info_str, encoding="utf-8")
    print(f"  full info JSON: {out_json}")

    info = json.loads(info_str)
    # ``Layers`` may not exist at top level depending on TRT version.
    # Try common keys.
    layers = info.get("Layers") or info.get("layers") or []
    if not layers and isinstance(info, list):
        layers = info

    # Categorize tactic names.
    sm_pat = re.compile(r"sm[_]?(\d+)", re.I)
    type_pat = re.compile(r"(fp8|fp4|bf16|bfloat16|fp16|fp32|int8|nvfp4|e4m3|e5m2|e2m1)", re.I)

    matmul_layers = []
    by_sm = Counter()
    by_type = Counter()
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        name = layer.get("Name") or layer.get("name", "")
        ltype = layer.get("LayerType") or layer.get("layerType", "")
        tactic = layer.get("TacticName") or layer.get("tacticName") or layer.get("tactic", "")
        # Look for gemm / matmul layers.
        if isinstance(ltype, str) and ("gemm" in ltype.lower() or "matmul" in ltype.lower() or "kgen" in ltype.lower()):
            matmul_layers.append({"name": name, "type": ltype, "tactic": tactic})
        # Aggregate sm / dtype hits across ALL layers.
        if tactic:
            for m in sm_pat.findall(tactic):
                by_sm[f"sm{m}"] += 1
            for m in type_pat.findall(tactic):
                by_type[m.lower()] += 1

    print(f"  total layers: {len(layers)}")
    print(f"  matmul/gemm-like layers: {len(matmul_layers)}")
    print(f"  SM bucket counts (across all tactics): {dict(by_sm)}")
    print(f"  dtype hits in tactic names:          {dict(by_type)}")

    # Show top distinct matmul tactic names.
    distinct = Counter(l["tactic"] for l in matmul_layers if l["tactic"])
    print(f"  distinct matmul tactic names: {len(distinct)}")
    print(f"  top tactic names:")
    for tac, count in distinct.most_common(10):
        disp = tac if len(tac) <= 120 else tac[:117] + "..."
        print(f"    {count:>4} x  {disp}")

    no_tactic = sum(1 for l in matmul_layers if not l["tactic"])
    if no_tactic:
        print(f"  matmul layers with NO tactic name: {no_tactic}")

    return {
        "engine_path": str(engine_path),
        "n_layers": len(layers),
        "n_matmul": len(matmul_layers),
        "sm_counts": dict(by_sm),
        "type_counts": dict(by_type),
        "top_tactics": distinct.most_common(10),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="*", help="Engine paths or variant tags (variants/<tag>/engine.engine).")
    args = ap.parse_args()

    variants_root = Path(__file__).parent / "variants"
    targets = []
    if not args.engines:
        targets = [
            variants_root / "w8a8_absmax_cal2" / "engine.engine",
            variants_root / "bf16" / "engine.engine",
        ]
    else:
        for e in args.engines:
            p = Path(e)
            if p.exists():
                targets.append(p)
            else:
                tagged = variants_root / e / "engine.engine"
                if tagged.exists():
                    targets.append(tagged)
                else:
                    print(f"NOT FOUND: {e}")
                    return

    for p in targets:
        print("=" * 72)
        inspect(p)
        print()


if __name__ == "__main__":
    main()
