#!/usr/bin/env python3
"""Build one steering bundle (``bundle.safetensors``) from a pack directory.

Reads every per-file pack under ``--packs`` (one model), joins each with
its per-sign ship quality bar from ``--bar``, and writes
``<out>/bundle.safetensors`` plus a human-readable ``bundle.json`` beside
it (the same manifest). See ``acestep.steering.packs`` (Bundles) for the
format.

``--bar`` is a JSON object keyed by pack name or by the pack's
``provenance.knob_source`` (the measured vector id; tried first). Each value is either
``{"pos": bool, "neg": bool}`` or the ship pass's measured record,
``{"pos": {"good": bool, ...}, "neg": {"good": bool, ...}, ...}``.

Refuses (exit 1, nothing written) when any pack lacks the full
calibrated-gain shape (per sign median/min/max/reached/seeds/cutoff) or a
bar entry for both signs.

Usage::

    python scripts/steering/build_bundle.py \\
        --packs ~/.daydream-scope/models/demon/steering_packs/sa3/medium \\
        --bar E:/Projects/DEMON/steering-bench/many_knobs_v2/ship/measured.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from acestep.steering.packs import (  # noqa: E402
    BUNDLE_NAME,
    PACK_SUFFIX,
    SIGNS,
    build_bundle,
    load_pack,
)


def _sign_pass(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, dict) and isinstance(value.get("good"), bool):
        return value["good"]
    return None


def bar_for(pack, bar: dict) -> dict:
    """``{"pos": bool, "neg": bool}`` for one pack, or ValueError."""
    prov = pack.provenance if isinstance(pack.provenance, dict) else {}
    # knob_source first: a measured file can hold a different (unshipped)
    # vector under the bare pack name.
    for key in (prov.get("knob_source"), pack.name):
        if key and key in bar:
            entry = bar[key]
            signs = {s: _sign_pass(entry.get(s)) if isinstance(entry, dict) else None
                     for s in SIGNS}
            if any(v is None for v in signs.values()):
                raise ValueError(
                    f"knob {pack.name!r}: bar entry {key!r} lacks a pass flag per sign"
                )
            return signs
    raise ValueError(f"knob {pack.name!r}: no bar entry (name or knob_source)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--packs", required=True, type=Path,
                    help="directory of per-file packs for ONE model")
    ap.add_argument("--bar", required=True, type=Path,
                    help="per-sign quality-bar JSON (see module doc)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: --packs)")
    args = ap.parse_args(argv)

    out_dir = args.out or args.packs
    bar = json.loads(args.bar.read_text(encoding="utf-8"))
    paths = sorted(p for p in args.packs.glob(f"*{PACK_SUFFIX}") if p.name != BUNDLE_NAME)
    if not paths:
        print(f"no packs in {args.packs}", file=sys.stderr)
        return 1
    packs, bars, errors = [], {}, []
    for path in paths:
        try:
            pack = load_pack(path)
            bars[pack.name] = bar_for(pack, bar)
            packs.append(pack)
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")
    if errors:
        print("refusing to build:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    target = out_dir / BUNDLE_NAME
    try:
        manifest = build_bundle(packs, bars, target)
    except ValueError as exc:
        print(f"refusing to build: {exc}", file=sys.stderr)
        return 1
    (out_dir / "bundle.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8",
    )
    one_sided = sum(1 for k in manifest["knobs"] if sum(k["bar_pass"].values()) == 1)
    print(f"wrote {target} ({target.stat().st_size} bytes): "
          f"{len(manifest['knobs'])} knobs, {one_sided} one-sided")
    return 0


if __name__ == "__main__":
    sys.exit(main())
