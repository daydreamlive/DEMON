"""Merged label table for a corpus extended by capture_resid.py --render-from (v1 rows +
new-seed rows): concatenate the earlier corpus's merged.parquet with the new rows' merged.parquet.
Text-scorer columns (clap_music / clap_general / muq) are kept for the earlier rows only when the
column's anchor texts are identical in both anchors files (else NaN: the new definition was never
computed on the earlier audio). -> $CAP/labels/merged_all.parquet (rows in prompts.json order).

    python merge_corpora.py --cap $CAP --old-labels $V1/labels/merged.parquet \
        --old-anchors anchors_v1.json --anchors scripts/steering_bench/anchors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_corpus as L  # noqa: E402

TEXT = ("clap_music", "clap_general", "muq")


def col_defs(path) -> dict:
    texts, plan = L.text_plan(path)
    return {c: (texts[i], texts[j] if j is not None else None) for c, i, j in plan}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--old-labels", required=True)
    ap.add_argument("--old-anchors", required=True)
    ap.add_argument("--anchors", default=str(L.DEFAULT_ANCHORS))
    args = ap.parse_args()
    import pandas as pd

    cap = Path(args.cap)
    new = pd.read_parquet(cap / "labels" / "merged.parquet")
    old = pd.read_parquet(args.old_labels)
    d_old, d_new = col_defs(args.old_anchors), col_defs(args.anchors)
    keep, dropped = [], []
    for c in old.columns:
        if c == "id":
            keep.append(c)
            continue
        sc, name = c.split(".", 1)
        if sc in TEXT and d_old.get(name) != d_new.get(name):
            dropped.append(c)
            continue
        if c in new.columns:
            keep.append(c)
    old = old[keep]
    out = pd.concat([old, new], ignore_index=True, sort=False)
    ids = [int(r["id"]) for r in json.loads((cap / "prompts.json").read_text())]
    out = out.set_index("id").reindex(ids).reset_index()
    out.to_parquet(cap / "labels" / "merged_all.parquet", index=False)
    print(f"merged_all {out.shape}: old {len(old)} rows ({len(dropped)} text columns redefined -> NaN for them), "
          f"new {len(new)} rows; NaN rows per column median {int(out.isna().sum().median())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
