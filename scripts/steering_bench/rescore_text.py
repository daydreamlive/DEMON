"""Re-score the text scorers (clap_music, clap_general, muq) for a new anchors.json
from the audio embeddings label_corpus.py saved per shard (labels/.emb/<scorer>/shard_k.npz),
so new catalogue anchors need no audio. Rewrites labels/<scorer>.parquet (all columns of the
new plan, one row per clip id, ordered like meta.json ids) and labels/<scorer>.done.

    python scripts/steering_bench/rescore_text.py --cap $CAP --scorer clap_music --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_corpus as L  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--scorer", required=True, choices=("clap_music", "clap_general", "muq"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--anchors", default=str(L.DEFAULT_ANCHORS))
    ap.add_argument("--template", default="{}")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    import pandas as pd
    import torch

    cap = Path(args.cap)
    t0 = time.time()
    anchors = L.text_plan(args.anchors)
    if args.scorer == "muq":
        sc = L.MuqScorer(args.device, anchors, args.template)
    else:
        sc = L.ClapScorer(args.scorer, args.device, anchors, args.ckpt, args.template)
    edir = cap / "labels" / ".emb" / args.scorer
    files = sorted(edir.glob("shard_*.npz"))
    if not files:
        raise SystemExit(f"no embeddings under {edir}")
    parts = []
    for f in files:
        with np.load(f) as z:
            emb, ids = z["emb"], z["ids"]
        sim = L._cos(torch.from_numpy(emb.astype(np.float32)).to(sc.text.device), sc.text)
        cols = L._text_cols(sim, sc.plan)
        df = pd.DataFrame({f"{args.scorer}.{c}": np.asarray(v, dtype=np.float32) for c, v in cols.items()})
        df.insert(0, "id", ids.astype(np.int64))
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    meta = json.loads((cap / "meta.json").read_text())
    order = {int(i): r for r, i in enumerate(meta.get("ids", []))}
    df = df.iloc[np.argsort([order.get(int(i), len(order) + int(i)) for i in df["id"]], kind="stable")]
    df = df.reset_index(drop=True)
    final = cap / "labels" / f"{args.scorer}.parquet"
    tmp = final.with_name(final.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(final)
    (cap / "labels" / f"{args.scorer}.done").write_text(json.dumps({
        "rows": len(df), "cols": len(df.columns) - 1, "shards": len(files), "rescored_from": "embeddings",
        "anchors": args.anchors, "t": time.strftime("%Y-%m-%d %H:%M:%S")}))
    print(f"[{args.scorer}] rescored {df.shape} from {len(files)} embedding shards in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
