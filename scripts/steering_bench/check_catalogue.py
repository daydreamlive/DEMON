"""Check that every catalogue label resolves to a column label_corpus.py writes.

No audio and no models: the column list comes from label_corpus.scorer_columns
(desc / dyn / timbral column lists, the AudioSet class csv for passt, and
anchors.json anchors + ``concepts`` for clap_music / clap_general / muq).
Checks primary_label, second_scorer and third_scorer (empty third = fine).
Prints every unresolved row and ends with ``unresolved: N`` (exit 1 if N > 0).

``--sync-anchors`` first rewrites the ``concepts`` section of anchors.json
from the catalogue's name / pos_anchor / neg_anchor (the other groups are
kept), so every row gets ``<clap_music|clap_general|muq>.<name>`` columns.
Run it after regenerating the catalogue.

    python scripts/steering_bench/check_catalogue.py --catalogue notes/steering_pr/concept_catalogue.csv
    python scripts/steering_bench/check_catalogue.py --catalogue $CAP/concept_catalogue.csv --sync-anchors
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import label_corpus as lc  # noqa: E402

FIELDS = ("primary_label", "second_scorer", "third_scorer")
#: local fallbacks for the AudioSet class list (passt column names)
LABEL_CSVS = (lc.AUDIOSET_LABELS, str(Path.home() / "panns_data" / "class_labels_indices.csv"))


def read_rows(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if (r.get("name") or "").strip()]


def sync_anchors(rows, anchors: Path) -> int:
    """Replace the trailing ``"concepts"`` block of anchors.json (one concept
    per line), leaving the hand-formatted anchor groups above it untouched."""
    concepts = {r["name"].strip(): [r["pos_anchor"].strip(), r["neg_anchor"].strip()]
                for r in rows if (r.get("pos_anchor") or "").strip() and (r.get("neg_anchor") or "").strip()}
    txt = anchors.read_text(encoding="utf-8")
    i = txt.find('\n "concepts": {')
    if i >= 0:
        txt = txt[:txt.rfind(",", 0, i)] + "\n}\n"
    body = txt.rstrip()
    assert body.endswith("}"), anchors
    lines = [f"  {json.dumps(n, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
             for n, v in concepts.items()]
    txt = body[:-1].rstrip() + ',\n "concepts": {\n' + ",\n".join(lines) + "\n }\n}\n"
    assert json.loads(txt)["concepts"] == concepts
    anchors.write_text(txt, encoding="utf-8")
    return len(concepts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--anchors", default=str(lc.DEFAULT_ANCHORS))
    ap.add_argument("--labels-csv", default=None, help="AudioSet class_labels_indices.csv (passt column names)")
    ap.add_argument("--sync-anchors", action="store_true", help="rewrite anchors.json concepts from the catalogue")
    args = ap.parse_args()

    rows = read_rows(args.catalogue)
    if args.sync_anchors:
        print(f"synced {sync_anchors(rows, Path(args.anchors))} concepts into {args.anchors}")
    cols, per = set(), {}
    for s in lc.SCORERS:
        if s == "passt":
            got = None
            for p in ([args.labels_csv] if args.labels_csv else LABEL_CSVS):
                got = lc.scorer_columns(s, args.anchors, p)
                if got is not None:
                    break
            if got is None:
                raise SystemExit("no AudioSet class_labels_indices.csv found (pass --labels-csv)")
        else:
            got = lc.scorer_columns(s, args.anchors)
        per[s] = len(got)
        cols.update(got)
    print("columns per scorer:", per)
    bad = 0
    for r in rows:
        miss = [f"{k}={r.get(k, '')!r}" for k in FIELDS
                if (r.get(k) or "").strip() and r[k].strip() not in cols]
        if not (r.get("primary_label") or "").strip():
            miss.insert(0, "primary_label empty")
        if miss:
            bad += 1
            print(f"UNRESOLVED {r['name']}: {'; '.join(miss)}")
    print(f"rows: {len(rows)}  unresolved: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
