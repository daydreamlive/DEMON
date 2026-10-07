"""Zero-strength no-op check for TADA steering on SA3 (offline path):
a render with every CAA / AUSteer vector hooked at strength 0 must be
bit-identical to a render with no hooks (same prompts, seed, batch)."""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

#: Working root for the replication data (audio, vectors, packs, the
#: steer-audio checkout); set TADA_ROOT to relocate it.
TADA_ROOT = Path(os.environ.get("TADA_ROOT", "tada-replication"))


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "sa3"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from acestep.engine import sa3_tada  # noqa: E402
from acestep.tada import concepts as C  # noqa: E402

OUT = TADA_ROOT / "sa3"


def main() -> int:
    from sa3_reference_generate import checkpoint_dir, load_local_model

    sam = load_local_model(checkpoint_dir("medium"), device="cuda", model_half=True)
    sam.model.eval()
    prompts = C.benchmark_prompts(holdout=True)[0][:8]
    base = sa3_tada.generate(sam, prompts, seed=2115)
    res = {}
    for f in sorted((OUT / "caa").glob("*.pt")):
        if f.name.endswith(".acts.pt"):
            continue
        d = torch.load(f, weights_only=False)
        vecs = d["vectors"]["loc"] if f.name.endswith(".austeer.pt") else d["vectors"]
        for renorm in (False, True):
            with sa3_tada.steer_offline(sam, vecs, 0.0, renorm=renorm):
                out = sa3_tada.generate(sam, prompts, seed=2115)
            res[f"{f.stem}{'_renorm' if renorm else ''}"] = bool(torch.equal(out, base))
    (OUT / "zero_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0 if all(res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
