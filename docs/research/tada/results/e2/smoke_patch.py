import sys, types
sys.argv = ["x"]; sys.path.insert(0, "scripts/tada")
import torch
import sa3_tada_run as R
from acestep.engine import sa3_tada
sam = R._load_sam()
target = sa3_tada.sa3_target(sam)
cp, xp = ["fast upbeat drum and bass", "fast techno"], ["slow ambient pad", "slow techno"]
for site in ("xattn_out", "resid"):
    a = types.SimpleNamespace(patch_site=site)
    rec = R._make_patcher(a, sam, target, range(24))
    with rec.record():
        clean = sa3_tada.generate(sam, cp, seed=222, duration=10.0, steps=8)
    p = R._make_patcher(a, sam, target, range(24)); p.cache = {b: list(rec.cache[b]) for b in range(24)}
    with p.patch():
        allp = sa3_tada.generate(sam, xp, seed=222, duration=10.0, steps=8)
    p = R._make_patcher(a, sam, target, [7]); p.cache = {7: list(rec.cache[7])}
    with p.patch():
        one = sa3_tada.generate(sam, xp, seed=222, duration=10.0, steps=8)
    corr = sa3_tada.generate(sam, xp, seed=222, duration=10.0, steps=8)
    print(site, "all==clean", torch.equal(allp, clean), "maxdiff", float((allp - clean).abs().max()),
          "| tf7 dist to corr", float((one - corr).pow(2).mean().sqrt()), "to clean", float((one - clean).pow(2).mean().sqrt()),
          "| leftover", p.leftover, "calls/block", len(rec.cache[0]))
