"""prompts_v2.json = v1 prompts.json (rows 0..N-1 unchanged, seed_offset 0) followed by the same
prompts with seed + s for s in 1..3; id = base_id + s * N."""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
v1 = json.load(open(src, encoding="utf-8"))
n = len(v1)
assert [int(r["id"]) for r in v1] == list(range(n)), "v1 ids must be 0..N-1"
out = []
for s in range(4):
    for r in v1:
        q = dict(r)
        q["id"] = int(r["id"]) + s * n
        q["seed"] = int(r["seed"]) + s
        q["base_id"] = int(r["id"])
        q["seed_offset"] = s
        out.append(q)
json.dump(out, open(dst, "w", encoding="utf-8"), indent=0)
print(len(out), "rows ->", dst)
