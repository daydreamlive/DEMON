"""E5: per-step audio-token MEAN of every block's cross-attention output
for the SAE cache's captions (same order, seeds and batches as
``sa3_sae.py cache``: MusicCaps shuffled seed 42, batch 16, seed 1000 +
batch index, ARC medium, 8 steps, cfg 1 = cond pass only). Memory tokens
and padding are excluded (:mod:`acestep.engine.sa3_tada_tokens`).
Writes ``means.npy`` fp16 ``[n_prompts, steps, blocks, d]`` plus
``meta.json`` (caption index, sigmas)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "sa3"))
sys.path.insert(0, str(_REPO / "scripts" / "tada"))
sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    import sa3_sae as M
    from acestep.engine import sa3_tada
    from acestep.engine.sa3_tada_tokens import sa3_audio_tokens

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="E:/Projects/tada-replication")
    ap.add_argument("--out", default="D:/tada-replication/e5_means")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    caps = M.captions(Path(args.root))
    order = M.prompt_order(len(caps))
    prompts = [caps[i] for i in order][: args.limit]
    sam = M.load_sam()
    mods = list(sa3_tada.cross_attn_modules(sam))
    nb = len(mods)
    audio = sa3_audio_tokens(sam)
    probe = M.SigmaProbe(sam)
    state = {"step": -1, "buf": None}
    sigmas = {}

    def pre0(_m, _a):
        state["step"] += 1
        sigmas.setdefault(state["step"], probe())

    def make(b):
        @torch.no_grad()
        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            mask = audio(hs).to(hs.dtype).unsqueeze(-1)
            mean = (hs * mask).sum(1) / mask.sum(1)
            state["buf"][:, state["step"], b] = mean.float().cpu()
        return hook

    hs = [mods[0].register_forward_pre_hook(pre0)] + [m.register_forward_hook(make(b))
                                                      for b, m in enumerate(mods)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arr = None
    t0 = time.perf_counter()
    with torch.no_grad():
        for j, i in enumerate(range(0, len(prompts), args.batch)):
            batch = prompts[i:i + args.batch]
            state["buf"] = torch.zeros(len(batch), args.steps, nb, 1536)
            state["step"] = -1
            sa3_tada.generate(sam, batch, seed=1000 + j, duration=10.0, steps=args.steps)
            if state["step"] != args.steps - 1:
                raise RuntimeError(f"saw {state['step'] + 1} forwards, expected {args.steps}")
            if arr is None:
                arr = np.lib.format.open_memmap(out / "means.npy", mode="w+", dtype=np.float16,
                                                shape=(len(prompts), args.steps, nb, 1536))
            arr[i:i + len(batch)] = state["buf"].numpy().astype(np.float16)
            if j % 40 == 0:
                print(f"[e5] {i + len(batch)}/{len(prompts)} {time.perf_counter() - t0:.0f}s", flush=True)
    arr.flush()
    for h in hs:
        h.remove()
    audio.remove()
    probe.remove()
    (out / "meta.json").write_text(json.dumps({
        "caption_index": [int(x) for x in order[:len(prompts)]], "steps": args.steps,
        "blocks": nb, "sigmas": sigmas, "batch": args.batch, "seed": "1000 + batch index",
        "tokens": "audio only (memory and padding excluded)", "checkpoint": "medium (ARC)"}, indent=1))
    print(f"done {len(prompts)} prompts {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
