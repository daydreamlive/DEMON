"""Closed-loop steering prototype (many-knobs v2, P6), demonenv.

Readout = a ridge probe (``ridge_probe.py``: standardised block token-mean at one
sampler step, or the step mean, -> one label column). Loop = per-prompt gain
control between renders (the stream-window timescale): render the open loop at
the calibrated alpha, read each prompt's predicted label, rescale that prompt's
alpha toward a common target (baseline prediction + the open loop's mean
predicted effect) and re-render, ``--iters`` times. Renders the unsteered
baseline, the open loop and the last closed-loop pass to npz (``set_tools.py
score`` then scores them), and writes alphas and readouts per pass to
``--out``/readout.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from set_tools import SR, _save_inplace, load_set  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--gain", type=float, required=True, help="calibrated knob gain (alpha = gain x magnitude)")
    ap.add_argument("--ridge", required=True, help="ridge_probe weights npz (w, mu, sd, ym, block, step)")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--seed", type=int, default=2115)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts" / "tada"))
    sys.path.insert(0, str(HERE))
    import torch

    import sa3_tada_run as R
    from acestep.engine import sa3_tada
    from acestep.engine.sa3_internals import trunk_module
    from capture_resid import AudioTokens

    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_set(args.sets, args.set)
    B = len(prompts)
    rz = np.load(args.ridge)
    w, mu, sd, ym = (torch.tensor(rz[k], dtype=torch.float64) for k in ("w", "mu", "sd", "ym"))
    rblock, rstep = int(rz["block"]), str(rz["step"])
    sam = R._load_sam()
    pv = R._pack_vectors(args.pack, 8)
    alpha_cal = args.gain * float(R._PACK["magnitude"])
    blocks = list(sa3_tada.sa3_blocks(sam))
    tok = AudioTokens(trunk_module(sam).transformer)
    st = {"step": -1, "x": {}}

    def pre0(_m, _a):
        st["step"] += 1

    @torch.no_grad()
    def read(_m, _i, o):
        hs = o[0] if isinstance(o, tuple) else o
        m = tok(hs).to(hs.dtype).unsqueeze(-1)
        st["x"][st["step"]] = ((hs * m).sum(1) / m.sum(1)).double().cpu()

    hooks = [blocks[0].register_forward_pre_hook(pre0)]

    def render(alpha: np.ndarray):
        st["step"], st["x"] = -1, {}
        a = torch.tensor(alpha, dtype=torch.float32).view(B, 1, 1)
        vec = {s: {b: v.view(1, 1, -1) * a for b, v in d.items()} for s, d in pv.items()}
        # the readout hook is registered after the steering hooks, so it sees the steered residual
        with sa3_tada.steer_offline(sam, vec, 1.0, hook="post_block_residual"):
            h = blocks[rblock].register_forward_hook(read)
            try:
                audio = sa3_tada.generate(sam, prompts, seed=args.seed, duration=10.0, steps=8)
            finally:
                h.remove()
        x = torch.stack([st["x"][k] for k in sorted(st["x"])], 1)          # [B, steps, H]
        x = x.mean(1) if rstep == "mean" else x[:, int(rstep)]
        pred = (((x - mu) / sd) @ w + ym).numpy()
        wav = (audio.mean(dim=1) * 32768.0).round().clamp(-32768, 32767).to(torch.int16).numpy()
        return pred, wav

    def save(tag, wav, alpha):
        _save_inplace(out / f"{tag}.npz", wav=wav, sr=np.int64(SR), seed=np.int64(args.seed), set=np.array(args.set),
                      steer=np.array(json.dumps({"pack": args.pack, "alpha": alpha.tolist()})), scale=np.float64(1))

    log = {"pack": args.pack, "gain": args.gain, "alpha_cal": alpha_cal, "ridge": args.ridge, "set": args.set,
           "readout": {"block": rblock, "step": rstep}, "passes": []}
    a0 = np.zeros(B)
    p0, wav = render(a0)
    save("base", wav, a0)
    a = np.full(B, alpha_cal)
    p, wav = render(a)
    save("open", wav, a)
    target = float(np.mean(p - p0))
    log["target_delta"] = target
    log["passes"].append({"alpha": a.tolist(), "pred_delta": (p - p0).tolist()})
    for _ in range(args.iters):
        e = p - p0
        ratio = np.where(e * np.sign(target) > 1e-9, target / np.where(e == 0, 1, e), 2.0)
        a = np.clip(a * np.clip(ratio, 0.5, 2.0), 0.25 * alpha_cal, 3.0 * alpha_cal)
        p, wav = render(a)
        log["passes"].append({"alpha": a.tolist(), "pred_delta": (p - p0).tolist()})
    save("closed", wav, a)
    for h in hooks:
        h.remove()
    (out / "readout.json").write_text(json.dumps(log, indent=1))
    d_open, d_cl = np.array(log["passes"][0]["pred_delta"]), np.array(log["passes"][-1]["pred_delta"])
    print(f"target {target:.4f}: predicted delta open {d_open.mean():.4f}+-{d_open.std():.4f}, "
          f"closed {d_cl.mean():.4f}+-{d_cl.std():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
