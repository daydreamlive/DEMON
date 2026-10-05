"""Per-step audio-token MEAN of every SA3 block's OUTPUT (the production
steering site ``post_block_residual``), for direction estimation.

Adapted from DEMON-tada-sae-sa3 ``scripts/tada/sa3_e5_capture.py`` (commit
40ece042 lineage); the only change of substance is the hooked module:
``sa3_tada.sa3_blocks(sam)[b]`` (whole block, after its residual adds and
FF) instead of ``cross_attn_modules(sam)[b]``. Everything else matches E5:
ARC medium, 8 steps, cfg 1 (every forward is the conditional pass), 10 s,
batch 16, seed ``seed0 + batch index``, 64 memory tokens and padding
excluded from the mean.

Modes:

* default (MusicCaps): the 5521 captions in the seed-42 shuffled order
  (same as E5 / the SAE cache), so ``meta.json caption_index`` lines up
  with the MusicCaps CSV rows for keyword labels. seed0 1000.
* ``--self-label``: N renders of the 100 TADA benchmark TEST prompts
  cycled (row r uses prompt r % 100; repeats land in other batches, so
  other seeds), seed0 5000. The audio is saved too (``audio/batch_*.npz``,
  int16 mono, reference layout) so descriptors can be measured afterwards
  (``build_directions.py --self-label``). The 20 holdout prompts used by the
  block sweep are NOT used here.

Outputs in ``--out``: ``means.npy`` float16 ``[N, steps, 24, 1536]``
(memmap) and ``meta.json``.

    python scripts/steering_bench/capture_resid.py --out D:/steer-bench/resid_mc
    python scripts/steering_bench/capture_resid.py --self-label --n 800 --out D:/steer-bench/resid_self
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_REPO / "scripts" / "sa3")):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_REPO / "scripts" / "sa3"))
sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

SR = 44100
DEFAULT_CAPTIONS = "E:/Projects/tada-replication/data/musiccaps-public.csv"
BENCH_PROMPTS = _REPO / "acestep" / "tada" / "data" / "benchmark_prompts.json"


class AudioTokens:
    """``mask(hs) -> bool [B, seq]``: audio rows only (memory rows and padding
    dropped). Copy of DEMON-tada-sae-sa3 ``acestep/engine/sa3_tada_tokens.py``
    (not on this branch). ``transformer`` = ``trunk_module(sam).transformer``,
    whose forward receives ``padding_mask``."""

    def __init__(self, transformer: torch.nn.Module):
        self.num_memory_tokens = int(getattr(transformer, "num_memory_tokens", 0) or 0)
        self.padding_mask = None
        self._h = transformer.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, _m, _args, kwargs):
        pm = kwargs.get("padding_mask")
        self.padding_mask = None if pm is None else pm.detach().to(torch.bool)

    def __call__(self, hs: torch.Tensor) -> torch.Tensor:
        b, seq = int(hs.shape[0]), int(hs.shape[1])
        n_audio = seq - self.num_memory_tokens
        if n_audio <= 0:
            raise ValueError(f"sequence of {seq} has no audio rows after {self.num_memory_tokens} memory tokens")
        audio = torch.ones(b, n_audio, dtype=torch.bool, device=hs.device)
        pm = self.padding_mask
        if pm is not None:
            if pm.shape[-1] != n_audio:
                raise ValueError(f"padding_mask covers {pm.shape[-1]} rows, expected {n_audio}")
            pm = pm.to(hs.device)
            audio = pm.expand(b, -1) if pm.shape[0] == 1 else pm
        mem = torch.zeros(b, self.num_memory_tokens, dtype=torch.bool, device=hs.device)
        return torch.cat([mem, audio], dim=1)

    def remove(self) -> None:
        self._h.remove()


class SigmaProbe:
    """Noise level of the current DiT forward (its ``t`` argument)."""

    def __init__(self, trunk: torch.nn.Module):
        self.sigma = float("nan")
        self._h = trunk.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, _m, args, kwargs):
        t = kwargs.get("t", args[1] if len(args) > 1 else None)
        if t is not None:
            self.sigma = float(torch.as_tensor(t).flatten()[0])

    def __call__(self) -> float:
        return self.sigma

    def remove(self) -> None:
        self._h.remove()


def prompt_order(n: int, seed: int = 42) -> list:
    """The reference (and E5) shuffles the caption set with seed 42."""
    return np.random.default_rng(seed).permutation(n).tolist()


def musiccaps_prompts(csv_path: str, n):
    import pandas as pd

    caps = pd.read_csv(csv_path)["caption"].astype(str).tolist()
    order = prompt_order(len(caps))[:n]
    return [caps[i] for i in order], order


def self_label_prompts(n: int):
    d = json.loads(BENCH_PROMPTS.read_text(encoding="utf-8"))
    test = list(d["test_prompts"])
    idx = [r % len(test) for r in range(n)]
    return [test[i] for i in idx], idx


def save_audio(path: Path, audio: torch.Tensor, rows, prompts) -> None:
    """int16 mono, the reference ``audios.npz`` layout plus row indices."""
    a = audio.float().mean(dim=1, keepdim=True)
    arr = (a * 32768.0).round().clamp(-32768, 32767).to(torch.int16).numpy()
    np.savez(path, audio=arr, sr=np.int64(SR), names=np.array(list(prompts)),
             rows=np.array(list(rows), dtype=np.int64))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", default=DEFAULT_CAPTIONS, help="MusicCaps CSV (column 'caption')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None, help="prompts (default: all 5521 captions; 800 self-label)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default="medium")
    ap.add_argument("--self-label", action="store_true",
                    help="benchmark test prompts cycled, audio saved for descriptor labels")
    ap.add_argument("--seed0", type=int, default=None, help="default 1000 (MusicCaps) / 5000 (self-label)")
    args = ap.parse_args()

    from sa3_reference_generate import checkpoint_dir, load_local_model
    from acestep.engine import sa3_tada
    from acestep.engine.sa3_internals import trunk_module

    if args.self_label:
        prompts, prompt_index = self_label_prompts(args.n or 800)
        seed0 = 5000 if args.seed0 is None else args.seed0
    else:
        prompts, prompt_index = musiccaps_prompts(args.captions, args.n)
        seed0 = 1000 if args.seed0 is None else args.seed0

    torch.backends.cuda.matmul.allow_tf32 = True
    sam = load_local_model(checkpoint_dir(args.checkpoint), device=args.device, model_half=True)
    sam.model.eval()
    blocks = list(sa3_tada.sa3_blocks(sam))
    nb = len(blocks)
    trunk = trunk_module(sam)
    audio_tok = AudioTokens(trunk.transformer)
    probe = SigmaProbe(trunk)
    state = {"step": -1, "buf": None, "hidden": None}
    sigmas = {}

    def pre0(_m, _a):
        state["step"] += 1
        sigmas.setdefault(state["step"], probe())

    def make(b):
        @torch.no_grad()
        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            mask = audio_tok(hs).to(hs.dtype).unsqueeze(-1)
            mean = (hs * mask).sum(1) / mask.sum(1)
            state["buf"][:, state["step"], b] = mean.float().cpu()
        return hook

    hidden = int(getattr(blocks[0], "dim", 1536))
    handles = [blocks[0].register_forward_pre_hook(pre0)]
    handles += [m.register_forward_hook(make(b)) for b, m in enumerate(blocks)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.self_label:
        (out / "audio").mkdir(exist_ok=True)
    arr = np.lib.format.open_memmap(out / "means.npy", mode="w+", dtype=np.float16,
                                    shape=(len(prompts), args.steps, nb, hidden))
    t0 = time.perf_counter()
    with torch.no_grad():
        for j, i in enumerate(range(0, len(prompts), args.batch)):
            batch = prompts[i:i + args.batch]
            state["buf"] = torch.zeros(len(batch), args.steps, nb, hidden)
            state["step"] = -1
            audio = sa3_tada.generate(sam, batch, seed=seed0 + j, duration=args.duration, steps=args.steps)
            if state["step"] != args.steps - 1:
                raise RuntimeError(f"saw {state['step'] + 1} forwards, expected {args.steps}")
            arr[i:i + len(batch)] = state["buf"].numpy().astype(np.float16)
            if args.self_label:
                save_audio(out / "audio" / f"batch_{j:04d}.npz", audio, range(i, i + len(batch)), batch)
            if j % 40 == 0:
                print(f"[resid] {i + len(batch)}/{len(prompts)} {time.perf_counter() - t0:.0f}s", flush=True)
    arr.flush()
    for h in handles:
        h.remove()
    audio_tok.remove()
    probe.remove()
    meta = {
        "site": "post_block_residual (output of sa3_blocks(sam)[b])",
        "mode": "self_label" if args.self_label else "musiccaps",
        "n": len(prompts), "steps": args.steps, "blocks": nb, "hidden": hidden,
        "sigmas": sigmas, "batch": args.batch, "seed": f"{seed0} + batch index",
        "duration_s": args.duration, "sampler": "ARC sam.generate default (cfg 1)",
        "tokens": f"audio only ({audio_tok.num_memory_tokens} memory tokens and padding excluded)",
        "checkpoint": f"{args.checkpoint} (ARC)" if args.checkpoint == "medium" else args.checkpoint,
        "date": time.strftime("%Y-%m-%d"),
    }
    if args.self_label:
        meta["prompt_index"] = [int(x) for x in prompt_index]
        meta["prompts_source"] = "acestep/tada/data/benchmark_prompts.json test_prompts, cycled"
        meta["audio"] = "audio/batch_<j>.npz int16 mono [B,1,T], sr, names, rows"
    else:
        meta["caption_index"] = [int(x) for x in prompt_index]
        meta["captions"] = str(args.captions)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done {len(prompts)} prompts {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
