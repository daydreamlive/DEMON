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

* ``--prompts-json FILE`` (many-knobs corpus, master plan "Interfaces"):
  rows ``{id, prompt, seed, category, tags}``. Rendered shard by shard
  (``--shard`` rows, default 32, each shard generated in ``--batch``
  sized calls). Per shard, in this order: the means rows are flushed to
  ``means.npy``, ``audio_{k:04d}.npz`` (int16 ``wav`` [B, T] mono, ``ids``
  [B]) is written atomically, then ``audio_{k:04d}.done`` appears, so
  ``label_corpus.py`` can start on it. ``meta.json`` is written before the
  first shard (``n``, ``n_shards``, ``complete: false``) and rewritten with
  the sigmas and ``complete: true`` at the end. ``--resume`` keeps the
  existing ``means.npy`` and skips shards whose ``.done`` exists. Seeds: SA3
  takes one seed per generate call, so every call uses the ``seed`` of its
  first row (rows inside a call get distinct noise by batch position);
  ``meta.json gen_seed`` records the seed each row was actually rendered with.
  ``--part I/N`` splits the shards over N processes (one per GPU) writing the
  same ``means.npy``; start part 0 first (it creates the memmap and owns
  ``meta.json``; it sets ``complete`` only after every part's shards exist).

Outputs in ``--out``: ``means.npy`` float16 ``[N, steps, 24, 1536]``
(memmap) and ``meta.json``.

    python scripts/steering_bench/capture_resid.py --out D:/steer-bench/resid_mc
    python scripts/steering_bench/capture_resid.py --self-label --n 800 --out D:/steer-bench/resid_self
    python scripts/steering_bench/capture_resid.py --prompts-json $CAP/prompts.json --out $CAP [--resume]
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


def load_prompts_json(path: str, n):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("prompts", rows.get("rows"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty list of {{id, prompt, seed, category, tags}}")
    for r in rows:
        missing = {"id", "prompt", "seed"} - set(r)
        if missing:
            raise ValueError(f"{path}: row {r!r} lacks {sorted(missing)}")
    ids = [int(r["id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids")
    return rows[:n] if n else rows


def to_mono_int16(audio: torch.Tensor) -> np.ndarray:
    """``[B, C, T]`` float in [-1, 1] -> int16 ``[B, T]`` (channel mean)."""
    a = audio.float().mean(dim=1)
    return (a * 32768.0).round().clamp(-32768, 32767).to(torch.int16).numpy()


def _atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def run_prompts_json(args, sam, nb, state, sigmas, audio_tok, hidden, generate) -> int:
    """Sharded corpus render + capture (see module docstring). ``generate``
    = ``sa3_tada.generate`` (passed in so the shard logic is testable).
    ``--part I/N`` renders shards ``k % N == I``; part 0 creates ``means.npy``
    and owns ``meta.json`` (it waits for every shard before ``complete``),
    the other parts wait for part 0's ``meta.json`` and write their rows
    into the same memmap."""
    rows_all = load_prompts_json(args.prompts_json, args.n)
    start = int(args.render_from or 0)
    rows = rows_all[start:]
    n_total = len(rows_all)
    n, shard = len(rows), int(args.shard)
    n_shards = (n + shard - 1) // shard
    sel = list(state["sel"])
    ring, ring_scorers = int(args.ring or 0), [x for x in (args.ring_scorers or "").split(",") if x]
    part, n_parts = (int(x) for x in str(args.part or "0/1").split("/"))
    if not 0 <= part < n_parts:
        raise SystemExit(f"--part {args.part}: expected I/N with 0 <= I < N")
    owner = part == 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath, meta_path = out / "means.npy", out / "meta.json"
    shape = (n_total, args.steps, len(sel), hidden)
    if ring and ring % n_parts:
        raise SystemExit(f"--ring {ring} must be a multiple of the part count {n_parts}")
    old = {}
    if owner:
        if args.resume and mpath.exists():
            arr = np.load(mpath, mmap_mode="r+")
        else:
            if any(out.glob("audio_*.done")):
                raise SystemExit(f"{out} already has finished shards; pass --resume or use a fresh --out")
            arr = np.lib.format.open_memmap(mpath, mode="w+", dtype=np.float16, shape=shape)
            if start:
                if not args.init_means:
                    raise SystemExit("--render-from needs --init-means (the earlier corpus's means.npy)")
                src = np.load(args.init_means, mmap_mode="r")
                src_meta = Path(args.init_means).with_name("meta.json")
                src_ids = list(range(src.shape[2]))
                if src_meta.exists():
                    src_ids = json.loads(src_meta.read_text()).get("block_ids") or src_ids
                if src.shape[0] < start or src.shape[1] != args.steps or any(b not in src_ids for b in sel):
                    raise SystemExit(f"--init-means {src.shape} (blocks {src_ids}) cannot fill {start} rows x blocks {sel}")
                pick = [src_ids.index(b) for b in sel]
                for a in range(0, start, 250):
                    arr[a:min(start, a + 250)] = src[a:min(start, a + 250)][:, :, pick]
                arr.flush()
                print(f"[resid] copied {start} rows x blocks {sel} from {args.init_means}", flush=True)
        if args.resume and meta_path.exists():
            old = json.loads(meta_path.read_text())
    else:
        t_wait = time.perf_counter()
        while not (meta_path.exists() and mpath.exists()):
            if time.perf_counter() - t_wait > 1800:
                raise SystemExit("--part: part 0 never wrote meta.json / means.npy")
            time.sleep(2)
        arr = np.load(mpath, mmap_mode="r+")
    if tuple(arr.shape) != shape or arr.dtype != np.float16:
        raise SystemExit(f"{mpath} is {arr.dtype}{tuple(arr.shape)}, expected float16{shape}")
    old_sig = old.get("steps")
    if isinstance(old_sig, list):
        for k, v in enumerate(old_sig):
            sigmas.setdefault(k, v)
    # The seed every row is rendered with follows from the rule, so any part
    # can record it: each generate call = rows [i, i + batch) inside a shard.
    gen_seed = [None] * n
    for lo in range(0, n, shard):
        hi = min(n, lo + shard)
        for i in range(lo, hi, args.batch):
            for r in range(i, min(hi, i + args.batch)):
                gen_seed[r] = int(rows[i]["seed"])
    gen_seed_all = [None] * start + gen_seed

    def meta(complete: bool) -> dict:
        return {
            "n": n_total, "n_rendered": n, "render_from": start, "n_shards": n_shards, "shard_size": shard,
            "complete": bool(complete),
            "steps": [sigmas[k] for k in sorted(sigmas)], "n_steps": args.steps, "blocks": len(sel),
            "block_ids": sel, "model_blocks": nb,
            "init_means": str(args.init_means) if start else None,
            "ring": ring, "ring_scorers": ring_scorers,
            "hidden": hidden, "hook": "post_block_residual",
            "site": "post_block_residual (output of sa3_blocks(sam)[b])",
            "sr": SR, "duration_s": args.duration, "mode": "prompts_json",
            "prompts": str(args.prompts_json), "ids": [int(r["id"]) for r in rows_all],
            "batch": args.batch, "parts": n_parts, "gen_seed": gen_seed_all,
            "seed_rule": "one seed per generate call = seed of the call's first row",
            "sampler": "ARC sam.generate default (cfg 1)",
            "tokens": f"audio only ({audio_tok.num_memory_tokens} memory tokens and padding excluded)",
            "means": "float16 [n, n_steps, len(block_ids), hidden], row order = prompts order; "
                     "rows < render_from copied from init_means",
            "audio": "audio_{k:04d}.npz: wav int16 [B, T] mono, ids [B] (shard k = rendered rows "
                     "[k*shard, (k+1)*shard) after render_from); complete when audio_{k:04d}.done exists; "
                     "with ring R the file of shard k-R is recycled (renamed, overwritten) for shard k once "
                     "every ring scorer has labels/.parts/<scorer>/shard_{k-R}.parquet",
            "checkpoint": f"{args.checkpoint} (ARC)" if args.checkpoint == "medium" else args.checkpoint,
            "date": time.strftime("%Y-%m-%d"),
        }

    if owner:
        _write_json(meta_path, meta(False))
    t0 = time.perf_counter()
    resumed = 0
    mine = [k for k in range(n_shards) if k % n_parts == part]
    with torch.no_grad():
        for j, k in enumerate(mine):
            done = out / f"audio_{k:04d}.done"
            if done.exists():
                resumed += 1
                continue
            lo, hi = k * shard, min(n, (k + 1) * shard)
            wavs = []
            for i in range(lo, hi, args.batch):
                part_rows = rows[i:min(hi, i + args.batch)]
                state["buf"] = torch.zeros(len(part_rows), args.steps, len(sel), hidden)
                state["step"] = -1
                audio = generate(sam, [str(r["prompt"]) for r in part_rows], seed=gen_seed[i],
                                 duration=args.duration, steps=args.steps)
                if state["step"] != args.steps - 1:
                    raise RuntimeError(f"saw {state['step'] + 1} forwards, expected {args.steps}")
                arr[start + i:start + i + len(part_rows)] = state["buf"].numpy().astype(np.float16)
                wavs.append(to_mono_int16(audio))
            arr.flush()
            if ring and k >= ring:
                old_k = k - ring
                t_ring = time.perf_counter()
                while not all((out / "labels" / ".parts" / sc / f"shard_{old_k:04d}.parquet").exists()
                              for sc in ring_scorers):
                    time.sleep(2)
                old_f = out / f"audio_{old_k:04d}.npz"
                if old_f.exists():
                    old_f.replace(out / f"audio_{k:04d}.npz.tmp")
                waited = time.perf_counter() - t_ring
                if waited > 5:
                    print(f"[resid {part}/{n_parts}] ring wait {waited:.0f}s for shard {old_k}", flush=True)
            _atomic_npz(out / f"audio_{k:04d}.npz", wav=np.concatenate(wavs, axis=0),
                        ids=np.array([int(r["id"]) for r in rows[lo:hi]], dtype=np.int64))
            done.write_text(json.dumps({"rows": [lo, hi], "part": part, "t": time.strftime("%H:%M:%S")}))
            if j % 10 == 0 or j == len(mine) - 1:
                if owner:
                    _write_json(meta_path, meta(False))
                print(f"[resid {part}/{n_parts}] shard {k} ({j + 1}/{len(mine)} of mine) rows {hi}/{n} "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
    arr.flush()
    del arr
    if owner:
        while not all((out / f"audio_{k:04d}.done").exists() for k in range(n_shards)):
            time.sleep(5)
        _write_json(meta_path, meta(True))
    print(f"done part {part}/{n_parts}: {len(mine)} shards ({resumed} resumed) "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    return 0


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
    ap.add_argument("--prompts-json", default=None,
                    help="corpus prompts.json [{id, prompt, seed, category, tags}]: sharded audio + .done markers")
    ap.add_argument("--shard", type=int, default=32, help="--prompts-json: clips per audio shard")
    ap.add_argument("--resume", action="store_true", help="--prompts-json: skip shards whose .done exists")
    ap.add_argument("--part", default="0/1",
                    help="--prompts-json: I/N, render shards k %% N == I (one process per GPU; part 0 owns meta)")
    ap.add_argument("--blocks", default=None,
                    help="--prompts-json: comma list of block indices to store (default all); meta block_ids")
    ap.add_argument("--render-from", type=int, default=0,
                    help="--prompts-json: render rows from this index on; earlier rows come from --init-means")
    ap.add_argument("--init-means", default=None,
                    help="earlier corpus means.npy [N0, steps, all blocks, hidden]; part 0 copies the --blocks slice")
    ap.add_argument("--ring", type=int, default=0,
                    help="--prompts-json: recycle the audio file of shard k-R for shard k (0 = keep every shard)")
    ap.add_argument("--ring-scorers", default="desc,dyn,timbral,passt,clap_music,clap_general,muq",
                    help="scorers whose labels/.parts must hold shard k-R before its file is recycled")
    args = ap.parse_args()
    if args.prompts_json and args.self_label:
        ap.error("--prompts-json and --self-label are exclusive")

    from sa3_reference_generate import checkpoint_dir, load_local_model
    from acestep.engine import sa3_tada
    from acestep.engine.sa3_internals import trunk_module

    if args.prompts_json:
        prompts = prompt_index = seed0 = None
    elif args.self_label:
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

    def make(j):
        @torch.no_grad()
        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            mask = audio_tok(hs).to(hs.dtype).unsqueeze(-1)
            mean = (hs * mask).sum(1) / mask.sum(1)
            state["buf"][:, state["step"], j] = mean.float().cpu()
        return hook

    sel = sorted({int(x) for x in args.blocks.split(",")}) if args.blocks else list(range(nb))
    if args.blocks and not args.prompts_json:
        raise SystemExit("--blocks is only supported with --prompts-json")
    if any(not 0 <= b < nb for b in sel):
        raise SystemExit(f"--blocks {sel}: model has {nb} blocks")
    state["sel"] = sel
    hidden = int(getattr(blocks[0], "dim", 1536))
    handles = [blocks[0].register_forward_pre_hook(pre0)]
    handles += [blocks[b].register_forward_hook(make(j)) for j, b in enumerate(sel)]
    if args.prompts_json:
        try:
            return run_prompts_json(args, sam, nb, state, sigmas, audio_tok, hidden, sa3_tada.generate)
        finally:
            for h in handles:
                h.remove()
            audio_tok.remove()
            probe.remove()
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
