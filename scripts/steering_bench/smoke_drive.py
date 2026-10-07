"""v3 smoke, deliverable B: render and score the descriptor-ridge grid against the shipped tag knobs.

Runbook: notes/steering_pr/v3_smoke_runbook.md (DEMON). Question: does a ridge direction (ridge_directions.py)
steer its descriptor monotonically, within the PCI budget, and at least as well as the shipped knob that proxies it?

Vectors: every ``provenance.method == "ridge"`` pack in ``--packs`` (ridge_<d>, ridge_resid_<d>; ``--variant-b`` adds
``--packs``/variant_b as vb_<name>) plus the three control knobs (percussive, bright, dense_arrangement) read from the
installed bundle with DEMON-steer's ``load_bundle`` (``--bundle``; when no bundle file exists, the loose pack of the
same name beside it, read with DEMON-steer's ``load_pack``), re-saved as format-1 loose packs under ``<out>/controls``.

Per vector and sign:
* PCI cutoff: the control knob's calibrated cutoff for that sign (median over seeds 0,1,2, from the pack's
  ``calibrated_gain``); ridge vectors use their descriptor's control (same sign).
* cutoff alpha a_c: controls = calibrated gain median x magnitude (the alpha that ships); ridge = probe on the ship
  probe grid (|alpha| 2, 4, ... 128, ascending, the 12 prompts, LPAPS only, stops at the first point past the
  cutoff), interpolated with ``ship_tools.interp_side`` (the ship pass rule) from bench-layout lpaps.csv files.
* sweep: 5 alphas, sign x a_c x (0.2, 0.4, 0.6, 0.8, 1.0), the 12 screening prompts (first 12 TADA holdout
  prompts, as screen_knobs.py), seed EVAL_SEED + ``--seed-offset`` (2115 + 0), 10 s, 8 steps. One alpha-0 render
  of the same batch is the shared reference.
Renders: ``sa3_tada_run._pack_vectors`` x alpha under ``sa3_tada.steer_offline`` (the set_tools.py render call), in
this process (DEMON env). Scores, in a worker on ``--eval-python`` (evalenv): LPAPS vs alpha 0 and CLAP to the
control's anchors.json pair (screen_knobs.RealBackend), ``dyn.lra`` through label_corpus.py (the corpus labeller);
here: the five proxies with score_descriptors.measure_dir (descriptors.csv per sweep dir).
Primary descriptor: centroid and onset_rate from score_descriptors (= corpus desc.centroid / desc.onset_rate),
lra from label_corpus dyn.lra.

Pass, per ridge vector and descriptor direction (pos = descriptor up, neg = down):
* median over the 12 prompts of Spearman(descriptor, |alpha|) x direction over the 5 alphas >= 0.9;
* mean LPAPS at the top alpha <= 1.1 x the PCI cutoff;
* direction-signed mean descriptor delta vs alpha 0 at matched LPAPS >= the control's, where matched LPAPS
  L* = min(cutoff, ridge max LPAPS, control max LPAPS), deltas interpolated linearly in LPAPS from (0, 0), and the
  control sign is the one whose top-alpha delta goes furthest in that direction (the control's best case).

Outputs: ``<out>/results.json``, ``<out>/results.md``, audio and per-dir scores under ``<out>/renders``.

``--dry-run``: screen_knobs FakeGenerator (1 s tone-plus-noise clips) and FakeBackend (worker), a stubbed descriptor
pool, stub ridge packs when ``--packs`` holds none; everything else (pack loading, control export, probe, interp,
score_descriptors, analysis, writers) is the real path.

    python scripts/steering_bench/smoke_drive.py --dry-run --out E:/Projects/DEMON/steering-bench/v3_smoke/dryrun
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

V3 = Path("E:/Projects/DEMON/steering-bench/v3_smoke")
BUNDLE = Path.home() / ".daydream-scope/models/demon/steering_packs/sa3/medium/bundle.safetensors"
STEER_REPO = Path("C:/_dev/projects/DEMON-steer")
SR = 44100
STEPS = 8
DURATION = 10.0
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
PROBE_GRID = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)   # ship_item.sh PROBE, one sign
RHO_MIN, LPAPS_SLACK = 0.9, 1.1
#: descriptor -> (corpus label column, primary source, shipped control knob)
DESCRIPTORS = {
    "onset_rate": ("desc.onset_rate", "score_descriptors", "percussive"),
    "centroid": ("desc.centroid", "score_descriptors", "bright"),
    "lra": ("dyn.lra", "label_corpus", "dense_arrangement"),
}
WORKER_COLS = ["dyn.lra"]
SIDES = {1: "pos", -1: "neg"}
#: S1 (v3 smoke batch): seeds per render when --seed-offsets lists more than one; clips are ordered seed-major
#: (index = seed_i * n_prompts + prompt_i). 1 = the first smoke's single-seed behaviour.
N_SEEDS = 1


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _akey(a: float) -> str:
    return f"alpha_{float(a)}"


# ----------------------------------------------------------------------------------------------- control export

def export_controls(args) -> int:
    """Runs with DEMON-steer first on sys.path: its bundle / format-2 aware loader. Writes one npz + json per knob."""
    sys.path.insert(0, str(Path(args.steer_repo)))
    from acestep.steering import packs as P

    out = Path(args.export_controls)
    out.mkdir(parents=True, exist_ok=True)
    bundle = Path(args.bundle)
    found = {}
    if bundle.exists():
        for pk in P.load_bundle(bundle):
            found[pk.name] = (pk, f"bundle {bundle}")
    for name in args.controls:
        if name not in found:
            loose = bundle.parent / f"{name}.safetensors"
            if not loose.exists():
                raise SystemExit(f"control {name}: not in {bundle} and no {loose}")
            found[name] = (P.load_pack(loose), f"loose {loose} (no bundle file at {bundle})")
        pk, src = found[name]
        if pk.vector_neg is not None or pk.vectors_neg is not None or len(pk.all_blocks) != 1:
            raise SystemExit(f"control {name}: not a single-block, sign-symmetric pack; smoke_drive renders format 1 only")
        np.savez(out / f"{name}.npz", vector=pk.vector.float().numpy())
        meta = {"name": name, "block": int(pk.block), "hook": pk.hook, "magnitude": float(pk.magnitude),
                "norm": float(pk.norm), "policy": dict(pk.policy), "label": pk.label, "blurb": pk.blurb,
                "pos_anchor": pk.pos_anchor, "neg_anchor": pk.neg_anchor, "source": src,
                "calibrated_gain": (pk.provenance or {}).get("calibrated_gain"),
                "bar_pass": pk.bar_pass, "provenance_method": (pk.provenance or {}).get("method")}
        (out / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))
    return 0


def load_controls(args, out: Path) -> dict:
    """Export through DEMON-steer's loader (subprocess), re-save each control as a format-1 loose pack here."""
    import torch
    from acestep.steering.packs import SteeringPack, load_pack, save_pack

    tmp = out / "controls" / "export"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--export-controls", str(tmp), "--bundle", str(args.bundle),
           "--steer-repo", str(args.steer_repo), "--controls", *args.controls]
    env = dict(os.environ, PYTHONPATH="")
    subprocess.run(cmd, check=True, env=env, cwd=str(args.steer_repo))
    ctrl = {}
    for name in args.controls:
        m = json.loads((tmp / f"{name}.json").read_text())
        vec = np.load(tmp / f"{name}.npz")["vector"].astype(np.float32)
        g = m["calibrated_gain"]
        if not g or "pos" not in g or "neg" not in g:
            raise SystemExit(f"control {name}: no calibrated_gain per sign in {m['source']}")
        pk = SteeringPack(family="sa3", checkpoint="medium", block=m["block"], hidden_size=int(vec.shape[0]),
                          name=name, vector=torch.from_numpy(vec), label=m["label"], blurb=m["blurb"], hook=m["hook"],
                          norm=m["norm"], magnitude=m["magnitude"], policy=m["policy"],
                          provenance={"tool": "scripts/steering_bench/smoke_drive.py", "method": "control_copy",
                                      "source": m["source"], "calibrated_gain": g})
        path = save_pack(pk, out / "controls" / f"{name}.safetensors")
        back = load_pack(path)
        c = float(torch.dot(back.vector, pk.vector) / back.vector.norm() / pk.vector.norm())
        if back.block != m["block"] or c < 0.99999:
            raise SystemExit(f"{path}: re-read mismatch (cos {c:.6f})")
        side = {}
        for s, k in SIDES.items():
            side[k] = {"alpha_cut": float(g[k]["median"]) * m["magnitude"], "cutoff": statistics.median(g[k]["cutoff"]),
                       "gain_median": float(g[k]["median"])}
        ctrl[name] = {"path": str(path), "meta": m, "side": side}
        _log(f"control {name}: b{m['block']} mag {m['magnitude']:.3f} a_c +{side['pos']['alpha_cut']:.2f} "
             f"-{side['neg']['alpha_cut']:.2f} cutoff {side['pos']['cutoff']:.3f}/{side['neg']['cutoff']:.3f} ({m['source']})")
    return ctrl


# ----------------------------------------------------------------------------------------------- ridge packs

def stub_ridge_packs(dest: Path, hidden: int = 1536) -> None:
    """Dry run only: ridge_directions.py's names and provenance keys, random unit vectors."""
    import torch
    from acestep.steering.packs import SteeringPack, save_pack

    rng = np.random.default_rng(0)
    blocks = {"onset_rate": 1, "centroid": 23, "lra": 15}
    for d, (col, _src, control) in DESCRIPTORS.items():
        for kind in ("ridge", "ridge_resid"):
            v = rng.standard_normal(hidden).astype(np.float32)
            v /= np.linalg.norm(v)
            save_pack(SteeringPack(family="sa3", checkpoint="medium", block=blocks[d], hidden_size=hidden,
                                   name=f"{kind}_{d}", vector=torch.from_numpy(v), label=f"{d} ({kind})",
                                   norm=1.0, magnitude=0.1, provenance={
                                       "tool": "smoke_drive.py stub (dry run)", "method": "ridge", "label_col": col,
                                       "sign": 1, "control": control, "calibrated_gain": None, "stub": True}),
                      dest / f"{kind}_{d}.safetensors")


def find_ridge(args, packs: Path) -> dict:
    from acestep.steering.packs import load_pack

    col2d = {v[0]: k for k, v in DESCRIPTORS.items()}
    dirs = [(packs, "")] + ([(packs / "variant_b", "vb_")] if args.variant_b else [])
    out = {}
    for d, prefix in dirs:
        for p in sorted(d.glob("*.safetensors")):
            pk = load_pack(p)
            prov = pk.provenance or {}
            if prov.get("method") != "ridge":
                continue
            desc = col2d.get(prov.get("label_col"))
            if desc is None:
                raise SystemExit(f"{p}: label_col {prov.get('label_col')} is not one of {list(col2d)}")
            kind = "ridge_resid" if pk.name.startswith("ridge_resid") or prov.get("confounds") else "ridge"
            kind = prov.get("kind") or kind                  # --per-sign packs (S1) name their kind
            vid = prefix + pk.name
            out[vid] = {"path": str(p), "descriptor": desc, "kind": prefix + kind, "pack_sign": int(prov.get("sign", 1)),
                        "sides": [int(x) for x in prov.get("sides", [1, -1])],
                        "block": int(pk.block), "blocks": list(pk.target_blocks), "cv_r2": prov.get("cv_r2"),
                        "stub": bool(prov.get("stub"))}
    return out


# ----------------------------------------------------------------------------------------------- render

class Renderer:
    """SA3 medium, eager, as set_tools.py render: pack vectors x alpha under steer_offline."""

    def __init__(self, seed_offset: int, seed_offsets=None, chunk: int = 0):
        sys.path.insert(0, str(REPO / "scripts" / "tada"))
        import torch

        import sa3_tada_run as R
        from acestep.engine import sa3_tada

        torch.backends.cuda.matmul.allow_tf32 = True
        self.R, self.E, self.torch = R, sa3_tada, torch
        self.seed = R.EVAL_SEED + seed_offset
        self.seeds = [R.EVAL_SEED + o for o in (seed_offsets or [seed_offset])]
        self.chunk = chunk
        self.cache = {}
        self.clamp_stats = {"batch": set(), "frac": []}
        _log("loading SA3 medium")
        self.sam = R._load_sam()

    def _gen(self, prompts, seed):
        """One seed; prompts in chunks of ``self.chunk`` (0 = one batch), each chunk with the same seed."""
        n = self.chunk or len(prompts)
        return self.torch.cat([self.E.generate(self.sam, prompts[i:i + n], seed=seed, duration=DURATION, steps=STEPS)
                               for i in range(0, len(prompts), n)])

    def render(self, prompts, pack, alpha, cand, seeds=None) -> np.ndarray:
        """Clips seed-major over ``seeds`` (default ``self.seeds``)."""
        seeds = seeds or self.seeds
        if isinstance(pack, dict):                      # Q3 --clamp: set-projection hook, not the additive one
            with clamp_offline(self.sam, pack, self.clamp_stats):
                audio = self.torch.cat([self._gen(prompts, sd) for sd in seeds])
        elif pack is None or alpha == 0.0:
            audio = self.torch.cat([self._gen(prompts, sd) for sd in seeds])
        else:
            if pack not in self.cache:
                self.cache[pack] = (self.R._pack_vectors(pack, STEPS), self.R._PACK["hook"])
            pv, hook = self.cache[pack]
            vectors = {s: {b: v * float(alpha) for b, v in d.items()} for s, d in pv.items()}
            # one steer_offline context per generate call: its step counter runs on across calls in one
            # context, so a second call would get no steering (S1 bug, fixed 2026-10-07 14:05)
            n = self.chunk or len(prompts)
            outs = []
            for sd in seeds:
                for i in range(0, len(prompts), n):
                    with self.E.steer_offline(self.sam, vectors, 1.0, hook=hook):
                        outs.append(self.E.generate(self.sam, prompts[i:i + n], seed=sd, duration=DURATION,
                                                    steps=STEPS))
            audio = self.torch.cat(outs)
        self.torch.cuda.empty_cache()
        return (audio.mean(dim=1) * 32768.0).round().clamp(-32768, 32767).to(self.torch.int16).numpy()


class FakeRenderer:
    def __init__(self, seed_offset: int, seed_offsets=None, chunk: int = 0):
        from screen_knobs import FakeGenerator

        self.g = FakeGenerator()
        self.seed = 2115 + seed_offset
        self.seeds = [2115 + o for o in (seed_offsets or [seed_offset])]

    clamp_stats = {"batch": set(), "frac": []}

    def render(self, prompts, pack, alpha, cand, seeds=None) -> np.ndarray:
        self.g.cand = cand
        return np.concatenate([self.g.render(prompts, None, None, alpha) for _ in (seeds or self.seeds)])


# ----------------------------------------------------------------------------------------------- Q3 clamp (bench only)
#
# lit_review_v3_2026-10-07.md Q3, after Verdini et al. 2026 (arXiv 2609.33810): instead of h + alpha * u, set the
# projection of the residual on the unit pack direction u to a target t (mode "set": h + (t - <h,u>) u) or cap it
# (mode "min", the down side: h + (min(<h,u>, t) - <h,u>) u). Applied per audio token (memory tokens and padding left
# alone, the same rows capture_resid.py averages) at the pack's block output (post_block_residual) on every steered step.
# Targets: per sampler step, percentiles of the corpus projection <token mean, u> (music population, the rows the ridge
# was fit on). Render path of this bench only; the shipped runtime is untouched.

CLAMP_PCTS = (50.0, 40.0, 30.0, 20.0, 10.0)


def clamp_targets(vecs: dict, corpus: Path, pcts, cache: Path) -> dict:
    """``{vid: {"block", "pcts", "targets": [[t per pct] per step]}}`` from the corpus means (CPU, one pass)."""
    from acestep.steering.packs import load_pack
    from build_directions import block_ids, load_corpus, population_mask

    key = {"corpus": str(corpus), "pcts": list(pcts), "packs": sorted(v["path"] for v in vecs.values())}
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            return c["targets"]
    means, meta, _ids, cats = load_corpus(corpus)
    bid = block_ids(meta, means.shape[2])
    rows = np.flatnonzero(population_mask("music", cats))
    us = {}
    for vid, v in vecs.items():
        pk = load_pack(v["path"])
        if int(pk.block) not in bid:
            raise SystemExit(f"{vid}: block {pk.block} not in the corpus blocks {bid}")
        u = pk.vector.float().numpy().reshape(-1)
        us[vid] = (bid.index(int(pk.block)), u / np.linalg.norm(u), int(pk.block))
    proj = {vid: np.empty((len(rows), means.shape[1]), np.float32) for vid in us}
    for i in range(0, len(rows), 512):
        r = rows[i:i + 512]
        x = np.asarray(means[r[0]:r[-1] + 1], dtype=np.float32)[r - r[0]]          # [c, steps, blocks, H]
        for vid, (j, u, _b) in us.items():
            proj[vid][i:i + len(r)] = x[:, :, j, :] @ u
    out = {vid: {"block": b, "pcts": list(pcts), "n_rows": int(len(rows)),
                 "targets": [[float(np.percentile(proj[vid][:, s], p)) for p in pcts] for s in range(means.shape[1])],
                 "proj_mean": [float(proj[vid][:, s].mean()) for s in range(means.shape[1])],
                 "proj_std": [float(proj[vid][:, s].std()) for s in range(means.shape[1])]}
           for vid, (j, u, b) in us.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "targets": out}, indent=1))
    return out


@contextlib.contextmanager
def clamp_offline(sam, spec: dict, stats: dict):
    """``spec = {"path", "mode": set|min, "block", "targets": {step: t}}``. Records the hooked batch sizes (a CFG
    render would show 2 x the prompt batch) and, per call, the fraction of audio tokens whose projection changed."""
    import torch

    from acestep.engine import sa3_tada
    from acestep.engine.sa3_internals import trunk_module
    from acestep.steering.packs import load_pack
    from capture_resid import AudioTokens

    blocks = list(sa3_tada.sa3_blocks(sam))
    tok = AudioTokens(trunk_module(sam).transformer)
    u0 = load_pack(spec["path"]).vector.float().reshape(-1)
    u0 = u0 / u0.norm()
    state = {"step": -1}

    def tick(_m, _i):
        state["step"] += 1

    @torch.no_grad()
    def hook(_m, _i, out):
        t = spec["targets"].get(state["step"])
        if t is None:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        stats["batch"].add(int(hs.shape[0]))
        u = u0.to(hs.device)
        h = hs.float()
        p = h @ u                                                   # [B, S]
        new = torch.full_like(p, float(t)) if spec["mode"] == "set" else torch.clamp(p, max=float(t))
        m = tok(hs).to(p.dtype)
        d = (new - p) * m
        stats["frac"].append(float(((d != 0).float().sum() / m.sum()).item()))
        hs = (h + d.unsqueeze(-1) * u).to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    handles = [blocks[0].register_forward_pre_hook(tick), blocks[int(spec["block"])].register_forward_hook(hook)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
        tok.remove()


def save_audio(d: Path, wav: np.ndarray, prompts) -> Path:
    """score_descriptors / sa3_tada_run layout: alpha_<a>/audios.npz, audio int16 [n, 1, T]."""
    d.mkdir(parents=True, exist_ok=True)
    p = d / "audios.npz"
    np.savez(p, audio=wav[:, None, :], sr=np.int64(SR), names=np.array(list(prompts)))
    return p


# ----------------------------------------------------------------------------------------------- scoring worker

def _fake_gain(cand: str, desc: str) -> float:
    """Dry run: how strongly a vector moves a descriptor (its own descriptor 1 / 0.8 resid / 0.5 control)."""
    for d, (_c, _s, control) in DESCRIPTORS.items():
        if d != desc:
            continue
        if cand.endswith(f"ridge_resid_{d}"):
            return 0.8
        if cand.endswith(f"ridge_{d}"):
            return 1.0
        if cand == control:
            return 0.5
    return 0.1


def serve(args) -> int:
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    if args.fake_scorer:
        from screen_knobs import FakeBackend

        be = FakeBackend()
    else:
        if args.tada_root:
            os.environ["TADA_ROOT"] = args.tada_root
        from screen_knobs import RealBackend

        be = RealBackend(None, {"label_workers": args.label_workers})
    try:
        import pyloudnorm  # noqa: F401
        pyln = True
    except ImportError:
        pyln = False
    out.write(json.dumps({"ready": True, "pyloudnorm": pyln, "fake": bool(args.fake_scorer)}) + "\n")
    fake = bool(args.fake_scorer)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        if req.get("quit"):
            break
        try:
            z, r = np.load(req["npz"]), np.load(req["ref"])
            wav, ref, sr = z["audio"].reshape(len(z["audio"]), -1), r["audio"].reshape(len(r["audio"]), -1), int(z["sr"])
            res = {"id": req["id"], "lpaps": be.lpaps(ref, wav, sr)}
            cols, anchors = req.get("cols") or [], req.get("anchors") or []
            if fake:
                kw = {"cand": req["cand"], "signs": {c: 10.0 * _fake_gain(req["cand"], "lra") for c in cols}}
                res["cols"] = be.cols(cols, wav, sr, **kw) if cols else {}
                res["clap"] = be.anchors("clap", anchors, wav, sr, cand=req["cand"]) if anchors else {}
            else:
                res["cols"] = be.cols(cols, wav, sr) if cols else {}
                res["clap"] = be.anchors("clap", anchors, wav, sr) if anchors else {}
        except Exception as e:  # report, keep serving
            import traceback

            traceback.print_exc()
            res = {"id": req.get("id"), "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(res, default=float) + "\n")
        if "torch" in sys.modules:
            sys.modules["torch"].cuda.empty_cache()
    return 0


class _NanDict(dict):
    def __missing__(self, key):
        return float("nan")


class Worker:
    def __init__(self, python: str, fake: bool, tada_root: str | None, label_workers: int):
        cmd = [python, str(Path(__file__).resolve()), "--serve", "--label-workers", str(label_workers)]
        if fake:
            cmd.append("--fake-scorer")
        if tada_root:
            cmd += ["--tada-root", tada_root]
        env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=os.pathsep.join([str(REPO), str(HERE)]))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        first = self.proc.stdout.readline()
        if not first:
            raise SystemExit(f"scorer worker failed to start: {' '.join(cmd)}")
        self.info = json.loads(first)
        self.done, self.cv, self.n = {}, threading.Condition(), 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            if line.strip():
                r = json.loads(line)
                with self.cv:
                    self.done[r["id"]] = r
                    self.cv.notify_all()
        with self.cv:
            self.done["__eof__"] = True
            self.cv.notify_all()

    no_clap = False          # S1 --no-clap: CLAP left to rescore_independent.py; results carry NaN CLAP cells

    def submit(self, **req) -> str:
        if self.no_clap:
            req["anchors"] = []
        self.n += 1
        req["id"] = f"r{self.n}"
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return req["id"]

    def get(self, rid: str) -> dict:
        with self.cv:
            while rid not in self.done:
                if "__eof__" in self.done:
                    raise SystemExit("scorer worker exited")
                self.cv.wait()
            r = self.done.pop(rid)
        if "error" in r:
            raise SystemExit(f"scorer error on {rid}: {r['error']}")
        if self.no_clap:
            r["clap"] = _NanDict()
        return r

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"quit": True}) + "\n")
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
        except Exception:
            self.proc.kill()


# ----------------------------------------------------------------------------------------------- analysis

def write_lpaps_csv(path: Path, curve: dict) -> None:
    """``protocol_results/lpaps.csv`` as the TADA scorer writes it (alpha, mean, std)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "mean", "std"])
        for a in sorted(curve):
            v = curve[a]
            w.writerow([a, float(np.mean(v)), float(np.std(v))])


def _ranks(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x), dtype=float)
    for v in np.unique(x):                     # average ties
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = _ranks(x[ok]), _ranks(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def delta_at(lp: list, dl: list, level: float) -> float:
    """Descriptor delta at mean LPAPS ``level``: piecewise linear through (0, 0) and the alpha-ordered points,
    LPAPS made non-decreasing (running max) so the curve is a function."""
    xs, ys, m = [0.0], [0.0], 0.0
    for x, y in zip(lp, dl):
        m = max(m, x)
        xs.append(m)
        ys.append(y)
    if level >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= level:
            if xs[i] == xs[i - 1]:
                return ys[i]
            t = (level - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def side_metrics(rec: dict, desc: str, direction: int, ref_desc: np.ndarray) -> dict:
    """One vector-sign sweep in one descriptor direction."""
    alphas = rec["alphas"]                                  # signed, ascending |alpha|
    vals = np.array([rec["desc"][desc][_akey(a)] for a in alphas])          # [5, n]
    lp = [float(np.mean(rec["lpaps"][_akey(a)])) for a in alphas]
    rho = [direction * spearman(vals[:, i], np.abs(alphas)) for i in range(vals.shape[1])]
    rho_f = [r for r in rho if np.isfinite(r)]
    dl = [float(direction * np.nanmean(vals[k] - ref_desc)) for k in range(len(alphas))]
    out = {"rho_per_prompt": rho, "rho_median": float(np.median(rho_f)) if rho_f else float("nan"),
           "lpaps": lp, "lpaps_top": lp[-1], "lpaps_max": max(lp), "delta": dl, "delta_top": dl[-1]}
    if N_SEEDS > 1:
        # S1: per-clip Spearman kept as rho_clip_median; the pass statistic becomes the median over prompts of
        # Spearman on the seed-mean descriptor (clips seed-major).
        sm = np.nanmean(vals.reshape(len(alphas), N_SEEDS, -1), axis=1)
        rs = [direction * spearman(sm[:, i], np.abs(alphas)) for i in range(sm.shape[1])]
        rs_f = [r for r in rs if np.isfinite(r)]
        out.update(rho_clip_median=out["rho_median"], rho_per_prompt_seedmean=rs,
                   rho_median=float(np.median(rs_f)) if rs_f else float("nan"))
    return out


def analyse(vecs: dict, ctrl: dict, recs: dict, ref: dict, args) -> dict:
    rows = []
    for d, (col, src, control) in DESCRIPTORS.items():
        refd = np.asarray(ref["desc"][d])
        for direction in (1, -1):
            dname = SIDES[direction]
            # control: the knob sign whose top-alpha delta goes furthest in this direction
            cands = []
            for s in (1, -1):
                if (control, s) not in recs:
                    continue
                m = side_metrics(recs[(control, s)], d, direction, refd)
                cands.append((m["delta_top"], s, m))
            _, cs, cm = max(cands, key=lambda x: x[0])
            ccut = ctrl[control]["side"][SIDES[cs]]["cutoff"]
            crow = {"descriptor": d, "direction": dname, "vector": control, "kind": "control",
                    "knob_sign": SIDES[cs], "alpha_top": recs[(control, cs)]["alphas"][-1], "cutoff": ccut,
                    "rho_median": cm["rho_median"], "lpaps_top_over_cut": cm["lpaps_top"] / ccut,
                    "delta_top": cm["delta_top"], "metrics": cm}
            for vid, v in vecs.items():
                if v["descriptor"] != d:
                    continue
                s = direction * v["pack_sign"]
                if (vid, s) not in recs:                     # --per-sign packs render their own side only
                    continue
                rec = recs[(vid, s)]
                m = side_metrics(rec, d, direction, refd)
                cut = rec["cutoff"]
                lstar = min(cut, m["lpaps_max"], cm["lpaps_max"])
                dr, dc = delta_at(m["lpaps"], m["delta"], lstar), delta_at(cm["lpaps"], cm["delta"], lstar)
                ok_rho = bool(np.isfinite(m["rho_median"]) and m["rho_median"] >= RHO_MIN)
                ok_lp = bool(m["lpaps_top"] <= LPAPS_SLACK * cut)
                ok_dc = bool(dr >= dc)
                rows.append({"descriptor": d, "direction": dname, "vector": vid, "kind": v["kind"],
                             "knob_sign": SIDES[s], "alpha_top": rec["alphas"][-1], "alpha_cut_source": rec["a_c_source"],
                             "cutoff": cut, "rho_median": m["rho_median"], "lpaps_top_over_cut": m["lpaps_top"] / cut,
                             "matched_lpaps": lstar, "delta_matched": dr, "control_delta_matched": dc,
                             "control_sign": SIDES[cs], "delta_top": m["delta_top"],
                             "rho_clip_median": m.get("rho_clip_median"),
                             "pass_rho": ok_rho, "pass_lpaps": ok_lp, "pass_vs_control": ok_dc,
                             "pass": ok_rho and ok_lp and ok_dc, "metrics": m})
            rows.append(crow)
    return rows


def clap_summary(recs: dict, ref: dict, ctrl_of: dict) -> dict:
    """Per vector-sign: mean (cos pos - cos neg) to the control's anchors.json pair, delta vs alpha 0 per alpha."""
    out = {}
    for (vid, s), rec in recs.items():
        c = ctrl_of[vid]
        pos, neg = rec["anchors"]
        base = np.asarray(ref["clap"][c][pos]) - np.asarray(ref["clap"][c][neg])
        out[f"{vid}|{SIDES[s]}"] = {"control": c, "anchors": [pos, neg], "delta": [
            float(np.mean(np.asarray(rec["clap"][_akey(a)][pos]) - np.asarray(rec["clap"][_akey(a)][neg]) - base))
            for a in rec["alphas"]]}
    return out


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{x:.{nd}g}" if abs(x) >= 1000 else f"{x:.{nd}f}"


def write_md(path: Path, rows: list, meta: dict, clap: dict) -> None:
    L = [f"# v3 smoke results{' (DRY RUN: stub renderer, scorer and packs; numbers are meaningless)' if meta['dry_run'] else ''}",
         "", f"Generated {meta['finished']} by scripts/steering_bench/smoke_drive.py. Runbook: notes/steering_pr/"
         "v3_smoke_runbook.md.", "",
         f"{meta['n_prompts']} prompts ({meta['prompts_source']}), seed(s) {meta.get('seeds', meta['seed'])}, "
         f"{meta['duration']} s, "
         f"{STEPS} steps; 5 alphas = sign x a_c x {list(FRACTIONS)}; a_c: controls = calibrated gain x magnitude, "
         "ridge = probe on the ship grid interpolated with ship_tools.interp_side; PCI cutoff = the control's "
         "calibrated cutoff for that sign (median of seeds 0,1,2).",
         ("Multiple seeds: Spearman med = median over prompts of Spearman on the seed-mean descriptor (pass "
          "statistic); (clip x) = median over every prompt x seed clip. " if meta.get("seeds") and
          len(meta["seeds"]) > 1 else "") +
         f"Pass (ridge rows): median Spearman x direction >= {RHO_MIN}; LPAPS at top alpha <= {LPAPS_SLACK} x cutoff; "
         "delta at matched LPAPS >= control's (control sign = the one moving furthest in that direction).",
         f"Descriptor source: centroid, onset_rate = score_descriptors.py (proxies); lra = label_corpus.py dyn.lra "
         f"(pyloudnorm in scorer env: {meta['pyloudnorm']}).", ""]
    for dname in ("pos", "neg"):
        L += [f"## Direction {dname} (descriptor {'up' if dname == 'pos' else 'down'})", "",
              "| descriptor | vector | knob sign | Spearman med | LPAPS top / cut | matched LPAPS | delta @ matched | "
              "control delta @ matched | pass |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["direction"] != dname:
                continue
            if r["kind"] == "control":
                L.append(f"| {r['descriptor']} | {r['vector']} (control) | {r['knob_sign']} | {fmt(r['rho_median'])} | "
                         f"{fmt(r['lpaps_top_over_cut'])} | | (top) {fmt(r['delta_top'])} | | ref |")
            else:
                flags = "".join(k for k, f in (("r", r["pass_rho"]), ("l", r["pass_lpaps"]), ("c", r["pass_vs_control"]))
                                if not f)
                clip = f" (clip {fmt(r['rho_clip_median'])})" if r.get("rho_clip_median") is not None else ""
                L.append(f"| {r['descriptor']} | {r['vector']} | {r['knob_sign']} | {fmt(r['rho_median'])}{clip} | "
                         f"{fmt(r['lpaps_top_over_cut'])} | {fmt(r['matched_lpaps'])} | {fmt(r['delta_matched'])} | "
                         f"{fmt(r['control_delta_matched'])} ({r['control_sign']}) | "
                         f"{'PASS' if r['pass'] else 'fail ' + flags} |")
        L.append("")
    L += ["fail flags: r = Spearman, l = LPAPS over budget, c = below control at matched LPAPS. Deltas are "
          "direction-signed raw descriptor units (Hz, onsets/s, LU), mean over prompts, paired vs alpha 0.", "",
          "## CLAP to the control's anchors.json pair (cos pos - cos neg, delta vs alpha 0, per alpha)", "",
          "| vector, sign | control | deltas |", "|---|---|---|"]
    for k, v in clap.items():
        L.append(f"| {k} | {v['control']} | {' '.join(fmt(x) for x in v['delta'])} |")
    L += ["", "## Notes", ""] + [f"- {n}" for n in meta["notes"]]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def clamp_compare(out: Path, rows: list, recs: dict, ref: dict, vecs: dict, args, batches: list) -> None:
    """Q3: add (earlier --compare-results run) vs clamp rows on the down side, each re-evaluated at one common matched
    LPAPS level per descriptor (min of the cutoff and every row's and the control's largest LPAPS)."""
    if not args.compare_results.exists():
        _log(f"no {args.compare_results}: add vs clamp table skipped")
        return
    old = json.loads(args.compare_results.read_text())
    oref = {d: np.asarray(v) for d, v in old["ref"]["desc"].items()}
    L = ["", "## Q3: add vs clamp, descriptor down side", "",
         f"Add rows from {args.compare_results} (same prompts and seed). All rows re-evaluated at one matched LPAPS "
         "per descriptor (min over cutoff, each row's max LPAPS and the control's). Hooked batch sizes during clamp "
         f"renders: {batches} (prompt batch {len(json.loads((out / 'results.json').read_text())['meta']['prompts'])}).",
         "", "| descriptor | mode | Spearman med | LPAPS top / cut | delta top | common LPAPS | delta @ common | "
         "control delta @ common | token frac changed (weak..strong) |", "|---|---|---|---|---|---|---|---|---|"]
    summary = []
    for d in DESCRIPTORS:
        base = [v["base"] for v in vecs.values() if v["descriptor"] == d]
        if not base:
            continue
        b = base[0]
        cr = next(r for r in rows if r["descriptor"] == d and r["direction"] == "neg" and r["kind"] == "control")
        cm = cr["metrics"]
        ent = []
        oc = next((r for r in old["rows"] if r["descriptor"] == d and r["direction"] == "neg" and r["vector"] == b), None)
        sw = old["sweeps"].get(f"{b}|neg")
        if oc and sw:
            ent.append(("add", side_metrics(sw, d, -1, oref[d]), sw["cutoff"], None))
        for vid, v in vecs.items():
            if v["descriptor"] == d and (vid, -1) in recs:
                rc = recs[(vid, -1)]
                ent.append((f"clamp_{v['clamp']}", side_metrics(rc, d, -1, np.asarray(ref["desc"][d])), rc["cutoff"],
                            [rc["clamp_frac"].get(_akey(a)) for a in rc["alphas"]]))
        lvl = min([cr["cutoff"], cm["lpaps_max"]] + [m["lpaps_max"] for _n, m, _c, _f in ent])
        dc = delta_at(cm["lpaps"], cm["delta"], lvl)
        for name, m, cut, fr in ent:
            dm = delta_at(m["lpaps"], m["delta"], lvl)
            L.append(f"| {d} | {name} | {fmt(m['rho_median'])} | {fmt(m['lpaps_top'] / cut)} | {fmt(m['delta_top'])} | "
                     f"{fmt(lvl)} | {fmt(dm)} | {fmt(dc)} | "
                     f"{' '.join(fmt(x, 2) for x in fr) if fr else ''} |")
            summary.append({"descriptor": d, "mode": name, "rho_median": m["rho_median"],
                            "lpaps_top_over_cut": m["lpaps_top"] / cut, "delta_top": m["delta_top"],
                            "common_lpaps": lvl, "delta_common": dm, "control_delta_common": dc, "token_frac": fr})
    L += ["", "Deltas: descriptor-down signed (positive = descriptor went down), raw units, mean over prompts vs "
          "alpha 0. Spearman over the 5 strengths (add: |alpha|; clamp: target rank, 50th to 10th percentile)."]
    with open(out / "results.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    (out / "q3_compare.json").write_text(json.dumps({"batches": batches, "rows": summary}, indent=1, default=float))


# ----------------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packs", type=Path, default=V3 / "packs", help="ridge_directions.py --out")
    ap.add_argument("--out", type=Path, default=V3)
    ap.add_argument("--bundle", type=Path, default=BUNDLE)
    ap.add_argument("--steer-repo", type=Path, default=STEER_REPO, help="checkout with load_bundle (DEMON-steer)")
    ap.add_argument("--controls", nargs="*", default=[v[2] for v in DESCRIPTORS.values()])
    ap.add_argument("--anchors", type=Path, default=HERE / "anchors.json")
    ap.add_argument("--variant-b", action="store_true", help="also the top-3 block packs in <packs>/variant_b")
    ap.add_argument("--only", nargs="*", default=[], help="ridge vector ids to keep")
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--seed-offsets", type=int, nargs="*", default=None,
                    help="S1: render every sweep at these seed offsets (clips seed-major); default = --seed-offset only")
    ap.add_argument("--n-test-prompts", type=int, default=0,
                    help="S1: append the first N TADA test prompts (the protocol set) to the --n-prompts holdout ones")
    ap.add_argument("--probe-small", action="store_true",
                    help="S1: run the a_c probe on the first --n-prompts holdout prompts at the first seed only "
                         "(the first smoke's probe set), against its own alpha-0 reference")
    ap.add_argument("--render-chunk", type=int, default=0, help="max prompts per generate call (0 = all)")
    ap.add_argument("--no-clap", action="store_true", help="skip CLAP in the scorer worker (NaN CLAP columns)")
    ap.add_argument("--resume", action="store_true", help="reuse audios.npz already on disk (same args only)")
    ap.add_argument("--control-sides", type=int, nargs="*", default=[1, -1], choices=[1, -1],
                    help="S1 extras: control knob signs to render (default both)")
    ap.add_argument("--eval-python", help="evalenv python for the scorer worker (laion_clap, steer-audio)")
    ap.add_argument("--tada-root", default=os.environ.get("TADA_ROOT"), help="TADA_ROOT for the scorer worker")
    ap.add_argument("--label-workers", type=int, default=8)
    ap.add_argument("--allow-lra-fallback", action="store_true",
                    help="run although pyloudnorm is missing in the scorer env (lra then differs from the corpus)")
    ap.add_argument("--clamp", nargs="*", choices=("set", "min"), default=None,
                    help="Q3: render the ridge vectors' descriptor-down side with the set-projection hook (one row "
                         "per mode) instead of the additive sweep; 5 targets = corpus projection percentiles")
    ap.add_argument("--clamp-pcts", type=float, nargs="*", default=list(CLAMP_PCTS),
                    help="Q3: target percentiles, weakest first (descending)")
    ap.add_argument("--clamp-match-add", action="store_true",
                    help="Q3: targets = corpus mean projection - |alpha| of the --compare-results add sweep (neg)")
    ap.add_argument("--corpus", type=Path, default=Path("E:/Projects/DEMON/steering-bench/many_knobs_v2/corpus"),
                    help="Q3: v2 corpus (means.npy, meta.json, prompts.json) for the clamp targets")
    ap.add_argument("--compare-results", type=Path, default=V3 / "results.json",
                    help="Q3: earlier additive results.json (same prompts / seed) for the add vs clamp table")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--fake-scorer", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--export-controls", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.serve:
        return serve(args)
    if args.export_controls:
        return export_controls(args)
    t0 = time.time()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    notes = []
    if not args.dry_run and not args.eval_python:
        ap.error("--eval-python <evalenv python> is required for a real run")

    import score_descriptors as SD
    import ship_tools as ST

    if args.dry_run:
        import desc_pool

        from screen_knobs import FakeBackend

        cur = {"cand": ""}

        def fake_measure(clips, sr, workers=None):
            wav = np.stack([np.asarray(c).reshape(-1) for c in clips])
            amp = np.array(FakeBackend._amp(wav, cur["cand"], sr))
            noise = wav.astype(np.float64).std(1) / 32768.0
            scale = {"centroid": 1e4, "lowhigh_db": 10.0, "flatness": 1.0, "onset_rate": 10.0, "perc_ratio": 1.0}
            return np.stack([1000 * noise + amp * scale[k] * _fake_gain(cur["cand"], k) for k in SD.DESCRIPTORS], 1)

        desc_pool.measure_clips = fake_measure
    else:
        cur = {"cand": ""}

    # vectors
    packs = args.packs.resolve()
    vecs = find_ridge(args, packs) if packs.exists() else {}
    if not vecs:
        if not args.dry_run:
            raise SystemExit(f"no ridge packs (provenance.method ridge) under {packs}")
        packs = out / "stub_packs"
        stub_ridge_packs(packs)
        vecs = find_ridge(args, packs)
        notes.append(f"dry run: no ridge packs under {args.packs}; stub packs written to {packs}")
    if args.only:
        vecs = {k: v for k, v in vecs.items() if k in set(args.only)}
    if args.clamp:
        if any(v["pack_sign"] != 1 for v in vecs.values()):
            raise SystemExit("--clamp expects packs with sign +1 (the down side lowers the projection)")
        targets = clamp_targets(vecs, args.corpus, args.clamp_pcts, out / "clamp_targets.json")
        if args.clamp_match_add:
            # targets = per-step corpus mean projection minus the additive sweep's |alpha| (same 5 shifts as the add
            # rows of --compare-results), i.e. the add's displacement of an average clip, as a clamp
            old = json.loads(args.compare_results.read_text())["sweeps"]
            for vid in targets:
                al = [abs(a) for a in old[f"{vid}|neg"]["alphas"]]
                t = targets[vid]
                t["pcts"] = [f"mean-{a:.2f}" for a in al]
                t["targets"] = [[m - a for a in al] for m in t["proj_mean"]]
            notes.append("Q3 --clamp-match-add: clamp target = corpus mean projection (per step) - |add alpha|")
        vecs = {f"{vid}__clamp_{m}": dict(v, kind=f"{v['kind']}_clamp_{m}", clamp=m, base=vid, sides=[-1],
                                         clamp_targets=targets[vid])
                for vid, v in vecs.items() for m in args.clamp}
        notes.append(f"Q3 clamp: modes {args.clamp}, targets = corpus projection percentiles {args.clamp_pcts} "
                     f"per step (music rows of {args.corpus}); the 'alphas' of a clamp row are target ranks "
                     "-1..-5 (weakest to strongest)")
    elif args.only:
        pass
    elif not args.dry_run:
        want = {f"{k}_{d}" for d in DESCRIPTORS for k in ("ridge", "ridge_resid")}
        if want - set(vecs):
            raise SystemExit(f"ridge packs missing under {packs}: {sorted(want - set(vecs))} (or pass --only)")
    ctrl = load_controls(args, out)
    anchors = json.loads(args.anchors.read_text(encoding="utf-8"))["concepts"]
    ctrl_of = {c: c for c in ctrl}
    for vid, v in vecs.items():
        ctrl_of[vid] = DESCRIPTORS[v["descriptor"]][2]
    for c in ctrl:
        a = anchors.get(c)
        if not a:
            raise SystemExit(f"anchors.json concepts has no {c}")
        m = ctrl[c]["meta"]
        if m.get("pos_anchor") and m["pos_anchor"] != a[0]:
            notes.append(f"{c}: anchors.json pair {a} differs from the shipped pack's own anchors "
                         f"[{m['pos_anchor']!r}, {m['neg_anchor']!r}]; CLAP column uses anchors.json as specified")
    _log(f"{len(vecs)} ridge vectors {sorted(vecs)}; controls {sorted(ctrl)}")

    worker = Worker(args.eval_python if not args.dry_run else (args.eval_python or sys.executable), args.dry_run,
                    args.tada_root, args.label_workers)
    worker.no_clap = bool(args.no_clap)
    if not worker.info.get("pyloudnorm"):
        msg = "pyloudnorm missing in the scorer env: dyn.lra falls back to RMS windows (corpus used pyloudnorm)"
        if not args.dry_run and not args.allow_lra_fallback:
            worker.close()
            raise SystemExit(msg + "; install it in the evalenv or pass --allow-lra-fallback")
        notes.append(msg)
    global N_SEEDS
    N_SEEDS = len(args.seed_offsets) if args.seed_offsets else 1
    gen = (FakeRenderer if args.dry_run else Renderer)(args.seed_offset, args.seed_offsets, args.render_chunk)
    if args.dry_run:
        prompts, psrc = [f"dry prompt {i}" for i in range(args.n_prompts)], "dry-run stub prompts"
        if args.n_test_prompts:
            prompts += [f"dry test prompt {i}" for i in range(args.n_test_prompts)]
            psrc += f" + {args.n_test_prompts} stub test prompts"
    else:
        from acestep.tada import concepts as C

        prompts, psrc = C.benchmark_prompts(holdout=True)[0][: args.n_prompts], "first TADA holdout prompts"
        if args.n_test_prompts:
            test = C.benchmark_prompts(holdout=False)[0][: args.n_test_prompts]
            if set(test) & set(prompts):
                raise SystemExit("test and holdout prompts overlap")
            prompts += test
            psrc = f"first {args.n_prompts} TADA holdout + first {args.n_test_prompts} TADA test prompts"
    probe_prompts = prompts[: args.n_prompts] if args.probe_small else prompts
    probe_seeds = gen.seeds[:1] if args.probe_small else None

    R = out / "renders"
    nrend = 0

    def render_to(d: Path, pack, alpha, cand, probe: bool = False) -> Path:
        nonlocal nrend
        nrend += 1
        if args.resume and (d / _akey(alpha) / "audios.npz").exists():   # S1: restart after a kill
            return d / _akey(alpha) / "audios.npz"
        if probe and args.probe_small:
            return save_audio(d / _akey(alpha), gen.render(probe_prompts, pack, alpha, cand, seeds=probe_seeds),
                              probe_prompts)
        return save_audio(d / _akey(alpha), gen.render(prompts, pack, alpha, cand),
                          [p for _ in gen.seeds for p in prompts])

    # reference
    ref_dir = R / "ref"
    ref_npz = render_to(ref_dir, None, 0.0, "")
    all_anchors = sorted({t for c in ctrl for t in anchors[c][:2]})
    rr = worker.get(worker.submit(npz=str(ref_npz), ref=str(ref_npz), cols=WORKER_COLS, anchors=all_anchors, cand=""))
    ref = {"clap": {c: {t: rr["clap"][t] for t in anchors[c][:2]} for c in ctrl}, "cols": rr["cols"]}
    cur["cand"] = ""
    rd = SD.measure_dir(ref_dir, force=True)[0.0]
    ref["desc"] = {d: (rr["cols"][col] if src == "label_corpus" else rd[:, SD.DESCRIPTORS.index(d)].tolist())
                   for d, (col, src, _c) in DESCRIPTORS.items()}
    probe_ref = render_to(R / "ref_probe", None, 0.0, "", probe=True) if args.probe_small else ref_npz
    _log(f"reference rendered and scored ({len(prompts)} prompts x {len(gen.seeds)} seeds)")

    # probe (ridge) + sweeps
    recs, pending = {}, []
    order = list(vecs) + list(ctrl)
    for vid in order:
        isc = vid in ctrl
        pack = ctrl[vid]["path"] if isc else vecs[vid]["path"]
        control = ctrl_of[vid]
        vdir = R / vid
        probe_curve = {0.0: [0.0] * len(prompts)}
        pci = {}
        for s in (tuple(args.control_sides) if isc else vecs[vid]["sides"]):
            side = SIDES[s]
            cut = ctrl[control]["side"][side]["cutoff"]
            pci[s * 1.0] = [cut]
            if isc:
                a_c, src = ctrl[vid]["side"][side]["alpha_cut"], "calibrated gain x magnitude"
            elif vecs[vid].get("clamp"):
                ct = vecs[vid]["clamp_targets"]
                alphas = [s * float(k + 1) for k in range(len(ct["pcts"]))]
                rec = {"vid": vid, "sign": s, "a_c": float(len(alphas)), "cutoff": cut, "alphas": alphas,
                       "a_c_source": f"clamp {vecs[vid]['clamp']} at pcts {ct['pcts']}", "clamp_targets": ct,
                       "anchors": anchors[control][:2], "lpaps": {}, "clap": {}, "cols": {}, "clamp_frac": {}}
                for k, a in enumerate(alphas):
                    spec = {"path": pack, "mode": vecs[vid]["clamp"], "block": ct["block"],
                            "targets": {st: row[k] for st, row in enumerate(ct["targets"])}}
                    n0 = len(gen.clamp_stats["frac"])
                    npz = render_to(vdir / "eval" / f"pack_pack_{vid}", spec, a, vid)
                    fr = gen.clamp_stats["frac"][n0:]
                    rec["clamp_frac"][_akey(a)] = float(np.mean(fr)) if fr else None
                    pending.append((rec, a, worker.submit(npz=str(npz), ref=str(ref_npz), cols=WORKER_COLS,
                                                          anchors=rec["anchors"], cand=vid)))
                recs[(vid, s)] = rec
                _log(f"{vid} {side}: clamp {vecs[vid]['clamp']}, {len(alphas)} targets, hooked batch sizes "
                     f"{sorted(gen.clamp_stats['batch'])}, token frac changed {rec['clamp_frac']}")
                continue
            else:
                for a in PROBE_GRID:
                    npz = render_to(vdir / "calib" / f"pack_pack_{vid}", pack, s * a, vid, probe=True)
                    lp = worker.get(worker.submit(npz=str(npz), ref=str(probe_ref), cand=vid))["lpaps"]
                    probe_curve[s * a] = lp
                    if float(np.mean(lp)) >= cut:
                        break
                write_lpaps_csv(vdir / "calib" / f"pack_pack_{vid}" / "protocol_results" / "lpaps.csv", probe_curve)
                write_lpaps_csv(vdir / "eval" / f"pci_all_{vid}" / "protocol_results" / "lpaps.csv", pci)
                r = ST.interp_side(vdir, vid, s)
                a_c = r["alpha"]
                src = f"probe {'reached' if r['reached'] else 'NOT reached (largest probe)'}, ship_tools.interp_side"
            alphas = [s * a_c * f for f in FRACTIONS]
            rec = {"vid": vid, "sign": s, "a_c": a_c, "a_c_source": src, "cutoff": cut, "alphas": alphas,
                   "anchors": anchors[control][:2], "lpaps": {}, "clap": {}, "cols": {}}
            for a in alphas:
                npz = render_to(vdir / "eval" / f"pack_pack_{vid}", pack, a, vid)
                pending.append((rec, a, worker.submit(npz=str(npz), ref=str(ref_npz), cols=WORKER_COLS,
                                                      anchors=rec["anchors"], cand=vid)))
            recs[(vid, s)] = rec
            _log(f"{vid} {side}: a_c {a_c:.3f} ({src}), cutoff {cut:.3f}, rendered {len(alphas)} alphas")
    for rec, a, rid in pending:
        r = worker.get(rid)
        k = _akey(a)
        rec["lpaps"][k], rec["clap"][k], rec["cols"][k] = r["lpaps"], r["clap"], r["cols"]
    worker.close()
    # descriptors: score_descriptors over each sweep dir (both signs), label_corpus columns from the worker
    for vid in order:
        cur["cand"] = vid
        sd = SD.measure_dir(R / vid / "eval" / f"pack_pack_{vid}", force=True)
        for s in (1, -1):
            if (vid, s) not in recs:
                continue
            rec = recs[(vid, s)]
            rec["desc"] = {d: {_akey(a): (rec["cols"][_akey(a)][col] if src == "label_corpus" else
                                          sd[float(a)][:, SD.DESCRIPTORS.index(d)].tolist()) for a in rec["alphas"]}
                           for d, (col, src, _c) in DESCRIPTORS.items()}
            rec["proxies_top"] = {k: float(np.mean(sd[float(rec["alphas"][-1])][:, i] - rd[:, i]))
                                  for i, k in enumerate(SD.DESCRIPTORS)}

    rows = analyse(vecs, ctrl, recs, ref, args)
    clap = clap_summary(recs, ref, ctrl_of)
    meta = {"dry_run": args.dry_run, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
            "renders": nrend, "clips": nrend * len(prompts) * len(gen.seeds), "n_prompts": len(prompts),
            "prompts_source": psrc, "seeds": gen.seeds, "probe_small": args.probe_small,
            "prompts": prompts, "seed": gen.seed, "duration": DURATION if not args.dry_run else 1.0,
            "fractions": FRACTIONS, "probe_grid": PROBE_GRID, "packs": str(packs), "bundle": str(args.bundle),
            "pyloudnorm": worker.info.get("pyloudnorm"), "descriptors": DESCRIPTORS, "notes": notes}
    res = {"meta": meta, "vectors": vecs,
           "controls": {c: {"path": v["path"], "side": v["side"], "source": v["meta"]["source"]} for c, v in ctrl.items()},
           "rows": rows, "clap": clap,
           "sweeps": {f"{vid}|{SIDES[s]}": {k: v for k, v in rec.items() if k not in ("cols", "clap")}
                      for (vid, s), rec in recs.items()},
           "ref": {"desc": ref["desc"]}}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_md(out / "results.md", rows, meta, clap)
    if args.clamp:
        clamp_compare(out, rows, recs, ref, vecs, args, sorted(gen.clamp_stats["batch"]))
    npass = sum(1 for r in rows if r["kind"] != "control" and r["pass"])
    nrid = sum(1 for r in rows if r["kind"] != "control")
    _log(f"done: {nrend} renders ({meta['clips']} clips) in {meta['seconds']} s; {npass}/{nrid} ridge "
         f"vector-directions pass -> {out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
