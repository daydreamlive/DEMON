"""Cheap screening of many candidate knobs (phase 5): ONE process per GPU.

The SA3 model loads once; the process loops over its share of the candidates
(``--gpu-index i --gpu-count n``: rows i, i+n, ... of the candidate list;
by default only the cluster keepers of dedupe.py's ``clusters.csv``).
Per candidate: the step-mean unit direction at its best block (from
``dirs/<name>.npz``, what make_packs.py ships), applied at
``post_block_residual`` with ``sa3_tada.steer_offline(hook=...)`` at every
step, renders the first ``--n-prompts`` (12) holdout prompts of the
candidate's ``screen_set`` (music: the TADA holdout set,
``acestep/tada/data/benchmark_prompts.json``; sfx: ``--sfx-prompts``,
default ``$CAP/holdout_sfx_prompts.json``) at alpha = K x std,
K in ``--k`` (2 4 8), std = the projection std of the corpus along the
direction at that block (step mean). Alpha 0 is the same render for every
candidate of a prompt set (same seed, same batch): once per process and set.

Audio goes to a ring of ``--ring`` (8) shard files per process,
``screen/ring/g<i>_s<k>.npz`` (one alpha x 12 clips each), overwritten in
place; nothing is ever deleted. A slot is reused only after the scorer has
returned the candidate that used it.

Scoring (``--scorer``): ``inproc`` when laion_clap imports in this env,
else ``subprocess``: a long-lived worker (this file, ``--serve``) run with
``--eval-python`` (the evalenv), JSON lines over stdin/stdout, models loaded
once; the generator renders the next candidate while the worker scores.
Scores: LPAPS vs alpha 0 (reference ``editing.eval.get_lpaps``: LAION-CLAP
music checkpoint, 10 s windows, as the TADA protocol), the candidate's own
label column and second scorer columns through code-A's label_corpus.py
(``score(scorer_name, wav_int16, sr) -> dict`` if it defines one, else its
``make_scorer(args).score(wav, sr)``, one object per scorer kept loaded;
columns ``<scorer>.<label>``; ``--label-template`` must match the template
the corpus was labelled with), and MuQ + CLAP cosine to the concept's text anchors
when it has any (catalogue column ``pos_anchor`` / ``anchor`` /
``anchors``, ``;``-separated, or ``anchors.json`` keyed by name).

Acceptance (many_knobs_master_plan.md), on the positive side:
* in-band alphas = those with mean LPAPS <= the cutoff (``--lpaps-cut``, or a
  catalogue column ``lpaps_cut``); at least ``--min-band`` (2) of them;
* own column: sign-adjusted paired delta vs alpha 0 > 0 at every in-band
  alpha, non-decreasing with alpha, and |mean / se| >= ``--min-z`` at the
  largest in-band alpha;
* an independent second scorer of a DIFFERENT family agrees (right-signed
  and monotone in band): the first of index.csv ``second_cols`` (catalogue
  second then third scorer, those present in the labels) of another family,
  expected sign = sign of its corpus correlation (``second_corrs``); else
  the MuQ anchor;
* best-block |cosine| < ``--cos-max`` (0.8) to every already-accepted knob.
``pass_signal`` = the first three; ``pass`` adds the cosine gate against what
summary.csv holds at that moment (other GPUs append too, so it is order
dependent); ``--finalize`` recomputes it deterministically (greedy by slope
per unit LPAPS) into ``screen/accepted.csv``. Concepts whose own column is a
CLAP/MuQ text anchor get the flag ``needs_ear`` (human spot check).

Outputs: ``screen/<name>.json`` and one line per candidate appended to
``screen/summary.csv``.

``--dry-run``: fake generator and fake scorer (no model, no GPU) through the
real loop, ring, worker protocol and outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SR = 44100
DURATION = 10.0
STEPS = 8
HOOK = "post_block_residual"
ANCHOR_COLS = ("pos_anchor", "anchor", "anchors", "text_anchor", "anchor_text")
FAMILY = {
    "descriptors": "descriptor", "descriptor": "descriptor", "desc": "descriptor", "timbral": "descriptor",
    "loudness": "descriptor", "tempo": "descriptor", "key": "descriptor", "onsets": "descriptor",
    "rhythm": "descriptor", "essentia": "descriptor",
    "passt": "classifier", "panns": "classifier", "audioset": "classifier",
    "clap_music": "clap", "clap_general": "clap", "clap": "clap",
    "muq": "muq", "musetimbre": "embedding",
}
SUMMARY_COLS = ("name", "label_col", "best_block", "std", "alphas", "lpaps", "own_delta", "own_z",
                "second", "second_delta", "n_band", "slope_per_lpaps", "alpha_at_cut", "cut_reached",
                "lpaps_cut", "pass_signal", "max_cos_accepted", "pass", "flags", "gpu", "seconds")


def scorer_of(col: str) -> str:
    return col.split(".", 1)[0] if "." in col else col


def family_of(col: str) -> str:
    s = scorer_of(col)
    return FAMILY.get(s, s)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _save_inplace(path: Path, **arrays) -> None:
    """Overwrite ``path`` in place (same file, truncated; never unlinked)."""
    with open(path, "r+b" if path.exists() else "wb") as f:
        f.truncate(0)
        np.savez(f, **arrays)


# ---------------------------------------------------------------------------
# scoring backends (run in the worker, or in-process)
# ---------------------------------------------------------------------------

def _per_clip(values, b: int) -> list:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if a.size == b:
        return a.tolist()
    if a.size == 1 and b == 1:
        return a.tolist()
    raise ValueError(f"scorer returned {a.size} values for {b} clips")


class RealBackend:
    """Reference LPAPS / MuQ / CLAP (steer-audio, evalenv) + code-A label scorers."""

    def __init__(self, labels_module: str | None, opts: dict | None = None):
        lm = Path(labels_module).resolve() if labels_module else HERE / "label_corpus.py"
        self.opts = dict(opts or {})
        sys.path.insert(0, str(REPO / "scripts" / "tada"))
        import sa3_tada_score as S  # noqa: F401  (sets up the reference paths, chdir to REF)

        S._cache_models()
        import torch

        # the reference protocol module patches torch.load (weights_only=False) for the
        # CLAP checkpoints at import; the TADA scorer gets it the same way
        import src.steering.eval.eval_steering_protocol  # noqa: F401

        import editing.eval as ev

        self.ev, self.torch = ev, torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_score, self.lc, self._scorers = None, None, {}
        if lm.exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location("label_corpus", lm)
            mod = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(lm.parent))
            sys.modules['label_corpus'] = mod      # its process-pool workers pickle by module name
            spec.loader.exec_module(mod)
            self.lc = mod
            if callable(getattr(mod, "score", None)):
                self.label_score = mod.score
            elif callable(getattr(mod, "make_scorer", None)):
                self.label_score = self._make_scorer_adapter
        if self.label_score is None:
            print(f"WARN: {lm} has neither score(scorer_name, wav_int16, sr) -> dict nor make_scorer(args); "
                  "own/second columns will be NaN (every candidate fails)", file=sys.stderr, flush=True)

    def _make_scorer_adapter(self, scorer: str, wav, sr) -> dict:
        """code-A's label_corpus.py: one scorer object per name (``make_scorer``
        with its CLI defaults; anchors / template / workers as the corpus was
        labelled), ``.score(wav int16 [B, T], sr) -> {label: [B]}``."""
        import argparse as _ap

        if scorer not in self._scorers:
            dev = self.opts.get("label_device") or ("cuda:0" if self.device == "cuda" else "cpu")
            ns = _ap.Namespace(
                scorer=scorer, device=dev, workers=int(self.opts.get("label_workers") or 8),
                anchors=self.opts.get("anchors") or str(getattr(self.lc, "DEFAULT_ANCHORS", HERE / "anchors.json")),
                template=self.opts.get("label_template") or "{}", ckpt=None,
                labels_csv=getattr(self.lc, "AUDIOSET_LABELS", None))
            self._scorers[scorer] = self.lc.make_scorer(ns)
        return self._scorers[scorer].score(np.asarray(wav), sr)

    def _tensors(self, wav):
        return [self.torch.from_numpy(w.astype(np.float32)[None] / 32768.0) for w in wav]

    def lpaps(self, ref, wav, sr) -> list:
        df = self.ev.get_lpaps(source_audios=self._tensors(ref), edits=self._tensors(wav),
                               srs_src=[sr] * len(ref), srs_edit=[sr] * len(wav), device=self.device)
        return df["lpaps"].astype(float).tolist()

    def cols(self, cols, wav, sr) -> dict:
        out = {}
        for scorer in sorted({scorer_of(c) for c in cols}):
            want = [c for c in cols if scorer_of(c) == scorer]
            if self.label_score is None:
                res = {}
            else:
                res = self.label_score(scorer, wav, sr)
                res = {(k if "." in k else f"{scorer}.{k}"): v for k, v in (res or {}).items()}
                if res and np.asarray(next(iter(res.values()))).size != len(wav):
                    per = [self.label_score(scorer, w, sr) for w in wav]
                    res = {(k if "." in k else f"{scorer}.{k}"): [float(np.asarray(p[k]).reshape(-1)[0])
                                                                  for p in per] for k in per[0]}
            for c in want:
                out[c] = _per_clip(res[c], len(wav)) if c in res else [float("nan")] * len(wav)
        return out

    def anchors(self, kind, texts, wav, sr) -> dict:
        out = {}
        for t in texts:
            if kind == "muq":
                df = self.ev.get_mulan([f"This is a music of {t}"] * len(wav), self._tensors(wav),
                                       [sr] * len(wav), self.device, verbose=False)
                out[t] = df["muqt_sim_p0"].astype(float).tolist()
            else:
                df = self.ev.get_clap([t] * len(wav), self._tensors(wav), [sr] * len(wav), self.device)
                out[t] = df["clap"].astype(float).tolist()
        return out


class FakeBackend:
    """Dry run: LPAPS = scaled rms of the difference; any column = the
    amplitude of the candidate's fake signature tone (or of its sign-flipped
    noise for names containing ``noise``); anchors the same, halved."""

    def lpaps(self, ref, wav, sr) -> list:
        d = (wav.astype(np.float64) - ref.astype(np.float64)) / 32768.0
        return (np.sqrt((d ** 2).mean(1)) * 25.0).tolist()

    @staticmethod
    def _amp(wav, cand, sr):
        f = FakeGenerator.freq(cand)
        t = np.arange(wav.shape[1]) / sr
        return ((wav.astype(np.float64) / 32768.0) @ np.sin(2 * np.pi * f * t) * 2 / wav.shape[1]).tolist()

    def cols(self, cols, wav, sr, cand="", signs=None) -> dict:
        a = np.array(self._amp(wav, cand, sr))
        return {c: (a * (signs or {}).get(c, 1)).tolist() for c in cols}

    def anchors(self, kind, texts, wav, sr, cand="", signs=None) -> dict:
        a = [x * 0.5 for x in self._amp(wav, cand, sr)]
        return {t: a for t in texts}


def score_request(backend, req: dict, ref_cache: dict) -> dict:
    """Score one candidate: ``req`` names the ref file, the shard files
    (one per alpha) and what to score. Ref (alpha 0) scores are cached per
    ref file."""
    fake = isinstance(backend, FakeBackend)
    kw = {"cand": req["cand"], "signs": req.get("signs", {})} if fake else {}
    ref = np.load(req["ref"])
    rwav, sr = ref["wav"], int(ref["sr"])
    cols, anchors = list(req["cols"]), list(req["anchors"])
    key = req["ref"]
    rc = ref_cache.setdefault(key, {"cols": {}, "muq": {}, "clap": {}})
    miss = [c for c in cols if c not in rc["cols"]]
    if miss and not fake:
        rc["cols"].update(backend.cols(miss, rwav, sr))
    ref_cols = backend.cols(cols, rwav, sr, **kw) if fake else {c: rc["cols"][c] for c in cols}
    ref_anchor = {}
    for kind in ("muq", "clap"):
        if fake:
            ref_anchor[kind] = backend.anchors(kind, anchors, rwav, sr, **kw)
        else:
            m = [t for t in anchors if t not in rc[kind]]
            if m:
                rc[kind].update(backend.anchors(kind, m, rwav, sr))
            ref_anchor[kind] = {t: rc[kind][t] for t in anchors}
    out = {"cand": req["cand"], "ref": {"cols": ref_cols, **ref_anchor}, "alphas": []}
    for shard, alpha in zip(req["shards"], req["alphas"]):
        z = np.load(shard)
        wav = z["wav"]
        if str(z["cand"]) != req["cand"] or abs(float(z["alpha"]) - alpha) > 1e-9:
            raise RuntimeError(f"{shard} holds {z['cand']} @ {float(z['alpha'])}, expected {req['cand']} @ {alpha}")
        out["alphas"].append({
            "alpha": alpha, "lpaps": backend.lpaps(rwav, wav, sr),
            "cols": backend.cols(cols, wav, sr, **kw),
            "muq": backend.anchors("muq", anchors, wav, sr, **kw),
            "clap": backend.anchors("clap", anchors, wav, sr, **kw)})
    return out


def serve(args) -> int:
    """Worker: JSON lines in on stdin, JSON lines out on the real stdout;
    every print of the libraries goes to stderr."""
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    backend = FakeBackend() if args.fake_scorer else RealBackend(args.labels_module, _label_opts(args))
    out.write(json.dumps({"ready": True}) + "\n")
    cache = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("quit"):
            break
        try:
            res = score_request(backend, req, cache)
        except Exception as e:  # report and keep serving
            import traceback

            traceback.print_exc()
            res = {"cand": req.get("cand"), "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(res) + "\n")
    return 0


def _label_opts(args) -> dict:
    return {"label_workers": args.label_workers, "label_template": args.label_template,
            "label_device": args.label_device,
            "anchors": str(Path(args.anchors).resolve()) if args.anchors else None}


class Scorer:
    """``submit(req)`` / ``get()``; subprocess worker or in-process."""

    def __init__(self, mode: str, python: str | None, fake: bool, labels_module: str | None, opts: dict):
        self.mode = mode
        self.q: queue.Queue = queue.Queue()
        if mode == "inproc":
            self.backend = FakeBackend() if fake else RealBackend(labels_module, opts)
            self.cache = {}
            return
        cmd = [python or sys.executable, str(Path(__file__).resolve()), "--serve"]
        if fake:
            cmd.append("--fake-scorer")
        if labels_module:
            cmd += ["--labels-module", str(Path(labels_module).resolve())]
        for k, flag in (("label_workers", "--label-workers"), ("label_template", "--label-template"),
                        ("label_device", "--label-device"), ("anchors", "--anchors")):
            if opts.get(k) is not None:
                cmd += [flag, str(opts[k])]
        env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=os.pathsep.join(
            [str(REPO)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                     bufsize=1, env=env)
        first = self.proc.stdout.readline()
        if not first or not json.loads(first).get("ready"):
            raise SystemExit(f"scorer worker failed to start: {' '.join(cmd)}")
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            if line.strip():
                self.q.put(json.loads(line))
        self.q.put({"eof": True})

    def submit(self, req: dict) -> None:
        if self.mode == "inproc":
            try:
                self.q.put(score_request(self.backend, req, self.cache))
            except Exception as e:
                self.q.put({"cand": req["cand"], "error": f"{type(e).__name__}: {e}"})
            return
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()

    def get(self) -> dict:
        r = self.q.get()
        if r.get("eof"):
            raise SystemExit("scorer worker exited")
        return r

    def close(self):
        if self.mode != "inproc":
            self.proc.stdin.write(json.dumps({"quit": True}) + "\n")
            self.proc.stdin.close()
            self.proc.wait(timeout=60)


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------

class SA3Generator:
    """SA3 medium (eager, as the TADA bench): ``sa3_tada_run._load_sam``,
    ``sa3_tada.generate`` (10 s, 8 steps, cfg 1, seed EVAL_SEED, one batch),
    steered with ``steer_offline(hook="post_block_residual")``."""

    def __init__(self):
        sys.path.insert(0, str(REPO / "scripts" / "tada"))
        import torch

        import sa3_tada_run as R
        from acestep.engine import sa3_tada

        torch.backends.cuda.matmul.allow_tf32 = True
        self.R, self.E, self.torch = R, sa3_tada, torch
        _log("loading SA3 medium")
        self.sam = R._load_sam()

    def render(self, prompts, unit, block, alpha) -> np.ndarray:
        v = self.torch.from_numpy(np.asarray(unit, dtype=np.float32))
        vectors = {s: {int(block): v} for s in range(STEPS)}
        with self.E.steer_offline(self.sam, vectors, float(alpha), hook=HOOK):
            audio = self.E.generate(self.sam, prompts, seed=self.R.EVAL_SEED, duration=DURATION, steps=STEPS)
        mono = audio.mean(dim=1)
        return (mono * 32768.0).round().clamp(-32768, 32767).to(self.torch.int16).numpy()


class FakeGenerator:
    """Dry run: per-prompt noise + alpha x the candidate's signature tone
    (sign flips per prompt for names containing ``noise``); 1 s clips."""

    sr_len = SR

    @staticmethod
    def freq(cand: str) -> float:
        return 200.0 + int(hashlib.md5(cand.encode()).hexdigest()[:4], 16) % 2000

    def __init__(self):
        self.cand = ""

    def render(self, prompts, unit, block, alpha) -> np.ndarray:
        t = np.arange(self.sr_len) / SR
        out = []
        for i, p in enumerate(prompts):
            r = np.random.default_rng(i)
            base = 0.1 * r.standard_normal(self.sr_len)
            s = (-1) ** i if "noise" in self.cand else 1.0
            out.append(base + s * 0.004 * alpha * np.sin(2 * np.pi * self.freq(self.cand) * t))
        return (np.clip(np.stack(out), -1, 1) * 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# candidates, analysis
# ---------------------------------------------------------------------------

def load_candidates(args) -> list:
    with open(args.candidates, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status", "ok") == "ok"]
    clusters = Path(args.candidates).parent / "clusters.csv"
    if not args.all_candidates and clusters.exists():
        with open(clusters, newline="", encoding="utf-8") as f:
            keep = {r["name"] for r in csv.DictReader(f) if r["is_keeper"] == "1"}
        rows = [r for r in rows if r["name"] in keep]
    if args.only:
        rows = [r for r in rows if r["name"] in set(args.only)]
    rows = rows[args.gpu_index::args.gpu_count]
    return rows[: args.limit] if args.limit else rows


def anchors_for(row: dict, table: dict) -> list:
    for c in ANCHOR_COLS:
        if row.get(c):
            return [t.strip() for t in row[c].replace("|", ";").split(";") if t.strip()]
    v = table.get(row["name"])
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x if isinstance(x, str) else x.get("text", "") for x in v][:2]
    if isinstance(v, dict) and v.get("text"):
        return [v["text"]]
    return []


def load_anchor_table(path) -> dict:
    p = Path(path) if path else HERE / "anchors.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(d, list):
        return {x.get("name"): x.get("text") or x.get("anchor") for x in d if isinstance(x, dict)}
    return d if isinstance(d, dict) else {}


def direction(npz_path: str, per_step: bool = False):
    """``(unit [H] step-mean at the best block, block, std at that block)``."""
    z = np.load(npz_path)
    b = int(z["best_block"])
    k = list(z["blocks"]).index(b)
    sm = (z["unit"][:, k].astype(np.float64) * z["norm"][:, k][:, None]).mean(0)
    unit = sm / max(np.linalg.norm(sm), 1e-12)
    return unit.astype(np.float32), b, float(z["std"][:, b].mean()), float(np.linalg.norm(sm))


def _stats(ref, alt, sign):
    d = sign * (np.asarray(alt, dtype=np.float64) - np.asarray(ref, dtype=np.float64))
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan"), float("nan")
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))


def _monotone_pos(vals) -> bool:
    v = np.asarray(vals, dtype=np.float64)
    return bool(v.size and np.all(np.isfinite(v)) and np.all(v > 0) and np.all(np.diff(v) >= 0))


def analyse(row: dict, meta: dict, res: dict, args) -> dict:
    sign = int(meta["sign"])
    own = row["label_col"]
    lab_std = float(row.get("label_std") or 1.0) or 1.0
    cut = float(row.get("lpaps_cut") or args.lpaps_cut)
    per_alpha = []
    for a in res["alphas"]:
        e = {"alpha": a["alpha"], "k": a["alpha"] / meta["std"] if meta["std"] else None,
             "lpaps_mean": float(np.mean(a["lpaps"])), "lpaps_std": float(np.std(a["lpaps"])),
             "own_scores": {}, "muq": {}, "clap": {}}
        for c, vals in a["cols"].items():
            sgn = sign if c == own else meta["second_sign"].get(c, 1)
            m, se = _stats(res["ref"]["cols"][c], vals, sgn)
            e["own_scores"][c] = {"delta": m, "se": se, "mean": float(np.nanmean(vals)), "sign": sgn}
        for kind in ("muq", "clap"):
            for t, vals in a[kind].items():
                m, se = _stats(res["ref"][kind][t], vals, 1)
                e[kind][t] = {"delta": m, "se": se}
        per_alpha.append(e)
    pos = sorted([e for e in per_alpha if e["alpha"] > 0], key=lambda e: e["alpha"])
    band = [e for e in pos if e["lpaps_mean"] <= cut]
    flags = [f"second_weak_corr:{c}" for c in meta.get("weak_second", [])]
    own_d = [e["own_scores"][own]["delta"] for e in band]
    own_ok = len(band) >= args.min_band and _monotone_pos(own_d)
    z = (band[-1]["own_scores"][own]["delta"] / band[-1]["own_scores"][own]["se"]
         if band and band[-1]["own_scores"][own]["se"] and np.isfinite(band[-1]["own_scores"][own]["se"])
         else float("nan"))
    z_ok = args.min_z <= 0 or (np.isfinite(z) and z >= args.min_z)
    if len(band) < args.min_band:
        flags.append("too_few_in_band")
    # independent second scorer: first second col of another family, else the MuQ anchor
    second, sec_d = None, []
    for c in meta["second_cols"]:
        if family_of(c) != family_of(own):
            second, sec_d = c, [e["own_scores"][c]["delta"] for e in band]
            break
        flags.append(f"second_same_family:{c}")
    if second is None and meta["anchors"] and family_of(own) != "muq":
        t = meta["anchors"][0]
        second, sec_d = f"muq:{t}", [e["muq"][t]["delta"] for e in band]
    if second is None:
        flags.append("no_second_scorer")
    sec_ok = second is not None and len(band) >= args.min_band and _monotone_pos(sec_d)
    if family_of(own) in ("clap", "muq"):
        flags.append("needs_ear")
    lp = np.array([e["lpaps_mean"] for e in band])
    dn = np.array([e["own_scores"][own]["delta"] for e in band]) / lab_std
    slope = float((dn * lp).sum() / (lp * lp).sum()) if len(band) and (lp * lp).sum() > 0 else float("nan")
    # alpha where LPAPS crosses the cutoff (piecewise linear from (0, 0))
    xs = [0.0] + [e["alpha"] for e in pos]
    ys = [0.0] + [e["lpaps_mean"] for e in pos]
    a_cut, reached = xs[-1], False
    for i in range(1, len(xs)):
        if ys[i] >= cut:
            f = (cut - ys[i - 1]) / max(ys[i] - ys[i - 1], 1e-12)
            a_cut, reached = xs[i - 1] + f * (xs[i] - xs[i - 1]), True
            break
    neg = sorted([e for e in per_alpha if e["alpha"] < 0], key=lambda e: -e["alpha"])
    a_cut_neg, reached_neg = None, None
    if neg:
        xs = [0.0] + [-e["alpha"] for e in neg]
        ys = [0.0] + [e["lpaps_mean"] for e in neg]
        a_cut_neg, reached_neg = xs[-1], False
        for i in range(1, len(xs)):
            if ys[i] >= cut:
                f = (cut - ys[i - 1]) / max(ys[i] - ys[i - 1], 1e-12)
                a_cut_neg, reached_neg = xs[i - 1] + f * (xs[i] - xs[i - 1]), True
                break
    return {
        "per_alpha": per_alpha, "lpaps_cut": cut, "in_band": [e["alpha"] for e in band],
        "own": {"col": own, "deltas": own_d, "z_at_max_band": z, "ok": bool(own_ok and z_ok)},
        "second": {"col": second, "deltas": sec_d, "ok": bool(sec_ok)},
        "slope_per_lpaps": slope, "alpha_at_cut": {"pos": a_cut, "neg": a_cut_neg},
        "cut_reached": {"pos": reached, "neg": reached_neg},
        "pass_signal": bool(own_ok and z_ok and sec_ok), "flags": flags,
    }


def accepted_units(screen: Path, dirs_by_name: dict) -> dict:
    """``{name: (unit, block)}`` of rows with pass = 1 in summary.csv."""
    out = {}
    s = screen / "summary.csv"
    if not s.exists():
        return out
    with open(s, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("pass") == "1" and r["name"] in dirs_by_name:
                u, b, _, _ = direction(dirs_by_name[r["name"]])
                out[r["name"]] = (u, b, dirs_by_name[r["name"]])
    return out


def cos_between(path_a: str, path_b: str) -> float:
    """|cos| of two concepts' step-mean units, at block(a) and block(b) where
    both npz store that block; max of the two (NaN if no common block)."""
    za, zb = np.load(path_a), np.load(path_b)
    best = []
    for b in {int(za["best_block"]), int(zb["best_block"])}:
        vs = []
        for z in (za, zb):
            bl = list(z["blocks"])
            if b not in bl:
                break
            k = bl.index(b)
            sm = (z["unit"][:, k].astype(np.float64) * z["norm"][:, k][:, None]).mean(0)
            vs.append(sm / max(np.linalg.norm(sm), 1e-12))
        if len(vs) == 2:
            best.append(abs(float(vs[0] @ vs[1])))
    return max(best) if best else float("nan")


class CosLookup:
    """|cos| between two concepts: dedupe.py's ``cosine_matrix.npy`` (each
    pair compared at both best blocks, recomputed from the capture) when it
    covers both names, else :func:`cos_between` on the stored npz blocks."""

    def __init__(self, dirs_dir: Path):
        self.idx, self.m = {}, None
        f, n = dirs_dir / "cosine_matrix.npy", dirs_dir / "cosine_names.json"
        if f.exists() and n.exists():
            self.m = np.load(f)
            self.idx = {x: i for i, x in enumerate(json.loads(n.read_text()))}

    def __call__(self, a: str, pa: str, b: str, pb: str) -> float:
        if self.m is not None and a in self.idx and b in self.idx:
            return abs(float(self.m[self.idx[a], self.idx[b]]))
        return cos_between(pa, pb)


def append_summary(path: Path, row: dict) -> None:
    try:
        with open(path, "x", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(SUMMARY_COLS)
    except FileExistsError:
        pass
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(c, "") for c in SUMMARY_COLS])


def finalize(args) -> int:
    screen = Path(args.out)
    rows = {}
    with open(screen / "summary.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["name"]] = r                                         # last line wins (reruns)
    with open(args.candidates, newline="", encoding="utf-8") as f:
        paths = {r["name"]: r["path"] for r in csv.DictReader(f)}
    sig = [r for r in rows.values() if r["pass_signal"] == "1" and r["name"] in paths]
    sig.sort(key=lambda r: -float(r["slope_per_lpaps"] or 0))
    cos = CosLookup(Path(args.candidates).parent)
    acc = []
    with open(screen / "accepted.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "accepted", "slope_per_lpaps", "max_cos", "nearest", "flags"])
        for r in sig:
            cs = [(cos(r["name"], paths[r["name"]], a, paths[a]), a) for a in acc]
            cs = [x for x in cs if np.isfinite(x[0])]
            mx, near = max(cs) if cs else (0.0, "")
            ok = mx < args.cos_max
            if ok:
                acc.append(r["name"])
            w.writerow([r["name"], int(ok), r["slope_per_lpaps"], f"{mx:.3f}", near, r["flags"]])
    print(f"{len(sig)} pass the signal rule; {len(acc)} accepted after the cosine gate -> "
          f"{screen / 'accepted.csv'}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", help="corpus root; outputs go to $CAP/screen")
    ap.add_argument("--candidates", help="default $CAP/dirs/index.csv (keepers of clusters.csv if present)")
    ap.add_argument("--out", help="default $CAP/screen")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--k", type=float, nargs="+", default=[2.0, 4.0, 8.0], help="alpha = K x projection std")
    ap.add_argument("--signs", choices=("pos", "both"), default="pos",
                    help="both: also -K (per-sign calibrated gain for make_packs)")
    ap.add_argument("--n-prompts", type=int, default=12, help="first N TADA holdout prompts")
    ap.add_argument("--ring", type=int, default=8, help="audio shard files per process")
    ap.add_argument("--sfx-prompts", help="screen_set sfx prompts (default $CAP/holdout_sfx_prompts.json)")
    ap.add_argument("--lpaps-cut", type=float, default=4.0,
                    help="LPAPS band (SA3 PCI-all cutoffs of the five knobs: 4.0 to 4.8)")
    ap.add_argument("--min-band", type=int, default=2)
    ap.add_argument("--min-z", type=float, default=2.0, help="own delta / se at the largest in-band alpha")
    ap.add_argument("--cos-max", type=float, default=0.8)
    ap.add_argument("--min-second-corr", type=float, default=0.1,
                    help="skip second scorer columns whose corpus |corr| with the label is below this "
                         "(their expected sign is unknown)")
    ap.add_argument("--scorer", choices=("auto", "inproc", "subprocess"), default="auto")
    ap.add_argument("--eval-python", help="evalenv python for the scorer worker")
    ap.add_argument("--labels-module", help="default scripts/steering_bench/label_corpus.py")
    ap.add_argument("--label-template", default="{}",
                    help="anchor text template, as label_corpus.py --template was run for the corpus")
    ap.add_argument("--label-workers", type=int, default=8, help="label_corpus CPU scorer workers")
    ap.add_argument("--label-device", default=None, help="label_corpus GPU scorers (default cuda:0)")
    ap.add_argument("--anchors", help="default scripts/steering_bench/anchors.json")
    ap.add_argument("--all-candidates", action="store_true", help="ignore clusters.csv")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-screen candidates with a json")
    ap.add_argument("--dry-run", action="store_true", help="fake generator + fake scorer")
    ap.add_argument("--finalize", action="store_true", help="greedy cosine gate -> accepted.csv")
    ap.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--fake-scorer", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.serve:
        return serve(args)
    if not (args.cap or (args.candidates and args.out)):
        ap.error("--cap (or --candidates and --out)")
    cap = Path(args.cap) if args.cap else None
    args.candidates = args.candidates or str(cap / "dirs" / "index.csv")
    args.out = args.out or str(cap / "screen")
    if args.finalize:
        return finalize(args)

    screen = Path(args.out)
    ring_dir = screen / "ring"
    ring_dir.mkdir(parents=True, exist_ok=True)
    cands = load_candidates(args)
    with open(args.candidates, newline="", encoding="utf-8") as f:
        dirs_by_name = {r["name"]: r["path"] for r in csv.DictReader(f) if r.get("path")}
    todo = [r for r in cands if args.force or not (screen / f"{r['name']}.json").exists()]
    _log(f"gpu {args.gpu_index}/{args.gpu_count}: {len(cands)} candidates, {len(todo)} to screen")
    if not todo:
        return 0

    mode = args.scorer
    if mode == "auto":
        if args.dry_run:
            mode = "subprocess"
        else:
            try:
                import laion_clap  # noqa: F401
                mode = "inproc"
            except ImportError:
                mode = "subprocess"
    if mode == "subprocess" and not args.dry_run and not args.eval_python:
        ap.error("laion_clap does not import here: pass --eval-python <evalenv python>")
    scorer = Scorer(mode, args.eval_python if not args.dry_run else (args.eval_python or sys.executable),
                    args.dry_run, args.labels_module, _label_opts(args))
    gen = FakeGenerator() if args.dry_run else SA3Generator()
    sets = {}

    def prompt_set(row) -> list:
        """Holdout prompts of the candidate's ``screen_set``: music = TADA
        holdout (benchmark_prompts.json), sfx = ``--sfx-prompts``."""
        name = (row.get("screen_set") or "music").strip().lower()
        if name not in sets:
            if args.dry_run:
                sets[name] = [f"dry {name} prompt {i}" for i in range(args.n_prompts)]
            elif name == "sfx":
                sp = Path(args.sfx_prompts) if args.sfx_prompts else Path(args.out).parent / "holdout_sfx_prompts.json"
                d = json.loads(sp.read_text(encoding="utf-8"))
                sets[name] = [x["prompt"] if isinstance(x, dict) else str(x) for x in d][: args.n_prompts]
            else:
                from acestep.tada import concepts as C

                sets[name] = C.benchmark_prompts(holdout=True)[0][: args.n_prompts]
        return sets[name]

    refs = {}

    def render_ref(set_name: str, prompts: list) -> Path:
        """Alpha 0 of a prompt set: rendered once per process (same seed and
        batch as every steered render, so it is the unsteered twin)."""
        path = ring_dir / f"g{args.gpu_index}_ref_{set_name}.npz"
        gen.cand = ""
        _save_inplace(path, wav=gen.render(prompts, np.zeros(1536, np.float32), 0, 0.0), sr=np.int64(SR),
                      alpha=np.float64(0.0), cand=np.array(""))
        refs[set_name] = path
        _log(f"alpha 0 reference ({set_name}, {len(prompts)} prompts): {path}")
        return path

    anchor_table = load_anchor_table(args.anchors)

    ring = [ring_dir / f"g{args.gpu_index}_s{k}.npz" for k in range(args.ring)]
    busy = {}                      # slot -> candidate name
    inflight = {}                  # name -> (row, meta, t0, slots)
    accepted = accepted_units(screen, dirs_by_name)
    cos = CosLookup(Path(args.candidates).parent)
    next_slot = 0

    def handle(res: dict) -> None:
        name = res.get("cand")
        row, meta, t0, slots = inflight.pop(name)
        for s in slots:
            busy.pop(s, None)
        if "error" in res:
            _log(f"{name}: scorer error {res['error']}")
            append_summary(screen / "summary.csv", {"name": name, "label_col": row["label_col"],
                                                    "flags": f"error:{res['error'][:80]}", "gpu": args.gpu_index})
            return
        an = analyse(row, meta, res, args)
        cs = [(cos(name, meta["path"], n, a[2]), n) for n, a in accepted.items() if n != name]
        cs = [x for x in cs if np.isfinite(x[0])]
        mx = max(cs) if cs else (0.0, "")
        ok = an["pass_signal"] and mx[0] < args.cos_max
        if ok:
            accepted[name] = (meta["unit"], meta["block"], meta["path"])
        secs = time.time() - t0
        doc = {"name": name, "label_col": row["label_col"], "sign": meta["sign"], "block": meta["block"],
               "std": meta["std"], "norm": meta["norm"], "hook": HOOK, "n_prompts": len(meta["prompts"]),
               "prompts": ("dry" if args.dry_run else
                           "sfx holdout" if (row.get("screen_set") or "") == "sfx" else "tada holdout"),
               "screen_set": row.get("screen_set") or "music", "k": args.k,
               "alphas": meta["alphas"], "anchors": meta["anchors"], "second_cols": meta["second_cols"],
               **an, "max_cos_accepted": {"cos": mx[0], "to": mx[1]}, "pass": bool(ok),
               "seconds": round(secs, 1), "gpu": args.gpu_index, "dry_run": args.dry_run,
               "date": time.strftime("%Y-%m-%d %H:%M:%S")}
        (screen / f"{name}.json").write_text(json.dumps(doc, indent=1, default=float))
        band = an["per_alpha"]
        append_summary(screen / "summary.csv", {
            "name": name, "label_col": row["label_col"], "best_block": meta["block"],
            "std": f"{meta['std']:.4f}", "alphas": " ".join(f"{e['alpha']:.3f}" for e in band),
            "lpaps": " ".join(f"{e['lpaps_mean']:.3f}" for e in band),
            "own_delta": " ".join(f"{e['own_scores'][row['label_col']]['delta']:.4g}" for e in band),
            "own_z": f"{an['own']['z_at_max_band']:.2f}", "second": an["second"]["col"] or "",
            "second_delta": " ".join(f"{x:.4g}" for x in an["second"]["deltas"]),
            "n_band": len(an["in_band"]), "slope_per_lpaps": f"{an['slope_per_lpaps']:.5g}",
            "alpha_at_cut": f"{an['alpha_at_cut']['pos']:.3f}", "cut_reached": int(an["cut_reached"]["pos"]),
            "lpaps_cut": an["lpaps_cut"], "pass_signal": int(an["pass_signal"]),
            "max_cos_accepted": f"{mx[0]:.3f}", "pass": int(ok), "flags": ";".join(an["flags"]),
            "gpu": args.gpu_index, "seconds": f"{secs:.1f}"})
        _log(f"{name}: lpaps {[round(e['lpaps_mean'], 2) for e in band]} own {an['own']['ok']} "
             f"second {an['second']['ok']} pass {ok} ({secs:.1f} s)")

    for i, row in enumerate(todo):
        name = row["name"]
        unit, block, std, norm = direction(row["path"])
        ks = list(args.k) + ([-k for k in args.k] if args.signs == "both" else [])
        alphas = [float(k * std) for k in ks]
        if row.get("second_cols") is not None and "second_corrs" in row:
            # build_directions: second/third scorer columns present in the labels + corpus correlation
            second_cols = [c for c in row["second_cols"].split(";") if c]
            corrs = [float(x) for x in row["second_corrs"].split(";") if x]
        else:
            second_cols = [c.strip() for c in (row.get("second_scorer_col") or "").replace("|", ";").split(";")
                           if c.strip()]
            corrs = [1.0] * len(second_cols)
        weak = [c for c, x in zip(second_cols, corrs) if abs(x) < args.min_second_corr]
        second_cols = [c for c in second_cols if c not in weak]
        second_sign = {c: (-1 if x < 0 else 1) for c, x in zip(second_cols, corrs) if c not in weak}
        prompts = prompt_set(row)
        ref_path = refs.get(row.get("screen_set") or "music")
        if ref_path is None:
            ref_path = render_ref(row.get("screen_set") or "music", prompts)
        meta = {"sign": int(float(row.get("sign") or 1)), "unit": unit, "block": block, "std": std, "norm": norm,
                "alphas": alphas, "anchors": anchors_for(row, anchor_table), "second_cols": second_cols,
                "second_sign": second_sign, "weak_second": weak, "path": row["path"]}
        if len(alphas) > len(ring):
            raise SystemExit(f"--ring {len(ring)} is smaller than {len(alphas)} alphas per candidate")
        slots = []
        for _a in alphas:
            while next_slot in busy:                   # wait for the scorer to free it
                handle(scorer.get())
            slots.append(next_slot)
            busy[next_slot] = name
            next_slot = (next_slot + 1) % len(ring)
        t0 = time.time()
        gen.cand = name
        shards = []
        for s, a in zip(slots, alphas):
            _save_inplace(ring[s], wav=gen.render(prompts, unit, block, a), sr=np.int64(SR),
                          alpha=np.float64(a), cand=np.array(name))
            shards.append(str(ring[s]))
        inflight[name] = (row, {**meta, "prompts": prompts}, t0, slots)
        scorer.submit({"cand": name, "ref": str(ref_path), "shards": shards, "alphas": alphas,
                       "cols": [row["label_col"]] + second_cols, "anchors": meta["anchors"],
                       "signs": {row["label_col"]: meta["sign"], **second_sign}})
    while inflight:
        handle(scorer.get())
    scorer.close()
    _log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
