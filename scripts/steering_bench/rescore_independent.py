"""Independent-scorer rescoring of the v3 smoke renders (Q2) and the inverted-direction
census of the shipped knobs (Q4). Lit review v3 2026-10-07 (notes/steering_pr).

Q2 asks whether the ridge descriptor directions win on the descriptor they were fit to
(metric gaming) or on scorers that did not take part in the fit:
  centroid   -> CLAP to the control anchors (bright / dark), hiss checks (top-band spectral
                flatness, HF stationary floor, centroid after removing the stationary floor),
                CLAP noise anchors
  onset_rate -> librosa beat_track tempo, superflux onset rate (log-mel, lag 2, max filter 3),
                CLAP percussive and fast-tempo anchors
  lra        -> ffmpeg ebur128 LRA (second EBU R128 implementation; the target is pyloudnorm),
                CLAP dense / dynamic / uncompressed anchors

Subcommands (all write under --out):
  q2-cpu   per-clip CPU metrics          -> q2_clips.csv          (evalenv: librosa, pyloudnorm, ffmpeg)
  q2-clap  per-clip CLAP anchor cosines   -> q2_clap.csv           (evalenv, GPU, TADA_ROOT set)
  q2-sum   summary tables                 -> q2.json, q2.md
  q4       inverted-direction census      -> q4.json, q4.md        (CPU; E2 env = local many_knobs_v2)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

V3 = Path("E:/Projects/DEMON/steering-bench/v3_smoke")
DESC = {"onset_rate": "percussive", "centroid": "bright", "lra": "dense_arrangement"}
CLAP_PAIRS = {  # concept name in anchors.json "concepts" -> used for which descriptor
    "bright": "centroid", "static_noise": "centroid", "noise_floor": "centroid", "white_noise": "centroid",
    "percussive": "onset_rate", "fast_tempo": "onset_rate",
    "dense_arrangement": "lra", "dynamic": "lra", "uncompressed": "lra",
}


# ------------------------------------------------------------------ render index

def index(renders: Path) -> list[dict]:
    """One row per (vector, sign, alpha) npz plus the alpha-0 reference."""
    rows = [{"vec": "ref", "sign": 0, "alpha": 0.0, "npz": renders / "ref" / "alpha_0.0" / "audios.npz"}]
    for vd in sorted(p for p in renders.iterdir() if p.is_dir() and p.name != "ref"):
        ev = vd / "eval" / f"pack_pack_{vd.name}"
        for ad in ev.glob("alpha_*"):
            a = float(ad.name[len("alpha_"):])
            if (ad / "audios.npz").exists():
                rows.append({"vec": vd.name, "sign": 1 if a > 0 else -1, "alpha": a, "npz": ad / "audios.npz"})
    return rows


def load(npz: Path):
    z = np.load(npz)
    return z["audio"].reshape(len(z["audio"]), -1), int(z["sr"]), [str(n) for n in z["names"]]


# ------------------------------------------------------------------ CPU metrics

def _ffmpeg_lra(clip: np.ndarray, sr: int) -> float:
    try:
        p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-f", "s16le", "-ar", str(sr), "-ac", "1",
                            "-i", "pipe:0", "-af", "ebur128", "-f", "null", "-"],
                           input=clip.astype("<i2").tobytes(), capture_output=True, timeout=120)
        m = re.findall(r"LRA:\s+(-?[\d.]+)\s+LU", p.stderr.decode("utf-8", "replace"))
        return float(m[-1]) if m else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def clip_metrics(job) -> dict:
    import librosa
    import proxies
    import label_corpus as lc

    clip, sr = job
    y = clip.astype(np.float32) / 32768.0
    out = {"centroid": proxies.centroid(y, sr), "onset_rate": proxies.onset_rate(y, sr)}
    # target LRA definition (label_corpus dyn.lra, pyloudnorm short-term 3 s / 0.5 s hop)
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
    except Exception:  # noqa: BLE001
        meter = None
    out["lra"] = lc._lra(lc._short_term(y, sr, meter)) if meter is not None else float("nan")
    out["lra_ffmpeg"] = _ffmpeg_lra(clip, sr)
    # hiss checks
    p = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = np.linspace(0, sr / 2, p.shape[0])
    e = p.sum(axis=0)
    act = e > 1e-4 * e.max() if e.max() > 0 else np.ones_like(e, bool)
    pa = p[:, act]
    hb = pa[f >= 8000.0] + 1e-12
    w = hb.sum(axis=0)
    fl = np.exp(np.log(hb).mean(axis=0)) / hb.mean(axis=0)
    out["hf_flatness"] = float((fl * w).sum() / max(w.sum(), 1e-20))      # 1 = white noise in 8 kHz+
    floor = np.percentile(pa, 10, axis=1)                                     # stationary per-bin floor
    tot = float(pa.mean(axis=1).sum())
    out["hf_floor_db"] = float(10 * np.log10(max(floor[f >= 8000.0].sum(), 1e-20) / max(tot, 1e-20)))
    out["floor_share_db"] = float(10 * np.log10(max(floor.sum(), 1e-20) / max(tot, 1e-20)))
    q = np.clip(pa - floor[:, None], 0, None)
    out["centroid_nofloor"] = float((f[:, None] * q).sum() / max(q.sum(), 1e-20))
    out["hf_share_db"] = float(10 * np.log10(max(pa[f >= 8000.0].sum(), 1e-20) / max(pa.sum(), 1e-20)))
    # tempo / onsets, different from proxies.onset_rate (librosa default flux, lag 1, no max filter)
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        out["tempo_bt"] = float(np.asarray(tempo).reshape(-1)[0])
        out["beat_rate"] = float(len(beats) / (len(y) / sr))
    except Exception:  # noqa: BLE001
        out["tempo_bt"] = out["beat_rate"] = float("nan")
    try:
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512,
                                                                 n_mels=138, fmin=27.5, fmax=16000.0))
        env = librosa.onset.onset_strength(S=mel, sr=sr, hop_length=512, lag=2, max_size=3)
        on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=512)
        out["onset_superflux"] = float(len(on) / (len(y) / sr))
    except Exception:  # noqa: BLE001
        out["onset_superflux"] = float("nan")
    return out


def cmd_q2_cpu(a) -> None:
    rows = index(a.renders)
    jobs, keys = [], []
    names0 = None
    for r in rows:
        wav, sr, names = load(r["npz"])
        names0 = names0 or names
        assert names == names0, f"prompt order differs in {r['npz']}"
        for i, c in enumerate(wav):
            jobs.append((c, sr))
            keys.append((r["vec"], r["sign"], r["alpha"], i))
    print(f"{len(jobs)} clips from {len(rows)} npz", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(clip_metrics, jobs, chunksize=4))
    cols = list(res[0].keys())
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "q2_clips.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["vec", "sign", "alpha", "clip"] + cols)
        for k, m in zip(keys, res):
            w.writerow(list(k) + [m[c] for c in cols])
    print("wrote", a.out / "q2_clips.csv")


# ------------------------------------------------------------------ CLAP (GPU)

def cmd_q2_clap(a) -> None:
    """Same CLAP model, checkpoint and 10 s / 0.1-overlap window-mean as editing.eval.get_clap (the smoke's
    CLAP column), but each clip's window embeddings are computed once and dotted with all anchor texts."""
    anchors = json.loads((HERE / "anchors.json").read_text(encoding="utf-8"))["concepts"]
    texts = sorted({t for c in CLAP_PAIRS for t in anchors[c][:2]})
    if a.tada_root:
        os.environ["TADA_ROOT"] = a.tada_root
    sys.path.insert(0, str(HERE.parents[1] / "scripts" / "tada"))
    import sa3_tada_score  # noqa: F401  (reference paths, chdir to steer-audio)
    import src.steering.eval.eval_steering_protocol  # noqa: F401  (torch.load patch for the CLAP ckpt)
    import torch
    import editing.eval as ev
    from editing.AudioEditingCode.evals.meta_clap_consistency import convert_audio

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ev.CLAPTextConsistencyMetric(model_path=ev._CLAP_PATH, model_arch=ev._MODEL_ARCH,
                                     enable_fusion=ev._FUSION).to(dev).eval()
    win = ev._WIN_LENGTH
    with torch.no_grad():
        te = m.model.get_text_embedding(texts, tokenizer=m._tokenizer, use_tensor=True)
        te = torch.nn.functional.normalize(te.float(), dim=-1)
        rows = index(a.renders)
        a.out.mkdir(parents=True, exist_ok=True)
        with open(a.out / "q2_clap.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["vec", "sign", "alpha", "clip"] + texts)
            for n, r in enumerate(rows):
                wav, sr, _ = load(r["npz"])
                x = torch.from_numpy(wav.astype(np.float32) / 32768.0)
                ws = int(win * sr) if win else x.shape[-1]
                starts = list(range(0, x.shape[-1], int(ws * 0.9))) if win else [0]
                sims = []
                for s0 in starts:
                    seg = x[:, s0:s0 + ws][:, None].to(dev)                        # [B, 1, T]
                    seg = convert_audio(seg, from_rate=sr, to_rate=m.model_sample_rate, to_channels=1).mean(dim=1)
                    ae = m.model.get_audio_embedding_from_data(seg, use_tensor=True)
                    ae = torch.nn.functional.normalize(ae.float(), dim=-1)
                    sims.append((ae @ te.T).cpu().numpy())                          # [B, n_texts]
                sim = np.mean(sims, axis=0)
                for i in range(len(wav)):
                    w.writerow([r["vec"], r["sign"], r["alpha"], i] + [float(v) for v in sim[i]])
                fh.flush()
                print(f"{n + 1}/{len(rows)}", flush=True)
    print("wrote", a.out / "q2_clap.csv")


# ------------------------------------------------------------------ summary

def _read(p: Path) -> dict:
    out = {}
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            k = (r.pop("vec"), int(r.pop("sign")), float(r.pop("alpha")), int(r.pop("clip")))
            out[k] = {c: float(v) for c, v in r.items()}
    return out


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr
    if np.allclose(y, y[0]) or not np.all(np.isfinite(y)):
        return float("nan")
    return float(spearmanr(x, y).correlation)


def cmd_q2_sum(a) -> None:
    clips = _read(a.out / "q2_clips.csv")
    clap_p = a.out / "q2_clap.csv"
    clap = _read(clap_p) if clap_p.exists() else {}
    anchors = json.loads((HERE / "anchors.json").read_text(encoding="utf-8"))["concepts"]
    for k, v in clap.items():
        for c in CLAP_PAIRS:
            pos, neg = anchors[c][:2]
            clips.setdefault(k, {})[f"clap.{c}"] = v[pos] - v[neg]
    res = json.loads((a.renders.parent / "results.json").read_text())
    lp_of = {(r["vector"], r["knob_sign"]): r["lpaps_top_over_cut"] for r in res["rows"]}
    ncl = 1 + max(k[3] for k in clips)
    metrics = [c for c in next(iter(clips.values())).keys()]
    groups = sorted({(k[0], k[1]) for k in clips if k[0] != "ref"})
    out = {}
    for vec, sign in groups:
        alphas = sorted({k[2] for k in clips if k[0] == vec and k[1] == sign}, key=abs)
        g = {"alphas": alphas, "lpaps_top_over_cut": lp_of.get((vec, "pos" if sign > 0 else "neg")), "m": {}}
        for m in metrics:
            ref = np.array([clips[("ref", 0, 0.0, i)][m] for i in range(ncl)])
            mat = np.array([[clips[(vec, sign, al, i)][m] for i in range(ncl)] for al in alphas])  # [5, n]
            d = mat - ref[None]
            xs = np.array([0.0] + [abs(x) for x in alphas])
            rho = [_spearman(xs, np.concatenate([[ref[i]], mat[:, i]])) for i in range(ncl)]
            g["m"][m] = {"delta_top": float(np.nanmean(d[-1])), "delta_curve": np.nanmean(d, axis=1).tolist(),
                         "rho_med": float(np.nanmedian(rho)) if np.any(np.isfinite(rho)) else float("nan"),
                         "frac_up_top": float(np.mean(d[-1] > 0)), "ref_mean": float(np.nanmean(ref))}
        out[f"{vec}|{'pos' if sign > 0 else 'neg'}"] = g
    (a.out / "q2.json").write_text(json.dumps(out, indent=1, default=float))
    write_q2_md(a.out / "q2.md", out)
    print("wrote", a.out / "q2.md")


TABLES = {
    "centroid": (["ridge_centroid", "ridge_resid_centroid", "bright"],
                 ["centroid", "centroid_nofloor", "hf_flatness", "hf_floor_db", "floor_share_db", "hf_share_db",
                  "clap.bright", "clap.static_noise", "clap.noise_floor", "clap.white_noise"]),
    "onset_rate": (["ridge_onset_rate", "ridge_resid_onset_rate", "percussive"],
                   ["onset_rate", "onset_superflux", "tempo_bt", "beat_rate", "clap.percussive",
                    "clap.fast_tempo"]),
    "lra": (["ridge_lra", "ridge_resid_lra", "dense_arrangement"],
            ["lra", "lra_ffmpeg", "clap.dense_arrangement", "clap.dynamic", "clap.uncompressed"]),
}


def write_q2_md(path: Path, out: dict) -> None:
    L = ["# Q2: independent-scorer rescoring of the v3 smoke renders", "",
         "Renders: v3_smoke/renders (12 prompts, seed 2115, 5 alphas per vector-sign, alpha 0 ref). No new renders.",
         "Cells: mean delta vs alpha 0 at the top alpha (LPAPS ~ the PCI cutoff for every row, see LPAPS/cut), "
         "then [median per-prompt Spearman over the 6 alphas incl. 0]. First column is the fit target.",
         "clap.X = cos(pos anchor) - cos(neg anchor) of anchors.json concept X. hf_flatness = energy-weighted "
         "spectral flatness above 8 kHz (1 = white noise); hf_floor_db / floor_share_db = 10th-percentile-over-time "
         "per-bin floor (>8 kHz / all) relative to mean energy; centroid_nofloor = centroid after subtracting that "
         "stationary floor; tempo_bt = librosa beat_track BPM; onset_superflux = log-mel superflux onset rate; "
         "lra_ffmpeg = ffmpeg ebur128 LRA (target lra = pyloudnorm short-term, label_corpus definition).", ""]
    for d, (vecs, ms) in TABLES.items():
        L += [f"## {d}", "", "| vector | sign | LPAPS/cut | " + " | ".join(ms) + " |",
              "|---|---|---|" + "---|" * len(ms)]
        for v in vecs:
            for s in ("pos", "neg"):
                g = out.get(f"{v}|{s}")
                if not g:
                    continue
                cells = []
                for m in ms:
                    x = g["m"].get(m)
                    cells.append("" if x is None else f"{x['delta_top']:+.3g} [{x['rho_med']:+.2f}]")
                lp = g["lpaps_top_over_cut"]
                L.append(f"| {v} | {s} | {'' if lp is None else f'{lp:.2f}'} | " + " | ".join(cells) + " |")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------ Q4

def cmd_q4(a) -> None:
    os.environ.setdefault("E2", str(a.e2))
    import ship_tools as T

    fs = T.jl(T.SH / "final_set.json")
    meas = T.jl(T.SH / "measured.json")
    prod = T.jl(T.SH / "product.json")
    cfx = prod["crossfx"]
    gains = T.jl(T.SH / "gains.json")
    rows, flagged = [], []
    for vid in sorted(fs["shipped_vids"]):
        m = meas[vid]
        per = {o: T.fixed_metrics(vid, o, gains) for o in (0, 1, 2, 3)}
        for side, ch in (("pos", "+"), ("neg", "-")):
            cx = cfx.get(f"{vid}{ch}") or {}
            oz = cx.get("own_z")
            # own_z is the raw (not direction-signed) label shift; fixed-gain effects are direction-signed
            cfx_wrong = oz is not None and (oz < 0 if side == "pos" else oz > 0)
            eff = {o: (p[side]["effect_clap"], p[side]["effect_muq"], p[side]["frac_clap"], p[side]["frac_muq"])
                   for o, p in per.items() if p}
            clap_wrong = [o for o, e in eff.items() if e[0] < 0]
            muq_wrong = [o for o, e in eff.items() if e[1] < 0]
            r = {"vid": vid, "side": side, "good_sign": side in m.get("good_signs", []),
                 "own_label": cx.get("own_label"), "own_z_seed2115": oz, "cfx_wrong": cfx_wrong,
                 "effect_clap_by_seed": {o: e[0] for o, e in eff.items()},
                 "effect_muq_by_seed": {o: e[1] for o, e in eff.items()},
                 "frac_clap_by_seed": {o: e[2] for o, e in eff.items()},
                 "frac_muq_by_seed": {o: e[3] for o, e in eff.items()},
                 "clap_wrong_seeds": clap_wrong, "muq_wrong_seeds": muq_wrong}
            r["any_wrong"] = bool(cfx_wrong or clap_wrong or muq_wrong)
            rows.append(r)
            if r["any_wrong"]:
                flagged.append(r)
    good = [r for r in rows if r["good_sign"]]
    summ = {
        "n_shipped_vids": len(fs["shipped_vids"]), "n_knob_signs": len(rows), "n_good_signs": len(good),
        "cfx_wrong_all": sum(r["cfx_wrong"] for r in rows), "cfx_wrong_good": sum(r["cfx_wrong"] for r in good),
        "clap_wrong_any_seed_all": sum(bool(r["clap_wrong_seeds"]) for r in rows),
        "clap_wrong_any_seed_good": sum(bool(r["clap_wrong_seeds"]) for r in good),
        "muq_wrong_any_seed_all": sum(bool(r["muq_wrong_seeds"]) for r in rows),
        "muq_wrong_any_seed_good": sum(bool(r["muq_wrong_seeds"]) for r in good),
        "any_wrong_all": len(flagged), "any_wrong_good": sum(r["any_wrong"] for r in good),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "q4.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=float))
    L = ["# Q4: inverted-direction census of the shipped knobs (ship pass data, CPU)", "",
         "Knob-sign = one side of a shipped vector (97 vectors x 2). good = side in measured.json good_signs "
         "(passes the quality bar). Own-label checks at the SHIPPED gain:",
         "- cfx: cross-effect own z (primary label, screen_music 12 prompts, seed 2115; ship/product.json) has "
         "the wrong sign (pos side < 0, neg side > 0).",
         "- clap / muq: fixed-gain direction-signed effect (mean over 50 test prompts) < 0 on any of seeds 0,1,2,3 "
         "(fixed_s{0..3}, ship_tools.fixed_metrics).", "",
         "## Counts", "", "| check | all knob-signs | good knob-signs |", "|---|---|---|"]
    for k, lab in (("cfx_wrong", "cfx own z wrong sign"), ("clap_wrong_any_seed", "CLAP effect < 0 on any seed"),
                   ("muq_wrong_any_seed", "MuQ effect < 0 on any seed"), ("any_wrong", "any of the three")):
        L.append(f"| {lab} | {summ[k + '_all']} / {summ['n_knob_signs']} | {summ[k + '_good']} / "
                 f"{summ['n_good_signs']} |")
    L += ["", "## Flagged knob-signs", "",
          "| knob | side | good | own z (s2115) | CLAP effect s0/s1/s2/s3 | MuQ effect s0/s1/s2/s3 | wrong on |",
          "|---|---|---|---|---|---|---|"]

    def fmt(dd):
        return " / ".join(f"{dd[o]:+.3f}" if o in dd else "-" for o in (0, 1, 2, 3))

    for r in sorted(flagged, key=lambda r: (not r["good_sign"], r["vid"], r["side"])):
        why = (["cfx"] if r["cfx_wrong"] else []) + \
              ([f"clap s{','.join(map(str, r['clap_wrong_seeds']))}"] if r["clap_wrong_seeds"] else []) + \
              ([f"muq s{','.join(map(str, r['muq_wrong_seeds']))}"] if r["muq_wrong_seeds"] else [])
        oz = r["own_z_seed2115"]
        L.append(f"| {r['vid']} | {r['side']} | {'y' if r['good_sign'] else 'n'} | "
                 f"{'' if oz is None else f'{oz:+.2f}'} | {fmt(r['effect_clap_by_seed'])} | "
                 f"{fmt(r['effect_muq_by_seed'])} | {'; '.join(why)} |")
    (a.out / "q4.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["q2-cpu", "q2-clap", "q2-sum", "q4"])
    ap.add_argument("--renders", type=Path, default=V3 / "renders")
    ap.add_argument("--out", type=Path, default=V3 / "q2q4")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 8))
    ap.add_argument("--tada-root", default=os.environ.get("TADA_ROOT"))
    ap.add_argument("--e2", type=Path, default=Path("E:/Projects/DEMON/steering-bench/many_knobs_v2"))
    a = ap.parse_args()
    {"q2-cpu": cmd_q2_cpu, "q2-clap": cmd_q2_clap, "q2-sum": cmd_q2_sum, "q4": cmd_q4}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
