"""Label the many-knobs capture corpus, one scorer per invocation.

Consumes ``$CAP/audio_{k:04d}.npz`` (int16 ``wav`` [B, T] mono, ``ids`` [B])
as soon as ``audio_{k:04d}.done`` appears (written by ``capture_resid.py
--prompts-json``), polls for more, and exits once ``meta.json`` reports
``n_shards`` and every one of them is labelled. Per shard it writes
``$CAP/labels/.parts/<scorer>/shard_{k:04d}.parquet`` (so a crash or restart
resumes where it stopped); at the end the parts are concatenated into
``$CAP/labels/<scorer>.parquet`` (column ``id`` + ``<scorer>.<label>``
float32, one row per clip id) and ``labels/<scorer>.done`` is written.

Scorers:

* ``desc``     the five proxies.py descriptors (centroid, lowhigh_db, flatness,
               onset_rate, perc_ratio) + ``centroid_st`` (centroid as a MIDI
               note number, 69 + 12 log2(c / 440)) + ``hf_ratio`` (energy above
               4 kHz over total) + spectral shape from the same STFT: band
               shares in dB of total (sub_db 20-60 Hz, bass_db 60-250, lowmid_db
               250-500, nasal_db 700-1500, presence_db 2-5 kHz, hf_db 8-16 kHz,
               air_db 12-20 kHz), tilt_db_oct (slope of octave-band dB per
               octave, 63 Hz-16 kHz), bandwidth (energy-weighted spectral
               bandwidth, Hz), rolloff85 (median 85% rolloff, Hz), floor_db
               (10th-percentile frame power, dB). CPU, multiprocess.
* ``timbral``  AudioCommons timbral_models: brightness, warmth, hardness, depth,
               roughness, sharpness, boominess, reverb (NaN per clip/feature on
               failure). numpy 2 / librosa 0.11 shim applied first (box_status.md
               "discriminators"). CPU, multiprocess.
* ``dyn``      lufs (pyloudnorm integrated, NaN if not importable), rms_db, lra
               (loudness range: 10-95 percentile of 3 s short-term loudness after
               EBU gating; pyloudnorm, else RMS dB windows), crest_db, onset_count,
               onset_rate, tempo, beat_strength (mean onset envelope on beats over
               mean overall), key (0 = C .. 11 = B), mode (1 major / 0 minor),
               key_strength (best Krumhansl correlation), major_minus_minor (best
               major minus best minor correlation), flux_mean (mean librosa onset
               strength = mel spectral flux), hp_ratio_db (10 log10 harmonic /
               percussive HPSS energy), loudness_slope (LU/s, short-term
               loudness vs time), momentary_std (dB std of 400 ms windows),
               silence_frac (share of 50 ms windows 40 dB under the loudest),
               peak_to_lufs, clip_frac (|x| >= 0.999), am_tremolo (4-8 Hz share
               of the 0.5-20 Hz RMS-envelope modulation spectrum), pulse_clarity
               (max onset-envelope autocorrelation at 0.25-2 s lags),
               tempo_stability (1 - CV of inter-beat intervals), offbeat_share
               (onset envelope at beat midpoints over on + off), hf_onset_rate
               (onsets/s of the >6 kHz mel onset envelope), pitch_centroid (CQT
               magnitude-weighted mean MIDI note, C1-B7), chroma_entropy (bits,
               mean chroma). CPU, multiprocess.
* ``passt``    AudioSet 527-class sigmoid probabilities (hear21passt, 32 kHz),
               columns ``passt.<display_name>`` from class_labels_indices.csv.
               If hear21passt is missing it writes nothing and exits 2.
* ``clap_music`` / ``clap_general``  laion_clap cosine to every anchor text in
               anchors.json (music_audioset_epoch_15_esc_90.14.pt HTSAT-base /
               630k-audioset-best.pt HTSAT-tiny, 48 kHz), plus one column per
               entry of anchors.json ``concepts`` ({name: [pos, neg]}, synced
               from the catalogue by check_catalogue.py --sync-anchors):
               ``<scorer>.<name>`` = cos(pos) - cos(neg). An anchor text equal
               to a concept name is written as ``<scorer>.anchor:<text>``.
* ``muq``      MuQ-MuLan-large cosine to the same anchors and concepts (24 kHz).
* ``musetimbre`` interface stub (NotImplementedError; reads MUSETIMBRE_REPO).

``--merge`` outer-joins every ``labels/<scorer>.parquet`` on id into
``labels/merged.parquet`` (row order: meta.json ``ids`` first, then any
extra ids).

    python scripts/steering_bench/label_corpus.py --cap $CAP --scorer dyn --workers 48
    python scripts/steering_bench/label_corpus.py --cap $CAP --scorer clap_music --device cuda:2
    python scripts/steering_bench/label_corpus.py --cap $CAP --merge
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

DEFAULT_ANCHORS = _HERE / "anchors.json"
CLAP_CKPT = {
    "clap_music": (os.environ.get("CLAP_MUSIC_CKPT",
                                  "/root/.cache/audio_metrics/music_audioset_epoch_15_esc_90.14.pt"), "HTSAT-base"),
    "clap_general": (os.environ.get("CLAP_GENERAL_CKPT",
                                    "/dev/shm/steerbench/ckpt/clap/630k-audioset-best.pt"), "HTSAT-tiny"),
}
AUDIOSET_LABELS = os.environ.get("AUDIOSET_LABELS",
                                 "/dev/shm/steerbench/ckpt/audioset/class_labels_indices.csv")
MUQ_MODEL = os.environ.get("MUQ_MODEL", "OpenMuQ/MuQ-MuLan-large")

DESC_COLS = ["centroid", "lowhigh_db", "flatness", "onset_rate", "perc_ratio", "centroid_st", "hf_ratio",
             "sub_db", "bass_db", "lowmid_db", "nasal_db", "presence_db", "hf_db", "air_db",
             "tilt_db_oct", "bandwidth", "rolloff85", "floor_db"]
DESC_BANDS = {"sub_db": (20, 60), "bass_db": (60, 250), "lowmid_db": (250, 500), "nasal_db": (700, 1500),
              "presence_db": (2000, 5000), "hf_db": (8000, 16000), "air_db": (12000, 20000)}
TIMBRAL_COLS = ["brightness", "warmth", "hardness", "depth", "roughness", "sharpness", "boominess", "reverb"]
TIMBRAL_FN = {"boominess": "timbral_booming"}
DYN_COLS = ["lufs", "rms_db", "lra", "crest_db", "onset_count", "onset_rate", "tempo", "beat_strength",
            "key", "mode", "key_strength", "major_minus_minor", "flux_mean", "hp_ratio_db",
            "loudness_slope", "momentary_std", "silence_frac", "peak_to_lufs", "clip_frac", "am_tremolo",
            "pulse_clarity", "tempo_stability", "offbeat_share", "hf_onset_rate", "pitch_centroid",
            "chroma_entropy"]
CPU_SCORERS = {"desc", "timbral", "dyn"}
GPU_SCORERS = {"passt", "clap_music", "clap_general", "muq", "musetimbre"}
SCORERS = sorted(CPU_SCORERS | GPU_SCORERS)

# Krumhansl-Kessler key profiles (C major / C minor).
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ----------------------------------------------------------------- anchors

def load_anchors(path) -> tuple[list[str], dict[str, str]]:
    """Tolerant loader: every non-underscore top-level key is a group holding
    a list of strings or of ``{"text": ...}``; a top-level list is one group
    "all". Duplicate texts keep their first group (warned)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, list):
        d = {"all": d}
    texts, group = [], {}
    for g, items in d.items():
        if str(g).startswith("_") or not isinstance(items, list):
            continue
        for it in items:
            t = it.get("text") if isinstance(it, dict) else it
            if not isinstance(t, str) or not t.strip():
                continue
            t = t.strip()
            if t in group:
                print(f"[anchors] duplicate {t!r} in {g} (kept in {group[t]})", file=sys.stderr)
                continue
            group[t] = g
            texts.append(t)
    if not texts:
        raise ValueError(f"{path}: no anchor texts")
    return texts, group


def load_concepts(path) -> dict:
    """anchors.json ``concepts``: {name: [pos_text, neg_text]} (empty if absent)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    c = d.get("concepts") if isinstance(d, dict) else None
    out = {}
    for name, v in (c or {}).items():
        pos, neg = (v.get("pos"), v.get("neg")) if isinstance(v, dict) else v
        out[str(name)] = (str(pos).strip(), str(neg).strip())
    return out


def text_plan(path) -> tuple[list[str], list[tuple[str, int, int | None]]]:
    """Texts to encode once, and the output columns ``(col, pos_idx, neg_idx)``:
    every anchor text (cosine; renamed ``anchor:<text>`` when it equals a
    concept name) then every concept (cos(pos) - cos(neg))."""
    anchors, _ = load_anchors(path)
    concepts = load_concepts(path)
    texts, idx = [], {}
    for t in anchors + [t for pn in concepts.values() for t in pn]:
        if t not in idx:
            idx[t] = len(texts)
            texts.append(t)
    cols = [(f"anchor:{t}" if t in concepts else t, idx[t], None) for t in anchors]
    cols += [(n, idx[p], idx[q]) for n, (p, q) in concepts.items()]
    return texts, cols


def _text_cols(sim: np.ndarray, plan) -> dict:
    return {c: (sim[:, i] - sim[:, j] if j is not None else sim[:, i]) for c, i, j in plan}


def scorer_columns(scorer: str, anchors=DEFAULT_ANCHORS, labels_csv=AUDIOSET_LABELS) -> list | None:
    """Column names (with the ``<scorer>.`` prefix) that ``--scorer`` writes,
    without audio or models. passt needs the AudioSet csv (None if missing)."""
    if scorer in CPU_FN:
        cols = CPU_FN[scorer][1]
    elif scorer == "passt":
        if not labels_csv or not Path(labels_csv).exists():
            return None
        import csv

        with open(labels_csv, newline="", encoding="utf-8") as f:
            cols = [r["display_name"].strip() for r in csv.DictReader(f)]
    elif scorer in ("clap_music", "clap_general", "muq"):
        cols = [c for c, _, _ in text_plan(anchors)[1]]
    else:
        cols = []
    return [f"{scorer}.{c}" for c in cols]


# ------------------------------------------------------------- CPU scorers

def _f32(clip) -> np.ndarray:
    return np.asarray(clip).reshape(-1).astype(np.float32) / 32768.0


def desc_clip(job) -> list:
    import librosa
    import proxies

    clip, sr = job
    y = _f32(clip)
    out = []
    for k in ("centroid", "lowhigh_db", "flatness", "onset_rate", "perc_ratio"):
        try:
            out.append(proxies.measure(k, y, sr))
        except Exception:
            out.append(float("nan"))
    c = out[0]
    out.append(69.0 + 12.0 * np.log2(c / 440.0) if c > 0 else float("nan"))
    p = np.abs(librosa.stft(y, n_fft=proxies.N_FFT, hop_length=proxies.HOP)) ** 2
    f = np.linspace(0, sr / 2, p.shape[0])
    tot = float(p.sum())
    out.append(float(p[f > 4000.0].sum()) / tot if tot > 0 else float("nan"))
    nan = float("nan")
    for lo, hi in DESC_BANDS.values():
        out.append(_db(float(p[(f >= lo) & (f < hi)].sum()) / tot) if tot > 0 else nan)
    if tot <= 0:
        return out + [nan] * 4
    octs = 63.0 * 2.0 ** np.arange(9)                        # 63 Hz .. 16 kHz
    band = np.array([float(p[(f >= c / np.sqrt(2)) & (f < c * np.sqrt(2))].sum()) for c in octs])
    ok = band > 0
    out.append(float(np.polyfit(np.arange(9)[ok], 10 * np.log10(band[ok]), 1)[0]) if ok.sum() > 1 else nan)
    e = p.sum(axis=0)
    act = e > 1e-4 * e.max()
    pa, ea = p[:, act], e[act]
    c = (f[:, None] * pa).sum(axis=0) / ea
    bw = np.sqrt((((f[:, None] - c[None]) ** 2) * pa).sum(axis=0) / ea)
    out.append(float((bw * ea).sum() / ea.sum()))
    cum = np.cumsum(pa, axis=0) / ea[None]
    out.append(float(np.median(f[np.argmax(cum >= 0.85, axis=0)])))
    out.append(_db(float(np.percentile(e, 10)) / p.shape[0]))
    return out


_TIMBRAL = None


def _timbral_module():
    """Import timbral_models behind the numpy 2 / librosa 0.11 shim."""
    global _TIMBRAL
    if _TIMBRAL is not None:
        return _TIMBRAL
    import librosa
    import librosa.onset as lon

    if not hasattr(np.lib, "pad"):
        np.lib.pad = np.pad
    if not getattr(lon.onset_detect, "_sb_shim", False):
        _od, _os = lon.onset_detect, lon.onset_strength

        def onset_detect(*a, **kw):
            if a:
                kw.setdefault("y", a[0])
            if len(a) > 1:
                kw.setdefault("sr", a[1])
            return _od(**kw)

        def onset_strength(*a, **kw):
            if a:
                kw.setdefault("y", a[0])
            if len(a) > 1:
                kw.setdefault("sr", a[1])
            return _os(**kw)

        onset_detect._sb_shim = onset_strength._sb_shim = True
        lon.onset_detect = librosa.onset_detect = onset_detect
        lon.onset_strength = librosa.onset_strength = onset_strength
    import timbral_models

    _TIMBRAL = timbral_models
    return _TIMBRAL


def timbral_clip(job) -> list:
    clip, sr = job
    tm = _timbral_module()
    y = _f32(clip).astype(np.float64)
    out = []
    for c in TIMBRAL_COLS:
        try:
            v = getattr(tm, TIMBRAL_FN.get(c, f"timbral_{c}"))(y, fs=sr)
            out.append(float(np.asarray(v, dtype=np.float64).reshape(-1)[0]))
        except Exception:
            out.append(float("nan"))
    return out


def _db(x: float) -> float:
    return float(10.0 * np.log10(max(x, 1e-12)))


def _short_term(y: np.ndarray, sr: int, meter) -> np.ndarray:
    win, hop = int(3.0 * sr), int(0.5 * sr)
    if len(y) < win:
        return np.array([])
    vals = []
    for s in range(0, len(y) - win + 1, hop):
        seg = y[s:s + win]
        if meter is not None:
            try:
                vals.append(meter.integrated_loudness(seg))
                continue
            except Exception:
                pass
        vals.append(_db(float(np.mean(seg.astype(np.float64) ** 2))))
    return np.array(vals, dtype=np.float64)


def _lra(st: np.ndarray) -> float:
    st = st[np.isfinite(st) & (st > -70.0)]
    if st.size < 2:
        return float("nan")
    rel = _db(float(np.mean(10.0 ** (st / 10.0)))) - 20.0
    st = st[st > rel]
    if st.size < 2:
        return float("nan")
    return float(np.percentile(st, 95) - np.percentile(st, 10))


def dyn_clip(job) -> list:
    import librosa

    clip, sr = job
    y = _f32(clip)
    nan = float("nan")
    r = dict.fromkeys(DYN_COLS, nan)
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
    except Exception:
        meter = None
    ms = float(np.mean(y.astype(np.float64) ** 2))
    r["rms_db"] = _db(ms)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if ms > 0 and peak > 0:
        r["crest_db"] = float(20.0 * np.log10(peak / np.sqrt(ms)))
    if meter is not None and peak > 0:
        try:
            v = float(meter.integrated_loudness(y))
            r["lufs"] = v if np.isfinite(v) else nan
        except Exception:
            pass
    st = _short_term(y, sr, meter)
    r["lra"] = _lra(st)
    if peak == 0:
        return [r[c] for c in DYN_COLS]
    r["clip_frac"] = float(np.mean(np.abs(y) >= 0.999))
    r["peak_to_lufs"] = float(20.0 * np.log10(peak)) - (r["lufs"] if np.isfinite(r["lufs"]) else r["rms_db"])
    ok = np.isfinite(st) & (st > -70.0)
    if ok.sum() >= 2:
        r["loudness_slope"] = float(np.polyfit(0.5 * np.arange(len(st))[ok], st[ok], 1)[0])
    yy = y.astype(np.float64) ** 2

    def _win_db(w, h):
        n = 1 + max(0, (len(yy) - w) // h)
        return np.array([_db(float(yy[i * h:i * h + w].mean())) for i in range(n)])

    mom = _win_db(int(0.4 * sr), int(0.1 * sr))
    mom = mom[mom > -70.0]
    if mom.size >= 2:
        r["momentary_std"] = float(mom.std())
    w50 = _win_db(int(0.05 * sr), int(0.05 * sr))
    r["silence_frac"] = float(np.mean(w50 < w50.max() - 40.0))
    try:
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0].astype(np.float64)
        spec = np.abs(np.fft.rfft((rms - rms.mean()) * np.hanning(len(rms)))) ** 2
        fq = np.fft.rfftfreq(len(rms), 512.0 / sr)
        den = spec[(fq >= 0.5) & (fq <= 20.0)].sum()
        if den > 0:
            r["am_tremolo"] = float(spec[(fq >= 4.0) & (fq <= 8.0)].sum() / den)
    except Exception:
        pass
    try:
        env = librosa.onset.onset_strength(y=y, sr=sr)
        r["flux_mean"] = float(env.mean())
        on = librosa.onset.onset_detect(onset_envelope=env, sr=sr)
        r["onset_count"] = float(len(on))
        r["onset_rate"] = float(len(on) / max(len(y) / sr, 1e-6))
        tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr)
        r["tempo"] = float(np.asarray(tempo).reshape(-1)[0])
        if len(beats) and env.mean() > 0:
            r["beat_strength"] = float(env[np.asarray(beats, dtype=int)].mean() / env.mean())
        beats = np.asarray(beats, dtype=int)
        if len(beats) >= 3:
            ibi = np.diff(beats).astype(np.float64)
            r["tempo_stability"] = float(1.0 - ibi.std() / ibi.mean())
            mid = (beats[:-1] + beats[1:]) // 2
            on_b, off_b = float(env[beats[:-1]].sum()), float(env[mid].sum())
            if on_b + off_b > 0:
                r["offbeat_share"] = off_b / (on_b + off_b)
        e0 = env - env.mean()
        ac = np.correlate(e0, e0, mode="full")[len(e0) - 1:]
        fr = sr / 512.0
        lo, hi = int(0.25 * fr), min(int(2.0 * fr), len(ac) - 1)
        if ac[0] > 0 and hi > lo:
            r["pulse_clarity"] = float(ac[lo:hi + 1].max() / ac[0])
        # >6 kHz bands of the full-band mel dB spectrogram (ref = clip max, 80 dB floor), so a clip
        # with no HF content gives a flat envelope and no onsets (no per-band renormalisation)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
        hf = librosa.mel_frequencies(n_mels=128, fmax=sr / 2.0) >= 6000.0
        envh = librosa.onset.onset_strength(S=mel[hf], sr=sr)
        onh = librosa.onset.onset_detect(onset_envelope=envh, sr=sr, normalize=False, delta=0.5)
        r["hf_onset_rate"] = float(len(onh) / max(len(y) / sr, 1e-6))
    except Exception:
        pass
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
        if np.std(chroma) > 0:
            maj = np.array([np.corrcoef(chroma, np.roll(KK_MAJOR, k))[0, 1] for k in range(12)])
            mnr = np.array([np.corrcoef(chroma, np.roll(KK_MINOR, k))[0, 1] for k in range(12)])
            if maj.max() >= mnr.max():
                r["key"], r["mode"], r["key_strength"] = float(maj.argmax()), 1.0, float(maj.max())
            else:
                r["key"], r["mode"], r["key_strength"] = float(mnr.argmax()), 0.0, float(mnr.max())
            r["major_minus_minor"] = float(maj.max() - mnr.max())
        if chroma.sum() > 0:
            q = chroma / chroma.sum()
            r["chroma_entropy"] = float(-(q[q > 0] * np.log2(q[q > 0])).sum())
    except Exception:
        pass
    try:
        cq = np.abs(librosa.cqt(y, sr=sr, hop_length=1024, fmin=librosa.note_to_hz("C1"), n_bins=84))
        w = cq.sum(axis=1)
        if w.sum() > 0:
            r["pitch_centroid"] = float(24.0 + (np.arange(84) * w).sum() / w.sum())
    except Exception:
        pass
    try:
        s = librosa.stft(y, n_fft=2048, hop_length=512)
        h, p = librosa.decompose.hpss(s)
        r["hp_ratio_db"] = _db(float((np.abs(h) ** 2).sum())) - _db(float((np.abs(p) ** 2).sum()))
    except Exception:
        pass
    return [r[c] for c in DYN_COLS]


CPU_FN = {"desc": (desc_clip, DESC_COLS), "timbral": (timbral_clip, TIMBRAL_COLS), "dyn": (dyn_clip, DYN_COLS)}


def _worker_init():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"


class CpuScorer:
    def __init__(self, name: str, workers: int):
        self.name = name
        self.fn, self.cols = CPU_FN[name]
        self.workers = max(1, int(workers))
        self.pool = None
        if name == "timbral":
            _timbral_module()  # fail fast in the parent if it cannot import
        if self.workers > 1:
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ.setdefault(k, "1")  # inherited by spawned workers
            self.pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"),
                                            initializer=_worker_init)

    def score(self, wav: np.ndarray, sr: int) -> dict:
        jobs = [(w, sr) for w in wav]
        if self.pool is None:
            rows = [self.fn(j) for j in jobs]
        else:
            rows = list(self.pool.map(self.fn, jobs, chunksize=1))
        a = np.asarray(rows, dtype=np.float64).reshape(len(jobs), len(self.cols))
        return {c: a[:, i] for i, c in enumerate(self.cols)}

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


# ------------------------------------------------------------- GPU scorers

def _resample(wav: np.ndarray, sr: int, target: int, device: str):
    import torch

    y = wav.astype(np.float32) / 32768.0
    if sr == target:
        return torch.from_numpy(y).to(device)
    try:
        import torchaudio.functional as AF
    except ImportError:  # no torchaudio in the env: resample on CPU
        import librosa

        return torch.from_numpy(librosa.resample(y, orig_sr=sr, target_sr=target, axis=-1)).to(device)
    return AF.resample(torch.from_numpy(y).to(device), sr, target)


def _cos(a, b):
    import torch

    a = torch.nn.functional.normalize(a.float(), dim=-1)
    b = torch.nn.functional.normalize(b.float(), dim=-1)
    return (a @ b.T).cpu().numpy()


class PasstScorer:
    def __init__(self, device: str, labels_csv: str, chunk: int = 8):
        try:
            from hear21passt.base import get_basic_model
        except ImportError as e:
            _fatal(f"passt: hear21passt is not importable ({e}); pip install hear21passt. Nothing written.")
            raise SystemExit(2)
        import pandas as pd

        if not Path(labels_csv).exists():
            _fatal(f"passt: AudioSet label csv {labels_csv} missing (set AUDIOSET_LABELS)")
            raise SystemExit(2)
        names = pd.read_csv(labels_csv).sort_values("index")["display_name"].astype(str).str.strip().tolist()
        if len(names) != 527:
            _fatal(f"passt: {labels_csv} has {len(names)} classes, expected 527")
            raise SystemExit(2)
        self.cols = names
        self.device, self.chunk = device, chunk
        self.model = get_basic_model(mode="logits").to(device).eval()

    def score(self, wav, sr):
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(wav), self.chunk):
                x = _resample(wav[i:i + self.chunk], sr, 32000, self.device)
                out.append(torch.sigmoid(self.model(x)).float().cpu().numpy())
        a = np.concatenate(out, 0)
        return {c: a[:, j] for j, c in enumerate(self.cols)}

    def close(self):
        pass


class ClapScorer:
    def __init__(self, name: str, device: str, anchors, ckpt: str | None, template: str, chunk: int = 16):
        import laion_clap
        import torch

        path, amodel = CLAP_CKPT[name]
        path = ckpt or path
        if not Path(path).exists():
            _fatal(f"{name}: checkpoint {path} missing (pass --ckpt or set CLAP_*_CKPT)")
            raise SystemExit(2)
        orig = torch.load
        torch.load = lambda *a, **kw: orig(*a, **{**kw, "weights_only": False})
        try:
            m = laion_clap.CLAP_Module(enable_fusion=False, amodel=amodel)
            m.load_ckpt(path, verbose=False)
        finally:
            torch.load = orig
        self.model = m.to(device).eval()
        self.device, self.chunk = device, chunk
        texts, self.plan = anchors
        self.cols = [c for c, _, _ in self.plan]
        texts = [template.format(t) for t in texts]
        with torch.no_grad():
            embs = [self.model.get_text_embedding(texts[i:i + 64], use_tensor=True)
                    for i in range(0, len(texts), 64)]
        self.text = torch.cat(embs, 0).to(device)

    def score(self, wav, sr):
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(wav), self.chunk):
                x = _resample(wav[i:i + self.chunk], sr, 48000, self.device)
                emb = self.model.get_audio_embedding_from_data(x=x, use_tensor=True)
                out.append(_cos(emb, self.text))
        return _text_cols(np.concatenate(out, 0), self.plan)

    def close(self):
        pass


class MuqScorer:
    def __init__(self, device: str, anchors, template: str, chunk: int = 8):
        import torch
        from muq import MuQMuLan

        self.model = MuQMuLan.from_pretrained(MUQ_MODEL).to(device).eval()
        self.device, self.chunk = device, chunk
        texts, self.plan = anchors
        self.cols = [c for c, _, _ in self.plan]
        texts = [template.format(t) for t in texts]
        with torch.no_grad():
            self.text = torch.cat([self.model(texts=texts[i:i + 64]) for i in range(0, len(texts), 64)], 0)

    def score(self, wav, sr):
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(wav), self.chunk):
                x = _resample(wav[i:i + self.chunk], sr, 24000, self.device)
                out.append(_cos(self.model(wavs=x), self.text))
        return _text_cols(np.concatenate(out, 0), self.plan)

    def close(self):
        pass


class MuseTimbreScorer:
    """TODO(planner): MuseTimbre timbre-similarity labels. The checkout path comes
    from env MUSETIMBRE_REPO (being located); implement load + score here with
    the same ``cols`` / ``score(wav int16 [B, T], sr) -> {col: [B]}`` interface."""

    def __init__(self, device: str, anchors):
        self.repo = os.environ.get("MUSETIMBRE_REPO")
        raise NotImplementedError(
            f"musetimbre scorer not implemented yet (MUSETIMBRE_REPO={self.repo!r}); "
            "TODO: load the model from that repo and map clips to label columns")


def _fatal(msg: str):
    print(f"ERROR {msg}", file=sys.stderr, flush=True)
    return RuntimeError(msg)


def make_scorer(args):
    if args.scorer in CPU_SCORERS:
        return CpuScorer(args.scorer, args.workers)
    anchors = None
    if args.scorer in ("clap_music", "clap_general", "muq", "musetimbre"):
        anchors = text_plan(args.anchors)
    if args.scorer == "passt":
        return PasstScorer(args.device, args.labels_csv)
    if args.scorer in CLAP_CKPT:
        return ClapScorer(args.scorer, args.device, anchors, args.ckpt, args.template)
    if args.scorer == "muq":
        return MuqScorer(args.device, anchors, args.template)
    return MuseTimbreScorer(args.device, anchors)


# ------------------------------------------------------------- driver

def _read_meta(cap: Path):
    try:
        return json.loads((cap / "meta.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _n_shards(meta) -> int | None:
    if not meta:
        return None
    if "n_shards" in meta:
        return int(meta["n_shards"])
    if "n" in meta:
        b = int(meta.get("shard_size", 32))
        return (int(meta["n"]) + b - 1) // b
    return None


def _write_part(path: Path, ids, cols: dict, scorer: str) -> None:
    import pandas as pd

    df = pd.DataFrame({f"{scorer}.{c}": np.asarray(v, dtype=np.float32) for c, v in cols.items()})
    df.insert(0, "id", np.asarray(ids, dtype=np.int64))
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def run_scorer(args) -> int:
    import pandas as pd

    cap = Path(args.cap)
    labels = cap / "labels"
    parts = labels / ".parts" / args.scorer
    final, done_flag = labels / f"{args.scorer}.parquet", labels / f"{args.scorer}.done"
    if done_flag.exists() and not args.force:
        print(f"{done_flag} exists (pass --force to redo)")
        return 0
    scorer = None
    t0, last_new, n_done = time.perf_counter(), time.perf_counter(), 0
    try:
        while True:
            meta = _read_meta(cap)
            total = _n_shards(meta)
            sr = int((meta or {}).get("sr", 44100))
            new = False
            for d in sorted(cap.glob("audio_*.done")):
                k = int(d.stem.split("_")[1])
                part = parts / f"shard_{k:04d}.parquet"
                if part.exists():
                    continue
                if scorer is None:
                    scorer = make_scorer(args)
                with np.load(cap / f"audio_{k:04d}.npz") as z:
                    wav, ids = z["wav"], z["ids"]
                if wav.ndim == 3:
                    wav = wav.mean(axis=1).astype(np.int16)
                ts = time.perf_counter()
                parts.mkdir(parents=True, exist_ok=True)
                _write_part(part, ids, scorer.score(wav, sr), args.scorer)
                n_done += 1
                new, last_new = True, time.perf_counter()
                print(f"[{args.scorer}] shard {k} ({len(ids)} clips) {time.perf_counter() - ts:.1f}s"
                      f" total {n_done} new, {time.perf_counter() - t0:.0f}s", flush=True)
            have = len(list(parts.glob("shard_*.parquet")))
            if total is not None and have >= total:
                break
            if args.timeout and time.perf_counter() - last_new > args.timeout:
                _fatal(f"{args.scorer}: no new shard for {args.timeout}s ({have}/{total}); parts kept, no .done")
                return 3
            if not new:
                time.sleep(args.poll)
    finally:
        if scorer is not None:
            scorer.close()
    files = sorted(parts.glob("shard_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if meta and "ids" in meta:
        order = {int(i): r for r, i in enumerate(meta["ids"])}
        df = df.iloc[np.argsort([order.get(int(i), len(order) + int(i)) for i in df["id"]], kind="stable")]
    df = df.reset_index(drop=True)
    tmp = final.with_name(final.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(final)
    done_flag.write_text(json.dumps({"rows": len(df), "cols": len(df.columns) - 1, "shards": len(files),
                                     "t": time.strftime("%Y-%m-%d %H:%M:%S")}))
    print(f"[{args.scorer}] wrote {final} {df.shape} in {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


def merge(cap: Path) -> int:
    import pandas as pd

    labels = cap / "labels"
    files = sorted(f for f in labels.glob("*.parquet") if f.stem != "merged")
    if not files:
        _fatal(f"no label parquet in {labels}")
        return 2
    out = None
    for f in files:
        df = pd.read_parquet(f).set_index("id")
        out = df if out is None else out.join(df, how="outer")
    meta = _read_meta(cap)
    if meta and "ids" in meta:
        have = set(out.index)
        first = [i for i in meta["ids"] if i in have]
        seen = set(first)
        rest = [i for i in out.index if i not in seen]
        out = out.loc[first + rest]
    out = out.reset_index()
    out.to_parquet(labels / "merged.parquet", index=False)
    print(f"merged {[f.stem for f in files]} -> {labels / 'merged.parquet'} {out.shape}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", required=True, help="corpus root ($CAP)")
    ap.add_argument("--scorer", choices=SCORERS)
    ap.add_argument("--merge", action="store_true", help="outer-join labels/*.parquet into labels/merged.parquet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--anchors", default=str(DEFAULT_ANCHORS))
    ap.add_argument("--template", default="{}", help="text template for anchors, e.g. 'This is a music of {}'")
    ap.add_argument("--ckpt", default=None, help="CLAP checkpoint override")
    ap.add_argument("--labels-csv", default=AUDIOSET_LABELS, help="AudioSet class_labels_indices.csv (passt)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="CPU scorers")
    ap.add_argument("--poll", type=float, default=10.0, help="seconds between shard scans")
    ap.add_argument("--timeout", type=float, default=0.0, help="give up after this many idle seconds (0 = never)")
    ap.add_argument("--force", action="store_true", help="relabel even if <scorer>.done exists (parts are reused)")
    args = ap.parse_args()
    if args.merge:
        return merge(Path(args.cap))
    if not args.scorer:
        ap.error("--scorer or --merge required")
    return run_scorer(args)


if __name__ == "__main__":
    raise SystemExit(main())
