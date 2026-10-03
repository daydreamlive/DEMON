"""TADA's evaluation protocol (arXiv 2602.11910, Sec. 5.2 and App. H).

Ported from the reference code (github.com/luk-st/steer-audio, MIT):
``src/steering/eval/auc.py`` (AUC, smoothness, quality sampling),
``eval_steering_protocol.py`` (alignment and LPAPS per strength),
``mir_preservation.py`` (decomposed preservation axes) and the LPAPS of
Manor and Michaeli (``editing/AudioEditingCode/evals``, MIT; LPIPS-style
licence). Pure functions on numbers and tensors are tested on CPU; the
model-backed scorers (MuQ-MuLan, music CLAP, LPAPS backbone) import their
packages lazily and are exercised by the family lanes on the GPU.

Protocol summary:

* preservation: LPAPS between the steered output and the ``alpha = 0``
  output of the same prompt and seed (lower = closer);
* alignment: sign-corrected delta ``sign(alpha) * (sim(alpha) - sim(0))``
  with ``sim`` = MuQ-MuLan or CLAP similarity to the concept query;
* AUC: trapezoid of delta over LPAPS, sweep order (increasing ``|alpha|``),
  truncated at a common LPAPS cutoff with one interpolated point at the
  cutoff; the cutoff is the maximum LPAPS of the prompt-swap baseline
  (PCI at full strength), the distance of a different prompt;
* smoothness: std of consecutive gaps of the delta normalised by its peak;
* quality: Audiobox Aesthetics sampled at 15 uniform LPAPS points per
  direction and averaged.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numpy import trapezoid as _trapezoid
except ImportError:  # numpy < 2
    from numpy import trapz as _trapezoid  # type: ignore

from .patching import aggregate_impacts, impact_score, select_layers  # noqa: F401

#: Quality sample points per direction (reference ``N_QUALITY_POINTS``).
N_QUALITY_POINTS = 15
#: MuQ-MuLan checkpoint and the reference similarity templates.
MUQ_MODEL_ID = "OpenMuQ/MuQ-MuLan-large"
CLAP_MUSIC_CKPT = "music_audioset_epoch_15_esc_90.14.pt"
#: Window used by windowed CLAP and LPAPS (seconds) and its overlap.
WINDOW_S = 10
WINDOW_OVERLAP = 0.1


# ---------------------------------------------------------------------------
# Alignment and the alignment-preservation AUC
# ---------------------------------------------------------------------------

def sign_corrected_delta(alphas: Sequence[float], values: Sequence[float],
                         direction: str) -> np.ndarray:
    """``delta = sign * (value - value_at_smallest_|alpha|)`` for one
    direction (``pos``: sign +1, ``neg``: sign -1)."""
    a = np.asarray(alphas, dtype=float)
    v = np.asarray(values, dtype=float)
    base = float(v[int(np.argmin(np.abs(a)))])
    sign = 1.0 if direction == "pos" else -1.0
    return sign * (v - base)


def filter_direction(alphas: Sequence[float], *cols: Sequence[float],
                     direction: str) -> Tuple[np.ndarray, ...]:
    """Rows with ``alpha >= 0`` (``pos``) or ``alpha <= 0`` (``neg``),
    sorted by alpha; alpha 0 belongs to both directions."""
    a = np.round(np.asarray(alphas, dtype=float), 6)
    keep = a >= 0 if direction == "pos" else a <= 0
    order = np.argsort(a[keep], kind="stable")
    return (a[keep][order],) + tuple(np.asarray(c, dtype=float)[keep][order] for c in cols)


def compute_auc(lpaps: Sequence[float], delta: Sequence[float], cutoff: float,
                abs_alpha: Optional[Sequence[float]] = None) -> float:
    """Path integral of ``delta`` over LPAPS up to ``cutoff``, points in
    increasing ``|alpha|`` (reference ``compute_auc``). nan below 2 points."""
    lp = np.asarray(lpaps, dtype=float)
    dv = np.asarray(delta, dtype=float)
    mask = lp <= cutoff + 1e-9
    lp, dv = lp[mask], dv[mask]
    if len(lp) < 2:
        return float("nan")
    if abs_alpha is not None:
        order = np.argsort(np.asarray(abs_alpha, dtype=float)[mask], kind="stable")
        lp, dv = lp[order], dv[order]
    return float(_trapezoid(dv, lp))


def truncate_at_cutoff(alphas: np.ndarray, lpaps: np.ndarray, values: np.ndarray,
                       cutoff: float, add_boundary: bool) -> Tuple[np.ndarray, ...]:
    """Rows with LPAPS <= cutoff (sorted by alpha), plus, when
    ``add_boundary`` and a point above the cutoff exists, one point at
    exactly ``lpaps == cutoff`` with the value linearly interpolated between
    the last point below and the first point above (reference
    ``load_and_merge_alignment``)."""
    below = lpaps <= cutoff + 1e-9
    if not below.any():
        return np.array([]), np.array([]), np.array([])
    a, lp, v = alphas[below], lpaps[below], values[below]
    above = ~below
    if add_boundary and above.any():
        bi = int(np.argmax(np.where(below, lpaps, -np.inf)))
        ai = int(np.argmin(np.where(above, lpaps, np.inf)))
        if lpaps[ai] - lpaps[bi] > 1e-12:
            frac = (cutoff - lpaps[bi]) / (lpaps[ai] - lpaps[bi])
            v_cut = float(values[bi]) + frac * (float(values[ai]) - float(values[bi]))
            a = np.append(a, alphas[ai])
            lp = np.append(lp, cutoff)
            v = np.append(v, v_cut)
    order = np.argsort(a, kind="stable")
    return a[order], lp[order], v[order]


def alignment_auc(alphas: Sequence[float], lpaps: Sequence[float],
                  values: Sequence[float], cutoff: float, direction: str) -> float:
    """AUC for one direction and one alignment metric (reference
    ``compute_alignment_auc_direction``)."""
    a, lp, v = filter_direction(alphas, lpaps, values, direction=direction)
    a, lp, v = truncate_at_cutoff(a, lp, v, cutoff, add_boundary=True)
    if len(a) == 0:
        return float("nan")
    delta = sign_corrected_delta(a, v, direction)
    return compute_auc(lp, delta, cutoff, abs_alpha=np.abs(a))


def smoothness(alphas: Sequence[float], lpaps: Sequence[float],
               values: Sequence[float], cutoff: float, direction: str) -> float:
    """CSM: within the cutoff, ``delta / max(clip(delta, 0))`` ordered by
    ``|alpha|``, std of consecutive gaps; lower is smoother. nan when the
    concept never moves the right way (reference ``compute_csm_direction``)."""
    a, lp, v = filter_direction(alphas, lpaps, values, direction=direction)
    a, lp, v = truncate_at_cutoff(a, lp, v, cutoff, add_boundary=False)
    if len(a) == 0:
        return float("nan")
    delta = sign_corrected_delta(a, v, direction)
    peak = float(np.max(np.clip(delta, 0, None)))
    if peak <= 1e-9:
        return float("nan")
    norm = delta / peak
    gaps = np.diff(norm[np.argsort(np.abs(a))])
    return float(np.std(gaps)) if len(gaps) else float("nan")


def lpaps_cutoff(max_lpaps_per_run: Sequence[float]) -> float:
    """Common preservation cutoff: the smallest of the reference runs'
    maximum LPAPS (the paper uses the PCI all-layer and localised runs)."""
    vals = [float(x) for x in max_lpaps_per_run if not math.isnan(float(x))]
    return min(vals) if vals else float("nan")


def quality_at_lpaps(alphas: Sequence[float], lpaps: Sequence[float],
                     quality: Sequence[float], cutoff: float, direction: str,
                     n_points: int = N_QUALITY_POINTS) -> float:
    """Mean quality interpolated at ``n_points`` uniform LPAPS values in
    ``[0, cutoff]`` within the measured range (reference
    ``interpolate_quality_at_lpaps``)."""
    _, lp, q = filter_direction(alphas, lpaps, quality, direction=direction)
    if len(lp) == 0:
        return float("nan")
    grid = np.linspace(0, cutoff, n_points)
    order = np.argsort(lp)
    lp, q = lp[order], q[order]
    grid = grid[(grid >= lp[0] - 1e-9) & (grid <= lp[-1] + 1e-9)]
    if len(grid) == 0:
        return float("nan")
    return float(np.mean(np.interp(grid, lp, q)))


def both_directions(fn: Callable[..., float], *args, **kwargs) -> Dict[str, float]:
    """``{"pos", "neg", "avg"}`` for a per-direction metric; avg is nan
    when either side is."""
    pos = fn(*args, direction="pos", **kwargs)
    neg = fn(*args, direction="neg", **kwargs)
    avg = (pos + neg) / 2.0 if not (math.isnan(pos) or math.isnan(neg)) else float("nan")
    return {"pos": pos, "neg": neg, "avg": avg}


# ---------------------------------------------------------------------------
# LPAPS (Manor and Michaeli), backbone-agnostic math
# ---------------------------------------------------------------------------

def lpaps_from_features(feats0: Sequence, feats1: Sequence):
    """LPAPS between two feature stacks (one tensor per backbone stage,
    shaped ``[B, N, C]``), exactly as the reference ``LPAPS.forward``:
    each stage is normalised along dim 1, squared differences are summed
    along dim 1, averaged over the remaining axes, and summed over stages.
    Returns a ``[B]`` tensor."""
    import torch

    val = 0
    for f0, f1 in zip(feats0, feats1):
        n0 = f0 / (torch.sqrt(torch.sum(f0 ** 2, dim=1, keepdim=True)) + 1e-10)
        n1 = f1 / (torch.sqrt(torch.sum(f1 ** 2, dim=1, keepdim=True)) + 1e-10)
        d = ((n0 - n1) ** 2).sum(dim=1, keepdim=True)
        val = val + d.mean([1, 2], keepdim=True)
    return val.reshape(-1)


def windowed(score: Callable, a, b, sr_a: int, sr_b: int,
             win_s: Optional[float] = WINDOW_S, overlap: float = WINDOW_OVERLAP,
             reduce: str = "mean") -> float:
    """Reference ``compute_lpaps_with_windows``: score ``win_s``-second
    windows stepped by ``win_s * (1 - overlap)`` and combine. ``a`` and
    ``b`` are ``[C, T]``; ``score(window_a, window_b)`` returns a float."""
    wa = int(sr_a * (win_s if win_s is not None else 10))
    wb = int(sr_b * (win_s if win_s is not None else 10))
    scores = [
        float(score(a[:, i:i + wa], b[:, j:j + wb]))
        for i, j in zip(range(0, a.shape[-1], int(wa * (1 - overlap))),
                        range(0, b.shape[-1], int(wb * (1 - overlap))))
    ]
    fn = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min}[reduce]
    return float(fn(scores))


# ---------------------------------------------------------------------------
# Decomposed preservation axes (App. H.3; reference mir_preservation.py)
# ---------------------------------------------------------------------------

def harmony_distance(chroma_a: np.ndarray, chroma_b: np.ndarray) -> float:
    """1 - mean per-frame cosine similarity of two CQT chromagrams."""
    T = min(chroma_a.shape[1], chroma_b.shape[1])
    b, s = chroma_a[:, :T], chroma_b[:, :T]
    num = (b * s).sum(0)
    den = np.linalg.norm(b, axis=0) * np.linalg.norm(s, axis=0) + 1e-9
    return 1.0 - float((num / den).mean())


def structure_distance(chroma_a: np.ndarray, chroma_b: np.ndarray) -> float:
    """1 - max(Pearson r, 0) between the chroma self-similarity matrices."""
    T = min(chroma_a.shape[1], chroma_b.shape[1])
    b = chroma_a[:, :T] / (np.linalg.norm(chroma_a[:, :T], axis=0) + 1e-9)
    s = chroma_b[:, :T] / (np.linalg.norm(chroma_b[:, :T], axis=0) + 1e-9)
    iu = np.triu_indices(T, k=1)
    r = np.corrcoef((b.T @ b)[iu], (s.T @ s)[iu])[0, 1]
    return 1.0 - max(float(r) if np.isfinite(r) else 0.0, 0.0)


def rhythm_distance(beats_a: np.ndarray, beats_b: np.ndarray) -> float:
    """1 - mir_eval beat F-measure (beat times in seconds)."""
    import mir_eval

    if len(beats_a) == 0 or len(beats_b) == 0:
        return float("nan")
    return 1.0 - float(mir_eval.beat.f_measure(beats_a, beats_b))


def melody_distance(f0_a: np.ndarray, f0_b: np.ndarray) -> float:
    """1 - raw pitch accuracy of pYIN f0 contours (nan = unvoiced)."""
    import mir_eval

    T = min(len(f0_a), len(f0_b))
    fb, fs = f0_a[:T], f0_b[:T]
    vb, vs = ~np.isnan(fb), ~np.isnan(fs)
    cb = mir_eval.melody.hz2cents(np.nan_to_num(fb))
    cs = mir_eval.melody.hz2cents(np.nan_to_num(fs))
    return 1.0 - float(mir_eval.melody.raw_pitch_accuracy(vb, cb, vs, cs))


def mir_features(y: np.ndarray, sr: int = 22050, axes: Sequence[str] = ("harmony", "rhythm", "melody", "ssm")) -> dict:
    """Per-clip features for the decomposed axes (mono float32 at ``sr``)."""
    import librosa

    out = {}
    if "harmony" in axes or "ssm" in axes:
        out["chroma"] = librosa.feature.chroma_cqt(y=y, sr=sr)
    if "rhythm" in axes:
        _, frames = librosa.beat.beat_track(y=y, sr=sr)
        out["beats"] = librosa.frames_to_time(frames, sr=sr)
    if "melody" in axes:
        fmin, fmax = librosa.note_to_hz("C2"), librosa.note_to_hz("C7")
        out["f0"] = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)[0]
    return out


# ---------------------------------------------------------------------------
# Model-backed similarity (lazy; GPU lanes)
# ---------------------------------------------------------------------------

def muq_similarity(wavs_24k: Sequence[np.ndarray], texts: Sequence[str],
                   device: str = "cuda", template: str = "{p}",
                   batch_size: int = 32) -> np.ndarray:
    """MuQ-MuLan audio-text similarity ``[n_audio, n_text]`` (reference
    ``calculate_muqt``: ``OpenMuQ/MuQ-MuLan-large``, mono 24 kHz input;
    the localisation run uses template ``"A {p} track"``). Needs ``muq``."""
    import torch
    from muq import MuQMuLan

    mulan = MuQMuLan.from_pretrained(MUQ_MODEL_ID).to(device).eval()
    with torch.no_grad():
        t = mulan(texts=[template.format(p=x) for x in texts])
        sims = []
        for i in range(0, len(wavs_24k), batch_size):
            batch = torch.from_numpy(np.stack(wavs_24k[i:i + batch_size])).to(device)
            sims.append(mulan.calc_similarity(mulan(wavs=batch), t).cpu())
    return torch.cat(sims).float().numpy()


def clap_similarity(wav_paths: Sequence[str], texts: Sequence[str], ckpt_path: str,
                    device: str = "cuda", template: str = "{p}") -> np.ndarray:
    """Music-CLAP cosine similarity ``[n_audio, n_text]`` (reference
    ``calculate_clap`` with the music checkpoint: HTSAT-base, no fusion;
    the localisation run uses template ``"This is a music of {p}"``).
    Needs ``laion_clap`` and the checkpoint file."""
    import laion_clap
    import torch

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    _load = torch.load
    torch.load = lambda *a, **kw: _load(*a, **{**kw, "weights_only": False})  # type: ignore
    try:
        model.load_ckpt(ckpt_path, verbose=False)
    finally:
        torch.load = _load
    model = model.to(torch.device(device)).eval()
    with torch.no_grad():
        t = torch.tensor(model.get_text_embedding([template.format(p=x) for x in texts]))
        a = torch.tensor(model.get_audio_embedding_from_filelist(x=list(wav_paths)))
        a = torch.nn.functional.normalize(a, p=2, dim=1)
        t = torch.nn.functional.normalize(t, p=2, dim=1)
    return (a @ t.t()).float().numpy()
