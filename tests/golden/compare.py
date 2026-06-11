"""Compare a scenario run against its canonical reference bundle.

Two tiers:

* **Tier 1: identity.** sha256 of ``canonical.f32.raw`` matches the
  reference: the run is byte-identical, nothing else to discuss.
* **Tier 2: tolerance.** When hashes differ (different GPU, driver,
  engine build, or a nondeterministic path), score perceptual/numeric
  distance and compare against per-scenario thresholds. Thresholds live
  in the refs manifest and MUST be calibrated from observed same-code
  variance (``runner --repeat``), never guessed: see README.

Metrics are computed on the canonical generation stream (mono mixdown):

* ``mel_l2``      mean L2 distance between log-mel frames (the workhorse)
* ``rms_db_diff`` overall level shift in dB
* ``win_cos_min`` worst per-second spectral cosine (localizes a glitch
                  that a global mean would average away)
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .client import SAMPLE_RATE

# Calibration fallbacks used only when a manifest entry carries no
# thresholds yet; deliberately strict so an uncalibrated comparison
# fails loudly rather than silently passing.
DEFAULT_THRESHOLDS = {"mel_l2": 0.10, "rms_db_diff": 0.5,
                      "win_cos_min": 0.995}

# 1s windows straddling a scenario action are masked from the
# win_cos_min gate (action window ± this many neighbours). A knob /
# prompt / swap firing mid-region is an intentional discontinuity, and
# exactly where it catches a pipeline block boundary is stable within a
# capture session but drifts between sessions (measured 2026-06-04:
# knob_step win_cos_min 0.98532 vs the same-build morning ref, cos
# 1.000000 everywhere outside the transition windows) — spectral
# identity there is not owed. The masked windows' worst cosine is still
# reported (win_cos_min_action) so a genuine glitch stays visible.
ACTION_WINDOW_PAD = 1

# log-mel framing (kept in one place so the mel_l2 action mask can map a
# 1 s window index to the frames it covers). A frame at hop index j is
# centred on sample j*MEL_HOP, i.e. inside 1 s window j*MEL_HOP // SR.
MEL_HOP = 512


def action_window_indices(bundle_dir: Path,
                          pad: int = ACTION_WINDOW_PAD) -> set:
    """Region-relative 1s-window indices covered by the actions recorded
    in a bundle's metrics.json, padded ±pad windows. Empty when the
    bundle has no actions (baseline scenarios: nothing is masked)."""
    try:
        meta = json.loads((Path(bundle_dir) / "metrics.json")
                          .read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()
    start_s = float((meta.get("canonical_region") or {})
                    .get("start_s", 0.0))
    idxs: set = set()
    for act in meta.get("actions") or []:
        at = act.get("frontier_s", act.get("at_s"))
        if at is None:
            continue
        w = int(at - start_s)
        idxs.update(range(w - pad, w + pad + 1))
    return {i for i in idxs if i >= 0}


def load_canonical(bundle_dir: Path) -> np.ndarray:
    raw = (Path(bundle_dir) / "canonical.f32.raw").read_bytes()
    metrics = json.loads(
        (Path(bundle_dir) / "metrics.json").read_text(encoding="utf-8"))
    channels = int(metrics.get("ready", {}).get("channels", 2))
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, channels)


def sha256_of(bundle_dir: Path) -> str:
    return hashlib.sha256(
        (Path(bundle_dir) / "canonical.f32.raw").read_bytes()).hexdigest()


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _log_mel(mono: np.ndarray) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=mono, sr=SAMPLE_RATE, n_fft=2048, hop_length=MEL_HOP, n_mels=64)
    return librosa.power_to_db(mel, ref=np.max)


def audio_metrics(ref: np.ndarray, run: np.ndarray,
                  action_windows: set | frozenset = frozenset()) -> dict:
    """Tier-2 distance metrics between two canonical streams.

    ``action_windows``: region-relative 1s-window indices excluded from
    the ``win_cos_min`` gate (see ``action_window_indices``); their worst
    cosine is reported separately as ``win_cos_min_action``."""
    n = min(ref.shape[0], run.shape[0])
    a, b = _mono(ref[:n]), _mono(run[:n])

    eps = 1e-12
    rms_db_diff = abs(
        20 * np.log10(np.sqrt(np.mean(a ** 2)) + eps)
        - 20 * np.log10(np.sqrt(np.mean(b ** 2)) + eps))

    ma, mb = _log_mel(a), _log_mel(b)
    f = min(ma.shape[1], mb.shape[1])
    # Per-frame L2 across mel bins, normalized by the mel dimensionality so
    # the number is scale-comparable across n_mels choices.
    per_frame = (np.linalg.norm(ma[:, :f] - mb[:, :f], axis=0)
                 / np.sqrt(ma.shape[0]))
    # Mask the same action windows the cosine gate masks: a knob / prompt /
    # swap firing mid-region is an intentional discontinuity and is not owed
    # spectral identity here either. Without this, mel_l2 (averaged over the
    # whole stream) silently re-counts the very frames win_cos_min excludes,
    # so a clean run trips the global mel_l2 gate purely on the transition
    # (knob_step: ~0.105 from windows it was told to ignore). The masked
    # frames' mean is still reported as mel_l2_action so a real glitch in a
    # transition stays visible. With no actions the mask is empty and this
    # is bit-identical to the unmasked mean.
    frame_win = (np.arange(f) * MEL_HOP) // SAMPLE_RATE
    mel_gate = np.array([int(w) not in action_windows for w in frame_win])
    mel_l2 = float(per_frame[mel_gate].mean()) if mel_gate.any() else 0.0
    mel_l2_action = (float(per_frame[~mel_gate].mean())
                     if (~mel_gate).any() else None)

    win = SAMPLE_RATE  # 1 s windows
    cos = []
    for s in range(0, n - win + 1, win):
        fa = np.abs(np.fft.rfft(a[s:s + win]))
        fb = np.abs(np.fft.rfft(b[s:s + win]))
        denom = np.linalg.norm(fa) * np.linalg.norm(fb)
        cos.append(float(np.dot(fa, fb) / denom) if denom > eps else 1.0)

    gated = [c for i, c in enumerate(cos) if i not in action_windows]
    masked = [c for i, c in enumerate(cos) if i in action_windows]

    return {
        "compared_samples": int(n),
        "len_ref": int(ref.shape[0]),
        "len_run": int(run.shape[0]),
        "mel_l2": round(mel_l2, 5),
        # informational: mel distance over the masked transition frames
        "mel_l2_action": (round(mel_l2_action, 5)
                          if mel_l2_action is not None else None),
        "rms_db_diff": round(float(rms_db_diff), 4),
        "win_cos_min": round(min(gated), 6) if gated else None,
        "win_cos_mean": round(float(np.mean(gated)), 6) if gated else None,
        # informational: never silently drop what was masked
        "action_windows": sorted(i for i in action_windows if i < len(cos)),
        "win_cos_min_action": round(min(masked), 6) if masked else None,
    }


def compare_bundles(ref_dir: Path, run_dir: Path,
                    thresholds: dict | None = None) -> dict:
    """Full comparison report. ``passed`` is True on tier-1 identity or
    on every tier-2 metric landing inside its threshold."""
    report: dict = {
        "ref_sha256": sha256_of(ref_dir),
        "run_sha256": sha256_of(run_dir),
    }
    if report["ref_sha256"] == report["run_sha256"]:
        report.update(tier=1, identical=True, passed=True)
        return report

    # `calibrated` records whether these thresholds came from a
    # same-build variance probe (runner --repeat) or are the strict
    # uncalibrated fallbacks. A tier-2 failure against fallbacks on
    # hardware that differs from the capture box is far more likely
    # hardware variance than a real regression — callers surface this.
    calibrated = bool(thresholds)
    th = dict(DEFAULT_THRESHOLDS)
    th.update(thresholds or {})
    # Union of both bundles' action windows: if the trigger landed on a
    # window boundary in only one of them, both placements are masked.
    aw = action_window_indices(ref_dir) | action_window_indices(run_dir)
    m = audio_metrics(load_canonical(ref_dir), load_canonical(run_dir),
                      action_windows=aw)
    failures = []
    if m["len_run"] < m["len_ref"]:
        failures.append(f"run shorter than ref ({m['len_run']} < "
                        f"{m['len_ref']} samples)")
    if m["mel_l2"] > th["mel_l2"]:
        failures.append(f"mel_l2 {m['mel_l2']} > {th['mel_l2']}")
    if m["rms_db_diff"] > th["rms_db_diff"]:
        failures.append(f"rms_db_diff {m['rms_db_diff']} > "
                        f"{th['rms_db_diff']}")
    if m["win_cos_min"] is not None and m["win_cos_min"] < th["win_cos_min"]:
        failures.append(f"win_cos_min {m['win_cos_min']} < "
                        f"{th['win_cos_min']}")
    report.update(tier=2, identical=False, metrics=m,
                  thresholds=th, calibrated=calibrated,
                  failures=failures, passed=not failures)
    return report
