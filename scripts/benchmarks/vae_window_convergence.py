"""Measure the real (Oobleck) VAE decoder's boundary receptive field.

Question driving this: the streaming decoder keeps a center window and
throws away an overlap *margin* on each side so boundary frames see real
context instead of zero-pad. The default margin is 0.5 s (12 frames) and
was probably chosen as a safe round number. How small can it actually go
before the kept center diverges from a full-context decode?

Method (pure windowing error, no TRT, fp32 reference):

  * Load a real fixture latent (not random noise).
  * Pick an interior center frame ``c``.
  * Ground truth: decode a *wide* span [c-G, c+G] (G frames >> any
    plausible receptive field). The center of that decode sees full
    context on both sides, so it is the artifact-free reference.
  * For each candidate margin ``m`` (in frames), decode the narrow span
    [keep_start-m, keep_end+m], trim back to the kept center, and compare
    that center to the ground-truth center.

As ``m`` grows the error collapses to the fp32 noise floor; the smallest
``m`` that reaches the floor IS the boundary receptive field. Everything
is reported in ms (m frames * 40 ms) so it maps straight onto the
``vae_overlap`` knob.

Usage::

    uv run python scripts/benchmarks/vae_window_convergence.py
    uv run python scripts/benchmarks/vae_window_convergence.py --keep-frames 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force the DEMON repo to the FRONT of sys.path. A sibling ACE-Step is
# editable-installed and shadows ``acestep`` (it lacks fixtures.py); a
# plain "if not in sys.path" guard is not enough because the repo can
# already be present but *behind* the sibling.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import torch

from acestep.fixtures import fixture_sidecar
from acestep.paths import checkpoints_dir

FRAMES_PER_SEC = 25
SAMPLES_PER_FRAME = 1920
MS_PER_FRAME = 1000.0 / FRAMES_PER_SEC  # 40 ms

# Margins to sweep, in latent frames. 12 frames = 480 ms ~= the current
# 0.5 s default; 16 = 640 ms gives headroom past it.
MARGIN_FRAMES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20)


def load_real_vae(device: torch.device) -> torch.nn.Module:
    from diffusers.models import AutoencoderOobleck

    vae_path = checkpoints_dir() / "vae"
    if not vae_path.is_dir():
        raise FileNotFoundError(f"real VAE not found at {vae_path}")
    vae = AutoencoderOobleck.from_pretrained(str(vae_path))
    vae = vae.to(device).to(torch.float32).eval()
    return vae


@torch.no_grad()
def decode(vae: torch.nn.Module, lat_bdt: torch.Tensor) -> torch.Tensor:
    """Monolithic fp32 decode of a [1, 64, T] latent -> [1, 2, T*1920]."""
    out = vae.decode(lat_bdt)
    return out.sample.float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture", default="inside_confusion_loop_60s_gsm.wav")
    ap.add_argument("--keep-frames", type=int, default=8,
                    help="center keep window in frames (8 = 320 ms ~= 333 ms ask)")
    ap.add_argument("--ground-truth-margin", type=int, default=200,
                    help="frames of context each side for the reference decode")
    ap.add_argument("--center", type=int, default=0,
                    help="center frame; 0 = auto (latent midpoint)")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    sc = fixture_sidecar(args.fixture)
    if sc is None:
        raise SystemExit(f"fixture sidecar not available: {args.fixture}")
    # sidecar latent is [B, T, D]; VAE decode wants [B, D, T].
    lat_btd = sc.latent.float()
    T = lat_btd.shape[1]
    lat_bdt = lat_btd.transpose(1, 2).contiguous().to(device)  # [1, 64, T]
    print(f"fixture: {args.fixture}  latent T={T} frames ({T/FRAMES_PER_SEC:.1f}s)")

    keep = args.keep_frames
    G = args.ground_truth_margin
    c = args.center or (T // 2)
    keep_start = c - keep // 2
    keep_end = keep_start + keep
    if keep_start - max(MARGIN_FRAMES) < 0 or keep_end + max(MARGIN_FRAMES) > T:
        raise SystemExit("center too close to latent edge for the sweep")
    if keep_start - G < 0 or keep_end + G > T:
        raise SystemExit("ground-truth margin exceeds available latent context")

    print(f"keep window: frames [{keep_start},{keep_end}) = {keep} frames "
          f"({keep*MS_PER_FRAME:.0f} ms)")

    vae = load_real_vae(device)

    # --- Ground truth: wide-context decode, extract the kept center. ---
    gt_start, gt_end = keep_start - G, keep_end + G
    gt_audio = decode(vae, lat_bdt[:, :, gt_start:gt_end])
    gt_off = (keep_start - gt_start) * SAMPLES_PER_FRAME
    gt_center = gt_audio[:, :, gt_off:gt_off + keep * SAMPLES_PER_FRAME].clone()
    gt_flat = gt_center.reshape(-1)
    gt_rms = gt_center.pow(2).mean().sqrt().item()
    print(f"ground-truth context: {G} frames/side ({G*MS_PER_FRAME:.0f} ms); "
          f"center RMS={gt_rms:.4f}\n")

    # --- Sweep margins. ---
    rows = []
    print(f"{'margin_fr':>9}{'margin_ms':>10}{'decode_fr':>10}"
          f"{'max_diff':>12}{'mean_diff':>12}{'cos':>12}{'snr_dB':>9}")
    print("-" * 74)
    for m in MARGIN_FRAMES:
        d_start, d_end = keep_start - m, keep_end + m
        audio = decode(vae, lat_bdt[:, :, d_start:d_end])
        off = m * SAMPLES_PER_FRAME
        center = audio[:, :, off:off + keep * SAMPLES_PER_FRAME]
        diff = (center - gt_center).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        cos = torch.nn.functional.cosine_similarity(
            center.reshape(1, -1), gt_flat.reshape(1, -1)
        ).item()
        err_rms = (center - gt_center).pow(2).mean().sqrt().item()
        snr = 20.0 * torch.log10(torch.tensor(gt_rms / (err_rms + 1e-12))).item()
        rows.append({
            "margin_frames": m, "margin_ms": round(m * MS_PER_FRAME, 1),
            "decode_frames": keep + 2 * m,
            "max_diff": max_diff, "mean_diff": mean_diff,
            "cos": cos, "snr_db": snr,
        })
        print(f"{m:>9}{m*MS_PER_FRAME:>10.0f}{keep+2*m:>10}"
              f"{max_diff:>12.2e}{mean_diff:>12.2e}{cos:>12.8f}{snr:>9.1f}")

    # --- Verdicts: smallest margin under common thresholds. ---
    print()
    # fp16 wire LSB near a 0.3-RMS signal is ~2e-4; 16-bit PCM LSB ~3e-5.
    for label, thr in (("max_diff<1e-2", 1e-2), ("max_diff<1e-3", 1e-3),
                       ("max_diff<3e-4 (fp16 wire)", 3e-4),
                       ("max_diff<3e-5 (16-bit PCM)", 3e-5)):
        hit = next((r for r in rows if r["max_diff"] < thr), None)
        if hit:
            print(f"  {label:<28} -> margin {hit['margin_frames']} fr "
                  f"({hit['margin_ms']:.0f} ms), decode {hit['decode_frames']} fr")
        else:
            print(f"  {label:<28} -> not reached within {max(MARGIN_FRAMES)} fr")

    out = Path(PROJECT_ROOT) / "scripts" / "benchmarks" / "vae_window_convergence.json"
    out.write_text(json.dumps({
        "fixture": args.fixture, "keep_frames": keep,
        "keep_ms": keep * MS_PER_FRAME, "center_frame": c,
        "ground_truth_margin_frames": G, "rows": rows,
    }, indent=2))
    print(f"\nJSON -> {out}")


if __name__ == "__main__":
    main()
