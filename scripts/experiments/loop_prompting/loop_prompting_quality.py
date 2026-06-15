"""Does loop-focused prompting improve quality? — paired A/B harness.

Science task: "Determine if loop-focused prompting has any bearing on
quality" — e.g. prepending ``a short perfect loop of <prompt>``.

DEMON's loop-focused workflow loops a single generated section. The
hypothesis is that telling the model up front that the section is a
*loop* makes it (a) sound better as a standalone clip and (b) wrap more
seamlessly when the engine repeats it.

Design — fully paired so the only thing that varies is prompt wording:
for each base prompt we render every *template* (baseline vs. loop
prefixes) at the same fixed seeds, same duration, same diffusion knobs,
pure text-to-music (no source audio, so the prompt is the only driver).
Each render is scored with objective loop/quality *proxies* and saved as
a WAV for human listening — the proxies rank candidates, ears decide.

Proxies (per clip, see ``clip_metrics``):
  - ``seam_spec_dist``  spectral-profile L2 between the head and tail
    windows; low ⇒ the texture at the loop point matches ⇒ smoother wrap.
  - ``seam_rms_jump_db`` |loudness(head) − loudness(tail)|; energy step
    you hear at the wrap.
  - ``silence_frac``    fraction of near-silent frames; flags dropouts /
    empty renders.
  - ``rms_db`` / ``centroid_hz``  loudness / brightness sanity.

Run (Linux):
    .venv/bin/python scripts/experiments/loop_prompting/loop_prompting_quality.py
Run (Windows):
    .venv/Scripts/python.exe scripts\\experiments\\loop_prompting\\loop_prompting_quality.py

Outputs WAVs + metrics.json under test_output/experiments/loop_prompting/
(both gitignored) and prints a markdown summary for the results doc.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(
    p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists()
)
# Force OUR repo to the front (a sibling ACE-Step checkout shadows `acestep`).
while str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS  # noqa: E402
from acestep.engine.session import Session  # noqa: E402
from acestep.nodes.types import Curve  # noqa: E402


# Base prompts span genres where "loopiness" reads differently (groove
# vs. evolving texture). All instrumental so lyrics never confound.
BASE_PROMPTS = [
    {"slug": "lofi", "tags": "lo-fi hip hop, mellow piano, vinyl crackle, boom bap drums",
     "bpm": 80, "key": "C minor"},
    {"slug": "deephouse", "tags": "deep house, four on the floor, warm analog bass, hypnotic synth stabs",
     "bpm": 124, "key": "A minor"},
    {"slug": "ambient", "tags": "ambient, evolving warm pads, soft texture, no drums",
     "bpm": 90, "key": "D major"},
    {"slug": "techno", "tags": "driving techno, punchy kick, dark hypnotic groove, modular bleeps",
     "bpm": 130, "key": "F minor"},
]

# The arm under test is "loop_perfect" (the exact phrasing the task names);
# "loop_seamless" probes whether wording matters; "baseline" is the control.
TEMPLATES = {
    "baseline": "{p}",
    "loop_perfect": "a short perfect loop of {p}",
    "loop_seamless": "seamless repeating loop of {p}",
}

SEEDS = [1528, 42, 9999]
DURATION = 60.0
STEPS = 8
SHIFT = 3.0
CFG = 7.5

SR = 48000
WIN_S = 0.5  # head/tail window used for the seam proxies


def _stats(xs):
    xs = sorted(xs)
    n = len(xs)
    return {"mean": round(sum(xs) / n, 4), "min": round(xs[0], 4),
            "p50": round(xs[n // 2], 4), "max": round(xs[-1], 4), "n": n}


def _mono(wav: torch.Tensor) -> torch.Tensor:
    """[*, C, N] or [C, N] float waveform -> mono [N] on CPU float32."""
    w = wav.detach().float().cpu()
    while w.dim() > 2:
        w = w.squeeze(0)
    if w.dim() == 2:
        w = w.mean(0)
    return w


def _log_spec_profile(seg: torch.Tensor) -> torch.Tensor:
    """Time-averaged log-magnitude STFT profile of a 1-D segment."""
    n_fft = 2048
    if seg.numel() < n_fft:
        seg = torch.nn.functional.pad(seg, (0, n_fft - seg.numel()))
    spec = torch.stft(seg, n_fft=n_fft, hop_length=512,
                      window=torch.hann_window(n_fft), return_complex=True)
    mag = spec.abs().mean(dim=1)  # average over time -> [freq]
    return torch.log1p(mag)


def clip_metrics(wav: torch.Tensor) -> dict:
    """Loop/quality proxies for one decoded clip. See module docstring."""
    m = _mono(wav)
    n = m.numel()
    w = min(int(WIN_S * SR), n // 2)
    head, tail = m[:w], m[-w:]

    # Seam: how alike are the texture/energy across the wrap point.
    hp, tp = _log_spec_profile(head), _log_spec_profile(tail)
    seam_spec_dist = float(torch.linalg.vector_norm(hp - tp) / math.sqrt(hp.numel()))
    rms = lambda x: float(torch.sqrt(torch.clamp((x ** 2).mean(), min=1e-12)))
    to_db = lambda r: 20.0 * math.log10(max(r, 1e-9))
    seam_rms_jump_db = abs(to_db(rms(head)) - to_db(rms(tail)))

    # Frame-wise activity: catch dropouts / empty renders at the seam or globally.
    fr = 0.05  # 50 ms frames
    fl = int(fr * SR)
    frames = m[: (n // fl) * fl].reshape(-1, fl)
    frame_rms_db = torch.tensor([to_db(rms(f)) for f in frames])
    silence_frac = float((frame_rms_db < -45.0).float().mean())

    # Brightness (spectral centroid) as a coarse degradation sanity check.
    prof = _log_spec_profile(m)
    freqs = torch.linspace(0, SR / 2, prof.numel())
    centroid_hz = float((freqs * prof).sum() / torch.clamp(prof.sum(), min=1e-9))

    return {
        "seam_spec_dist": round(seam_spec_dist, 5),
        "seam_rms_jump_db": round(seam_rms_jump_db, 3),
        "silence_frac": round(silence_frac, 4),
        "rms_db": round(to_db(rms(m)), 2),
        "centroid_hz": round(centroid_hz, 1),
    }


def save_wav(wav: torch.Tensor, path: Path) -> None:
    import soundfile as sf
    w = wav.detach().float().cpu()
    while w.dim() > 2:
        w = w.squeeze(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), w.T.numpy(), SR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--shift", type=float, default=SHIFT)
    ap.add_argument("--cfg", type=float, default=CFG)
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "test_output" / "experiments" / "loop_prompting")
    args = ap.parse_args()

    t0 = time.time()
    print("[1] loading session (eager)...", flush=True)
    session = Session()
    print(f"    ready in {time.time() - t0:.1f}s", flush=True)

    T = int(round(args.duration * 25))
    guidance = Curve(tensor=torch.full((T,), args.cfg, dtype=torch.bfloat16))

    rows = []
    for bp in BASE_PROMPTS:
        for tname, tmpl in TEMPLATES.items():
            tags = tmpl.format(p=bp["tags"])
            cond = session.encode_text(
                tags=tags, lyrics="[instrumental]",
                instruction=TASK_INSTRUCTIONS["text2music"],
                bpm=bp["bpm"], duration=args.duration, key=bp["key"],
            )
            neg = session.null_conditioning(cond)
            for seed in args.seeds:
                tg = time.time()
                lat = session.generate(
                    conditioning=cond, negative=neg, guidance_curve=guidance,
                    seed=seed, duration=args.duration, steps=args.steps, shift=args.shift,
                )
                gen_s = time.time() - tg
                audio = session.decode(lat)
                met = clip_metrics(audio.waveform)
                wav_path = args.out / bp["slug"] / f"{tname}__seed{seed}.wav"
                save_wav(audio.waveform, wav_path)
                row = {"prompt": bp["slug"], "template": tname, "seed": seed,
                       "gen_s": round(gen_s, 3), **met}
                rows.append(row)
                print(f"  {bp['slug']:9s} {tname:13s} seed={seed:<5d} "
                      f"seam_spec={met['seam_spec_dist']:.4f} "
                      f"seam_db={met['seam_rms_jump_db']:.2f} "
                      f"sil={met['silence_frac']:.3f}", flush=True)

    # Aggregate per template across all prompts+seeds -> the headline comparison.
    agg = {}
    for tname in TEMPLATES:
        sub = [r for r in rows if r["template"] == tname]
        agg[tname] = {
            "seam_spec_dist": _stats([r["seam_spec_dist"] for r in sub]),
            "seam_rms_jump_db": _stats([r["seam_rms_jump_db"] for r in sub]),
            "silence_frac": _stats([r["silence_frac"] for r in sub]),
        }

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"seeds": args.seeds, "duration": args.duration,
                   "steps": args.steps, "shift": args.shift, "cfg": args.cfg,
                   "templates": TEMPLATES, "n_prompts": len(BASE_PROMPTS)},
        "rows": rows, "aggregate": agg,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(out, indent=2))

    # Markdown summary for the results doc.
    print("\n### Aggregate (mean across prompts × seeds; lower seam = smoother loop)\n")
    print("| template | seam_spec_dist | seam_rms_jump_db | silence_frac |")
    print("|---|---|---|---|")
    for tname, a in agg.items():
        print(f"| {tname} | {a['seam_spec_dist']['mean']} | "
              f"{a['seam_rms_jump_db']['mean']} | {a['silence_frac']['mean']} |")
    print(f"\nWAVs + metrics.json under {args.out}")


if __name__ == "__main__":
    main()
