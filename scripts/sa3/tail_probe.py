"""Tail probe: does an SA3 loop end in a fade/silence?

Runs headless PRIMARY sessions against a local DEMON pod, one per
(sa3_denoise, seed), using a server fixture (default: DreamSampler's
default track) as the source, waits for a full lap of fresh audio, then
reports the per-second RMS of the loop's last seconds against its body.
A run is flagged ENDING when the last 3 s sit more than 8 dB under the
body median. Every lap is also written to ``~/tail_probe_out/`` as a
wav for listening, plus a JSON summary per label.

  # 30 s cut of the fixture, both product denoise regimes, 3 seeds
  python scripts/sa3/tail_probe.py --label legacy --duration 30 \
      --denoise 1.0 0.9 --seeds 1 2 3
  # the fixture at its own length (what the plugin does): --duration 0
  python scripts/sa3/tail_probe.py --label full --duration 0 --denoise 0.9

This is the measurement behind the song-length label
(``DEMON_SA3_SONG_SECONDS``, see ``acestep/engine/sa3_context.py``):
restart the pod with the env var to A/B, then run the probe under a
new ``--label``.
"""

import argparse
import json
import os
import sys
import time
import wave

import numpy as np

# Repo root FIRST on sys.path (AGENTS.md: a sibling ACE-Step checkout can
# shadow ``acestep`` / ``demos`` otherwise).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from demos.realtime_motion_graph_web.headless_client import HeadlessClient  # noqa: E402

SR = 48000
PROMPT = ("upbeat electronic dance track, driving four-on-the-floor drums, "
          "punchy synth bass, bright arpeggios, 124 bpm")


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = (0.25 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    return np.stack([x, x])


def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-6))


def run_one(args, denoise: float, seed: int) -> dict:
    cfg = {
        "prompt": PROMPT, "sde": False, "lora": False,
        "depth": 4, "steps": 8,
        "client_id": f"tail-probe-{args.label}",
    }
    if args.duration > 0:
        cfg["sa3_duration_s"] = float(args.duration)
    wav = None
    if args.synthetic:
        wav = tone(args.duration)
    else:
        cfg["use_server_fixture"] = True
        cfg["fixture_name"] = args.fixture
    client = HeadlessClient(args.url, cfg, wav)
    client._raw = {"sa3_denoise": float(denoise), "seed": int(seed)}
    t0 = time.time()
    ready = client.start(timeout_s=900)
    dur = float(ready["duration"])
    settle = args.settle if args.settle > 0 else dur + 20.0
    while time.time() - t0 < settle and client.running:
        time.sleep(0.5)
    if not client.running:
        print(f"FAIL client died: {client.closed_reason}", flush=True)
        client.stop()
        return {"denoise": denoise, "seed": seed, "error": client.closed_reason}
    m = client.mirror.copy()
    slices = client.slice_count
    client.stop()

    mono = m.mean(axis=1)
    n_bins = int(len(mono) // SR)
    bins = [db(float(np.sqrt((mono[i * SR:(i + 1) * SR] ** 2).mean())))
            for i in range(n_bins)]
    body = bins[2:max(3, n_bins - 10)]
    body_med = float(np.median(body))
    tail8 = bins[-8:]
    last3 = float(np.mean(bins[-3:]))
    drop = body_med - last3
    ending = drop > 8.0
    out_dir = os.path.expanduser("~/tail_probe_out")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.label}_d{denoise:.2f}_s{seed}_{int(dur)}s.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(m.shape[1]); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(m, -1, 1) * 32767).astype("<i2").tobytes())
    rec = {
        "label": args.label, "denoise": denoise, "seed": seed, "duration": dur,
        "slices": slices, "body_median_db": round(body_med, 1),
        "last3_mean_db": round(last3, 1), "drop_db": round(drop, 1),
        "ending": ending,
        "tail8_db": [round(b, 1) for b in tail8], "wav": path,
    }
    print(
        f"[{args.label}] denoise={denoise:.2f} seed={seed} dur={dur:.1f}s "
        f"slices={slices} body={body_med:.1f}dB last3={last3:.1f}dB "
        f"drop={drop:+.1f}dB {'ENDING' if ending else 'ok'} "
        f"tail8={[round(b) for b in tail8]}", flush=True,
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", default="ws://127.0.0.1:1318/")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="sa3_duration_s (a cut of the fixture); 0 = the fixture's own length")
    ap.add_argument("--denoise", type=float, nargs="+", default=[1.0, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--settle", type=float, default=0.0,
                    help="seconds to wait per session (default: duration + 20)")
    ap.add_argument("--fixture", default="low_fi_Gm_loop_60s_gnm.wav")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    recs = []
    for d in args.denoise:
        for s in args.seeds:
            recs.append(run_one(args, d, s))
            time.sleep(1.0)
    good = [r for r in recs if "error" not in r]
    n_end = sum(1 for r in good if r["ending"])
    mean_drop = float(np.mean([r["drop_db"] for r in good])) if good else float("nan")
    print(f"SUMMARY label={args.label} runs={len(good)} endings={n_end} "
          f"mean_drop={mean_drop:+.1f}dB", flush=True)
    with open(os.path.expanduser(f"~/tail_probe_out/{args.label}.json"), "w") as f:
        json.dump(recs, f, indent=1)


main()
