"""Live check of TADA packs on a running DEMON server (127.0.0.1:1318).

Headless sessions like the /sa3 page (server fixture, prompt, 125 Hz
params). Checks: the ready message's knob manifest carries ``steer_*``
knobs; two sessions with no steer value and with the knob explicitly at
0 produce identical audio per slice position (zero is a no-op); the knob
at +/- full scale changes the audio. Reports per-session onset rate and
spectral centroid over the slices received after the first ``--settle``
seconds.

    .venv/Scripts/python.exe scripts/tada/sa3_tada_live_check.py --knob steer_tempo
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from demos.realtime_motion_graph_web.protocol import (  # noqa: E402
    SLICE_FLAG_DELTA, SLICE_HDR_FMT, SLICE_HDR_SIZE,
)

SR = 48000


def session(url, raw, secs, prompt, fixture):
    ws = connect(url, open_timeout=30, max_size=None)
    ws.send(json.dumps({"fixture_name": fixture, "use_server_fixture": True,
                        "prompt": prompt, "depth": 4, "steps": 8}))
    manifest, duration = {}, None
    while True:
        msg = ws.recv(timeout=300)
        if isinstance(msg, str):
            d = json.loads(msg)
            if d.get("type") == "ready":
                duration = float(d["duration"])
                manifest = (d.get("knob_manifest") or {}).get("knobs", {})
        else:
            break
    t0 = time.monotonic()
    slices, stop = {}, [False]

    def rx():
        pending = 0
        while not stop[0]:
            try:
                m = ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            except Exception:
                break
            if isinstance(m, str):
                d = json.loads(m)
                if d.get("type") == "stem_assets":
                    pending = len(d.get("stems", []))
                continue
            if pending:
                pending -= 1
                continue
            if len(m) <= SLICE_HDR_SIZE:
                continue
            h = struct.unpack(SLICE_HDR_FMT, m[:SLICE_HDR_SIZE])
            p = m[SLICE_HDR_SIZE:]
            if h[0] == SLICE_FLAG_DELTA:
                import zstandard as zstd  # same decoding as protocol.RemoteBackend.recv
                p = zstd.decompress(p)
            a = np.frombuffer(p, dtype=np.float16).astype(np.float32).reshape(h[2], h[3])
            slices.setdefault(h[1], []).append((time.monotonic() - t0, a))

    def tx():
        while not stop[0]:
            ws.send(json.dumps({"type": "params", "raw": dict(raw),
                                "playback_pos": (time.monotonic() - t0) % duration,
                                "client_time": time.monotonic()}))
            time.sleep(0.008)

    th = [threading.Thread(target=f, daemon=True) for f in (rx, tx)]
    for t in th:
        t.start()
    time.sleep(secs)
    stop[0] = True
    for t in th:
        t.join(timeout=3)
    ws.close()
    print(f"  session {raw}: {sum(len(v) for v in slices.values())} slices", flush=True)
    return manifest, slices


def features(slices, settle):
    import librosa

    late = [a for s, v in sorted(slices.items()) for (t, a) in v if t >= settle]
    if not late:
        return {}
    x = np.concatenate([a.mean(axis=1) for a in late])
    on = librosa.onset.onset_detect(y=x, sr=SR, units="time")
    return {"seconds": round(len(x) / SR, 1), "onsets_per_s": round(len(on) / (len(x) / SR), 2),
            "centroid_hz": round(float(librosa.feature.spectral_centroid(y=x, sr=SR).mean()), 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", default="steer_tempo")
    ap.add_argument("--secs", type=float, default=30)
    ap.add_argument("--settle", type=float, default=8)
    ap.add_argument("--prompt", default="lo-fi hip hop, mellow")
    ap.add_argument("--fixture", default="low_fi_Gm_loop_60s_gnm.wav")
    ap.add_argument("--url", default="ws://127.0.0.1:1318/")
    ap.add_argument("--quick", action="store_true", help="determinism control only")
    args = ap.parse_args()
    runs = {"unset": {}, "unset2": {}, "zero": {args.knob: 0.0}, "plus": {args.knob: 30.0},
            "minus": {args.knob: -30.0}}
    if args.quick:
        runs = {"unset": {}, "unset2": {}}
    out, sl = {}, {}
    for name, raw in runs.items():
        manifest, slices = session(args.url, raw, args.secs, args.prompt, args.fixture)
        sl[name] = slices
        out[name] = features(slices, args.settle)
        if name == "unset":
            out["steer_knobs"] = sorted(k for k in manifest if k.startswith("steer_"))
        print(name, out[name], flush=True)

    def compare(a, b):
        same = diff = 0
        for s in set(sl[a]) & set(sl[b]):
            x, y = sl[a][s][0][1], sl[b][s][0][1]
            if x.shape == y.shape and np.array_equal(x, y):
                same += 1
            else:
                diff += 1
        return {"identical": same, "different": diff}

    # Control: two identical sessions. Only if these match slice for slice
    # is the cross-session zero comparison meaningful.
    out["unset_vs_unset2"] = compare("unset", "unset2")
    if args.quick:
        print(json.dumps(out["unset_vs_unset2"]))
        return 0
    out["unset_vs_zero"] = compare("unset", "zero")
    out["unset_vs_plus"] = compare("unset", "plus")
    out["unset_vs_minus"] = compare("unset", "minus")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
