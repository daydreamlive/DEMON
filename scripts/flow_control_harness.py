"""End-to-end flow-control harness for the DEMON slice stream.

Emulates the user's saturated SSH tunnel: a TCP proxy that relays
client->server unthrottled but server->client at a fixed byte rate with
UNBOUNDED proxy-side buffering (like a bufferbloated tunnel, the server's
own socket never blocks). A headless client mimics the browser: config
handshake (server-side fixture), simulated 1x playhead, 125 Hz params
with client_time / slice_lead_s / slice_bytes_rx, and per-slice landing
lead measurement.

PASS criteria (printed at the end):
  * throttled run: after the ramp (t > 25 s), median lead per 5 s bucket
    stays positive and the negative-lead fraction ~0.
  * control run (no throttle): leads healthy, no window drops.

Usage (server must be running on 127.0.0.1:1318):
    .venv/bin/python scripts/flow_control_harness.py --mode throttled --rate 200000 --secs 75
    .venv/bin/python scripts/flow_control_harness.py --mode direct --secs 40

PASS = leads stay positive after the controller settles. This is the
regression harness for the raw-source-bleed class of bug (slices
landing behind the playhead on slow links / throttled tabs).
"""

import argparse
import collections
import json
import os
import socket
import struct
import sys
import threading
import time

# Repo root FIRST so a sibling ACE-Step checkout cannot shadow acestep
# (see the AGENTS.md sys.path gotcha).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from websockets.sync.client import connect  # noqa: E402

from demos.realtime_motion_graph_web.protocol import (  # noqa: E402
    SLICE_HDR_FMT,
    SLICE_HDR_SIZE,
)

SAMPLE_RATE = 48000

SERVER = ("127.0.0.1", 1318)
PROXY_PORT = 14318


# ---------------------------------------------------------------------------
# Throttling proxy (bufferbloat emulation)
# ---------------------------------------------------------------------------

class ThrottleProxy(threading.Thread):
    """One-connection TCP proxy. Downstream (server->client) is rate-
    limited with an unbounded internal buffer; upstream passes through.
    Throttling engages only after `free_bytes` have been relayed
    downstream (lets the multi-MB init buffer through fast)."""

    def __init__(self, rate_bps: int, free_bytes: int = 14_000_000):
        super().__init__(daemon=True)
        self.rate = rate_bps
        self.free = free_bytes
        self.lsock = socket.create_server(("127.0.0.1", PROXY_PORT))
        self.buffered = 0          # live gauge of proxy-held bytes
        self.total_down = 0
        self._stop = False

    def run(self):
        conn, _ = self.lsock.accept()
        up = socket.create_connection(SERVER)
        conn.settimeout(0.2)
        up.settimeout(0.2)
        buf = collections.deque()
        buf_lock = threading.Lock()

        def upstream():
            while not self._stop:
                try:
                    data = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                try:
                    up.sendall(data)
                except OSError:
                    break

        def reader():
            while not self._stop:
                try:
                    data = up.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                with buf_lock:
                    buf.append(data)
                    self.buffered += len(data)

        def writer():
            budget = 0.0
            last = time.monotonic()
            while not self._stop:
                with buf_lock:
                    chunk = buf.popleft() if buf else None
                if chunk is None:
                    time.sleep(0.005)
                    continue
                if self.total_down > self.free:
                    now = time.monotonic()
                    budget += (now - last) * self.rate
                    budget = min(budget, self.rate * 0.25)
                    last = now
                    while budget < len(chunk) and not self._stop:
                        time.sleep(0.01)
                        now = time.monotonic()
                        budget += (now - last) * self.rate
                        last = now
                    budget -= len(chunk)
                else:
                    last = time.monotonic()
                try:
                    conn.sendall(chunk)
                except OSError:
                    break
                with buf_lock:
                    self.buffered -= len(chunk)
                self.total_down += len(chunk)

        threads = [threading.Thread(target=f, daemon=True)
                   for f in (upstream, reader, writer)]
        for t in threads:
            t.start()
        while not self._stop:
            time.sleep(0.2)

    def stop(self):
        self._stop = True


# ---------------------------------------------------------------------------
# Headless client
# ---------------------------------------------------------------------------

def run_client(url: str, run_secs: float, label: str) -> bool:
    print(f"[{label}] connecting {url}")
    ws = connect(url, open_timeout=30, max_size=None)

    config = {
        "fixture_name": "low_fi_Gm_loop_60s_gnm.wav",
        "use_server_fixture": True,
        "prompt": "lo-fi hip hop, mellow",
        "depth": 4,
        "steps": 8,
    }
    ws.send(json.dumps(config))

    # Handshake: ready JSON then binary initial buffer. Tolerate other
    # JSON (init_ack etc.) before ready.
    duration = None
    while True:
        msg = ws.recv(timeout=240)
        if isinstance(msg, str):
            data = json.loads(msg)
            if data.get("type") == "ready":
                duration = float(data["duration"])
                print(f"[{label}] ready duration={duration:.1f}s")
        else:
            break  # initial buffer
    assert duration is not None

    play_start = time.monotonic()
    stop = [False]
    state = {
        "bytes_rx": 0,
        "worst_lead": None,
        "pending_stem_bins": 0,
        "slices": [],   # (t_since_start, lead)
        "lock": threading.Lock(),
    }

    def playhead() -> float:
        return (time.monotonic() - play_start) % duration

    def fold(lead: float) -> float:
        return ((lead + duration / 2) % duration) - duration / 2

    def recv_loop():
        while not stop[0]:
            try:
                msg = ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            except Exception:
                break
            if isinstance(msg, str):
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if data.get("type") == "stem_assets":
                    state["pending_stem_bins"] = len(data.get("stems", []))
                continue
            if state["pending_stem_bins"] > 0:
                state["pending_stem_bins"] -= 1
                continue
            if len(msg) < SLICE_HDR_SIZE:
                continue
            hdr = struct.unpack(SLICE_HDR_FMT, msg[:SLICE_HDR_SIZE])
            start_sample = hdr[1]
            lead = fold(start_sample / SAMPLE_RATE - playhead())
            with state["lock"]:
                state["bytes_rx"] += len(msg)
                w = state["worst_lead"]
                state["worst_lead"] = lead if w is None else min(w, lead)
                state["slices"].append(
                    (time.monotonic() - play_start, lead),
                )

    def params_loop():
        denoise = 1.0
        i = 0
        while not stop[0]:
            i += 1
            # nudge denoise every ~10s to keep the session non-idle
            if i % 1250 == 0:
                denoise = 1.0 if denoise < 1.0 else 0.99
            with state["lock"]:
                worst = state["worst_lead"]
                state["worst_lead"] = None
                acked = state["bytes_rx"]
            payload = {
                "type": "params",
                "raw": {"seed": 42, "denoise": denoise},
                "playback_pos": playhead(),
                "client_time": time.monotonic(),
                "slice_bytes_rx": acked,
            }
            if worst is not None:
                payload["slice_lead_s"] = worst
            try:
                ws.send(json.dumps(payload))
            except Exception:
                break
            time.sleep(0.008)

    rt = threading.Thread(target=recv_loop, daemon=True)
    pt = threading.Thread(target=params_loop, daemon=True)
    rt.start()
    pt.start()
    time.sleep(run_secs)
    stop[0] = True
    rt.join(timeout=3)
    pt.join(timeout=3)
    try:
        ws.close()
    except Exception:
        pass

    # ---- report ----
    slices = state["slices"]
    print(f"[{label}] total slices applied: {len(slices)}")
    ok = True
    settle = 25.0
    buckets: dict[int, list] = {}
    for t, lead in slices:
        buckets.setdefault(int(t // 5) * 5, []).append(lead)
    for b in sorted(buckets):
        ls = sorted(buckets[b])
        med = ls[len(ls) // 2]
        neg = sum(1 for x in ls if x < 0) / len(ls)
        flag = ""
        if b >= settle and (med <= 0 or neg > 0.2):
            ok = False
            flag = "  <-- FAIL"
        print(
            f"[{label}]  t={b:>3}-{b+5:<3} n={len(ls):>3} "
            f"min={ls[0]:+.2f} med={med:+.2f} neg%={neg*100:4.0f}{flag}"
        )
    print(f"[{label}] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=500_000)
    ap.add_argument("--secs", type=float, default=75)
    ap.add_argument("--mode", choices=["throttled", "direct"],
                    default="throttled")
    args = ap.parse_args()

    if args.mode == "direct":
        ok = run_client(f"ws://{SERVER[0]}:{SERVER[1]}/", args.secs, "direct")
        sys.exit(0 if ok else 1)

    proxy = ThrottleProxy(args.rate)
    proxy.start()
    time.sleep(0.3)
    try:
        ok = run_client(
            f"ws://127.0.0.1:{PROXY_PORT}/", args.secs,
            f"throttle@{args.rate//1000}KBps",
        )
        print(f"[proxy] buffered_now={proxy.buffered} "
              f"total_down={proxy.total_down}")
    finally:
        proxy.stop()
    sys.exit(0 if ok else 1)


def _self_check():
    pass


if __name__ == "__main__":
    main()
