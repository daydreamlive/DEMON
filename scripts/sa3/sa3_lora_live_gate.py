"""SA3 live LoRA refit proof + lofi denoise-sweep gate recording against a private host.

usage: python scripts/sa3/sa3_lora_live_gate.py <out_dir> <port> <lora_id> <sweep_s>

Phase 1 (strength proof): session with lora=True on the lofi fixture, 20 s warm,
enable_lora(1.0), 20 s, lora_str -> 0.5, 20 s, -> 0.0, 20 s. Records heard_strength.wav
and per-phase slice counts / stale ticks.
Phase 2 (merge-gate recipe, E:\\Projects\\DEMON\\sa3_garble\\record.py): LoRA back to 1.0,
sa3_denoise swept 1.0,0.85,0.7,0.5,0.3,0.15,0.3,0.5,0.7,0.85 at 15 s per step for
<sweep_s>, recorded heard.wav + session.json (denoise_changes relative to heard.wav t0).
"""
import json, os, sys, threading, time
import sys as _sys
from pathlib import Path as _Path

_REPO = next(p for p in _Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))
import numpy as np, soundfile as sf

OUT, PORT, LORA, SWEEP_S = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
os.makedirs(OUT, exist_ok=True)
from demos.realtime_motion_graph_web import headless_client as HC

SR = 48000
slices = []
_orig = HC.HeadlessClient._handle_binary


def _hb(self, msg):
    if not self._expectations:
        sl = HC.decode_slice_frame(msg)
        slices.append((time.monotonic(), sl.start_sample, sl.num_samples, sl.flags, sl.tick_ms, sl.dec_ms, sl.num_gens))
    return _orig(self, msg)


HC.HeadlessClient._handle_binary = _hb
cfg = {"prompt": "deep techno, rolling hypnotic bassline, 126 bpm, warehouse",
       "sde": False, "lora": True, "depth": 4, "steps": 8, "client_id": "d5-fp8-refit",
       "fixture_name": "low_fi_Gm_loop_60s_gnm.wav", "use_server_fixture": True,
       "sa3_duration_s": 54.0}
c = HC.HeadlessClient(f"ws://127.0.0.1:{PORT}/", cfg, None, params_hz=30.0)
ready = c.start(timeout_s=900)
print("ready", json.dumps({k: ready.get(k) for k in ("session_id", "duration", "checkpoint")}), flush=True)

heard = []
rec_on = [True]


def rec():
    last = c.player.sample()
    while rec_on[0]:
        time.sleep(0.01)
        m = c.mirror
        pos = c.player.sample()
        if pos >= last:
            heard.append(m[last:pos].copy())
        else:
            heard.append(np.concatenate([m[last:], m[:pos]]))
        last = pos


def setk(k, v):
    with c._state_lock:
        c._raw[k] = v


def window(label, secs):
    a = time.monotonic()
    n0 = len(slices)
    time.sleep(secs)
    b = time.monotonic()
    w = [s for s in slices[n0:]]
    gaps = np.diff([s[0] for s in w]) if len(w) > 1 else np.array([0.0])
    row = {"phase": label, "secs": round(b - a, 2), "slices": len(w),
           "max_gap_s": round(float(gaps.max()), 3) if len(gaps) else None,
           "tick_ms_p50": float(np.median([s[4] for s in w])) if w else None,
           "tick_ms_max": float(max(s[4] for s in w)) if w else None,
           "lora_desc": [d for d in (c.last_params_update or {}).items() if "lora" in d[0]][:4]}
    print(json.dumps(row, default=str), flush=True)
    return row


t = threading.Thread(target=rec, daemon=True)
t.start()
t_s0 = time.monotonic()
setk("sa3_denoise", 1.0)
phases = [window("warm_no_lora", 20)]
marks = [("enable_1.0", round(time.monotonic() - t_s0, 2))]
c.send_command({"type": "enable_lora", "id": LORA, "strength": 1.0})
setk(f"lora_str_{LORA}", 1.0)
phases.append(window("lora_1.0", 20))
marks.append(("str_0.5", round(time.monotonic() - t_s0, 2)))
setk(f"lora_str_{LORA}", 0.5)
phases.append(window("lora_0.5", 20))
marks.append(("str_0.0", round(time.monotonic() - t_s0, 2)))
setk(f"lora_str_{LORA}", 0.0)
phases.append(window("lora_0.0", 20))
rec_on[0] = False
t.join()
sf.write(os.path.join(OUT, "heard_strength.wav"), np.concatenate(heard), SR, subtype="FLOAT")
rep1 = c.tracker.report(window_s=None, stale_threshold_s=3.0)

# Phase 2: merge-gate sweep with the LoRA at 1.0
setk(f"lora_str_{LORA}", 1.0)
time.sleep(5)
heard = []
rec_on[0] = True
t = threading.Thread(target=rec, daemon=True)
t.start()
t0 = time.monotonic()
changes, minutes = [], []
SWEEP = [1.0, 0.85, 0.7, 0.5, 0.3, 0.15, 0.3, 0.5, 0.7, 0.85]
STEP_S = 15


def _toggle(k):
    v = SWEEP[k % len(SWEEP)]
    setk("sa3_denoise", v)
    changes.append((round(time.monotonic() - t0, 2), v))


_toggle(0)
k = 1
while time.monotonic() - t0 < SWEEP_S - 1:
    for _ in range(4):
        time.sleep(max(0.0, t0 + k * STEP_S - time.monotonic()))
        if time.monotonic() - t0 >= SWEEP_S - 1:
            break
        _toggle(k)
        k += 1
    r = c.tracker.report(window_s=60, stale_threshold_s=3.0)
    w = time.monotonic() - 60
    d = [s[5] for s in slices if s[0] >= w]
    row = {"min": len(minutes) + 1, "stale": r["ticks"]["stale_ticks"], "behind": r["slices"]["behind_playhead"],
           "slices": len(d), "dec_p50": float(np.median(d)) if d else None, "wall": time.strftime("%H:%M:%S")}
    minutes.append(row)
    print(json.dumps(row), flush=True)
rec_on[0] = False
t.join()
rep = c.tracker.report(window_s=None, stale_threshold_s=3.0)
st = c.status()
sf.write(os.path.join(OUT, "heard.wav"), np.concatenate(heard), SR, subtype="FLOAT")
c.stop()
json.dump({"ready": {k: v for k, v in ready.items() if k != "geometry"}, "strength_phases": phases,
           "strength_marks_s": marks, "strength_report": rep1, "report": rep, "status": st,
           "slices": len(slices), "minutes": minutes, "denoise_changes": changes},
          open(os.path.join(OUT, "session.json"), "w"), indent=1, default=str)
print("DONE", OUT, "slices", len(slices), "stale", rep["ticks"]["stale_ticks"], flush=True)
