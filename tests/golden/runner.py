"""Scenario runner: drives one streaming session per scenario and writes
a result bundle (canonical audio + wire transcript + timing metrics).

Usage (against a pod):
    python -m tests.golden.runner --pod-url ws://HOST:1318 --scenario all
    python -m tests.golden.runner --pod-url ws://HOST:1318 \
        --scenario baseline_stream --repeat 2     # determinism probe

Bundle layout (one dir per scenario run):
    canonical.f32.raw   the position-aligned comparison region of the
                        song buffer ([anchor + warmup_skip_s,
                        + canonical_s], float32 interleaved): the
                        tier-1 hash target
    canonical.wav       same audio, for ears
    buffer.wav          final mirrored song buffer state, for ears
    transcript.jsonl    every wire frame, timestamped (replay input for
                        the app-tier tests)
    blobs/              raw binary frames referenced by the transcript
    metrics.json        timing stats + env metadata + canonical sha256

Why a buffer region and not the raw slice stream: slice segmentation is
scheduling-dependent (same-box runs diverge in window batching), so only
position space is comparable at all. The warm-up skip and settle margin
trim the highest-variance parts; see scenarios.py for the
frontier-relative action semantics and the no-bit-exactness rationale.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .client import SAMPLE_RATE, GoldenClient, Recorder
from .scenarios import SCENARIOS, SCENARIOS_BY_NAME, Scenario

HEARTBEAT_S = 0.1  # params/playback_pos heartbeat cadence


# ── pod HTTP helpers ────────────────────────────────────────────────────

def http_base(pod_url: str) -> str:
    return (pod_url.replace("wss://", "https://")
                   .replace("ws://", "http://").rstrip("/"))


def fetch_server_info(pod_url: str) -> dict:
    try:
        with urllib.request.urlopen(
                f"{http_base(pod_url)}/api/server-info", timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}


def fetch_fixture_pcm(pod_url: str, name: str) -> tuple[np.ndarray, int]:
    """Download a fixture WAV from the pod's own /fixtures endpoint and
    return (interleaved float32 (samples, channels), channels). Keeps
    the harness free of local fixture state."""
    import io

    import soundfile as sf

    url = f"{http_base(pod_url)}/fixtures/{name}"
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = r.read()
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"fixture {name} is {sr} Hz, expected "
                           f"{SAMPLE_RATE}")
    return np.ascontiguousarray(data), data.shape[1]


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _gpu_name() -> str:
    """The GPU this harness process can see (recorded in env metadata so
    reports/thresholds are attributable to hardware). Falls back to
    nvidia-smi so a torch-free client box still reports something."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "unknown"


# ── stats helpers ───────────────────────────────────────────────────────

def _pct(values, ps=(50, 95)) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    out = {f"p{p}": round(float(np.percentile(arr, p)), 3) for p in ps}
    out["mean"] = round(float(arr.mean()), 3)
    out["n"] = int(arr.size)
    return out


# Ack event each action kind waits for (latency attribution).
_ACK_EVENT = {
    "params": "params_update",
    "prompt": "prompt_applied",
    "swap": "swap_ready",
    "enable_lora": "lora_catalog",
}


def _action_audible(ready_at: float, slices: list, entry: dict) -> dict:
    """Knob-to-ear latency for one fired action, from recorded data.

    The playhead is the runner's simulated 1.0x clock (position =
    wall - ready_at, the same value sent as ``playback_pos``). An action
    becomes audible when the playhead reaches re-generated content, so:

    * ``audible_first_ms`` — lower bound: the playhead reaches the start
      of the first post-action slice that landed ahead of where the
      playhead was at send time. Windows already in flight refine their
      REMAINING denoise steps with the new params, so this is where the
      effect starts ramping in (partially refined).
    * ``audible_full_ms`` — upper bound: the playhead reaches the
      generation frontier as of the send. Windows past that point enter
      the pipeline after the action and get EVERY step with the new
      params — full effect from here on.

    Both take max(arrival, playhead-reaches-position): content can't be
    heard before it lands in the buffer, nor before the playhead gets
    there. Values are None when no qualifying slice was recorded.
    ``swap`` actions are excluded by the caller (the audible event for a
    swap is the buffer crossfade itself; ``ack_gap_ms`` covers it).
    """
    sent = entry["sent_wall"]
    pos_at_send = sent - ready_at
    later = slices[entry["slices_before"]:]
    out: dict = {"audible_first_ms": None, "audible_full_ms": None}

    first = next((s for s in later
                  if s.recv_at >= sent
                  and s.start_sample / SAMPLE_RATE > pos_at_send), None)
    if first is not None:
        heard = max(first.recv_at,
                    ready_at + first.start_sample / SAMPLE_RATE)
        out["audible_first_ms"] = round((heard - sent) * 1000.0, 1)

    frontier = entry.get("frontier_s")
    if frontier is not None:
        cover = next((s for s in later
                      if s.recv_at >= sent
                      and s.start_sample / SAMPLE_RATE >= frontier), None)
        if cover is not None:
            heard = max(cover.recv_at, ready_at + frontier)
            out["audible_full_ms"] = round((heard - sent) * 1000.0, 1)
    return out


# ── core ────────────────────────────────────────────────────────────────

def run_scenario(pod_url: str, sc: Scenario, out_dir: Path,
                 save_blobs: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = Recorder(out_dir, save_blobs=save_blobs)
    client = GoldenClient(pod_url, recorder=rec)
    result: dict = {"scenario": sc.name, "status": "ok"}
    try:
        config = sc.session_config()
        client.send_config(config)
        if sc.upload:
            pcm, channels = fetch_fixture_pcm(pod_url, sc.fixture)
            client.send_pcm(pcm, channels)
        t_cfg = client._now()
        ready = client.wait_ready(timeout=sc.timeout_s)
        result["ready"] = {
            k: ready.get(k) for k in
            ("duration", "channels", "sample_rate", "checkpoint",
             "pipeline_depth", "max_pipeline_depth", "bpm", "key")
        }
        result["t_config_to_ready_s"] = round(client.ready_at - t_cfg, 3)

        # lora_enable needs a catalog entry; skip cleanly when absent.
        catalog = ready.get("lora_catalog") or []
        for a in sc.actions:
            if a.kind == "enable_lora" and not catalog:
                result["status"] = "skipped"
                result["reason"] = "pod ships no LoRAs"
                return result

        pending = sorted(sc.actions, key=lambda a: a.at_s)
        fired: list[dict] = []
        deadline = time.monotonic() + sc.timeout_s
        last_beat = 0.0
        skip = int(sc.warmup_skip_s * SAMPLE_RATE)
        canon_n = int(sc.canonical_s * SAMPLE_RATE)

        def playhead() -> float:
            # Simulated 1.0x playback clock, starting at ready.
            return client._now() - client.ready_at

        def frontier_s() -> float:
            # Furthest song position slices have covered (post-reset).
            return (max(b for _, b in client.coverage) / SAMPLE_RATE
                    if client.coverage else 0.0)

        # The comparison region is anchored at an ABSOLUTE song
        # position: 0 for session-start scenarios, the swap trigger
        # position for swap scenarios. Anchoring at "first covered
        # frame" looks natural but that frame shifts by a window or two
        # between runs, which offsets the compared region and turns an
        # identical-content pair into a giant false diff.
        anchor_s = max((a.at_s for a in sc.actions if a.kind == "swap"),
                       default=0.0)
        region_start = int(anchor_s * SAMPLE_RATE) + skip

        def region() -> tuple[int, int] | None:
            if not client.coverage or client.buffer is None:
                return None
            end = min(region_start + canon_n, client.buffer.shape[0])
            return (region_start, end)

        def region_done() -> bool:
            # Done when coverage extends settle_s PAST the region end:
            # windows are re-emitted as they refine through the pipeline
            # depth, so a region right at the frontier is still a
            # partially-refined state. The settle margin lets the
            # compared region finish refining before we stop.
            r = region()
            if r is None:
                return False
            settle = int(sc.settle_s * SAMPLE_RATE)
            end = min(r[1] + settle, client.buffer.shape[0])
            run = client.covered_run_from(r[0])
            return run is not None and run[0] == r[0] and run[1] >= end

        while True:
            client.pump(timeout=0.05)
            pos = playhead()
            # Heartbeat: the browser rides playback_pos on its params
            # channel; the server's adaptive decode lead needs it.
            if pos - last_beat >= HEARTBEAT_S:
                client.send_params({}, playback_pos=pos)
                last_beat = pos
            # Fire actions when the generation frontier crosses their
            # song position (see scenarios.py for why not the playhead).
            while pending and frontier_s() >= pending[0].at_s:
                a = pending.pop(0)
                rec_entry = {"kind": a.kind, "at_s": a.at_s,
                             "sent_wall": client._now(),
                             "playhead": round(pos, 3),
                             "frontier_s": round(frontier_s(), 3),
                             "slices_before": len(client.slices),
                             "events_before": len(client.events)}
                if a.kind == "params":
                    client.send_params(a.payload["raw"], playback_pos=pos)
                elif a.kind == "prompt":
                    client.send_prompt(a.payload["tags"],
                                       a.payload.get("tags_b"))
                elif a.kind == "swap":
                    client.send_swap_to_fixture(a.payload["fixture"])
                elif a.kind == "enable_lora":
                    client.send_enable_lora(
                        str(catalog[0].get("id")),
                        a.payload.get("strength"))
                else:
                    raise ValueError(f"unknown action kind: {a.kind}")
                fired.append(rec_entry)
            if not pending and region_done():
                break
            if time.monotonic() > deadline:
                result["status"] = "timeout"
                break

        # Drain briefly so in-flight acks land in the transcript.
        drain_until = time.monotonic() + 2.0
        while time.monotonic() < drain_until:
            if not client.pump(timeout=0.1):
                break

        result.update(_finalize(client, sc, fired, region(), out_dir))
    except Exception as exc:  # keep the bundle + report the failure
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        client.close()
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def _finalize(client: GoldenClient, sc: Scenario, fired: list,
              region: tuple[int, int] | None, out_dir: Path) -> dict:
    import soundfile as sf

    out: dict = {"gpu": _gpu_name()}

    # Canonical artifact: the position-aligned buffer comparison region.
    if region is not None and client.buffer is not None:
        start, end = region
        canon = client.buffer[start:end]
        raw = np.ascontiguousarray(canon, dtype=np.float32).tobytes()
        (out_dir / "canonical.f32.raw").write_bytes(raw)
        sf.write(out_dir / "canonical.wav", canon, SAMPLE_RATE)
        out["canonical_sha256"] = hashlib.sha256(raw).hexdigest()
        out["canonical_region"] = {
            "start_frame": int(start), "end_frame": int(end),
            "start_s": round(start / SAMPLE_RATE, 3),
            "len_s": round((end - start) / SAMPLE_RATE, 3),
        }
    if client.buffer is not None:
        sf.write(out_dir / "buffer.wav", client.buffer, SAMPLE_RATE)
    out["coverage"] = [[int(a), int(b)] for a, b in client.coverage]

    # Latency: ack + next-slice gap per fired action.
    for entry in fired:
        ack_name = _ACK_EVENT.get(entry["kind"])
        sent = entry["sent_wall"]
        ack = next((t for t, e in client.events[entry["events_before"]:]
                    if e.get("type") == ack_name and t >= sent), None)
        entry["ack_event"] = ack_name
        entry["ack_gap_ms"] = (round((ack - sent) * 1000.0, 1)
                               if ack is not None else None)
        nxt = next((s.recv_at for s in client.slices[entry["slices_before"]:]
                    if s.recv_at >= sent), None)
        entry["next_slice_gap_ms"] = (round((nxt - sent) * 1000.0, 1)
                                      if nxt is not None else None)
        # Knob-to-ear: when the simulated playhead reaches re-generated
        # content (first-touched lower bound / past-frontier full
        # effect). Not meaningful for swap (the crossfade IS the event).
        if entry["kind"] != "swap" and client.ready_at is not None:
            entry.update(_action_audible(
                client.ready_at, client.slices, entry))
    out["actions"] = fired

    # Stream-level timing.
    recvs = [s.recv_at for s in client.slices]
    out["t_ready_to_first_slice_s"] = (
        round(recvs[0] - client.ready_at, 3) if recvs else None)
    out["slice_gap_ms"] = _pct([
        (b - a) * 1000.0 for a, b in zip(recvs, recvs[1:])])
    out["dec_ms"] = _pct([s.dec_ms for s in client.slices])
    out["tick_ms"] = _pct([s.tick_ms for s in client.slices])
    out["lead_s"] = _pct([
        s.start_sample / SAMPLE_RATE - (s.recv_at - client.ready_at)
        for s in client.slices])
    out["n_slices"] = len(client.slices)
    out["gen_samples"] = int(client.gen_samples)
    out["wall_s"] = round(client._now(), 3)
    out["realtime_factor"] = (
        round(client.gen_samples / SAMPLE_RATE
              / (client._now() - client.ready_at), 2)
        if client.ready_at and client._now() > client.ready_at else None)
    return out


def run_all(pod_url: str, names: list[str], out_root: Path,
            repeat: int = 1, save_blobs: bool = True) -> list[dict]:
    env = {
        "harness_git_sha": _git_sha(),
        "pod_url": pod_url,
        # NOTE: the GPU the HARNESS box can see. Identical to the server
        # GPU for local runs; for remote pods set DEMON_SERVER_GPU to
        # the pod's card so the identity gate stays meaningful.
        "gpu": os.environ.get("DEMON_SERVER_GPU") or _gpu_name(),
        "server_info": fetch_server_info(pod_url),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "env.json").write_text(
        json.dumps(env, indent=2), encoding="utf-8")
    results = []
    for name in names:
        sc = SCENARIOS_BY_NAME[name]
        for i in range(repeat):
            run_dir = (out_root / name if repeat == 1
                       else out_root / f"{name}.run{i + 1}")
            print(f"[golden] {name} (run {i + 1}/{repeat}) ...",
                  flush=True)
            res = run_scenario(pod_url, sc, run_dir,
                               save_blobs=save_blobs)
            if res.get("canonical_sha256"):
                res["bundle_dir"] = str(run_dir)
            print(f"[golden]   -> {res['status']} "
                  f"sha={res.get('canonical_sha256', '-')[:12]} "
                  f"wall={res.get('wall_s')}s", flush=True)
            results.append(res)
    if repeat > 1:
        _determinism_report(results)
    return results


def _determinism_report(results: list[dict]) -> None:
    """Same-build variance probe: hash agreement + pairwise tier-2
    metrics + suggested thresholds (observed noise floor x3)."""
    import itertools

    from .compare import action_window_indices, audio_metrics, load_canonical

    by_name: dict = {}
    for r in results:
        if r.get("bundle_dir"):
            by_name.setdefault(r["scenario"], []).append(r)
    print("\n[golden] variance probe:")
    for name, runs in by_name.items():
        shas = [r.get("canonical_sha256") for r in runs]
        if len(set(shas)) == 1 and shas[0] is not None:
            print(f"  {name}: BIT-EXACT across {len(runs)} runs")
            continue
        worst: dict = {}
        for a, b in itertools.combinations(runs, 2):
            # Mask action windows here too: suggested thresholds must be
            # calibrated against the same population the gate scores.
            aw = (action_window_indices(Path(a["bundle_dir"]))
                  | action_window_indices(Path(b["bundle_dir"])))
            m = audio_metrics(load_canonical(Path(a["bundle_dir"])),
                              load_canonical(Path(b["bundle_dir"])),
                              action_windows=aw)
            worst["mel_l2"] = max(worst.get("mel_l2", 0), m["mel_l2"])
            worst["rms_db_diff"] = max(worst.get("rms_db_diff", 0),
                                       m["rms_db_diff"])
            worst["win_cos_min"] = min(worst.get("win_cos_min", 1.0),
                                       m["win_cos_min"]
                                       if m["win_cos_min"] is not None
                                       else 1.0)
        sug = {
            "mel_l2": round(max(worst["mel_l2"] * 3, 0.01), 4),
            "rms_db_diff": round(max(worst["rms_db_diff"] * 3, 0.1), 3),
            "win_cos_min": round(
                min(0.999, max(0.0, 1 - (1 - worst["win_cos_min"]) * 3)),
                6),
        }
        print(f"  {name}: not bit-exact; observed noise floor {worst}")
        print(f"    suggested refs.json thresholds: {sug}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pod-url", default=None,
                    help="ws://HOST:PORT (or set DEMON_POD_URL)")
    ap.add_argument("--local", action="store_true",
                    help="spawn the server on this machine's GPU for "
                         "the duration of the run")
    ap.add_argument("--scenario", default="all",
                    help="scenario name, comma list, or 'all'")
    ap.add_argument("--out", default=None,
                    help="output root (default runs/golden-<utc>)")
    ap.add_argument("--repeat", type=int, default=1,
                    help=">1 runs each scenario N times and reports "
                         "hash agreement (determinism probe)")
    ap.add_argument("--no-blobs", action="store_true",
                    help="skip raw binary blobs (smaller bundles, no "
                         "replay support)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for s in SCENARIOS:
            print(f"{s.name:20s} canon={s.canonical_s:>4.0f}s "
                  f"actions={len(s.actions)}  {s.notes}")
        return 0

    pod_url = args.pod_url or os.environ.get("DEMON_POD_URL")
    if not pod_url and not args.local:
        ap.error("--pod-url, DEMON_POD_URL, or --local is required")

    names = ([s.name for s in SCENARIOS] if args.scenario == "all"
             else [n.strip() for n in args.scenario.split(",")])
    unknown = [n for n in names if n not in SCENARIOS_BY_NAME]
    if unknown:
        ap.error(f"unknown scenario(s): {unknown}")

    out_root = Path(args.out) if args.out else Path(
        "runs") / f"golden-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

    server = None
    if not pod_url:
        from .local_server import LocalServer

        out_root.mkdir(parents=True, exist_ok=True)
        print("[golden] spawning local server ...", flush=True)
        server = LocalServer(log_path=out_root / "server.log")
        pod_url = server.url
        print(f"[golden] local server up at {pod_url}", flush=True)
    try:
        results = run_all(pod_url, names, out_root, repeat=args.repeat,
                          save_blobs=not args.no_blobs)
    finally:
        if server is not None:
            server.stop()
    bad = [r for r in results if r["status"] not in ("ok", "skipped")]
    print(f"\n[golden] bundles in {out_root}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
