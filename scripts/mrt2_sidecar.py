#!/usr/bin/env python
"""Magenta RT 2 generation sidecar for the DEMON ``mrt2`` backend family.

Runs NEXT TO the JAX model, not inside the DEMON server process: JAX has
no CUDA on native Windows, so this script lives in the MRT2 venv (WSL:
``~/.venvs/mrt2``, see notes/magenta-realtime) or on a Linux pod, and
serves frames to :class:`acestep.streaming.mrt2.backend.MRT2Backend`
over the tiny TCP protocol in ``acestep/streaming/mrt2/protocol.py``.

That protocol module is loaded BY FILE PATH (spec_from_file_location),
never via ``import acestep`` — the acestep package import would pull
torch, which this venv deliberately does not have.

Usage (from the repo root, inside the MRT2 venv):

    python scripts/mrt2_sidecar.py --model mrt2_small

The model is loaded and JIT-warmed once at startup (~30s); connections
are then served one at a time. Generation is credit-paced: the loop
only runs ``mrt.generate`` while the connected backend has granted
frames, so the (faster-than-real-time) model never runs ahead of the
backend's configured lead — for an append-only model, audio generated
ahead of need is latency, not safety.

The recurrent state persists across prompt/knob changes (that is the
whole point: one evolving stream of music, conditioned live) and is
reset only when a new connection arrives.
"""

import argparse
import importlib.util
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Protocol module, loaded by file path (NOT ``import acestep`` — no torch
# in this venv; see module docstring).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROTO_PATH = _REPO_ROOT / "acestep" / "streaming" / "mrt2" / "protocol.py"
_spec = importlib.util.spec_from_file_location("mrt2_protocol", _PROTO_PATH)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def log(msg: str) -> None:
    print(f"[mrt2-sidecar +{time.monotonic() - _T0:9.2f}s] {msg}", flush=True)


_T0 = time.monotonic()


class Conditioning:
    """Live conditioning shared between the socket reader and the
    generation loop. The reader thread writes; the generation loop
    snapshots before each chunk, so changes land at chunk granularity
    (a handful of 40 ms frames)."""

    def __init__(self, style_model, defaults: dict):
        self._style_model = style_model
        self._lock = threading.Lock()
        self._embed_cache: dict = {}
        self.tags = ""
        self.tags_b = None
        self.blend = 0.0
        self.knobs = dict(defaults)
        self._style = None
        self._style_dirty = True

    def _embed(self, tags: str):
        emb = self._embed_cache.get(tags)
        if emb is None:
            emb = self._style_model.embed_batch_text([tags])[0]
            # Unbounded growth is fine in practice (an operator types
            # dozens of prompts, not millions), but cap it anyway.
            if len(self._embed_cache) > 256:
                self._embed_cache.clear()
            self._embed_cache[tags] = emb
        return emb

    def set_prompt(self, tags: str, tags_b) -> None:
        with self._lock:
            self.tags = tags or ""
            self.tags_b = tags_b if tags_b else None
            self._style_dirty = True

    def set_blend(self, value: float) -> None:
        with self._lock:
            self.blend = max(0.0, min(1.0, float(value)))
            self._style_dirty = True

    def set_knobs(self, update: dict) -> None:
        with self._lock:
            self.knobs.update(update)

    def snapshot(self):
        """(style_embedding | None, knobs dict) for the next chunk.
        The style embedding is recomputed only when tags/blend moved;
        embeddings are cached per tags string so blend sweeps cost a
        lerp, not a MusicCoCa forward."""
        with self._lock:
            tags, tags_b, blend = self.tags, self.tags_b, self.blend
            knobs = dict(self.knobs)
            dirty = self._style_dirty
            self._style_dirty = False
        if dirty:
            emb_a = self._embed(tags) if tags else None
            if tags_b and blend > 0.0 and emb_a is not None:
                emb_b = self._embed(tags_b)
                style = (1.0 - blend) * emb_a + blend * emb_b
            elif tags_b and blend > 0.0:
                style = self._embed(tags_b) if blend >= 0.999 else None
            else:
                style = emb_a
            self._style = style
        return self._style, knobs


def serve_one(conn, mrt, cond_defaults: dict, chunk_frames: int) -> None:
    """Serve one backend connection until it drops."""
    conn.settimeout(None)
    ctrl: "queue.Queue" = queue.Queue()
    alive = [True]

    def _reader():
        try:
            while True:
                kind, payload = mp.recv_msg(conn)
                if kind == mp.MSG_JSON:
                    ctrl.put(mp.unpack_json(payload))
        except (ConnectionError, OSError):
            alive[0] = False
            ctrl.put(None)  # wake the main loop

    cond = Conditioning(mrt._style_model, cond_defaults)
    credit = 0
    frame_index = 0
    state = None  # fresh recurrent state per connection
    send_lock = threading.Lock()

    def _send(data: bytes) -> bool:
        try:
            with send_lock:
                conn.sendall(data)
            return True
        except (ConnectionError, OSError):
            alive[0] = False
            return False

    threading.Thread(target=_reader, daemon=True).start()

    while alive[0]:
        # Drain control messages. Out of credit -> block briefly (the
        # next message is the only thing that can change that); credit
        # in hand -> drain whatever is queued and get on with generating.
        try:
            msg = ctrl.get(timeout=0.05) if credit <= 0 else ctrl.get_nowait()
        except queue.Empty:
            msg = "none"
        while msg != "none":
            if msg is None:
                return
            mtype = msg.get("type")
            if mtype == "hello":
                _send(mp.pack_json({
                    "type": "meta",
                    "sample_rate": mp.SAMPLE_RATE,
                    "channels": mp.CHANNELS,
                    "frame_samples": mp.FRAME_SAMPLES,
                    "model": MODEL_NAME,
                }))
            elif mtype == "prompt":
                cond.set_prompt(msg.get("tags", ""), msg.get("tags_b"))
                log(f"prompt tags={msg.get('tags')!r} tags_b={msg.get('tags_b')!r}")
            elif mtype == "blend":
                cond.set_blend(msg.get("value", 0.0))
            elif mtype == "knobs":
                update = {
                    k: v for k, v in msg.items()
                    if k in ("temperature", "top_k", "cfg_musiccoca",
                             "cfg_notes", "cfg_drums")
                }
                cond.set_knobs(update)
                log(f"knobs {update}")
            elif mtype == "credit":
                credit += int(msg.get("frames", 0))
            elif mtype == "ping":
                _send(mp.pack_json({"type": "pong", "t": msg.get("t")}))
            try:
                msg = ctrl.get_nowait()
            except queue.Empty:
                msg = "none"

        if credit <= 0 or not alive[0]:
            continue

        n = min(credit, chunk_frames)
        style, knobs = cond.snapshot()
        t0 = time.monotonic()
        try:
            wav, state = mrt.generate(
                style=style,
                frames=n,
                state=state,
                temperature=float(knobs["temperature"]),
                top_k=int(knobs["top_k"]),
                cfg_musiccoca=float(knobs["cfg_musiccoca"]),
                cfg_notes=float(knobs["cfg_notes"]),
                cfg_drums=float(knobs["cfg_drums"]),
            )
        except Exception as exc:  # surface, don't die mid-connection
            log(f"generate failed: {exc!r}")
            _send(mp.pack_json({"type": "err", "message": str(exc)}))
            credit = 0
            continue
        gen_s = time.monotonic() - t0

        pcm = np.asarray(wav.samples, dtype=np.float32)
        if pcm.ndim == 1:
            pcm = pcm.reshape(-1, 1).repeat(mp.CHANNELS, axis=1)
        pcm = np.ascontiguousarray(pcm[:, : mp.CHANNELS], dtype=np.float32)
        # Pin the header invariant exactly: num_frames * FRAME_SAMPLES
        # samples on the wire (the backend's credit accounting and the
        # runner's rolling-window math both lean on it).
        want = n * mp.FRAME_SAMPLES
        if pcm.shape[0] != want:
            log(f"unexpected chunk length {pcm.shape[0]} != {want}; pinning")
            if pcm.shape[0] > want:
                pcm = pcm[:want]
            else:
                pcm = np.vstack(
                    [pcm, np.zeros((want - pcm.shape[0], mp.CHANNELS), np.float32)],
                )
        if not _send(mp.pack_audio(frame_index, n, pcm.tobytes())):
            return
        frame_index += n
        credit -= n
        if frame_index % 250 < n:  # ~every 10 s of audio
            log(
                f"frontier={frame_index * mp.FRAME_SECONDS:.1f}s "
                f"chunk={n}f gen={gen_s * 1000:.0f}ms "
                f"rt_x={n * mp.FRAME_SECONDS / max(gen_s, 1e-9):.2f}"
            )


MODEL_NAME = ""


def main() -> int:
    global MODEL_NAME
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=mp.DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=mp.DEFAULT_PORT)
    ap.add_argument("--model", default="mrt2_small",
                    help="Model variant (mrt2_small fits real-time on a "
                         "5090; mrt2_base is ~0.93x RT as shipped).")
    ap.add_argument("--chunk-frames", type=int, default=2,
                    help="Frames per generate call (2 = 80 ms; smaller = "
                         "lower control latency, more per-call overhead).")
    args = ap.parse_args()
    MODEL_NAME = args.model

    log(f"loading {args.model} (JAX) …")
    from magenta_rt import MagentaRT2Jax

    mrt = MagentaRT2Jax(size=args.model)
    cond_defaults = {
        "temperature": 1.3, "top_k": 40,
        "cfg_musiccoca": 3.0, "cfg_notes": 1.0, "cfg_drums": 1.0,
    }
    log("warmup generate (JIT compile) …")
    t0 = time.monotonic()
    _wav, _state = mrt.generate(frames=args.chunk_frames, state=None)
    log(f"warm in {time.monotonic() - t0:.1f}s; "
        f"serving on {args.host}:{args.port}")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log(f"backend connected from {addr}")
        try:
            serve_one(conn, mrt, cond_defaults, args.chunk_frames)
        except Exception as exc:
            log(f"connection error: {exc!r}")
        finally:
            try:
                conn.close()
            except OSError:
                pass
            log("backend disconnected")


if __name__ == "__main__":
    sys.exit(main())
