"""WebSocket protocol shared by the client, the server, and the full demo.

Torch-free. Only depends on numpy, websockets, and zstandard.
"""

import json
import struct
import threading

import numpy as np

SAMPLE_RATE = 48000
T = 1500  # 60s at 25fps latents
CROSSFADE_SECONDS = 0.05

# Binary slice header: flags, start_sample, num_samples, channels, tick_ms, dec_ms, num_gens
# flags: 0 = raw float16, 1 = zstd-compressed float16 delta
SLICE_HDR_FMT = "<BIIHffI"
SLICE_HDR_SIZE = struct.calcsize(SLICE_HDR_FMT)
SLICE_FLAG_RAW = 0
SLICE_FLAG_DELTA = 1


class RemoteBackend:
    """WebSocket connection to a remote GPU pipeline server.

    Accepts the source audio as either a torch.Tensor with shape
    ``(channels, samples)`` (via ``.numpy()``) or a plain numpy array with
    the same shape. This keeps the thin client torch-free while remaining
    compatible with the full local+remote demo.
    """

    def __init__(self, url: str, waveform, config: dict):
        from websockets.sync.client import connect as ws_connect

        print(f"[Remote] Connecting to {url}...")
        self.ws = ws_connect(url, max_size=50 * 1024 * 1024, open_timeout=30)
        self._send_lock = threading.Lock()

        print("[Remote] Sending config...")
        self.ws.send(json.dumps(config))

        if hasattr(waveform, "numpy"):
            audio_np = waveform.numpy()
        else:
            audio_np = np.asarray(waveform)
        # Normalize to (samples, channels) layout for the wire format.
        # Callers pass (channels, samples); transpose here.
        audio_np = audio_np.T
        hdr = struct.pack("<II", audio_np.shape[1], audio_np.shape[0])
        self.ws.send(hdr + audio_np.astype(np.float32).tobytes())
        print(f"[Remote] Uploaded audio ({audio_np.nbytes / 1024 / 1024:.1f} MB)")

        print("[Remote] Waiting for server init...")
        ready = json.loads(self.ws.recv())
        self.duration = ready["duration"]
        self.channels = ready["channels"]
        self.sample_rate = ready["sample_rate"]
        self.lora_count = ready.get("lora_count", 0)

        init_bytes = self.ws.recv()
        self.initial_buffer = (
            np.frombuffer(init_bytes, dtype=np.float16)
            .astype(np.float32)
            .reshape(-1, self.channels)
        )
        print(f"[Remote] Ready: {self.duration:.1f}s, {self.channels}ch")

    def send_raw(self, raw_dict: dict, playback_pos: float):
        msg = {"type": "params", "raw": raw_dict, "playback_pos": playback_pos}
        try:
            with self._send_lock:
                self.ws.send(json.dumps(msg))
        except Exception:
            pass

    def send_prompt(self, tags: str):
        try:
            with self._send_lock:
                self.ws.send(json.dumps({"type": "prompt", "tags": tags}))
        except Exception:
            pass

    def recv(self, timeout: float = 0.01):
        try:
            msg = self.ws.recv(timeout=timeout)
            if isinstance(msg, bytes) and len(msg) > SLICE_HDR_SIZE:
                hdr = struct.unpack(SLICE_HDR_FMT, msg[:SLICE_HDR_SIZE])
                flags = hdr[0]
                payload = msg[SLICE_HDR_SIZE:]
                if flags == SLICE_FLAG_DELTA:
                    import zstandard as zstd
                    payload = zstd.decompress(payload)
                audio_f16 = np.frombuffer(payload, dtype=np.float16)
                return ("audio", {
                    "flags": flags,
                    "start_sample": hdr[1],
                    "num_samples": hdr[2],
                    "channels": hdr[3],
                    "tick_ms": hdr[4],
                    "dec_ms": hdr[5],
                    "num_gens": hdr[6],
                    "audio": audio_f16.astype(np.float32).reshape(hdr[2], hdr[3]),
                })
            elif isinstance(msg, str):
                return ("json", json.loads(msg))
        except TimeoutError:
            return None
        except Exception:
            return None
        return None

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass
