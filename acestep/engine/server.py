"""Thin HTTP server wrapping Session for VST/external client integration.

Starts a Session on launch, exposes endpoints for the generation pipeline.
Intermediate state (source, conditioning, LoRAs) is held server-side;
clients send lightweight JSON requests and receive audio back.

Usage:
    python -m acestep.engine.server
    python -m acestep.engine.server --port 8731 --project-root /path/to/checkpoints
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Optional

from loguru import logger
import soundfile as sf
import torch


def _build_curve(spec: dict, length: int):
    """Build a Curve from a JSON spec.

    Supported formats:
        {"type": "constant", "value": 1.0}
        {"type": "ramp", "start": 0.2, "end": 1.5}
        {"type": "sine", "frames_per_cycle": 25, "amplitude": 0.25, "offset": 0.25}
        {"type": "pulse", "frames_per_cycle": 150, "amplitude": 0.5, "offset": 0.5}
        {"type": "square", "frames_per_cycle": 150, "amplitude": 0.5, "offset": 0.5}
    """
    from acestep.nodes.curve_nodes import CurveConstant, CurveRamp, CurveWave

    kind = spec.get("type", "constant")
    if kind == "constant":
        return CurveConstant().execute(
            value=spec.get("value", 1.0), length=length,
        )["curve"]
    elif kind == "ramp":
        return CurveRamp().execute(
            start=spec.get("start", 0.0),
            end=spec.get("end", 1.0),
            length=length,
        )["curve"]
    elif kind in ("sine", "pulse", "square"):
        return CurveWave().execute(
            wave_type=kind,
            frames_per_cycle=spec.get("frames_per_cycle", 150),
            amplitude=spec.get("amplitude", 1.0),
            offset=spec.get("offset", 0.0),
            length=length,
        )["curve"]
    elif kind == "raw":
        from acestep.nodes.types import Curve

        values = spec.get("values", [])
        t = torch.tensor(values, dtype=torch.float32)
        if t.numel() == 0:
            t = torch.ones(length)
        elif t.numel() != length:
            t = torch.nn.functional.interpolate(
                t.view(1, 1, -1), size=length, mode="linear", align_corners=True,
            ).view(length)
        return Curve(tensor=t)
    else:
        raise ValueError(f"Unknown curve type: {kind}")


class SessionState:
    """Mutable server-side state holding cached intermediates."""

    def __init__(self):
        from acestep.engine.session import Session

        self.session: Optional[Session] = None
        # Named slots for cached intermediates
        self.sources: dict[str, Any] = {}  # name -> PreparedSource
        self.conditionings: dict[str, Any] = {}  # name -> Conditioning
        self.latents: dict[str, Any] = {}  # name -> Latent

    def ensure_session(self, **kwargs) -> None:
        if self.session is None:
            from acestep.engine.session import Session
            self.session = Session(**kwargs)


def _audio_from_bytes(data: bytes, sr: int = 48000):
    """Decode audio bytes (wav/flac/mp3) to Audio type."""
    from acestep.nodes.types import Audio

    buf = io.BytesIO(data)
    samples, file_sr = sf.read(buf, dtype="float32")
    waveform = torch.from_numpy(
        samples.T if samples.ndim > 1 else samples.reshape(1, -1)
    )
    if file_sr != sr:
        import torchaudio
        waveform = torchaudio.transforms.Resample(file_sr, sr)(waveform)
    waveform = waveform[:2]
    return Audio(waveform=waveform, sample_rate=sr)


def _audio_to_wav_bytes(audio) -> bytes:
    """Encode Audio to WAV bytes."""
    wav = audio.waveform
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    buf = io.BytesIO()
    sf.write(buf, wav.detach().cpu().float().numpy().T, audio.sample_rate, format="WAV")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    state: SessionState

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body)

    def _respond_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_audio(self, wav_bytes: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.end_headers()
        self.wfile.write(wav_bytes)

    def _respond_error(self, msg: str, status: int = 400):
        self._respond_json({"error": msg}, status)

    def do_GET(self):
        if self.path == "/health":
            self._respond_json({
                "status": "ok",
                "session_ready": self.state.session is not None,
                "sources": list(self.state.sources.keys()),
                "conditionings": list(self.state.conditionings.keys()),
            })
        else:
            self._respond_error("Not found", 404)

    def do_POST(self):
        try:
            route = self.path.rstrip("/")
            handlers = {
                "/prepare_source": self._handle_prepare_source,
                "/audio_info": self._handle_audio_info,
                "/encode_text": self._handle_encode_text,
                "/apply_lora": self._handle_apply_lora,
                "/remove_loras": self._handle_remove_loras,
                "/generate": self._handle_generate,
                "/warmup": self._handle_warmup,
            }
            fn = handlers.get(route)
            if fn is None:
                self._respond_error("Not found", 404)
                return
            fn()
        except Exception as e:
            logger.exception("Request failed")
            self._respond_error(str(e), 500)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def _handle_prepare_source(self):
        """POST /prepare_source
        Body: raw audio bytes.
        Params via query string OR headers (X-Source-Name).
        """
        params = self._parse_query()
        name = params.get("name") or self.headers.get("X-Source-Name", "default")

        # Read audio from request body
        audio_bytes = self._read_body_bytes()
        audio = _audio_from_bytes(audio_bytes)

        # Snap to pool boundary (always trim, never pad, to stay within
        # TRT engine max T)
        samples_per_pool = 1920 * 5  # 9600
        n_samples = audio.waveform.shape[-1]
        remainder = n_samples % samples_per_pool
        if remainder != 0:
            audio = type(audio)(
                waveform=audio.waveform[:, :n_samples - remainder],
                sample_rate=audio.sample_rate,
            )

        t0 = time.perf_counter()
        source = self.state.session.prepare_source(audio)
        elapsed = time.perf_counter() - t0

        self.state.sources[name] = source
        self.state.latents[f"{name}_latent"] = source.latent
        self.state.latents[f"{name}_context"] = source.context_latent

        self._respond_json({
            "name": name,
            "frames": source.latent.tensor.shape[1],
            "elapsed": round(elapsed, 3),
        })

    def _handle_audio_info(self):
        """POST /audio_info  Body: raw audio bytes."""
        from acestep.engine.session import Session

        audio_bytes = self._read_body_bytes()
        audio = _audio_from_bytes(audio_bytes)
        info = Session.audio_info(audio)
        self._respond_json(info)

    def _handle_encode_text(self):
        """POST /encode_text  Body: JSON with text params."""
        req = self._read_json()
        name = req.pop("name", "default")
        source_name = req.pop("source_name", "default")

        # Use source latent as timbre reference unless overridden
        refer_latent = None
        if source_name in self.state.sources:
            refer_latent = self.state.sources[source_name].latent

        # Timbre blending. ``timbre_strength`` is no longer a node-level
        # knob — callers who want partial timbre reference blend it with
        # silence themselves. The HTTP surface keeps the convenience of
        # accepting a scalar for now by doing the LatentBlend here.
        timbre_strength = req.pop("timbre_strength", None)
        if timbre_strength is not None and timbre_strength < 1.0 and refer_latent is not None:
            # Match silence to actual latent frame count to avoid shape mismatch
            T_ref = refer_latent.tensor.shape[1]
            duration_from_frames = T_ref / 25.0
            silence = self.state.session.empty_latent(duration=duration_from_frames)
            sil_t = silence.tensor
            if sil_t.shape[1] != T_ref:
                sil_t = sil_t[:, :T_ref, :] if sil_t.shape[1] > T_ref else torch.nn.functional.pad(
                    sil_t, (0, 0, 0, T_ref - sil_t.shape[1])
                )
                from acestep.nodes.types import Latent
                silence = Latent(tensor=sil_t)
            refer_latent = self.state.session.blend_latents(
                silence, refer_latent, alpha=float(timbre_strength),
            )

        t0 = time.perf_counter()
        cond = self.state.session.encode_text(
            refer_latent=refer_latent, **req,
        )
        elapsed = time.perf_counter() - t0

        self.state.conditionings[name] = cond
        self._respond_json({"name": name, "elapsed": round(elapsed, 3)})

    def _handle_apply_lora(self):
        """POST /apply_lora  Body: {"path": "...", "scale": 1.0}"""
        req = self._read_json()
        path = req["path"]
        scale = req.get("scale", 1.0)

        t0 = time.perf_counter()
        self.state.session.apply_lora(path, scale=scale)
        elapsed = time.perf_counter() - t0

        self._respond_json({"elapsed": round(elapsed, 3)})

    def _handle_remove_loras(self):
        """POST /remove_loras"""
        self.state.session.remove_loras()
        self._respond_json({"ok": True})

    def _handle_generate(self):
        """POST /generate  Body: JSON with generation params.
        Returns: WAV audio bytes.
        """
        from acestep.nodes.cond_nodes import ConditioningAverage

        req = self._read_json()
        source_name = req.get("source_name", "default")
        cond_name = req.get("conditioning_name", "default")

        # Support dual conditioning blend
        cond_name_b = req.get("conditioning_name_b")
        blend = req.get("prompt_blend", 0.0)

        source = self.state.sources.get(source_name)
        if source is None:
            self._respond_error(f"Source '{source_name}' not prepared")
            return

        cond = self.state.conditionings.get(cond_name)
        if cond is None:
            self._respond_error(f"Conditioning '{cond_name}' not encoded")
            return

        # Blend conditionings if requested
        if cond_name_b and blend > 0.0:
            cond_b = self.state.conditionings.get(cond_name_b)
            if cond_b is not None:
                cond = ConditioningAverage().execute(
                    conditioning_a=cond, conditioning_b=cond_b, weight=blend,
                )["conditioning"]

        # Context latent (hint strength)
        context_latent = source.context_latent
        hint_strength = req.get("hint_strength")
        if hint_strength is not None and hint_strength < 1.0:
            # Match silence to actual source frame count (not duration) to
            # avoid shape mismatch when audio was snapped to pool boundaries.
            T_ctx = context_latent.tensor.shape[1]
            duration_from_frames = T_ctx / 25.0
            silence = self.state.session.empty_latent(duration=duration_from_frames)
            # Trim or pad silence to exactly match context_latent frames
            sil_t = silence.tensor
            if sil_t.shape[1] != T_ctx:
                sil_t = sil_t[:, :T_ctx, :] if sil_t.shape[1] > T_ctx else torch.nn.functional.pad(
                    sil_t, (0, 0, 0, T_ctx - sil_t.shape[1])
                )
                from acestep.nodes.types import Latent
                silence = Latent(tensor=sil_t)
            context_latent = self.state.session.blend_latents(
                silence, context_latent, alpha=float(hint_strength),
            )

        # Build per-frame curves from specs
        T = source.latent.tensor.shape[1]
        gen_kwargs = {}
        for curve_name in (
            "velocity_scale", "sde_denoise_curve",
            "initial_noise_curve", "ode_noise_curve",
        ):
            spec = req.get(curve_name)
            if spec is not None:
                gen_kwargs[curve_name] = _build_curve(spec, T)

        # x0 target blending
        x0_target_name = req.get("x0_target")
        if x0_target_name and x0_target_name in self.state.latents:
            gen_kwargs["x0_target"] = self.state.latents[x0_target_name]
            x0_spec = req.get("x0_target_curve")
            if x0_spec is not None:
                gen_kwargs["x0_target_curve"] = _build_curve(x0_spec, T)

        t0 = time.perf_counter()
        output_latent = self.state.session.generate(
            conditioning=cond,
            context_latent=context_latent,
            source_latent=source.latent,
            seed=req.get("seed"),
            denoise=req.get("denoise", 1.0),
            steps=req.get("steps", 8),
            shift=req.get("shift", 3.0),
            method=req.get("method", "ode"),
            **gen_kwargs,
        )
        t_gen = time.perf_counter() - t0

        # Optionally stash the output latent for x0_target use
        stash_name = req.get("stash_latent")
        if stash_name:
            self.state.latents[stash_name] = output_latent

        t0 = time.perf_counter()
        audio = self.state.session.decode(output_latent)
        t_dec = time.perf_counter() - t0

        wav_bytes = _audio_to_wav_bytes(audio)

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.send_header("X-Generate-Ms", str(int(t_gen * 1000)))
        self.send_header("X-Decode-Ms", str(int(t_dec * 1000)))
        self.end_headers()
        self.wfile.write(wav_bytes)

    def _handle_warmup(self):
        """POST /warmup  Trigger torch.compile warmup."""
        s = self.state.session
        silence = s.empty_latent(duration=60.0)
        cond = s.encode_text(tags="warmup", duration=60.0)
        ctx = s.extract_hints(silence)

        t0 = time.perf_counter()
        _ = s.generate(
            conditioning=cond, context_latent=ctx,
            source_latent=silence, seed=0,
        )
        elapsed = time.perf_counter() - t0
        self._respond_json({"warmup_seconds": round(elapsed, 1)})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_query(self) -> dict:
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        return {k: v[0] for k, v in qs.items()}

    def _read_body_bytes(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def log_message(self, format, *args):
        logger.info(format, *args)


def main():
    # Idempotent; safe if another entrypoint already configured loguru.
    from acestep.engine.obs import configure as _configure_logging
    _configure_logging()

    parser = argparse.ArgumentParser(description="ACE-Step Session Server")
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--project-root", default=None,
                        help="Path to checkpoints (default: ~/.daydream-scope/models/demon/checkpoints)")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--trt-decoder", default=None, help="Path to TRT decoder engine")
    parser.add_argument("--trt-vae-encode", default=None, help="Path to TRT VAE encode engine")
    parser.add_argument("--trt-vae-decode", default=None, help="Path to TRT VAE decode engine")
    args = parser.parse_args()

    trt_engines = {}
    if args.trt_decoder:
        trt_engines["decoder"] = args.trt_decoder
    if args.trt_vae_encode:
        trt_engines["vae_encode"] = args.trt_vae_encode
    if args.trt_vae_decode:
        trt_engines["vae_decode"] = args.trt_vae_decode

    state = SessionState()
    logger.info("Initializing session...")

    # Server defaults to compile for throughput; --no-compile picks eager.
    # When TRT engine paths are passed on the CLI, those components switch
    # to the tensorrt backend automatically.
    decoder_backend = "compile" if not args.no_compile else "eager"
    vae_backend = "compile" if not args.no_compile else "eager"
    if "decoder" in trt_engines:
        decoder_backend = "tensorrt"
    if "vae_encode" in trt_engines and "vae_decode" in trt_engines:
        vae_backend = "tensorrt"

    state.ensure_session(
        project_root=args.project_root,
        decoder_backend=decoder_backend,
        vae_backend=vae_backend,
        use_flash_attention=not args.no_flash_attn,
        trt_engines=trt_engines or None,
    )
    logger.info("Session ready.")

    # Bind state to handler class
    Handler.state = state

    server = HTTPServer((args.host, args.port), Handler)
    logger.info("Listening on http://{}:{}", args.host, args.port)
    logger.info("Endpoints: /health, /prepare_source, /audio_info, /encode_text, /apply_lora, /remove_loras, /generate, /warmup")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
