"""ACE-Step LEGO stem generation helpers for the realtime demo.

The realtime demo usually runs turbo/xl checkpoints for low-latency streaming,
but LEGO generation is a base-family task. This module keeps that path separate
and returns unmerged layer tensors that the browser can mix as overlays.
"""

from __future__ import annotations

import inspect
import gc
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

torch.set_grad_enabled(False)

from acestep.constants import SFT_GEN_PROMPT, TRACK_NAMES
from acestep.engine.session import Session
from acestep.nodes import Audio
from acestep.nodes.types import TextEmbed
from acestep.paths import checkpoints_dir


SAMPLE_RATE = 48_000
LATENT_FRAME_SAMPLES = 1_920
DEFAULT_MAX_DURATION_SECONDS = 240.0
SUPPORTED_MODELS = ("acestep-v15-base", "acestep-v15-xl-base")

ProgressCallback = Callable[[str, str, str | None], None]


@dataclass(frozen=True)
class LegoStemOptions:
    model: str = "acestep-v15-base"
    seed: int = 1528
    steps: int = 50
    shift: float = 1.0
    cfg_scale: float = 7.0
    lyrics: str = ""
    vocal_language: str = "unknown"
    bpm: int | None = None
    key_scale: str = ""
    time_signature: str = ""
    start: float = 0.0
    end: float | None = None
    device: str = "cuda"


def clear_trt_vae_cache() -> None:
    """Evict process-global TRT VAE engines before memory-heavy LEGO work."""
    try:
        from acestep.nodes import vae_nodes

        cache = getattr(vae_nodes, "_trt_vae_cache", {})
        for path in list(cache.keys()):
            try:
                vae_nodes._evict_trt_vae(path)
            except Exception:
                cache.pop(path, None)
        stream = getattr(vae_nodes, "_trt_stream", None)
        if stream is not None:
            try:
                stream.free()
            except Exception:
                pass
            vae_nodes._trt_stream = None
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def enable_lego_cpu_offload(session: Session) -> None:
    """Keep LEGO's heavy modules mutually exclusive on GPU."""
    handler = session.handler
    handler.offload_to_cpu = True
    handler.offload_dit_to_cpu = True
    if handler.model is not None:
        handler._recursive_to_device(handler.model, "cpu")
    if handler.silence_latent is not None:
        handler.silence_latent = handler.silence_latent.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_lego_prompts(prompts: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    valid = set(TRACK_NAMES)
    for raw_track, raw_prompt in prompts.items():
        track = str(raw_track).strip()
        prompt = str(raw_prompt).strip()
        if track not in valid:
            raise ValueError(f"Unsupported LEGO track: {track!r}")
        if not prompt:
            raise ValueError(f"Prompt for LEGO track {track!r} cannot be empty")
        normalized[track] = prompt
    if not normalized:
        raise ValueError("No LEGO tracks were selected")
    return normalized


def require_checkpoint(model_name: str) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported LEGO model {model_name!r}; expected one of "
            f"{', '.join(SUPPORTED_MODELS)}"
        )
    model_dir = checkpoints_dir() / model_name
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Checkpoint {model_name!r} is not available at {model_dir}. "
            f"Download it first with: uv run acestep-download --model {model_name}"
        )


def audio_from_waveform(
    waveform: torch.Tensor,
    *,
    max_duration: float = DEFAULT_MAX_DURATION_SECONDS,
) -> Audio:
    """Normalize a 48 kHz waveform to stereo and latent-frame alignment."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(f"Expected waveform [channels, samples], got {list(waveform.shape)}")
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    else:
        waveform = waveform[:2]

    max_samples = int(max_duration * SAMPLE_RATE)
    sample_count = min(int(waveform.shape[-1]), max_samples)
    sample_count = (sample_count // LATENT_FRAME_SAMPLES) * LATENT_FRAME_SAMPLES
    if sample_count < LATENT_FRAME_SAMPLES:
        raise ValueError("LEGO source audio is too short after frame alignment")
    return Audio(
        waveform=waveform[:, :sample_count].detach().cpu().float().contiguous(),
        sample_rate=SAMPLE_RATE,
    )


def audio_waveform_2d(audio: Audio) -> torch.Tensor:
    waveform = audio.waveform
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.dim() != 2:
        raise ValueError(f"Expected 2D audio waveform, got shape {list(waveform.shape)}")
    return waveform


def _match_source_length(waveform: torch.Tensor, samples: int) -> torch.Tensor:
    waveform = waveform.detach().cpu().float()
    if waveform.shape[-1] == samples:
        return waveform.contiguous()
    if waveform.shape[-1] > samples:
        return waveform[:, :samples].contiguous()
    pad = torch.zeros(
        waveform.shape[0],
        samples - waveform.shape[-1],
        dtype=waveform.dtype,
    )
    return torch.cat([waveform, pad], dim=-1).contiguous()


def encode_audio_eager(session: Session, audio: Audio) -> torch.Tensor:
    """VAE-encode without using the process-global TRT VAE node cache."""
    handler = session.handler
    with handler._load_model_context("vae"):
        latents = handler._encode_audio_to_latents(audio.waveform)
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    return latents


def decode_audio_eager(session: Session, latents: torch.Tensor) -> Audio:
    """VAE-decode without using the process-global TRT VAE node cache."""
    handler = session.handler
    lat_bdt = latents.transpose(1, 2)
    with handler._load_model_context("vae"):
        waveform = handler.tiled_decode(lat_bdt)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        waveform = waveform.detach().to("cpu").float()
    return Audio(waveform=waveform, sample_rate=SAMPLE_RATE)


def official_meta(
    *,
    duration_s: float,
    bpm: int | None,
    key_scale: str,
    time_signature: str,
) -> str:
    bpm_value: Any = bpm if bpm else "N/A"
    key_value = key_scale if key_scale.strip() else "N/A"
    time_value = (
        time_signature
        if time_signature.strip() and time_signature != "N/A"
        else "N/A"
    )
    return (
        f"- bpm: {bpm_value}\n"
        f"- timesignature: {time_value}\n"
        f"- keyscale: {key_value}\n"
        f"- duration: {int(duration_s)} seconds\n"
    )


def encode_text_official(
    *,
    session: Session,
    prompt: str,
    lyrics: str,
    instruction: str,
    duration_s: float,
    vocal_language: str,
    bpm: int | None,
    key_scale: str,
    time_signature: str,
) -> tuple[TextEmbed, str, str, str]:
    handler = session.handler
    meta = official_meta(
        duration_s=duration_s,
        bpm=bpm,
        key_scale=key_scale,
        time_signature=time_signature,
    )
    text_prompt = SFT_GEN_PROMPT.format(instruction, prompt, meta)
    lyrics_prompt = f"# Languages\n{vocal_language}\n\n# Lyric\n{lyrics}<|endoftext|>"

    with handler._load_model_context("text_encoder"):
        text_tokens = handler.text_tokenizer(
            text_prompt,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        lyric_tokens = handler.text_tokenizer(
            lyrics_prompt,
            padding="longest",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        text_ids = text_tokens.input_ids.to(handler.device)
        lyric_ids = lyric_tokens.input_ids.to(handler.device)
        text_embed = TextEmbed(
            text_hidden_states=handler.infer_text_embeddings(text_ids),
            text_attention_mask=text_tokens.attention_mask.to(handler.device).bool(),
            lyric_hidden_states=handler.infer_lyric_embeddings(lyric_ids),
            lyric_attention_mask=lyric_tokens.attention_mask.to(handler.device).bool(),
        )
    return text_embed, text_prompt, lyrics_prompt, meta


def guidance_kwarg_name(model: torch.nn.Module) -> str:
    parameters = inspect.signature(model.generate_audio).parameters
    if "diffusion_guidance_scale" in parameters:
        return "diffusion_guidance_scale"
    return "diffusion_guidance_sale"


def generate_lego_stems(
    waveform: torch.Tensor,
    prompts: Mapping[str, str],
    *,
    options: LegoStemOptions | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, torch.Tensor]:
    """Generate one or more unmerged LEGO layers from a source waveform."""
    opts = options or LegoStemOptions()
    prompt_map = normalize_lego_prompts(prompts)
    require_checkpoint(opts.model)
    clear_trt_vae_cache()

    source_audio = audio_from_waveform(waveform)
    duration_s = source_audio.waveform.shape[-1] / source_audio.sample_rate
    source_samples = int(source_audio.waveform.shape[-1])

    session: Session | None = None
    try:
        t0 = time.perf_counter()
        session = Session(
            project_root=str(checkpoints_dir()),
            config_path=opts.model,
            device=opts.device,
            decoder_backend="eager",
            vae_backend="eager",
        )
        print(f"[LEGO] base session loaded in {time.perf_counter() - t0:.1f}s")

        source_latent = encode_audio_eager(session, source_audio)
        handler = session.handler
        device = handler.device
        dtype = handler.dtype
        src_latents = source_latent.to(device=device, dtype=dtype)
        total_latent_frames = src_latents.shape[1]

        region_start = max(0.0, float(opts.start))
        region_end = duration_s if opts.end is None else min(float(opts.end), duration_s)
        if region_start >= region_end:
            raise ValueError(
                f"LEGO start ({region_start:.3f}) must be less than end "
                f"({region_end:.3f}) within source duration ({duration_s:.3f}s)"
            )
        start_frame = int(region_start * 25.0)
        end_frame = int(region_end * 25.0)
        start_frame = max(0, min(start_frame, total_latent_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_latent_frames))

        chunk_mask_2d = torch.zeros(
            src_latents.shape[0],
            total_latent_frames,
            device=device,
            dtype=torch.bool,
        )
        chunk_mask_2d[:, start_frame:end_frame] = True
        chunk_masks = chunk_mask_2d.to(dtype=dtype).unsqueeze(-1).expand_as(src_latents)
        repaint_mask = chunk_mask_2d.clone()
        is_covers = torch.zeros(src_latents.shape[0], device=device, dtype=torch.bool)
        attention_mask = torch.ones(
            src_latents.shape[0],
            src_latents.shape[1],
            device=device,
            dtype=dtype,
        )
        refer_audio = handler.silence_latent[:, :750, :].to(
            device=device,
            dtype=dtype,
        )
        refer_order_mask = torch.zeros(1, device=device, dtype=torch.long)
        guidance_name = guidance_kwarg_name(handler.model)

        out: dict[str, torch.Tensor] = {}
        for track, prompt in prompt_map.items():
            if progress is not None:
                progress(track, "running", None)
            instruction = f"Generate the {track.upper()} track based on the audio context:"
            text_embed, _, _, _ = encode_text_official(
                session=session,
                prompt=prompt,
                lyrics=opts.lyrics,
                instruction=instruction,
                duration_s=duration_s,
                vocal_language=opts.vocal_language,
                bpm=opts.bpm,
                key_scale=opts.key_scale,
                time_signature=opts.time_signature,
            )
            generate_kwargs = {
                "text_hidden_states": text_embed.text_hidden_states.to(dtype),
                "text_attention_mask": text_embed.text_attention_mask,
                "lyric_hidden_states": text_embed.lyric_hidden_states.to(dtype),
                "lyric_attention_mask": text_embed.lyric_attention_mask,
                "refer_audio_acoustic_hidden_states_packed": refer_audio,
                "refer_audio_order_mask": refer_order_mask,
                "src_latents": src_latents,
                "chunk_masks": chunk_masks,
                "is_covers": is_covers,
                "silence_latent": handler.silence_latent.to(device=device, dtype=dtype),
                "attention_mask": attention_mask,
                "seed": opts.seed,
                "infer_method": "ode",
                "infer_steps": opts.steps,
                guidance_name: opts.cfg_scale,
                "audio_cover_strength": 1.0,
                "cfg_interval_start": 0.0,
                "cfg_interval_end": 1.0,
                "shift": opts.shift,
                "repaint_mask": repaint_mask,
                "clean_src_latents": src_latents,
                "repaint_crossfade_frames": 10,
                "repaint_injection_ratio": 0.5,
                "use_progress_bar": False,
            }
            with handler._load_model_context("model"):
                outputs = handler.model.generate_audio(**generate_kwargs)
            target_latents = outputs["target_latents"]
            del outputs
            decoded = decode_audio_eager(session, target_latents)
            out[track] = _match_source_length(audio_waveform_2d(decoded), source_samples)
            del target_latents, text_embed, generate_kwargs
            del decoded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if progress is not None:
                progress(track, "ready", None)
        return out
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            clear_trt_vae_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def generate_lego_stems_isolated(
    waveform: torch.Tensor,
    prompts: Mapping[str, str],
    *,
    options: LegoStemOptions | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, torch.Tensor]:
    """Generate LEGO layers in short-lived subprocesses to release CUDA fully."""
    opts = options or LegoStemOptions()
    prompt_map = normalize_lego_prompts(prompts)
    require_checkpoint(opts.model)

    repo_root = Path(__file__).resolve().parents[2]
    waveform_cpu = waveform.detach().cpu().float().contiguous()
    out: dict[str, torch.Tensor] = {}
    with tempfile.TemporaryDirectory(prefix="demon_lego_") as tmp:
        tmp_dir = Path(tmp)
        waveform_path = tmp_dir / "waveform.pt"
        torch.save(waveform_cpu, waveform_path)
        for track, prompt in prompt_map.items():
            if progress is not None:
                progress(track, "running", None)
            request_path = tmp_dir / f"{track}_request.json"
            output_path = tmp_dir / f"{track}_output.pt"
            request_path.write_text(
                json.dumps({
                    "waveform_path": str(waveform_path),
                    "output_path": str(output_path),
                    "track": track,
                    "prompt": prompt,
                    "options": asdict(opts),
                }),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "demos.realtime_motion_graph_web.lego_stems",
                    "--worker",
                    str(request_path),
                ],
                cwd=str(repo_root),
                text=True,
                capture_output=True,
            )
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(
                    f"LEGO {track} worker failed"
                    + (f": {detail[-2000:]}" if detail else "")
                )
            payload = torch.load(output_path, map_location="cpu", weights_only=True)
            out[track] = payload["stem"].detach().cpu().float().contiguous()
            if progress is not None:
                progress(track, "ready", None)
    return out


def _run_worker(request_path: str) -> None:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    waveform = torch.load(request["waveform_path"], map_location="cpu", weights_only=True)
    opts = LegoStemOptions(**request["options"])
    track = str(request["track"])
    prompt = str(request["prompt"])
    stems = generate_lego_stems(waveform, {track: prompt}, options=opts)
    torch.save({"stem": stems[track]}, request["output_path"])


def _main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2])
        return
    raise SystemExit("Usage: python -m demos.realtime_motion_graph_web.lego_stems --worker REQUEST_JSON")


if __name__ == "__main__":
    _main()
