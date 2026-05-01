"""Interactive stream pipeline with denoise knob and real-time audio."""

import os, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import soundfile as sf
import sounddevice as sd
import gradio as gr

torch.set_grad_enabled(False)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRT_ENGINE = os.path.join(PROJECT_ROOT, "trt_engines", "decoder_mixed_b8_60s", "decoder_mixed_b8_60s.engine")
SOURCE_AUDIO = os.path.join(PROJECT_ROOT, "tests/fixtures", "new_order_confusion_60seconds.wav")
SAMPLE_RATE = 48000

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.engine.diffusion import DiffusionConfig, DiffusionEngine
from acestep.engine.stream import StreamPipeline, SlotRequest
from acestep.nodes.types import Audio, Latent

# ---------------------------------------------------------------
# Init
# ---------------------------------------------------------------
print("Loading model...")
session = Session(
    project_root=os.path.join(PROJECT_ROOT, "checkpoints"),
    use_flash_attention=True,
)
handler = session.handler
device = handler.device
dtype = handler.dtype

print("Loading source audio...")
data, sr = sf.read(SOURCE_AUDIO, dtype="float32")
waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
if sr != SAMPLE_RATE:
    import torchaudio
    waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
waveform = waveform[:2, :int(60.0 * SAMPLE_RATE)]
pool = 1920 * 5
rem = waveform.shape[-1] % pool
if rem:
    waveform = waveform[:, :waveform.shape[-1] - rem]
audio_in = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)

print("Preparing source...")
source = session.prepare_source(audio_in)

print("Encoding conditioning...")
cond = session.encode_text(
    tags="deathstep, heavy bass, dark atmosphere",
    instruction=TASK_INSTRUCTIONS["cover"],
    refer_latent=source.latent,
    bpm=136, duration=60.0, key="G# minor",
)
entry = cond.to_entries()[0]

ctx_lat = source.context_latent.tensor.to(device=device, dtype=dtype)
D_ctx = ctx_lat.shape[2]
T = ctx_lat.shape[1]
cm = torch.ones(1, T, D_ctx, device=device, dtype=dtype)
context_latents = torch.cat([ctx_lat, cm], dim=-1)
source_latents = source.latent.tensor.to(device=device, dtype=dtype)

print("Creating stream pipeline...")
diff_engine = DiffusionEngine(handler.model, trt_engine_path=TRT_ENGINE)
config = DiffusionConfig(infer_steps=8, shift=3.0, noise_on_cpu=True)
pipe = StreamPipeline(diff_engine, config)

# Warmup VAE decode so TRT engine gets found and cached
print("Warming up VAE decode...")
_warmup_lat = source.latent
_warmup_audio = session.decode(_warmup_lat)
print("VAE decode ready (TRT cached).")

# ---------------------------------------------------------------
# State
# ---------------------------------------------------------------
current_denoise = 0.5
current_seed = 1528
running = False
gen_count = 0
last_tick_ms = 0.0
last_vae_ms = 0.0

# Audio double buffer
source_wav = waveform.numpy().T.copy()  # [samples, 2]
audio_buf_a = source_wav.copy()
audio_buf_b = source_wav.copy()
active_buf = audio_buf_a
buf_lock = threading.Lock()
playback_pos = 0


def audio_callback(outdata, frames, time_info, status):
    global playback_pos
    with buf_lock:
        buf = active_buf
    total = len(buf)
    for i in range(frames):
        idx = (playback_pos + i) % total
        outdata[i] = buf[idx]
    playback_pos = (playback_pos + frames) % total


audio_stream = sd.OutputStream(
    samplerate=SAMPLE_RATE, channels=2,
    callback=audio_callback, blocksize=2048,
)


def pipeline_loop():
    global active_buf, audio_buf_a, audio_buf_b, running
    global gen_count, last_tick_ms, last_vae_ms

    use_a = True
    while running:
        pipe.submit(SlotRequest(
            encoder_hidden_states=entry.encoder_hidden_states,
            encoder_attention_mask=entry.encoder_attention_mask,
            context_latents=context_latents,
            seed=current_seed,
            source_latents=source_latents,
            denoise=current_denoise,
        ))

        t0 = time.perf_counter()
        result = pipe.tick()
        last_tick_ms = (time.perf_counter() - t0) * 1000

        if result is not None:
            t1 = time.perf_counter()
            audio_out = session.decode(Latent(tensor=result))
            last_vae_ms = (time.perf_counter() - t1) * 1000

            wav = audio_out.waveform.detach().cpu().float().squeeze(0).numpy().T

            # Write to inactive buffer, then swap
            if use_a:
                audio_buf_a = wav
                with buf_lock:
                    active_buf = audio_buf_a
            else:
                audio_buf_b = wav
                with buf_lock:
                    active_buf = audio_buf_b
            use_a = not use_a
            gen_count += 1


pipeline_thread = None


def start(denoise, seed):
    global running, pipeline_thread, current_denoise, current_seed, gen_count
    current_denoise = denoise
    current_seed = int(seed)
    if running:
        return "Already running"
    running = True
    gen_count = 0
    pipeline_thread = threading.Thread(target=pipeline_loop, daemon=True)
    pipeline_thread.start()
    audio_stream.start()
    return "Running"


def stop():
    global running
    running = False
    audio_stream.stop()
    return "Stopped"


def update_denoise(val):
    global current_denoise
    current_denoise = val
    return f"denoise={val:.2f} | gen={gen_count} | tick={last_tick_ms:.0f}ms | vae={last_vae_ms:.0f}ms"


def update_seed(val):
    global current_seed
    current_seed = int(val)
    return f"seed={current_seed}"


def get_status():
    return (f"denoise={current_denoise:.2f} | seed={current_seed} | "
            f"gen={gen_count} | tick={last_tick_ms:.0f}ms | vae={last_vae_ms:.0f}ms | "
            f"active={pipe.active_slots} | running={running}")


# ---------------------------------------------------------------
# Gradio
# ---------------------------------------------------------------
print("\nReady. Launching UI...")

with gr.Blocks(title="Stream Pipeline") as demo:
    gr.Markdown("# ACE-Step Stream Pipeline")

    with gr.Row():
        start_btn = gr.Button("Start", variant="primary", scale=1)
        stop_btn = gr.Button("Stop", variant="stop", scale=1)
        status_btn = gr.Button("Status", scale=1)

    denoise_slider = gr.Slider(
        minimum=0.1, maximum=1.0, value=0.5, step=0.01,
        label="Denoise",
    )
    seed_num = gr.Number(value=1528, label="Seed", precision=0)
    status_box = gr.Textbox(label="Status", interactive=False)

    start_btn.click(start, inputs=[denoise_slider, seed_num], outputs=status_box)
    stop_btn.click(stop, outputs=status_box)
    status_btn.click(get_status, outputs=status_box)
    denoise_slider.release(update_denoise, inputs=denoise_slider, outputs=status_box)

demo.launch(server_name="127.0.0.1", server_port=7861)
