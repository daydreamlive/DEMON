"""Offline stress test for the SA3 streaming pipeline — the SA3 analog of
``test_stream_cover_graph.py``: ONE coherent, continuously-evolving song.

How the ACE cover-graph demo stays coherent (re-derived from the code):
every emit is a partial-denoise *cover* of the SAME fixed source latent at
the SAME seed, just at a different denoise strength; the playback head
advances one slice per emit and each slice is read from a fresh emit. Because
all emits are anchored to one source song, position t is the same music in
every emit, so splicing advancing windows reconstructs the song playing
forward while the denoise character breathes. The continuity comes from the
source anchor + deterministic (ODE) solver + shared seed; the evolution comes
from the denoise sweep.

SA3 has the exact same mechanism. ACE ``source_latents`` + ``denoise`` maps
1:1 onto SA3 audio-to-audio ``init_audio`` + ``init_noise_level``:

    ACE  _init_slot:   xt = t_start*noise + (1-t_start)*source,  t_start≈denoise
    SA3  sample_diffusion:  noise = init_data*(1-sigma_max) + noise*sigma_max
                            sigma_max = init_noise_level

So this demo:
  1. generates ONE SA3 source song (text->audio) and uses its latent as the
     shared anchor (the SA3 equivalent of the encoded source audio);
  2. drives the standalone SA3 ringbuffer in ODE
     mode with that ``source_latents`` + a per-tick ``denoise`` swept on a
     cosine timeline + a constant seed — i.e. source-anchored audio-to-audio,
     the cover task for SA3;
  3. splices an advancing playback window from each emit into one montage WAV.

ACE stays untouched: this exercises the SA3-only streaming branch with the
same source-anchored ringbuffer semantics.

Run:
    .venv/Scripts/python.exe demos/test_stream_sa3_graph.py
    .venv/Scripts/python.exe demos/test_stream_sa3_graph.py --source-duration 24 --slice-sec 0.3 --cycles 2
    .venv/Scripts/python.exe demos/test_stream_sa3_graph.py --cover-prompt "lush ambient pads, reverb, slow"
    .venv/Scripts/python.exe demos/test_stream_sa3_graph.py --sampler sde   # stochastic pingpong (seams)
"""

if __name__ != "__main__":
    import sys
    sys.exit(0)

import sys
import time
from pathlib import Path

# --- sys.path: force THIS repo to the front (a sibling ACE-Step editable
#     install otherwise shadows our edited `acestep`), then add scripts/sa3
#     for the SA3 helpers and the vendored stable_audio_3 source. ---
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists())
_SA3_DIR = _REPO_ROOT / "scripts" / "sa3"
_SA3_SRC = _REPO_ROOT / "notes" / "SA3" / "stable-audio-3"
for _p in (str(_SA3_SRC), str(_SA3_DIR), str(_REPO_ROOT)):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_SA3_SRC))
sys.path.insert(0, str(_SA3_DIR))
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import soundfile as sf
import torch

torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

from acestep.fixtures import audio_fixture

from sa3_reference_generate import checkpoint_dir, load_local_model
from sa3_stream_spike import capture_recipe
from sa3_stream_pipeline import (
    SA3Request,
    SA3StreamPipeline,
    decode_sa3_latent,
    encode_sa3_source,
    prepare_sa3_conditioning,
)

OUTPUT_DIR = _REPO_ROOT / "demos" / "outputs" / "stream_sa3_graph"
SEED = 1528

# Default fixture for parity with test_stream_cover_graph.py: SAME-encode a
# real audio loop and use ITS latent as the fixed source anchor (SA3
# audio-to-audio via init_audio). This is the cover task SA3 was designed for.
# Pass --source-prompt instead to fall back to self-generated text->audio
# source (less faithful to the ACE parallel).
DEFAULT_FIXTURE = "low_fi_Gm_loop_60s_gnm.wav"

# Cover stylistic prompt — what the stream renders the source *as*. A cover
# is a transformation target, not a description of the input; SA3's only
# structured controls are the t5gemma text prompt and `seconds_total`, so
# tempo / key / meter MUST be written into this string or the model has no
# way to lock to them. The lowfi fixture is 152 BPM, G minor, 4/4 (per its
# sidecar) — paired here with a genuinely different target style.
DEFAULT_COVER_PROMPT = (
    "driving cinematic synthwave, analog arpeggios, gated reverb snare, "
    "wide saw-lead, 152 bpm, G minor, 4/4"
)

# ---------------------------------------------------------------------------
# CLI flags (same lightweight parser style as the ACE demo)
# ---------------------------------------------------------------------------
_args = sys.argv[1:]


def _get_arg(name, default=None, cast=str):
    # Accept both "--flag value" and "--flag=value" so the script is
    # invokable in environments where space-separated args trip permission
    # gates (e.g. some sandboxed Bash wrappers).
    if name in _args:
        return cast(_args[_args.index(name) + 1])
    pfx = name + "="
    for tok in _args:
        if tok.startswith(pfx):
            return cast(tok[len(pfx):])
    return default


depth = _get_arg("--depth", 8, int)
steps = _get_arg("--steps", 8, int)
source_duration = _get_arg("--source-duration", 60.0, float)
slice_sec = _get_arg("--slice-sec", 0.3, float)
playback_start = _get_arg("--playback-start", 1.0, float)
num_cycles = _get_arg("--cycles", 2, int)
dn_min = _get_arg("--dn-min", 0.4, float)
dn_max = _get_arg("--dn-max", 0.55, float)
# --denoise X holds denoise CONSTANT (no sweep). With ODE + a shared source +
# a constant seed this makes every emit bit-deterministic and identical, so
# the montage is the source song played straight through at strength X — the
# cleanest possible continuity (consecutive-emit MSE -> 0).
const_denoise = _get_arg("--denoise", None, float)
sampler = _get_arg("--sampler", "ode", str)  # ode (deterministic) | sde (pingpong)
# SA3 checkpoint to load: "small-music" (433M, default, fast) or "medium"
# (1.4B + SAME-L, higher quality, ~3x slower DiT). Both are post-trained so
# cfg_scale has no effect on either — quality bump only, no new steering knob.
sa3_model = _get_arg("--sa3-model", "small-music", str)
# Source = SAME-encoded fixture (parity with test_stream_cover_graph.py) by
# default; --source-prompt switches to self-generated text->audio.
fixture_name = _get_arg("--fixture", DEFAULT_FIXTURE, str)
source_prompt = _get_arg("--source-prompt", None, str)
cover_prompt = _get_arg("--cover-prompt", DEFAULT_COVER_PROMPT, str)
explicit_output = _get_arg("--output", None, str)

infer_method = "sde" if sampler == "sde" else "ode"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_dn_tag = f"_dn{const_denoise:g}" if const_denoise is not None else "_sweep"
_src_tag = "promptgen" if source_prompt is not None else Path(fixture_name).stem
OUTPUT_FILE = (
    Path(explicit_output) if explicit_output and Path(explicit_output).is_absolute()
    else (OUTPUT_DIR / explicit_output) if explicit_output
    else OUTPUT_DIR / f"stream_sa3_graph_{sa3_model}_{_src_tag}_{infer_method}{_dn_tag}_d{depth}_s{steps}_{int(source_duration)}s.wav"
)

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
timings = {}


def timed(label, quiet=False):
    class _Timer:
        def __enter__(self_):
            torch.cuda.synchronize()
            self_.t0 = time.perf_counter()
            return self_

        def __exit__(self_, *exc):
            torch.cuda.synchronize()
            self_.ms = (time.perf_counter() - self_.t0) * 1000
            timings.setdefault(label, []).append(self_.ms)
            if not quiet:
                print(f"  [{label}] {self_.ms:.1f}ms")

    return _Timer()


print("=" * 64)
print("Stream SA3 graph — one source-anchored, continuously-evolving song")
print("=" * 64)

# ------------------------------------------------------------------
# Setup: load model
# ------------------------------------------------------------------
with timed("model_load"):
    print(f"[Setup] Loading SA3 {sa3_model}...")
    sam = load_local_model(checkpoint_dir(sa3_model), device="cuda", model_half=True)

sr = sam.model.sample_rate
ds = sam.model.pretransform.downsampling_ratio
SAMPLE_RATE = sr
print(f"[Setup] sample_rate={sr}  latent_rate={sr/ds:.4f} Hz  solver={infer_method}")
print(f"[Setup] depth={depth} steps={steps} source_duration={source_duration}s "
      f"slice={slice_sec}s cycles={num_cycles} denoise=[{dn_min},{dn_max}]")
if source_prompt is not None:
    print(f"[Setup] source = self-generated text->audio  prompt={source_prompt!r}")
else:
    print(f"[Setup] source = SAME-encoded fixture  fixture={fixture_name}")
print(f"[Setup] cover_prompt={cover_prompt!r}")
print(f"[Setup] output={OUTPUT_FILE.name}")

# ------------------------------------------------------------------
# Capture the COVER recipe. This always runs: it gives us the cond bundle
# (cross-attn, padding mask, inpaint placeholders), the schedule machinery
# (`sched_args` lets the SA3 pipeline rebuild schedules at any
# init_noise_level), AND the target latent T from SA3's own
# `_adapt_sample_size(duration)`. The source latent's T MUST match this T or
# the DiT forward will mismatch shapes with the cond bundle.
# ------------------------------------------------------------------
with timed("encode_cover"):
    print(f"[Setup] Encoding cover conditioning (duration={source_duration}s)...")
    cover_cond = prepare_sa3_conditioning(
        sam, prompt=cover_prompt, duration=source_duration, steps=steps,
    )
cond_bundle = cover_cond.cond_bundle
sched_args = cover_cond.sched_args
target_T = cover_cond.latent_frames
target_samples = cover_cond.audio_sample_size

# ------------------------------------------------------------------
# Source acquisition: SAME-encoded fixture (parity with the ACE cover-graph,
# which VAE-encodes acestep/fixtures/inside_confusion_loop_60s_gsm.wav), or
# self-generated text->audio when --source-prompt is set.
# ------------------------------------------------------------------
if source_prompt is not None:
    with timed("gen_source"):
        print(f"[Setup] Generating source song (pingpong, prompt={source_prompt!r})...")
        src_recipe = capture_recipe(
            sam, prompt=source_prompt, duration=source_duration, seed=SEED,
            steps=steps, sampler_type="pingpong",
        )
        source_bcl = src_recipe["ref_latent"]
else:
    with timed("encode_fixture"):
        fixture_path = audio_fixture(fixture_name)
        print(f"[Setup] SAME-encoding fixture: {fixture_path}")
        data, in_sr = sf.read(str(fixture_path), dtype="float32")
        audio_t = torch.from_numpy(
            data.T if data.ndim > 1 else data.reshape(1, -1)
        ).float()
        # SA3/SAME source latent, native layout [1, 256, target_T].
        source_bcl = encode_sa3_source(sam, (in_sr, audio_t), target_samples)

T = source_bcl.shape[-1]
assert T == target_T, (
    f"source latent T ({T}) != cond-bundle T ({target_T}); they must match for the "
    f"DiT forward to align padding_mask and cross-attn against the noise tensor."
)
device, dtype = source_bcl.device, source_bcl.dtype
source_bcl = source_bcl.contiguous().to(device=device, dtype=dtype)
print(f"  latent T={T} (~{T/(sr/ds):.1f}s)   source_latents shape={tuple(source_bcl.shape)}")

# ------------------------------------------------------------------
# Build the standalone SA3 ringbuffer.
# ------------------------------------------------------------------
with timed("stream_setup"):
    pipeline = SA3StreamPipeline.from_sched_args(
        sam.model.model,
        sched_args,
        depth=depth,
        steps=steps,
        device=device,
        dtype=dtype,
        sampler=infer_method,
    )

    # Self-check: rebuilding at denoise=1.0 must reproduce the captured full
    # schedule path is wired; schedule[0] must preserve sigma_max.
    full = pipeline._schedule(1.0)
    print(f"  schedule self-check: len={full.numel()} start={float(full[0]):.3f} end={float(full[-1]):.3f}")

    def make_request(denoise: float) -> SA3Request:
        return SA3Request(
            cond_bundle=cond_bundle,
            latent_frames=T,
            source_latents=source_bcl,   # the shared anchor (init_audio)
            seed=SEED,                   # constant seed => shared noise realization
            denoise=denoise,             # init_noise_level / sigma_max
        )

print(f"  Pipeline ready (backend=sa3, solver={pipeline.sampler})")


# ------------------------------------------------------------------
# Decode helper: engine [1,T,256] -> audio [2, N] (SAME wants [1,256,T]).
# ------------------------------------------------------------------
def decode_latent(latent_btc):
    return decode_sa3_latent(sam, latent_btc).float().clamp(-1, 1)[0].cpu()  # [2, N]


# Decode the source song once — used as the "ideal" reference timeline for the
# objective continuity check (the montage should track it position-for-position).
with timed("decode_source"):
    source_wav = decode_latent(source_bcl)
src_len = source_wav.shape[1]
_src_name = f"source_{sa3_model}_{_src_tag}_{int(source_duration)}s.wav"
sf.write(str(OUTPUT_DIR / _src_name), source_wav.numpy().T, SAMPLE_RATE, format="WAV")
print(f"  source song decoded: {src_len/SAMPLE_RATE:.1f}s, saved {_src_name}")

# ------------------------------------------------------------------
# Denoise timeline (cosine sweep, exactly like the ACE cover graph). Sized so
# the advancing playback window stays inside the source song's valid region.
# ------------------------------------------------------------------
slice_samples = int(slice_sec * SAMPLE_RATE)
playback_offset_samples = int(playback_start * SAMPLE_RATE)
# Leave a slice of margin at the tail.
usable_sec = max(0.0, source_duration - playback_start - 2 * slice_sec)
total_submissions = max(1, int(usable_sec / slice_sec))

if const_denoise is not None:
    denoise_per_tick = [round(float(const_denoise), 4)] * total_submissions
else:
    dn_span = dn_max - dn_min
    denoise_per_tick = [
        round(dn_min + 0.5 * dn_span * (1.0 - np.cos(2 * np.pi * num_cycles * (i / total_submissions))), 4)
        for i in range(total_submissions)
    ]
total_ticks = total_submissions + steps + depth + 4

print(f"\n[Timeline] {total_submissions} submissions over ~{total_ticks} ticks  "
      f"(playback {playback_start:.1f}s -> {playback_start + total_submissions*slice_sec:.1f}s)")
if const_denoise is not None:
    print(f"  denoise CONSTANT: {const_denoise:.3f}  (no sweep — every emit deterministic & identical)")
else:
    print(f"  denoise sweep: {min(denoise_per_tick):.3f} - {max(denoise_per_tick):.3f}  "
          f"({num_cycles} cycle(s))")

# ------------------------------------------------------------------
# Run: submit one request/tick (active phase), then drain. On each emit,
# decode the finished latent and splice the advancing playback window.
# ------------------------------------------------------------------
output_chunks = []
num_completed = 0
prev_dn = None
last_latent = None
mse_values = []
src_cos_values = []        # cosine(montage chunk, source chunk) per emit
denoise_of_emit = []

print(f"\n[Run] Starting pipeline...")
run_start = time.time()

for tick_num in range(total_ticks):
    torch.cuda.synchronize()
    iter_t0 = time.perf_counter()

    if tick_num < total_submissions:
        pipeline.submit(make_request(denoise_per_tick[tick_num]))

    raw = pipeline.tick()  # [1, 256, T] SA3/SAME layout, or None

    torch.cuda.synchronize()
    timings.setdefault("tick", []).append((time.perf_counter() - iter_t0) * 1000)

    if raw is not None:
        # Emits come out in submission (FIFO) order, so the k-th emit carries
        # the k-th submitted denoise and lands at the k-th playback window.
        dn = denoise_per_tick[num_completed] if num_completed < total_submissions else -1.0
        denoise_of_emit.append(dn)

        # Latent-domain continuity: consecutive emits should be close (same
        # song, slowly-varying denoise) — not independent clips.
        if last_latent is not None:
            mse_values.append((raw.float() - last_latent).pow(2).mean().item())
        last_latent = raw.float().clone()

        dec_t0 = time.perf_counter()
        wav = decode_latent(raw)  # [2, N]
        torch.cuda.synchronize()
        timings.setdefault("vae_decode", []).append((time.perf_counter() - dec_t0) * 1000)

        start = playback_offset_samples + num_completed * slice_samples
        end = start + slice_samples
        if end <= wav.shape[1]:
            chunk = wav[:, start:end]
        else:
            chunk = torch.zeros(wav.shape[0], slice_samples)
            avail = max(0, wav.shape[1] - start)
            if avail > 0:
                chunk[:, :avail] = wav[:, start:start + avail]
        output_chunks.append(chunk)

        # Source-tracking: this chunk should resemble the source at the SAME
        # position (a low-denoise cover is near-identical; higher denoise
        # reinterprets). High cos => the montage follows one song.
        if end <= src_len:
            sc = source_wav[:, start:end]
            denomc = (chunk.float().norm() * sc.float().norm() + 1e-9)
            src_cos_values.append(float((chunk.float() * sc.float()).sum() / denomc))

        num_completed += 1
        if dn != prev_dn or num_completed % 20 == 0:
            rms = chunk.float().pow(2).mean().sqrt().item()
            print(f"  #{num_completed:3d} dn={dn:.3f}  rms={rms:.3f}  "
                  f"(playback {start/SAMPLE_RATE:.1f}s-{end/SAMPLE_RATE:.1f}s)")
            prev_dn = dn

    if pipeline.active_slots == 0 and tick_num >= total_submissions:
        break

run_ms = (time.time() - run_start) * 1000
print(f"\n[Run] {num_completed} generations in {run_ms:.0f}ms "
      f"({run_ms / max(num_completed, 1):.1f}ms avg incl. decode)")

# ------------------------------------------------------------------
# Save montage + source.
# ------------------------------------------------------------------
output_wav = torch.cat(output_chunks, dim=1)
total_dur = output_wav.shape[1] / SAMPLE_RATE
print(f"\n[Save] Montage: {total_dur:.1f}s, {tuple(output_wav.shape)}")
sf.write(str(OUTPUT_FILE), output_wav.numpy().T, SAMPLE_RATE, format="WAV")
print(f"  Saved: {OUTPUT_FILE}")

# ------------------------------------------------------------------
# Objective continuity assessment (since we can't listen).
# ------------------------------------------------------------------
print(f"\n{'=' * 64}\nCONTINUITY ASSESSMENT\n{'=' * 64}")

if mse_values:
    s = sorted(mse_values)
    print(f"  consecutive-emit latent MSE:  min={s[0]:.2e}  "
          f"median={s[len(s)//2]:.2e}  max={s[-1]:.2e}")
    print("    (small + smooth => emits are the same song at nearby denoise,")
    print("     not independent generations)")

if src_cos_values:
    a = np.array(src_cos_values)
    print(f"  montage-vs-source cosine/slice: min={a.min():.3f}  "
          f"mean={a.mean():.3f}  max={a.max():.3f}")
    print("    (high => each slice tracks the source song at its playback")
    print("     position; dips correspond to high-denoise reinterpretation)")

# Splice-boundary discontinuity: compare |jump| at each slice seam to the
# in-slice sample-to-sample diff. Ratio ~1 => seamless; >>1 => audible seams.
mono = output_wav.mean(0)
interior = mono[1:] - mono[:-1]
interior_rms = float(interior.pow(2).mean().sqrt()) + 1e-12
seam_idx = [k * slice_samples for k in range(1, len(output_chunks)) if k * slice_samples < mono.shape[0]]
if seam_idx:
    jumps = torch.tensor([abs(float(mono[i] - mono[i - 1])) for i in seam_idx])
    seam_rms = float(jumps.pow(2).mean().sqrt())
    print(f"  splice-seam jump / interior diff RMS: {seam_rms/interior_rms:.2f}x  "
          f"({len(seam_idx)} seams)")
    print("    (~1x => seamless; large => boundary clicks. ODE+shared-seed")
    print("     stays low; SDE/pingpong renoise raises it.)")

# ------------------------------------------------------------------
# Timing summary
# ------------------------------------------------------------------
print(f"\n{'=' * 64}\nTIMING SUMMARY\n{'=' * 64}")
for label in ["model_load", "encode_cover", "encode_fixture", "gen_source",
              "decode_source", "stream_setup", "tick", "vae_decode"]:
    vals = timings.get(label, [])
    if not vals:
        continue
    total = sum(vals)
    if len(vals) == 1:
        print(f"  {label:16s}  {total:8.1f}ms  (1 call)")
    else:
        print(f"  {label:16s}  {total:8.1f}ms total  avg={total/len(vals):6.1f}ms  "
              f"min={min(vals):6.1f}ms  max={max(vals):6.1f}ms  ({len(vals)} calls)")
print("\n" + "=" * 64)
