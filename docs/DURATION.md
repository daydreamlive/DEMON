# Duration Semantics in DEMON

DEMON uses several different "duration" values. They are related, but they are
not interchangeable:

- **Input audio duration** controls how much source audio is uploaded, encoded,
  and used as cover/context material.
- **Latent duration** is the number of ACE-Step latent frames the diffusion
  model generates. ACE-Step uses 25 latent frames per second.
- **ACE text metadata duration** is the `- duration: ...` line sent to ACE-Step's
  text encoder. In the realtime demo it follows the actual processed audio
  duration; there is no separate public override in this PR.
- **TRT profile duration** is the maximum sequence length an installed TensorRT
  engine can accept.
- **VAE window duration** controls how much generated audio is decoded per
  realtime patch. It does not change what the DiT generates.
- **Playback loop duration** controls how much of the generated client buffer is
  actually heard before wrapping. It can hide model-written outros/fades.
- **Playback head/tail guards** skip generated intro/outro material and loop a
  stable middle section for continuous playback.

For continuous playback, the important distinction is: **model length and audible
loop length are different controls.** The model can still generate a 60 second
latent while the browser loops only the stable middle of that buffer.

## ACE-Step Timing Basics

DEMON runs ACE-Step 1.5 in a 48 kHz stereo audio pipeline.

- Audio sample rate: `48000` samples/second.
- Latent frame rate: `25` frames/second.
- Samples per latent frame: `1920`.
- The DiT patch size is `2`, so a 60 second latent has about `1500` latent frames
  and about `750` DiT patch tokens.
- The realtime demo trims uploaded audio to a multiple of `1920 * 5` samples
  (5 latent frames, or 0.2 seconds) so VAE and TRT shapes stay aligned.

The generated sequence length is ultimately driven by the latent/source tensor
shape. The text prompt's `duration` line should describe that shape; it is not a
separate loop-control surface in the realtime demo.

## Argument and Config Reference

| Name | Where | Changes model tensor length? | Changes ACE text metadata? | Changes audible playback length? | Notes |
| --- | --- | --- | --- | --- | --- |
| `--duration` | Web demo, benchmarks | Yes | Yes, because metadata follows actual processed audio duration | Yes, because the uploaded/generated buffer is shorter | Caps source/input audio before VAE encode. |
| `--loop-head-guard` | Web demo | No | No | Yes | Skips the first N seconds during browser playback. `--loop-guard-head` is accepted as an alias; prefer this spelling. |
| `--loop-tail-guard` | Web demo | No | No | Yes | Skips the final N seconds during browser playback. |
| `engine.loop_head_guard` | `static/config.json` | No | No | Yes | Config version of `--loop-head-guard`. |
| `engine.loop_tail_guard` | `static/config.json` | No | No | Yes | Config version of `--loop-tail-guard`. |
| `--playback-loop-seconds` | Web demo | No | No | Yes | Sets exact browser loop length after the head guard; wins over tail guard when >0. |
| `engine.playback_loop_seconds` | `static/config.json` | No | No | Yes | Config version of `--playback-loop-seconds`. |
| `--vae-window` / `engine.vae_window` | Benchmarks / config | No | No | No | Changes windowed VAE decode size and latency. |
| `engine.crop` | `static/config.json` | No | No | Yes-ish | Older server-side buffer trim. Prefer playback loop controls for seamless sections. |
| `--checkpoint` | Web demo, benchmarks | No by itself | No | No | Selects ACE-Step 1.5 2B vs XL checkpoint; changes available TRT duration profiles. |

## Input Audio Duration

Input duration is the amount of source audio DEMON actually loads or uploads.
It affects:

- VAE encode length.
- Source latent length.
- Semantic/context latent length.
- Diffusion tensor shape `[B, T, 64]`.
- RoPE positions inside the DiT, because the sequence has `T / 2` patch tokens.
- TRT profile selection in TensorRT mode.
- The size of the initial client playback buffer.

In the realtime web demo:

```powershell
uv run python -u -m demos.realtime_motion_graph_web --duration 60
```

`--duration 60` tells the browser to upload at most 60 seconds of the selected
fixture/source. The backend still applies its own cap based on the largest
registered TRT profile for the selected checkpoint.

In the headless benchmark:

```powershell
uv run python -u -m demos.realtime_motion_graph_web.benchmark --duration 60
```

`--duration` controls how much fixture/local audio the benchmark loads.

## ACE Text Metadata Duration

`EncodeText` still builds ACE metadata that contains a duration line:

```text
# Metas
- bpm: ...
- timesignature: ...
- keyscale: ...
- duration: ...
```

For the realtime cover path, this value is now derived from the processed source
duration:

- The browser limits upload length with `--duration` when provided.
- The backend may cap that further to the selected engine's maximum supported
  profile.
- The backend trims to a VAE/TRT-friendly frame multiple.
- The resulting `audio_duration_s` is what `encode_text()` receives.

That keeps text metadata, source latent length, TRT profile selection, and VAE
decode length aligned. A separate metadata-duration override is intentionally
not part of this PR; the continuous-playback work is handled at the playback
window instead of by lying to the text encoder.

## Generated Latent Duration

For cover/realtime workflows, generated latent duration follows the source
latent. If the uploaded source is 56.6 seconds, the source latent is about
`56.6 * 25 = 1415` frames, and each completed generation has the same frame
count.

For text-to-music workflows with no source/context latent, `duration` may be
used to create a silence/context latent of `int(duration * 25)` frames. In those
flows, the same value often controls both metadata and latent length because
there is no uploaded source setting the shape.

## TensorRT Profile Duration

TensorRT engines are built for maximum durations. DEMON chooses the smallest
built profile that can fit the actual input/latent duration.

Registered profiles:

- `acestep-v15-turbo` (2B): 60s, 120s, 240s.
- `acestep-v15-xl-turbo` (XL): 60s, 120s.

Examples:

```powershell
# 2B checkpoint, 60 second source: uses a 60s profile when built.
uv run python -u -m demos.realtime_motion_graph_web --accel tensorrt --duration 60

# 2B checkpoint, longer source: can use up to the 240s registered profile.
uv run python -u -m demos.realtime_motion_graph_web --accel tensorrt --duration 180

# XL TRT currently has registered profiles up to 120s.
uv run python -u -m demos.realtime_motion_graph_web `
  --accel tensorrt --checkpoint acestep-v15-xl-turbo --duration 120
```

If the smallest fitting profile is not built, DEMON falls back to a larger built
profile when available and warns about extra VRAM. If no built profile can fit,
startup fails with a build command.

Build examples:

```powershell
# 2B, all components, 240s profile.
uv run python -m acestep.engine.trt.build --all --duration 240

# XL decoder profiles are built with the XL checkpoint.
uv run python -m acestep.engine.trt.build --all --duration 120 --checkpoint acestep-v15-xl-turbo
```

## VAE Window Duration

`vae_window` controls how much audio the VAE decodes around the current playback
position each realtime tick. It is a performance/decode setting, not a generation
length setting.

Config:

```json
{
  "engine": {
    "vae_window": 6.0
  }
}
```

Effects:

- Smaller windows decode faster.
- The DiT still generates the full latent.
- The client buffer is patched only around the decoded window.
- The demo uses overlap/context around the window to avoid VAE boundary
  artifacts.
- With `cyclic=True`, the VAE decode can draw boundary context from the opposite
  end of the latent for loop-friendly decoding.

Use `--vae-window` in benchmarks:

```powershell
uv run python -u -m demos.realtime_motion_graph_web.benchmark `
  --duration 60 --vae-window 6
```

## Playback Loop Duration

Playback loop duration is client-side only. It controls which part of the current
audio buffer is heard before the worklet wraps. The model still generates the
full latent.

This is the most direct tool for seamless sections. If ACE generates a fade-in at
the beginning or a fade-out/ending riff at the end, do not play those sections:
loop the middle of the generated buffer.

The staged loop-guard work adds a real audible window:

- `AudioPlayer` computes `loopStartFrame` and `loopEndFrame`.
- `audio-worklet.js` wraps between those frames instead of always wrapping the
  entire buffer.
- The UI progress display uses the audible window, not the full source length.
- Source swaps reset playback to the head guard and the backend's audio engine
  starts windowed decoding from the same guarded position.
- Remote playback starts/swap-gates on the first generated slice, so the raw
  source placeholder is not heard as the first "generated" region.

### Head and Tail Guards

```powershell
uv run python -u -m demos.realtime_motion_graph_web `
  --duration 60 --loop-head-guard 8 --loop-tail-guard 8
```

For a 60 second uploaded/generated buffer, `--loop-head-guard 8 --loop-tail-guard
8` makes the audible loop about 44 seconds. The first and final 8 seconds remain
in the buffer for generation and VAE context, but the listener never reaches
them.

Config:

```json
{
  "engine": {
    "loop_head_guard": 8.0,
    "loop_tail_guard": 8.0
  }
}
```

### Exact Playback Loop Length

```powershell
uv run python -u -m demos.realtime_motion_graph_web `
  --duration 60 --loop-head-guard 8 --playback-loop-seconds 44
```

`--playback-loop-seconds` starts after `loop_head_guard` and sets the audible loop
length directly. When it is greater than zero, it overrides `loop_tail_guard`.

Config:

```json
{
  "engine": {
    "playback_loop_seconds": 44.0
  }
}
```

## `crop`

`engine.crop` is an older/demo-level output trim. When greater than zero, the
server crops the initial buffer and full-decode outputs to that many seconds.
For continuous playback experiments, prefer `loop_head_guard` plus
`loop_tail_guard` or
`playback_loop_seconds` because they preserve the full generated buffer while
changing only the audible wrap point.

## Recommended Continuous Playback Recipes

### Baseline: Actual Duration Everywhere

```powershell
uv run python -u -m demos.realtime_motion_graph_web `
  --accel tensorrt --checkpoint acestep-v15-xl-turbo --duration 60 `
  --loop-head-guard 0 --loop-tail-guard 0
```

This is useful as a control. ACE sees a 60 second song and the browser plays the
full 60 second buffer.

### Seamless Section Loop

```powershell
uv run python -u -m demos.realtime_motion_graph_web `
  --accel tensorrt --checkpoint acestep-v15-xl-turbo --duration 60 `
  --loop-head-guard 8 --loop-tail-guard 8
```

This generates from a 60 second source and loops only the middle 44 seconds,
skipping generated intro/fade-in and outro/fade-out material. Increase either
guard if the section still feels like it starts or lands before the wrap.

### Fixed 48 Second Audible Section

```powershell
uv run python -u -m demos.realtime_motion_graph_web `
  --accel tensorrt --checkpoint acestep-v15-xl-turbo --duration 60 `
  --loop-head-guard 6 --playback-loop-seconds 48
```

This is useful for installation-style playback where the audible section should
have a stable length regardless of source duration.

### Headless Timing / Regression Check

```powershell
uv run python -u -m demos.realtime_motion_graph_web.benchmark `
  --accel tensorrt --checkpoint acestep-v15-xl-turbo `
  --duration 60 --no-decode
```

This validates source length, profile selection, and diffusion timing without
browser/audio playback. Playback loop guards are browser-side, so they are not
measured by this benchmark.

## Common Confusions

### Does ACE text metadata make a longer generation?

No. In the realtime demo, text metadata duration follows the processed source
duration. The generated tensor length follows the actual input/context latent.

### Does `--playback-loop-seconds 48` make the model generate 48 seconds?

No. It only changes where the browser audio worklet wraps. The model can still
generate a 60 second latent, and server patches can still update the full buffer.

### Does `vae_window` change song length?

No. It changes decode window size and latency.

### Why can a track still sound like it ends?

ACE-Step was trained on full songs/clips. Ending behavior may come from several
signals:

- Duration metadata.
- Absolute position inside the generated latent.
- Boundary/context effects near the tail.
- Source audio that itself has an ending.
- Prompt semantics such as "song", "track", or genres with strong cadences.

The current DEMON strategy is to loop a stable middle excerpt with
`loop_head_guard` and `loop_tail_guard` (or `playback_loop_seconds`). If the
middle still behaves too much like the end of a complete song, inspect the source
audio and prompts first; do not add another duration knob unless the model path
actually needs one.
