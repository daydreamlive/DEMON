# Realtime Motion-to-Music (Web)

Browser-based real-time motion-to-music demo. Single-port HTTP +
WebSocket server runs the GPU pipeline alongside the browser client:

- Upload a source audio file, get a live ACE-Step stream back
- Live-editable prompt
- Every knob visible at once in stacked Core / Groups / Keystones
  sections (no bank tab switching)
- Optional hardware MIDI input via the Web MIDI API with per-knob
  **MIDI learn**: click `CC ?` next to any knob, wiggle a physical
  control, and it rebinds live. Mappings persist per option-profile in
  localStorage; click `Reset MIDI map` to restore the auto-map.
- Optional webcam motion input (frame-diff driving `denoise`)
- HUD canvas with waveform background, history trails, playhead, SDE
  curve, and live stats
- Zstd-compressed delta slices decoded in the browser

## Requirements

- **Server**: the full ACE-Step install (CUDA GPU, `uv sync`, prebuilt
  TensorRT engines). No extra dependencies beyond the main project.
- **Client**: any modern Chromium or Firefox. Web MIDI and webcam
  support are optional. HTTPS is *not* required because the server
  binds on the same origin as the WebSocket endpoint.

## Run

From the remote 5090 box (the machine with the GPU):

```bash
uv run python -u -m demos.realtime_motion_graph_web
# or with explicit binds:
uv run python -u -m demos.realtime_motion_graph_web \
    --host 0.0.0.0 --port 8765
# pick the acceleration mode explicitly (default is tensorrt):
uv run python -u -m demos.realtime_motion_graph_web --accel tensorrt
uv run python -u -m demos.realtime_motion_graph_web --accel compile
uv run python -u -m demos.realtime_motion_graph_web --accel eager
```

`--accel {tensorrt,compile,eager}` sets BOTH `decoder_backend` and
`vae_backend` on the underlying `Session`. Default is `tensorrt`.

`--decoder-accel` and `--vae-accel` override `--accel` for one
component at a time. Useful when, for example, only one of the two
TRT engines exists for a given checkpoint, or when you want to debug
one component in eager while the other stays on TRT:

```bash
# Mix-and-match: TRT decoder, eager VAE.
uv run python -u -m demos.realtime_motion_graph_web \
    --accel tensorrt --vae-accel eager
```

The text encoder stays resident in VRAM by default so live prompt edits do not
pay CPU/GPU transfer cost. Add `--offload-text-encoder` on lower-VRAM GPUs to
restore the previous lower-memory behavior.

`--checkpoint <name>` selects which DiT checkpoint to load. The name
must match a directory under `<checkpoints_dir>/`. Full TensorRT mode is
registered for `acestep-v15-turbo` (default, 2B) and
`acestep-v15-xl-turbo` (XL). For XL, build the `b1` TRT decoder profile
first, then launch with:

```bash
uv run python -u -m demos.realtime_motion_graph_web \
    --accel tensorrt --checkpoint acestep-v15-xl-turbo
```

## Headless Performance Benchmark

The browser HUD receives `tick_ms` and `dec_ms` over WebSocket, but the
server does not print structured telemetry. To compare accelerators,
checkpoints, TRT profiles, or VAE settings without browser/audio-device
overhead, run the headless benchmark:

```bash
uv run python -u -m demos.realtime_motion_graph_web.benchmark \
    --accel tensorrt --checkpoint acestep-v15-xl-turbo
```

The benchmark mirrors the backend inference path: fixture/config defaults,
TRT engine selection, `Session(...)`, `prepare_source`, `encode_text`,
`session.stream(...)`, repeated `stream.tick(...)`, and optional VAE decode.
It reports setup timings, per-generation `tick`, `decode`, `tick+decode`
mean/P50/P90/P95/min/max, skip counts, and peak CUDA memory.

Useful variants:

```bash
# Compare compile mode against the same workload.
uv run python -u -m demos.realtime_motion_graph_web.benchmark --accel compile

# Mixed backend, same style as the server flags.
uv run python -u -m demos.realtime_motion_graph_web.benchmark \
    --decoder-accel tensorrt --vae-accel eager

# Persist raw samples and summary stats.
uv run python -u -m demos.realtime_motion_graph_web.benchmark \
    --accel tensorrt --checkpoint acestep-v15-xl-turbo \
    --json runs/xl-trt-bench.json

# Mirror PipelineRunner's decode-skip behavior.
uv run python -u -m demos.realtime_motion_graph_web.benchmark \
    --accel tensorrt --skip-threshold 1e-3
```

By default, `--skip-threshold -1` disables decode skipping so VAE decode
latency is measured on every completed generation. Set `--no-decode` for
decoder-only throughput.

For continuous playback, the browser can loop only the stable middle of the
generated buffer. `engine.loop_head_guard` skips generated intro/fade-in material
and `engine.loop_tail_guard` skips generated outro/fade-out material without
changing latent length or the TRT profile.

The web server also accepts runtime overrides for the seamless-section test:

```bash
uv run python -u -m demos.realtime_motion_graph_web \
    --duration 60 --loop-head-guard 10 --loop-tail-guard 10
```

`--duration` caps the uploaded source in the browser, while the head/tail guards
make the audio worklet wrap inside the generated buffer so trained intro/outro
material is not played. For an exact audible section length, use
`--loop-head-guard 8 --playback-loop-seconds 44` instead of the tail guard.

To isolate live prompt-change cost, use the prompt benchmark. It keeps setup
timings separate, then repeatedly runs the same server-side path used for
prompt edits: text encode, conditioning fusion with the current source latent,
and `stream.conditioning` swap.

```bash
uv run python -u -m demos.realtime_motion_graph_web.prompt_benchmark \
    --accel tensorrt --checkpoint acestep-v15-xl-turbo \
    --json runs/xl-prompt-bench.json
```

It reports `text_encoder`, `conditioning_fusion`, `apply_swap`, total apply
latency, and CUDA allocated/reserved peaks for each measured prompt change.
Add `--offload-text-encoder` to test the lower-VRAM, higher-latency fallback.

Then from any laptop on the same network:

1. Open `http://<server-host>:8765/`
2. Click **Play** — the demo loads the default fixture
   (`inside_confusion_loop_60s_gsm.wav`). Fixtures stream from the
   `daydreamlive/demon-fixtures` Hugging Face dataset on first request
   and are cached locally.
3. Switch fixtures any time using the selector at the top of the
   Advanced drawer; switching tears down the session and restarts with
   the new audio.
4. Cold start takes ~15 s while the server loads the model + TRT
   engines (or longer on `--accel compile`); once it's ready the UI
   switches to the live HUD view.

### Audio source vs. video

Audio is the **primary** source: the demo always loads from the
canonical fixture set (`daydreamlive/demon-fixtures` on Hugging Face,
listed in `acestep.fixtures.KNOWN_FIXTURES`), served by the web server
at `/fixtures/<name>` via lazy HF download.
Video is **optional and secondary** — drop any `.mp4`/`.webm`/`.mov`
into `static/videos/` to attach the audio-reactive shader pipeline.
With no videos present the demo runs audio-only (graph mode is the
default and looks the same).

## Layout

```
demos/realtime_motion_graph_web/
├── README.md
├── __init__.py
├── __main__.py               # `python -m demos.realtime_motion_graph_web`
├── server.py                 # HTTP (static) + WebSocket multiplex on one port
├── backend.py                # GPU handle_client coroutine
├── pipeline.py               # PipelineRunner (graph-driven streaming loop)
├── audio_engine.py           # server-side audio buffer
├── knobs.py                  # MIDI knob bank definitions
├── protocol.py               # wire format (Python source of truth for protocol.js)
└── static/
    ├── index.html            # launcher + live HUD DOM
    ├── style.css
    ├── main.js               # orchestration, UI, session loops
    ├── protocol.js           # wire format (float16, zstd delta, slice hdr)
    ├── audio.js              # main-thread wrapper around the worklet
    ├── audio-worklet.js      # realtime buffer / swap / patch / delta-add
    ├── knobs.js              # bank definitions + flat value store
    ├── motion.js             # webcam motion tracker (canvas frame diff)
    ├── hud.js                # canvas HUD (waveform, trails, stats)
    └── lib/
        └── fzstd.min.js      # bundled pure-JS zstd decoder
```

## Protocol

The WebSocket protocol is defined in `protocol.py` (the Python source
of truth that `static/protocol.js` mirrors):

- **Init**: JSON config -> binary audio upload
  (`<uint32 channels><uint32 samples>` + float32 PCM)
- **Server init**: JSON ready + binary float16 initial buffer
- **Streaming**: JSON params/prompt out, binary slice (raw float16 or
  zstd-compressed float16 delta) + `params_update` / `prompt_applied`
  JSON messages in

`server.py` multiplexes HTTP static-file serving and the WebSocket
upgrade onto one TCP port; the WS handshake hands off to
`backend.handle_client`.

## Audio-reactive video

The video is rendered through a small WebGL2 shader pipeline
(`static/effects.js`) so it visually responds to the music in real
time. Two effects:

- **Color parallax** — saturated regions drift horizontally with a
  slow sway plus a punch on every kick.
- **Bloom on kick** — luminance-thresholded bloom that brightens with
  the bass envelope.

Defaults live in `static/config.json` under `effects`:

```json
"effects": {
  "parallax_strength": 0.4,
  "bloom_on_kick": 0.3,
  "bloom_threshold": 0.15
}
```

The same kick amplitude is exposed to CSS as `--bloom-amount`, so the
perimeter HUD bars and the cursor halo glow in lockstep with the
shader bloom on the video. No knobs in the public UI — edit
`config.json` and refresh to retune.

**Curator setup: nothing.** Color parallax is saturation-driven, not
depth-driven, so there is no preprocessing step and no depth map
sidecars to generate. Drop the source video into `static/videos/`
and run the server as usual. If WebGL2 is unavailable the canvas is
hidden and the plain video plays as fallback.

## Browser notes

- **Web Audio**: an `AudioWorkletNode` drives a shared PCM buffer that
  the main thread patches in place on each slice. Same crossfade logic
  as the native `AudioEngine` (50 ms on swap, in-place delta add
  otherwise).
- **Web MIDI**: auto-attaches the first input. Values use the
  endless-encoder two's-complement CC semantics from the native client
  so existing controllers just work.
- **Webcam**: `getUserMedia` with a low-res capture canvas and a simple
  abs-diff detector, smoothed like the OpenCV version.
- **Zstd**: bundled `fzstd` UMD build under `static/lib/`. Falls back to
  jsdelivr CDN if the bundled copy is missing (e.g. while hot-iterating
  without a build step).

## Troubleshooting

- **"fzstd library not loaded"**: `static/lib/fzstd.min.js` did not
  download or load. Re-fetch from
  `https://cdn.jsdelivr.net/npm/fzstd@0.1.1/umd/index.min.js` and place
  it under `static/lib/`.
- **"WebSocket connection failed"**: verify `--ws-port` is reachable
  from the browser (firewall, reverse proxy). The page and the
  WebSocket are on different ports.
- **Audio plays silent on first connect**: browsers gate audio on a
  user gesture; `Connect & start` counts as one, so this should "just
  work" but if it doesn't, click anywhere in the HUD view.
- **No MIDI devices listed**: Web MIDI requires `localhost` or HTTPS in
  Chromium. Use `http://localhost:8080` locally, or run behind a
  reverse proxy with TLS for remote access.
- **Webcam permission denied**: same-origin constraint as MIDI. Switch
  to on-screen knobs mode if the browser blocks the camera.
- **Cold start long**: the GPU server rebuilds the pipeline on every
  new connection, same as the native server. Reuse connections when
  possible.
