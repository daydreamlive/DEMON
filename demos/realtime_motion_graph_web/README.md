# Realtime Motion-to-Music (Web)

Browser-based port of `demos.realtime_motion_graph`. The GPU server is
byte-identical to the native server, so the browser client is a drop-in
replacement for the pygame/OpenCV client with the same feature set:

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
  TensorRT engines). Pulls in Python `http.server` and the existing
  `demos.realtime_motion_graph.server.handle_client`; no extra
  dependencies.
- **Client**: any modern Chromium or Firefox. Web MIDI and webcam
  support are optional. HTTPS is *not* required because the server
  binds on the same origin as the WebSocket endpoint.

## Run

First-run only — fetch the bundled audio fixtures:

```bash
python tests/fixtures/download.py
```

Then from the remote 5090 box (the machine with the GPU):

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

Then from any laptop on the same network:

1. Open `http://<server-host>:8765/`
2. Click **Play** — the demo loads the default fixture
   (`new_order_confusion_60seconds.wav`).
3. Switch fixtures any time using the selector at the top of the
   Advanced drawer; switching tears down the session and restarts with
   the new audio.
4. Cold start takes ~15 s while the server loads the model + TRT
   engines (or longer on `--accel compile`); once it's ready the UI
   switches to the live HUD view.

### Audio source vs. video

Audio is the **primary** source: the demo always loads from
`tests/fixtures/`, served by the web server at `/fixtures/<name>`.
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
├── server.py                 # HTTP (static) + WebSocket (reuses handle_client)
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

The WebSocket protocol is the *same* one defined in
`demos/realtime_motion_graph/client/protocol.py`:

- **Init**: JSON config -> binary audio upload
  (`<uint32 channels><uint32 samples>` + float32 PCM)
- **Server init**: JSON ready + binary float16 initial buffer
- **Streaming**: JSON params/prompt out, binary slice (raw float16 or
  zstd-compressed float16 delta) + `params_update` / `prompt_applied`
  JSON messages in

All of `server.py` is imported unchanged: the web server just adds an
HTTP static file server next to the same `handle_client` coroutine.

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
