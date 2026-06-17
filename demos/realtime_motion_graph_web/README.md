# Realtime Motion-to-Music (Web)

Browser-based real-time motion-to-music demo. A Python backend runs the
GPU pipeline behind an HTTP + WebSocket server; a Next.js front-end
renders the live UI:

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

- **Server**: the full DEMON install — `uv sync` then `uv run
  demon-setup` (downloads the model checkpoints and builds the minimal
  TensorRT engine set; see the [repo README Quickstart](../../README.md#quickstart)
  and [docs/INSTALL.md](../../docs/INSTALL.md)). No extra dependencies
  beyond the main project.
- **Client**: any modern Chromium or Firefox. Web MIDI and webcam
  support are optional.

At boot the backend runs a preflight: it downloads checkpoints if they
are missing (visible in the terminal) and verifies the TensorRT
engines for the configured `--accel`, exiting with the exact fix
command when they are absent. Bypass with `--skip-preflight`.

## Run

A single launcher starts the Python backend on `:1318` and a Next.js
dev server on `:6660` with combined output:

```bash
uv run python -u -m demos.realtime_motion_graph_web.run
# forward backend args after `--`:
uv run python -u -m demos.realtime_motion_graph_web.run -- --accel eager
```

First run installs the web and shared SDK `node_modules` automatically
(Node.js 20+ required).
Open `http://localhost:6660`. Next.js rewrites `/api/*`, `/fixtures/*`,
`/loras/*`, and `/videos/*` to the backend at `:1318`; the WebSocket
URL comes from `NEXT_PUBLIC_POD_BASE_URL` (set by the launcher).

### External static demos

Static demo repos can be mounted at runtime without vendoring them into
DEMON. Add a `demon.demo.json` manifest to the external repo:

```json
{
  "route": "/arp",
  "entry": "index.html"
}
```

`entry` is optional and defaults to `index.html`. The demo is served as
static files only; DEMON does not run build scripts or import demo code.
Browser code should load the shared SDK from `/sdk/demon-client.js`.

Run the backend directly with one or more external demos:

```powershell
uv run python -u -m demos.realtime_motion_graph_web.server --demo C:\path\to\demo
```

Or pass the demo path to the launcher:

```powershell
uv run python -u -m demos.realtime_motion_graph_web.run --demo C:\path\to\demo
```

Example: [`daydreamlive/demon-example-apps`](https://github.com/daydreamlive/demon-example-apps)
collects no-build static demos that use the shared `/sdk/demon-client.js`
bundle. Its `apps/summon` demo uses server fixtures, live LoRA controls, and
MediaPipe hand tracking as a control surface:

```powershell
git clone https://github.com/daydreamlive/demon-example-apps.git
uv run python -u -m demos.realtime_motion_graph_web.run --demo C:\path\to\demon-example-apps\apps\summon
```

The backend prints each mounted static demo URL at startup, for example
`Static demo: http://localhost:1318/arp/`. `--demo` may be repeated.
The older forwarding form also works when you need to pass backend args
verbatim: `uv run python -u -m demos.realtime_motion_graph_web.run -- --demo C:\path\to\demo`.

DEMON treats these paths as already-built static files. It does not run
`npm install`, `uv sync`, build scripts, or demo code. No-build demos can
use browser-native modules, committed static assets, pinned CDN imports,
or DEMON's shared `/sdk/` browser bundle. Demos with their own build
tooling should build themselves before being mounted.

### Remote / headless server

To run the GPU server on one machine and open the UI from another, bind the
backend to all interfaces and tell the browser the server's address:

```bash
# launcher flags go BEFORE `--`; backend flags (e.g. --accel) go AFTER it.
uv run python -u -m demos.realtime_motion_graph_web.run \
  --host 0.0.0.0 --client-host 10.0.0.5 \
  -- --accel tensorrt
# (10.0.0.5 = the server's LAN IP)
```

On startup the launcher prints `engine base URL (for browser): …` — it must be
the server's address, not `127.0.0.1`. Then open `http://10.0.0.5:6660` from
the client. Note:

- **Argument order matters.** Everything after `--` is forwarded to the
  backend, so a launcher flag like `--client-host` placed after `--` silently
  has no effect (the base URL falls back to `127.0.0.1`, which a remote browser
  resolves to *itself*). Put launcher flags before `--`.
- `--client-host` sets the address the **browser** uses for both the HTTP API
  and the WebSocket — both connect straight to the backend, so it must be
  reachable from the client, not `localhost`.
- Open **both** ports on the server's firewall: `:6660` (UI) and `:1318`
  (backend HTTP + WebSocket).
- Changing `web/.env.local` requires restarting the dev server — it's read
  only at startup.

The full UI lives under `web/` (React + zustand, mirrored from the
internal `daydreamlive/demon-react` package). See `web/components/`,
`web/engine/`, `web/hooks/`, and `web/store/` for the source.

### Backend args

Anything after `--` on the launcher is forwarded to the backend.

`--accel {tensorrt,compile,eager}` sets BOTH `decoder_backend` and
`vae_backend` on the underlying `Session`. Default is `tensorrt`.

`--decoder-accel` and `--vae-accel` override `--accel` for one
component at a time. Useful when, for example, only one of the two
TRT engines exists for a given checkpoint, or when you want to debug
one component in eager while the other stays on TRT:

```bash
# Mix-and-match: TRT decoder, eager VAE.
uv run python -u -m demos.realtime_motion_graph_web.run -- \
    --accel tensorrt --vae-accel eager
```

The text encoder stays resident in VRAM by default so live prompt edits do not
pay CPU/GPU transfer cost. Add `--offload-text-encoder` on lower-VRAM GPUs to
restore the previous lower-memory behavior.

`--checkpoint <name>` selects which DiT checkpoint to load. The name
must match a directory under `<checkpoints_dir>/`. Full TensorRT mode is
registered for `acestep-v15-turbo` (default, 2B) and
`acestep-v15-xl-turbo` (XL). XL TRT uses dynamic-batch `b8` decoder
profiles; build one first, then launch with:

```bash
uv run python -u -m demos.realtime_motion_graph_web \
    --accel tensorrt --checkpoint acestep-v15-xl-turbo
```

### Slow links (server-side env knobs)

The slice stream is heavily overlapped: the write frontier emits ~0.36 s
windows only ~0.03–0.04 s apart, so each region is re-sent ~9× as the
frontier sweeps it. Two env toggles tune this for bandwidth-limited
deployments (SSH/IDE tunnels, weak uplinks):

- `DEMON_SLICE_WINDOW_BYTES` (default `262144` = 256 KiB) — in-flight
  flow-control window. The server holds back slice emission while
  sent-minus-acked exceeds this, so a slow link gets fresh slices at
  link rate instead of an ever-staler backlog. Reactive backpressure.

- `DEMON_SLICE_EMIT_TRIM` (default off; set `1` to enable) — transmit
  only the region the frontier just **finalized** each tick, instead of
  re-sending the full overlapping window. Every overlapping copy lands
  ~lead *before* the playhead reaches the region, so only the last is
  ever heard; trimming sends each region once, at its freshest.
  Measured ~**2.8× less downstream bandwidth** (420 → 150 KB/s) with
  **identical latency** — the generation lead is untouched; this is a
  transmission-only change (verified: `advance_s` byte-identical with
  and without the flag). 150 KB/s is near the irreducible audio entropy
  at fp16, so this extracts essentially all the *count* redundancy.

  Trade-off: it spends the overlap's redundancy, which was also a
  safety margin. The client buffers less pre-playhead audio (~the lead,
  vs ~0.6 s with full overlap), so it tolerates a shorter transient
  stall before falling back to the source, and flow-control drops have
  no redundancy left to mask. Pairs with the connection-quality
  indicator for the below-floor case. Default off pending broader soak.

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

Once it's running:

1. Open `http://localhost:6660/`
2. Click **Play** — the demo loads the default fixture
   (`inside_confusion_loop_60s_gsm.wav`). Fixtures stream from the
   `daydreamlive/demon-fixtures-v2` Hugging Face dataset on first request
   and are cached locally.
3. Switch fixtures any time using the selector at the top of the
   Advanced drawer; switching tears down the session and restarts with
   the new audio.
4. Cold start takes ~15 s while the server loads the model + TRT
   engines (or longer on `--accel compile`); once it's ready the UI
   switches to the live HUD view.

### Audio source vs. video

Audio is the **primary** source: the demo always loads from the
canonical fixture set (`daydreamlive/demon-fixtures-v2` on Hugging Face,
listed in `acestep.fixtures.KNOWN_FIXTURES`), served by the backend
at `/fixtures/<name>` via lazy HF download.
Video is **optional and secondary** — drop any `.mp4`/`.webm`/`.mov`
into `videos/` (sibling of `web/`) to attach the audio-reactive shader
pipeline. With no videos present the demo runs audio-only (graph mode
is the default and looks the same).

## Layout

```
demos/realtime_motion_graph_web/
├── README.md
├── __init__.py
├── __main__.py               # `python -m demos.realtime_motion_graph_web`
├── run.py                    # launcher: backend + Next.js dev server
├── server.py                 # HTTP API + WebSocket multiplex on one port
├── ws_adapter.py             # per-WebSocket coroutine; wraps StreamingSession
├── audio_codec.py            # per-subscriber slice codec + stem payload
├── protocol.py               # wire format (Python source of truth)
├── videos/                   # user-supplied .mp4/.webm/.mov drop-in (optional)
└── web/                      # Next.js front-end (React + zustand)
```

## Protocol

The WebSocket protocol is defined in `protocol.py` (the Python source
of truth that `web/engine/protocol.ts` mirrors):

- **Init**: JSON config -> binary audio upload
  (`<uint32 channels><uint32 samples>` + float32 PCM)
- **Server init**: JSON ready + binary float16 initial buffer
- **Streaming**: JSON params/prompt out, binary slice (raw float16 or
  zstd-compressed float16 delta) + `params_update` / `prompt_applied`
  JSON messages in

`server.py` multiplexes the JSON HTTP API, fixture/video file serving,
and the WebSocket upgrade onto one TCP port; the WS handshake hands
off to `ws_adapter.handle_client`, which wraps a
`acestep.streaming.session.StreamingSession`.

## Audio-reactive video

The video is rendered through a small WebGL2 shader pipeline so it
visually responds to the music in real time. Two effects:

- **Color parallax** — saturated regions drift horizontally with a
  slow sway plus a punch on every kick.
- **Bloom on kick** — luminance-thresholded bloom that brightens with
  the bass envelope.

The same kick amplitude is exposed to CSS as `--bloom-amount`, so the
perimeter HUD bars and the cursor halo glow in lockstep with the
shader bloom on the video.

**Curator setup: nothing.** Color parallax is saturation-driven, not
depth-driven, so there is no preprocessing step and no depth map
sidecars to generate. Drop the source video into `videos/` and run
the launcher as usual. If WebGL2 is unavailable the canvas is hidden
and the plain video plays as fallback.

## Test fixtures

The eight files in `acestep.fixtures.KNOWN_FIXTURES` ship with sidecar
files in the `daydreamlive/demon-fixtures-v2` HF dataset:

```
<track-id>/
  source.wav
  track.json                      # editable BPM/key/time-signature + asset manifest
  stems/
    vocals.wav                    # optional user-facing vocal stem
    instruments.wav               # optional user-facing instrumental stem
  sidecars/
    full.json
    full.safetensors              # full-track source latent + context_latent
    vocals.json
    vocals.safetensors            # optional pre-encoded vocal source
    instruments.json
    instruments.safetensors       # optional pre-encoded instrumental source
```

When the client sends `fixture_name` for a known fixture, the server
loads the cached source latent + context latent and reads BPM / key
from the JSON, skipping librosa beat tracking, the CNN key
classifier, and `Session.prepare_source`. `Session.encode_text` still
runs live every connect (it depends on the prompt and the demo's
blended-prompt UI typically diverges from any baked tags within
seconds of connecting; the ~60ms warm cost isn't worth the cache
complication). For ad-hoc uploads (no `fixture_name`), the full live
path runs as before.

The runtime checks `MODELS_DIR/fixtures/` first (so local edits are
tested without an upload round-trip) and falls through to the dataset.
User uploads use the same layout under `MODELS_DIR/user_uploads/`;
fixtures are effectively system-provided tracks.

If you want to override the BPM or key for a fixture, edit the
`<track-id>/track.json` and re-run the precompute script. Editing the
JSON's `bpm` / `key` / `time_signature` fields and re-running
preserves them (the script only re-derives values that aren't already
pinned). To
forcibly re-derive everything from scratch, pass `--force`:

```bash
uv run python -m scripts.calibration.precompute_fixture_sidecars
uv run python -m scripts.calibration.precompute_fixture_sidecars --with-stems
uv run python -m scripts.calibration.precompute_fixture_sidecars --force
uv run python -m scripts.calibration.precompute_fixture_sidecars --only \
    inside_confusion_loop_60s_gsm.wav
```

After editing, upload the regenerated track JSON, WAV stem assets, and
all sidecar JSON/safetensors pairs back to `daydreamlive/demon-fixtures-v2`.
The runtime still falls back to the legacy `daydreamlive/demon-fixtures`
dataset for old audio and full-track sidecar files when a v2 asset is
not present.

Fixture dataset v2 plan:

1. Create `daydreamlive/demon-fixtures-v2` as a Hugging Face dataset repo.
2. Run `uv run python -m scripts.calibration.precompute_fixture_sidecars --with-stems`.
3. Upload each fixture directory with `source.wav`, `track.json`, `stems/`,
   and `sidecars/`.
4. Keep `daydreamlive/demon-fixtures` available until deployed pods have
   warmed or downloaded the v2 layout.

## Onboard MCP server (drive the demo from an LLM)

`mcp_server.py` is a stdio MCP server that exposes every user-facing
demo action — prompt, knobs, LoRA enable/disable, timbre/structure
refs, source swap — as an MCP tool. Useful for letting Claude Code (or
any MCP client) drive the demo for automated testing.

How it works:

1. The user opens the demo in their browser as usual. The backend
   registers that session in a process-global registry and starts the
   control bus, a small HTTP server on `127.0.0.1:1319`.
2. The MCP server speaks HTTP to the control bus. It does **not** open
   its own WebSocket and does **not** start a separate GPU session, so
   running it alongside a browser tab is free.
3. Every command the MCP sends is dispatched through the same handler
   as the browser's own WebSocket frames. Acks (`prompt_applied`,
   `lora_catalog`, `timbre_set`, `swap_ready`, etc.) flow back to the
   browser's WS, so the front-end UI mirrors MCP-driven changes
   automatically. The new `params_echo` message handles knob changes
   (see `useMcpMirror`).

Run the backend the usual way:

```bash
uv run python -u -m demos.realtime_motion_graph_web.run
```

then point Claude Code at the MCP server:

```jsonc
{
  "mcpServers": {
    "demon": {
      "command": "uv",
      "args": [
        "run", "python", "-u",
        "-m", "demos.realtime_motion_graph_web.mcp_server"
      ],
      "cwd": "C:/path/to/DEMON"
    }
  }
}
```

Env-var overrides:

- `DEMON_HOST` / `DEMON_PORT` — backend's main HTTP+WS port
  (default `127.0.0.1:1318`).
- `DEMON_CONTROL_HOST` / `DEMON_CONTROL_PORT` — control bus
  (default `127.0.0.1:1319`).
- `DEMON_WIPE_USER_UPLOADS=1` - hosted rented-pod cleanup only. When set,
  the backend wipes `MODELS_DIR/user_uploads/` at startup and main-session
  teardown to prevent cross-user leakage. Leave unset for local installs.

Server-side flags:

- `--control-host <host>` — bind the control bus to a non-localhost
  interface (use sparingly: this lets remote clients write into a live
  session).
- `--control-port <port>` — override the control port (default 1319).
- `--no-control` — disable the control bus entirely.

Available tools: `list_sessions`, `session_state`, `list_fixtures`,
`list_loras`, `list_knobs`, `set_prompt`, `set_prompt_blend`,
`set_knob`, `set_knobs`, `get_knob`, `enable_lora`, `disable_lora`,
`set_timbre_strength`, `set_timbre_fixture`, `set_timbre_audio`,
`clear_timbre`, `set_structure_fixture`, `set_structure_audio`,
`clear_structure`, `swap_to_fixture`, `swap_to_audio`. Most tools take
an optional `session_id` parameter; they default to the
most-recently-started session if you omit it.

Open the browser tab first — the MCP will refuse to act when there's
no live session to attach to.

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

## Troubleshooting

- **"WebSocket connection failed"**: verify the backend is reachable on
  `:1318` (firewall, reverse proxy). The launcher logs `[backend]`
  output if the Python side crashed.
- **Audio plays silent on first connect**: browsers gate audio on a
  user gesture; `Connect & start` counts as one, so this should "just
  work" but if it doesn't, click anywhere in the HUD view.
- **No MIDI devices listed**: Web MIDI requires `localhost` or HTTPS in
  Chromium. Use the local `http://localhost:6660` URL, or run behind a
  reverse proxy with TLS for remote access.
- **Webcam permission denied**: same-origin constraint as MIDI. Switch
  to on-screen knobs mode if the browser blocks the camera.
- **Cold start long**: the GPU server rebuilds the pipeline on every
  new connection, same as the native server. Reuse connections when
  possible.
