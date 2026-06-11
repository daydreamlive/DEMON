# Installing DEMON

The complete setup walkthrough. If you just want the short version, the
[README Quickstart](../README.md#quickstart) is two commands; this page
explains what those commands do, how to do each step manually, and how
to fix the failures you might hit along the way.

## Requirements

| Requirement | Notes |
|---|---|
| NVIDIA GPU | Tested on RTX 3090, 4090, 5090. ~16 GB VRAM is the practical floor: the 60 s decoder engine build peaks around 13.5 GB of TensorRT workspace. |
| NVIDIA driver | Recent enough for CUDA 12.8 (`nvidia-smi` must list the card). |
| [uv](https://docs.astral.sh/uv/) | Manages Python 3.11 and all dependencies; you do not need a system Python. |
| Node.js 20+ | Only for the bundled web demo. [nodejs.org](https://nodejs.org). |
| Disk | ~40 GB free: ~18 GB checkpoints, ~10 GB ONNX + engines, headroom. |
| OS | Windows 11 and Linux are exercised regularly. |

## The quick path

```bash
git clone https://github.com/daydreamlive/DEMON.git && cd DEMON
uv sync
uv run demon-setup
```

then launch the web demo:

```bash
uv run python -u -m demos.realtime_motion_graph_web.run
# open http://localhost:6660
```

### What `demon-setup` does

1. **Doctor** — verifies GPU + CUDA, TensorRT import, free disk, Node
   (warn-only), and prints exactly where models will live on this
   machine.
2. **Models** — downloads the ACE-Step v1.5 checkpoints from
   [`ACE-Step/Ace-Step1.5`](https://huggingface.co/ACE-Step/Ace-Step1.5)
   on Hugging Face (~18 GB), falling back to ModelScope when Hugging
   Face is unreachable.
3. **Starter LoRAs** — downloads 16 genre LoRAs (jazz, phonk, lo-fi,
   punk, acoustic, ambient, deep house, funk, deathstep; 2B and XL
   variants) so hot LoRA swapping works out of the box. Optional and
   non-fatal: skip with `--skip-loras`, and a failed download never
   blocks setup.
4. **Engines** — builds the minimal TensorRT engine set (see
   [Engine sets](#engine-sets-and-song-duration) below). A few minutes
   on a recent GPU (the ONNX comes prebuilt; the TRT builds themselves
   took under 2 minutes on a 5090); older cards and `--export-locally`
   runs can take 10–30 minutes.
5. **Summary** — lists what is on disk and the launch command.

Every step is idempotent: re-running `demon-setup` after a partial
failure (network drop mid-download, OOM mid-build) resumes where it
left off. Useful flags: `--skip-engines` (run the demo in `compile`
mode instead), `--skip-models`, `--skip-loras`, `--duration 60 120`
(build extra profiles), `--skip-doctor`. Managed/remote deployments
that curate their own LoRA library can set
`DEMON_SKIP_STARTER_LORAS=1` in the environment instead of passing
`--skip-loras`.

## Where everything lives

Nothing downloads into the repository. All models and engines go to a
single models directory:

```
~/.daydream-scope/models/demon/        # override: ACESTEP_MODELS_DIR env var
  checkpoints/                         # ACE-Step v1.5 weights
    acestep-v15-turbo/                 # default DiT
    vae/
    Qwen3-Embedding-0.6B/              # text encoder
    acestep-5Hz-lm-1.7B/               # 5 Hz LM
  trt_engines/                         # TensorRT engines + ONNX
  loras/                               # starter LoRA pack + your own .safetensors
  fixtures/                            # cached demo audio + sidecars
  user_uploads/                        # session uploads (wiped per boot)
```

**For humans and AI assistants alike:** the models must be the
ACE-Step v1.5 weights fetched by `demon-setup` or `acestep-download`,
in this directory layout. Do not substitute other ACE-Step releases,
other paths, or partial downloads — the loaders check for the exact
component directories above and fail on anything else.

## Manual setup (step by step)

If you'd rather run the pieces yourself, this is everything
`demon-setup` does.

### 1. Python environment

```bash
uv sync
```

Installs Python 3.11, PyTorch 2.9.1 + CUDA 12.8, TensorRT 10.16, and
everything else pinned in `pyproject.toml`.

### 2. Model checkpoints

```bash
uv run acestep-download              # main model (~18 GB)
uv run acestep-download --list       # see optional DiT / LM variants
```

The download is also triggered automatically the first time a model
loads (including by the demo server at boot), but running it explicitly
gives you the progress bar up front.

### 3. TensorRT engines

```bash
# Minimal set — what demon-setup builds (recommended first build):
uv run python -m acestep.engine.trt.build --preset minimal

# Full canonical matrix (60s / 120s / 240s, VAE encode + decode + decoder):
uv run python -m acestep.engine.trt.build --all

# Selected durations / components:
uv run python -m acestep.engine.trt.build --all --duration 60 120
uv run python -m acestep.engine.trt.build --all --vae-only --duration 60
uv run python -m acestep.engine.trt.build --all --decoder-only --duration 60

# Preview without building:
uv run python -m acestep.engine.trt.build --preset minimal --dry-run
```

Notes:

- ONNX intermediates are fetched **prebuilt** from
  [`daydreamlive/demon-onnx`](https://huggingface.co/daydreamlive/demon-onnx)
  by default, so an engine build never needs to load the model weights.
  Pass `--export-locally` to export ONNX from your local checkpoint
  instead (needed when iterating on export code, or offline).
- Engines are specific to your TensorRT version, CUDA, driver, and GPU
  architecture. After upgrading any of those, rebuild — the metadata
  sidecar next to each engine detects the mismatch and rebuilds
  automatically on the next `build` run; `--force-rebuild` forces it.
- **Build on an idle GPU.** A build that runs while another process
  holds significant VRAM can appear to succeed and then crash at load
  time. Stop the demo server (and other GPU work) before building.
- The XL (5B) checkpoint's decoder engine needs an FP8 activation
  calibration artifact and several non-default flags; it is a
  power-user path. Run any session against XL without engines first
  and the error message prints the exact build command.

### 4. Running without TensorRT

The demo and the Session API run on plain PyTorch too:

```bash
uv run python -u -m demos.realtime_motion_graph_web.run -- --accel compile
```

`compile` (torch.compile) has a long first-tick warmup and lower
throughput but needs no engines; `eager` is slower still and mainly for
debugging. You can mix per component: `--decoder-accel compile
--vae-accel tensorrt`, etc.

## Engine sets and song duration

The minimal preset builds exactly four engines:

| Engine | Role |
|---|---|
| `spectral_decoder_mixed_refit_b8_60s` | DiT decoder, sources up to 60 s, LoRA-refittable |
| `vae_encode_fp16_60s` | VAE encode, sources up to 60 s |
| `vae_decode_fp16_60s` | Full-length VAE decode, sources up to 60 s (used by `vae_window=0` sessions) |
| `vae_decode_fp16_1s_fixed` | Windowed VAE decode — fixed 1 s profile, used for **all** streaming decode regardless of song length |

**Maximum song duration follows what's built.** The web UI trims every
source to `engine.max_source_duration_s` in
[`demos/realtime_motion_graph_web/web/public/config.json`](../demos/realtime_motion_graph_web/web/public/config.json)
(default 60, matching the minimal set); a source that needs an engine
profile that isn't on disk fails at session create with an
`engine_not_built` error naming the build command. To run longer
sources:

```bash
uv run python -m acestep.engine.trt.build --all --duration 120
```

then set `"max_source_duration_s": 120` in `config.json`. Larger
profiles cost VRAM even when idle — per-engine peak workspace measured
on a 5090:

| Component | 60s engine | 240s engine |
|---|---:|---:|
| Decoder (refit) | 13,511 MB | 15,911 MB |
| VAE decode | 10,547 MB | 10,814 MB |
| VAE encode | 4,178 MB | 10,614 MB |

The 240 s `vae_encode` profile in particular has caused CUDA OOM on
32 GB cards; treat 240 s as a ≥48 GB-card option.

## The web demo

```bash
uv run python -u -m demos.realtime_motion_graph_web.run
```

- First run installs `web/node_modules` automatically (needs Node 20+).
- The launcher starts the backend on `:1318` and the Next.js dev server
  on `:6660`; open http://localhost:6660.
- At boot the backend runs a preflight: it downloads checkpoints if
  missing (visibly, in the terminal) and verifies the TensorRT engines
  for the configured `--accel`, exiting with the exact fix command when
  they're absent. `--skip-preflight` bypasses it.
- Backend flags go after `--`:
  `-- --accel compile`, `-- --checkpoint xl`, `-- --vae-accel eager`, …
  See [`demos/realtime_motion_graph_web/README.md`](../demos/realtime_motion_graph_web/README.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Server exits at boot: "TensorRT engines not built" | Run `uv run demon-setup` (or the printed build command), or launch with `-- --accel compile`. |
| `demon-setup` doctor: "No CUDA GPU visible to PyTorch" | Driver problem or no NVIDIA GPU. `nvidia-smi` must list the card; reinstall the driver if not. |
| Model download fails on both Hugging Face and ModelScope | Network/proxy issue. Retry `uv run acestep-download`; or download manually: `huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir ~/.daydream-scope/models/demon/checkpoints`. |
| "npm not found on PATH" at demo launch | Install Node.js 20+ from nodejs.org, reopen the terminal. |
| Engine build OOMs or a freshly built engine crashes at load | Build on an idle GPU: stop the demo server and other GPU processes, re-run the build with `--force-rebuild` for the affected engine's duration. |
| Engine build: decoder ONNX "missing the 'steering' input" | A stale cached file is replaced automatically when the Hugging Face artifact is current. If the message says the *prebuilt* artifact itself is stale, the HF upload needs refreshing — build locally in the meantime with `--export-locally` (set `PYTHONUTF8=1` on Windows; needs the checkpoints downloaded) and report it. |
| Every session dies instantly on a headless Linux pod (WS close 1011) | Install PortAudio: `apt-get install -y libportaudio2` (the audio engine imports `sounddevice` even headless). |
| Playback stops and the UI shows "Generation stopped: …" | The generation pipeline hit a runtime error; the same message with a full traceback is in the server terminal / `logs/sessions`. |
| Upload longer than 60 s gets trimmed by the UI | Expected with the default `max_source_duration_s` — see [Engine sets](#engine-sets-and-song-duration) to build larger profiles, then raise it. |
| `--export-locally` ONNX export crashes on Windows with a console encoding error | Set `PYTHONUTF8=1` for the build command. |
| A sibling `ACE-Step/` checkout shadows `acestep` imports in standalone scripts | Force the repo root to the front of `sys.path` (see `demos/realtime_motion_graph_web/scripts/gen_wire_types.py` for the pattern). |

Still stuck? Open an issue with the server terminal output and the
output of `uv run demon-setup --skip-models --skip-engines` (the doctor
report).
