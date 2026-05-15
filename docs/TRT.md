# TensorRT Support In DEMON

This document describes the TensorRT implementation in DEMON as it exists now:
what was changed, which engines are supported, how to build them, how to run
them, and what is known about ACE-Step 1.5 XL support.

DEMON targets TensorRT 10.16.x. Engines are not portable across TensorRT
versions or GPU architectures by default, so treat generated `.engine` files as
local build artifacts.

## Current Scope

TensorRT acceleration is implemented for:

- ACE-Step DiT decoder inference.
- ACE-Step VAE encode and decode.
- Windowed VAE decode for low-VRAM streaming decode.
- ACE-Step 1.5 2B turbo decoder engines.
- ACE-Step 1.5 XL turbo decoder engines with dynamic-batch ONNX patching.

The default runtime still requires the ACE-Step text encoder and tokenizer for
conditioning. TensorRT replaces the heavy decoder and/or VAE execution paths
when selected.

## Runtime Stack

The project dependencies pin the tested TensorRT stack:

```toml
"torch==2.9.1+cu128"
"onnx>=1.20,<1.22"
"onnxscript>=0.7.0"
"tensorrt>=10.16,<10.17"
"polygraphy>=0.49.26"
"cuda-python>=13.2,<13.3"
```

`onnxscript` is required for PyTorch's dynamo ONNX exporter, which is used by
the bf16 XL export path.

The build script enforces TensorRT `>=10.16,<10.17` during preflight. It records
TensorRT, CUDA, PyTorch, ONNX, GPU name, compute capability, and driver metadata
next to every engine in `<engine>.metadata.json`.

Known-good local build environment used for the current artifacts:

- TensorRT: `10.16.1.11`
- CUDA Python: `13.2.0`
- CUDA toolkit wheel: `13.2.1`
- PyTorch: `2.9.1+cu128`
- GPU: RTX 4090, SM 8.9, 24 GB VRAM
- Driver seen by `nvidia-smi`: `595.97`

### RTX 5090 / Blackwell VAE Builds

TensorRT 10.15 and 10.16 can generate a Myelin fusion for the Oobleck VAE
graph that segfaults on RTX 5090 during the first `execute_async_v3` call.
DEMON pins VAE TensorRT builds to `builder_optimization_level=1` in
`acestep/engine/trt/vae_export.py` to avoid that fusion. Decoder engines do not
use this workaround; the 10.16 decoder path has been validated separately on
Blackwell.

Any VAE or DreamVAE `.engine` built before this opt-level pin should be treated
as unsafe on Blackwell. Rebuild those engines on the target TensorRT/CUDA/driver
stack before running full TRT mode on an RTX 5090.

## Storage Layout

Paths are resolved through `acestep.paths`.

Default model root:

```text
%USERPROFILE%\.daydream-scope\models\demon
```

Override it with:

```powershell
$env:ACESTEP_MODELS_DIR = "D:\models\demon"
```

Directory layout:

```text
models/demon/
  checkpoints/
    acestep-v15-turbo/
    acestep-v15-xl-turbo/
    vae/
    Qwen3-Embedding-0.6B/
  trt_engines/
    _onnx_vae/
    _onnx_acestep-v15-turbo/
    _onnx_acestep-v15-xl-turbo/
    decoder_mixed_refit_b8_60s/
    decoder_xl-turbo_fp8_refit_b4_60s/
    vae_decode_fp16_3to30s/
    build_report.csv
  calibration/
    decoder_xl_fp8/
      60s/   activation_absmax.json, calibration.npz, manifest.json
      120s/  activation_absmax.json, calibration.npz, manifest.json
      240s/  activation_absmax.json, calibration.npz, manifest.json
```

ONNX exports are cached under `trt_engines/_onnx_*`. Decoder ONNX is
checkpoint-specific. VAE ONNX is shared across DiT variants.

XL FP8 calibration artifacts live under `calibration/decoder_xl_fp8/<dur>s/`,
one subdirectory per profile (60s / 120s / 240s). Each subdir holds the
`.npz` snapshot of decoder inputs and the per-Linear `activation_absmax.json`
that drives the FP8 W8A8 scales for that profile's sequence length.

## Main Code Paths

- `acestep/engine/trt/build.py`
  - One entry point for TRT builds.
  - Performs TensorRT 10.16.x preflight.
  - Resolves or exports ONNX.
  - Builds decoder, VAE, windowed VAE, and DreamVAE engines.
  - Writes engine metadata and appends `build_report.csv`.

- `acestep/engine/trt/export.py`
  - Exports the DiT decoder to ONNX.
  - Builds decoder TensorRT engines.
  - Contains precision recipes for 2B and XL.

- `acestep/engine/trt/vae_export.py`
  - Exports VAE encode/decode ONNX.
  - Builds VAE encode/decode TensorRT engines.

- `acestep/engine/trt/runtime.py`
  - Runtime wrapper for decoder engines.
  - Allocates buffers using the dtype declared by the engine.

- `acestep/nodes/vae_nodes.py`
  - Runtime wrapper/cache for VAE TRT engines.
  - Handles windowed VAE decode selection.

- `acestep/engine/session.py`
  - User-facing API that wires TRT backends into a `Session`.

- `acestep/paths.py`
  - Canonical 2B and XL turbo engine profile registration and path helpers.

## Build Script Basics

Run all commands from the repo root:

```powershell
cd C:\sd\DEMON
```

Preview the build matrix:

```powershell
uv run python -m acestep.engine.trt.build --all --dry-run
```

Build the standard 2B turbo 60s engines:

```powershell
uv run python -m acestep.engine.trt.build --all --duration 60
```

Build the standard 2B turbo 120s engines:

```powershell
uv run python -m acestep.engine.trt.build --all --duration 120
```

Build only decoder engines:

```powershell
uv run python -m acestep.engine.trt.build --all --decoder-only --duration 60
```

Build only VAE engines:

```powershell
uv run python -m acestep.engine.trt.build --all --vae-only --duration 60
```

Force rebuild existing engines:

```powershell
uv run python -m acestep.engine.trt.build --all --duration 60 --force-rebuild
```

Force ONNX export from local weights and rebuild:

```powershell
uv run python -m acestep.engine.trt.build --all --duration 60 --force-onnx --force-rebuild
```

## ONNX Resolution

The build script resolves ONNX in this order:

1. Reuse local ONNX if it already exists.
2. If `--skip-onnx` is passed, fail when any required ONNX is missing.
3. If `--force-onnx` is passed, export from local checkpoint weights.
4. If `--export-locally` is passed, export missing ONNX from local weights.
5. Otherwise, fetch missing ONNX from Hugging Face via
   `acestep.engine.trt.onnx_hub`.

For 2B turbo, the default Hugging Face ONNX cache may be enough.

For ACE-Step XL turbo, use local export unless an XL ONNX bundle has already
been uploaded to the configured ONNX hub. The build step patches the exported
ONNX before engine construction so reshape constants that baked in `B=1` become
dynamic:

```powershell
uv run python -m acestep.engine.trt.build --all --checkpoint acestep-v15-xl-turbo --decoder-only --duration 60 --batch-max 4 --batch-opt 4 --builder-optimization-level 5 --workspace-gb 20 --export-locally
```

## Decoder Precision Recipes

The decoder build now accepts:

```powershell
--decoder-precision auto
--decoder-precision fp32
--decoder-precision fp16_mixed
--decoder-precision bf16_mixed
--decoder-precision fp8_mixed
```

`auto` behaves as follows:

- For XL checkpoints, `auto` resolves to `bf16_mixed`.
- For existing 2B mixed builds, `auto` resolves to `fp16_mixed`.
- For single non-mixed decoder exports, `auto` resolves to `fp32`.

Why XL uses `bf16_mixed`:

- XL turbo is trained/stored in bf16.
- The XL residual stream can exceed fp16 range.
- Some attention paths can overflow fp16.
- TensorRT does not provide a bf16 kernel for the XL `proj_out`
  `ConvTranspose1d` shape, so `bf16_mixed` keeps the model in bf16 while
  wrapping that deconvolution in fp32.

### FP8 W8A8 (`--decoder-precision fp8_mixed`)

`fp8_mixed` is the canonical XL recipe. It exports the same bf16-mixed
ONNX as the base graph, then patches every `nn.Linear`-with-bf16-initializer
`MatMul` with FP8 E4M3FN weights (per-output-channel symmetric scale) and
inserts activation `QuantizeLinear`/`DequantizeLinear` chains on each
unique input (per-tensor symmetric scale). On Blackwell, TensorRT 10.16's
strongly-typed builder picks FP8 GEMM tactics for these MatMuls.

W8A8 requires a per-Linear activation absmax JSON captured against the
PyTorch model. Activation distributions shift with sequence length, so
each engine profile (60s / 120s / 240s) gets its own calibration JSON,
captured at the matching `T` (1500 / 3000 / 6000 latent frames) against
the matching bf16 engine. The artifacts land in
`<MODELS_DIR>/calibration/decoder_xl_fp8/<dur>s/`.

Prerequisite: the bf16 XL engine for the target duration must already be
built (used as the inference engine that drives calibration capture):

```powershell
uv run python -m acestep.engine.trt.build --all --checkpoint acestep-v15-xl-turbo `
    --decoder-only --duration 60 --batch-max 4 --batch-opt 4 `
    --builder-optimization-level 5 --workspace-gb 20 `
    --export-locally --decoder-precision bf16_mixed
```

Then for each duration `<DUR>` in 60 / 120 / 180 / 240, with `<T>` =
`<DUR> * 25` (the latent frame count):

```powershell
$DUR = 60
$T   = 1500
$CAL_DIR = "$env:USERPROFILE\.daydream-scope\models\demon\calibration\decoder_xl_fp8\${DUR}s"
$BF16_ENGINE = "$env:USERPROFILE\.daydream-scope\models\demon\trt_engines\decoder_xl-turbo_mixed_refit_b4_${DUR}s\decoder_xl-turbo_mixed_refit_b4_${DUR}s.engine"

# 1. Snapshot real decoder inputs (.npz) by streaming through the matching
#    bf16 engine at T=$T. The per-duration sequence length is what makes
#    each JSON specific to its profile.
uv run python scripts/collect_decoder_calibration.py `
    --checkpoint acestep-v15-xl-turbo `
    --num-prompts 25 --max-calls 200 `
    --seq-len $T `
    --decoder-engine $BF16_ENGINE `
    --output "$CAL_DIR\calibration.npz"

# 2. Replay them through the PyTorch model and record per-Linear absmax.
#    --output-dir lands the JSON next to the matching .npz.
uv run python scripts/collect_activation_absmax.py `
    --checkpoint acestep-v15-xl-turbo --batch 4 `
    --calibration "$CAL_DIR\calibration.npz" `
    --output-dir $CAL_DIR

# 3. Build the FP8 engine. --skip-onnx reuses the cached bf16 ONNX; the
#    FP8 patch rewrites it in place from the new absmax JSON before TRT
#    parses the graph.
uv run python -m acestep.engine.trt.build --all `
    --checkpoint acestep-v15-xl-turbo --decoder-only --duration $DUR `
    --batch-max 4 --batch-opt 4 --builder-optimization-level 5 `
    --workspace-gb 20 --skip-onnx `
    --decoder-precision fp8_mixed `
    --activation-absmax-json "$CAL_DIR\activation_absmax.json"
```

Repeat for the other durations. The patched FP8 ONNX is a sibling of the
bf16 ONNX and is rewritten on each build, so the engines must be built
one duration at a time.

The calibration scripts dispatch on `--checkpoint`: XL writes to
`<MODELS_DIR>/calibration/decoder_xl_fp8/<dur>s/`, 2B writes to
`<MODELS_DIR>/calibration/decoder_2b_fp8/`.

Validated XL recipe at B=4 T=1500 L_enc=200, RTX 5090, TRT 10.16
(original 60s cal2 numbers; 120s / 240s use their own per-profile JSONs
with comparable scales):

- bf16_mixed: 142.8 ms / 7.0 steps/s / 7.8 GB engine
- fp8_mixed (W8A8 absmax): 91.4 ms / 10.9 steps/s / 4.0 GB engine
- Speedup: **1.56x**, engine size **0.52x**
- Quality: cosine sim 0.957-0.981 across calibration batches.

The 2B turbo decoder's bf16-hybrid recipe already routes Linears
through fp16 tensor cores, so FP8 on 2B only adds ~1.08x over the
hybrid baseline (see `archive/ryanontheinside/fp8-2b-research-vram`).
FP8 is most useful on 2B as a VRAM win, not a latency win.

## Engine Naming

Decoder engines use:

```text
decoder[_variant]_<precision>[_refit]_b<batch_max>_<duration>s.engine
```

Examples:

```text
decoder_mixed_refit_b8_60s.engine
decoder_mixed_refit_b8_120s.engine
decoder_xl-turbo_fp8_refit_b4_60s.engine
decoder_xl-turbo_fp8_refit_b4_120s.engine
decoder_xl-turbo_fp8_refit_b4_240s.engine
```

The XL profiles in `_XL_TURBO_TRT_ENGINE_PROFILES` resolve to the FP8
engines for 60s / 120s / 240s. The bf16_mixed engines are still used as
calibration drivers (see the FP8 section above) but are not the
production runtime target.

VAE engines use:

```text
vae_encode_fp16_60s.engine
vae_decode_fp16_60s.engine
vae_decode_fp16_120s.engine
vae_decode_fp16_3to30s.engine
```

The `mixed` tag means the TensorRT network is strongly typed and follows the
dtypes encoded in the ONNX graph. For XL `bf16_mixed`, that includes bf16 bulk
compute plus the fp32 deconvolution island.

## Current Local Build Results

The following artifacts have been built and validated locally:

- `decoder_mixed_refit_b8_60s`: 2B turbo decoder, about 3299.6 MB.
- `decoder_mixed_refit_b8_120s`: 2B turbo decoder, about 3299.4 MB.
- `decoder_xl-turbo_fp8_refit_b4_60s`: XL turbo decoder, FP8 W8A8 (canonical).
- `decoder_xl-turbo_fp8_refit_b4_120s`: XL turbo decoder, FP8 W8A8 (canonical).
- `decoder_xl-turbo_fp8_refit_b4_240s`: XL turbo decoder, FP8 W8A8 (canonical).
- `decoder_xl-turbo_mixed_refit_b4_60s`: XL turbo decoder, bf16_mixed (calibration driver).
- `decoder_xl-turbo_mixed_refit_b4_120s`: XL turbo decoder, bf16_mixed (calibration driver).
- `decoder_xl-turbo_mixed_refit_b4_240s`: XL turbo decoder, bf16_mixed (calibration driver).
- `vae_decode_fp16_60s`: VAE decode, about 179.4 MB.
- `vae_decode_fp16_120s`: VAE decode, about 347.2 MB.
- `vae_decode_fp16_3to30s`: windowed VAE decode, about 269.1 MB.
- `vae_encode_fp16_60s`: VAE encode, about 172.3 MB.
- `vae_encode_fp16_120s`: VAE encode, about 172.4 MB.

Recent build timings on the RTX 4090:

- 2B turbo 120s decoder: about 130 seconds.
- XL turbo 60s decoder: about 199 seconds after ONNX export.
- XL turbo 120s decoder: about 199 seconds, reusing the XL ONNX.
- XL turbo refit ONNX export: about 476 seconds.

## Building ACE-Step 1.5 XL Turbo

Download the XL checkpoint:

```powershell
uv run acestep-download --model acestep-v15-xl-turbo --skip-main
```

The canonical XL runtime engines are FP8 (`decoder_xl-turbo_fp8_refit_b4_*`).
Building them is a two-stage pipeline: first build the bf16_mixed engine
for the duration (it drives calibration capture), then run the FP8 pipeline
documented in [FP8 W8A8](#fp8-w8a8---decoder-precision-fp8_mixed) above.

Build 60s XL bf16 engine (calibration driver):

```powershell
uv run python -m acestep.engine.trt.build --all --checkpoint acestep-v15-xl-turbo --decoder-only --duration 60 --batch-max 4 --batch-opt 4 --builder-optimization-level 5 --workspace-gb 20 --export-locally --decoder-precision bf16_mixed
```

Build 120s XL bf16 engine after the 60s path validates:

```powershell
uv run python -m acestep.engine.trt.build --all --checkpoint acestep-v15-xl-turbo --decoder-only --duration 120 --batch-max 4 --batch-opt 4 --builder-optimization-level 5 --workspace-gb 20 --export-locally --decoder-precision bf16_mixed
```

Build 240s XL bf16 engine for the longest profile:

```powershell
uv run python -m acestep.engine.trt.build --all --checkpoint acestep-v15-xl-turbo --decoder-only --duration 240 --batch-max 4 --batch-opt 4 --builder-optimization-level 5 --workspace-gb 20 --export-locally --decoder-precision bf16_mixed
```

Then follow the FP8 W8A8 section for each duration (per-profile calibration
+ FP8 engine build). The runtime resolves to the FP8 engine names via
`_XL_TURBO_TRT_ENGINE_PROFILES` in `acestep/paths.py`, so a bf16-only
build leaves the canonical slot empty and the runtime will raise
`EngineNotBuiltError` with a build hint pointing at the FP8 recipe.

If changing precision recipe or regenerating the ONNX:

```powershell
uv run python -m acestep.engine.trt.build --all --checkpoint acestep-v15-xl-turbo --decoder-only --duration 60 --batch-max 4 --batch-opt 4 --builder-optimization-level 5 --workspace-gb 20 --force-onnx --decoder-precision bf16_mixed
```

Use `batch-max=4`, `batch-opt=4`, and builder optimization level `5` for the
registered XL profiles. During the build, `acestep.engine.trt.build` writes a sibling
`*_dynbatch.onnx` with `[1, ...]` Reshape shape constants rewritten to
`[-1, ...]`, then builds the engine from that patched graph.

## Running With TensorRT

2B turbo, canonical helper:

```python
from acestep.engine.session import Session
from acestep.paths import default_trt_engines

session = Session(
    config_path="acestep-v15-turbo",
    decoder_backend="tensorrt",
    vae_backend="tensorrt",
    trt_engines=default_trt_engines(),
    vae_window=15,
)
```

XL turbo with explicit engine paths:

```python
from acestep.engine.session import Session
from acestep.paths import trt_engine_path

session = Session(
    config_path="acestep-v15-xl-turbo",
    decoder_backend="tensorrt",
    vae_backend="tensorrt",
    trt_engines={
        "decoder": str(trt_engine_path("decoder_xl-turbo_fp8_refit_b4_120s")),
        "vae_encode": str(trt_engine_path("vae_encode_fp16_120s")),
        "vae_decode": str(trt_engine_path("vae_decode_fp16_3to30s")),
    },
    vae_window=15,
)
```

Use the windowed VAE decode engine (`vae_decode_fp16_3to30s`) when
`vae_window > 0`. It reduces context memory compared with full-duration VAE
decode engines and is the preferred streaming decode path.

## Running The Realtime Web Demo With TRT

The realtime browser demo is launched with:

```powershell
uv run python -u -m demos.realtime_motion_graph_web --port 8765
```

The entry point is `demos/realtime_motion_graph_web/__main__.py`, which calls
`demos/realtime_motion_graph_web/server.py`. The server uses a single TCP port
for both static HTTP and WebSocket traffic. Open the UI at:

```text
http://localhost:8765/
```

or, from another machine on the same network:

```text
http://<gpu-host>:8765/
```

The demo defaults to TensorRT:

```text
--accel tensorrt
```

That means both `decoder_backend` and `vae_backend` are set to `tensorrt` on
the underlying `Session`. These are equivalent:

```powershell
uv run python -u -m demos.realtime_motion_graph_web --port 8765

uv run python -u -m demos.realtime_motion_graph_web --port 8765 --accel tensorrt
```

You can also choose a backend explicitly:

```powershell
uv run python -u -m demos.realtime_motion_graph_web --port 8765 --accel compile
uv run python -u -m demos.realtime_motion_graph_web --port 8765 --accel eager
```

For mixed backend debugging:

```powershell
# TensorRT decoder, eager VAE.
uv run python -u -m demos.realtime_motion_graph_web --port 8765 --accel tensorrt --vae-accel eager

# Eager decoder, TensorRT VAE.
uv run python -u -m demos.realtime_motion_graph_web --port 8765 --accel eager --vae-accel tensorrt
```

### How The Demo Picks TRT Engines

`demos/realtime_motion_graph_web/backend.py` chooses engines at client
connection time. It measures the uploaded or fixture audio duration, then calls:

```python
available_trt_engines(
    duration_s=audio_duration_s,
    needs=tuple(needs),
    checkpoint=checkpoint,
)
```

`needs` is based on the selected backends:

- `decoder_backend == "tensorrt"` requires `decoder`.
- `vae_backend == "tensorrt"` requires `vae_encode` and `vae_decode`.

The picker walks the canonical profile set for the selected checkpoint and
selects the smallest built profile that can fit the audio duration. If the
smallest fitting profile is missing, it can fall back to the next larger built
profile and logs a warning about extra VRAM cost. If no suitable built profile
exists, the server sends an `engine_not_built` error to the UI and closes the
WebSocket.

For `acestep-v15-turbo`, the decoder profiles are:

```text
decoder_mixed_refit_b8_60s
decoder_mixed_refit_b8_120s
decoder_mixed_refit_b8_240s
vae_encode_fp16_*
vae_decode_fp16_*
```

For `acestep-v15-xl-turbo`, the decoder profiles are:

```text
decoder_xl-turbo_fp8_refit_b4_60s
decoder_xl-turbo_fp8_refit_b4_120s
decoder_xl-turbo_fp8_refit_b4_240s
vae_encode_fp16_*
vae_decode_fp16_*
```

VAE engines are shared across the 2B and XL turbo checkpoints. The demo caps
incoming audio at the largest registered profile for the selected checkpoint
before engine selection. XL registers 60s, 120s, and 240s decoder profiles,
all FP8 W8A8.

### Recommended 2B Turbo Demo Flow

Build at least the 60s 2B engines:

```powershell
uv run python -m acestep.engine.trt.build --all --duration 60
```

For the default fixture and longer source audio, the 120s engines are useful:

```powershell
uv run python -m acestep.engine.trt.build --all --duration 120
```

Start the demo:

```powershell
uv run python -u -m demos.realtime_motion_graph_web --port 8765 --accel tensorrt --checkpoint acestep-v15-turbo
```

The demo will:

1. Serve the web UI on port `8765`.
2. Wait for the browser to connect over WebSocket.
3. Load the selected fixture or uploaded audio.
4. Pick the smallest built TRT profile that fits that audio.
5. Construct `Session(decoder_backend="tensorrt", vae_backend="tensorrt")`.
6. Swap in the windowed VAE decode engine automatically when `vae_window > 0`
   and `vae_decode_fp16_3to30s` is built.

The browser-side default configuration lives in:

```text
demos/realtime_motion_graph_web/static/config.json
```

Relevant TRT/runtime fields in that config include:

```json
{
  "depth": 8,
  "vae_window": 6.0,
  "crop": 0,
  "steps": 8,
  "fast_vae": true
}
```

`vae_window > 0` enables windowed decode. `fast_vae=true` asks the backend to
use DreamVAE as the TRT VAE decode engine when a matching DreamVAE engine is
built; otherwise it logs a warning and falls back to the standard VAE decode
engine.

### Recommended XL Turbo Demo Flow

Build the XL checkpoint:

```powershell
uv run acestep-download --model acestep-v15-xl-turbo --skip-main
```

The XL canonical engines are FP8. Each duration needs (a) the bf16_mixed
engine as a calibration driver, (b) a per-profile activation absmax JSON,
(c) the FP8 engine. See the [FP8 W8A8](#fp8-w8a8---decoder-precision-fp8_mixed)
section above for the full per-duration pipeline. The minimum build for
the demo at 60s is the bf16 engine plus the FP8 engine + 60s calibration
artifacts; the calibration capture also produces the VAE engines for the
same duration as a side effect of the bf16 build.

Start the demo in full TRT mode:

```powershell
uv run python -u -m demos.realtime_motion_graph_web --port 8765 --accel tensorrt --checkpoint acestep-v15-xl-turbo
```

The backend passes `checkpoint` into `available_trt_engines()`, so this selects
`decoder_xl-turbo_fp8_refit_b4_60s`, `decoder_xl-turbo_fp8_refit_b4_120s`,
or `decoder_xl-turbo_fp8_refit_b4_240s` instead of the 2B decoder profiles.
The streaming decoder path submits the active ring-buffer rows as one
batched TRT execution, so XL profiles must be built with enough batch
capacity for the configured pipeline depth.

### Demo Troubleshooting

If the UI reports that a TRT engine is not built, build the smallest fitting
profile from the server log. For a 60s source:

```powershell
uv run python -m acestep.engine.trt.build --all --duration 60
```

For a 120s source:

```powershell
uv run python -m acestep.engine.trt.build --all --duration 120
```

If cold start uses more VRAM than expected:

- Prefer the smallest duration profile that fits the source audio.
- Keep `vae_window > 0` so the windowed VAE decode engine can be used.
- Disable DreamVAE or build the matching DreamVAE profile if `fast_vae=true`.
- Use `--decoder-accel tensorrt --vae-accel eager` or the inverse to isolate a
  backend.

If TensorRT reports a static batch or reshape mismatch such as:

```text
Static dimension mismatch while setting input shape for hidden_states.
Set dimensions are [2,...]. Expected dimensions are [1,-1,64].
```

or:

```text
IExecutionContext::inferShapes: IShuffleLayer node_view: reshaping failed
for tensor: linear_2
RESHAPE input dims{2, 15360} reshape dims{1, 6, 2560}
```

make sure the XL engine was built from the dynamic-batch patched ONNX. The build
logs should mention `*_dynbatch.onnx` and report patched `Reshape` constants.

## Validation Commands

Compile modified TRT modules:

```powershell
uv run python -m py_compile acestep/engine/trt/build.py acestep/engine/trt/export.py
```

Rebuild VAE engines after changing TensorRT, CUDA, driver, GPU architecture, or
the VAE builder optimization level:

```powershell
uv run python -m acestep.engine.trt.build --all --vae-only --duration 60 --force-rebuild
uv run python -m acestep.engine.trt.build --all --vae-only --duration 120 --force-rebuild
```

For a quick RTX 5090 smoke test, load the rebuilt VAE engines and execute one
encode/decode pass before launching the realtime demo. A crash on the first
`execute_async_v3` usually means the VAE engine was built without the
`builder_optimization_level=1` workaround or was carried across an incompatible
TensorRT/GPU stack.

Run XL vendored loader tests:

```powershell
uv run pytest tests/unit/test_vendored_xl_dit_loads.py -q
```

Directly execute the 120s XL decoder engine at max profile:

```powershell
@'
import torch
from acestep.engine.trt.runtime import TRTDecoder
from acestep.paths import trt_engine_path

engine = TRTDecoder(trt_engine_path("decoder_xl-turbo_fp8_refit_b4_120s"))
hs = torch.randn(2, 3000, 64, device="cuda", dtype=torch.bfloat16)
ts = torch.full((2,), 0.5, device="cuda", dtype=torch.bfloat16)
enc = torch.randn(2, 200, 2048, device="cuda", dtype=torch.bfloat16)
ctx = torch.randn(2, 3000, 128, device="cuda", dtype=torch.bfloat16)
out = engine(hs, ts, enc, ctx)
assert out.shape == (2, 3000, 64), out.shape
assert torch.isfinite(out).all()
print("XL 120s TRT decoder execution OK", tuple(out.shape), out.dtype, out.device)
'@ | uv run python -
```

Run a short XL `Session` generation and VAE decode:

```powershell
@'
import torch
from acestep.engine.session import Session
from acestep.paths import trt_engine_path
from acestep.constants import TASK_INSTRUCTIONS

trt_engines = {
    "decoder": str(trt_engine_path("decoder_xl-turbo_fp8_refit_b4_60s")),
    "vae_encode": str(trt_engine_path("vae_encode_fp16_60s")),
    "vae_decode": str(trt_engine_path("vae_decode_fp16_3to30s")),
}

session = Session(
    config_path="acestep-v15-xl-turbo",
    decoder_backend="tensorrt",
    vae_backend="tensorrt",
    trt_engines=trt_engines,
    vae_window=5,
)
cond = session.encode_text(
    tags="instrumental electronic test",
    lyrics="",
    instruction=TASK_INSTRUCTIONS["text2music"],
    duration=5.0,
)
latent = session.generate(
    conditioning=cond,
    seed=1234,
    steps=1,
    duration=5.0,
    denoise=1.0,
)
assert torch.isfinite(latent.tensor).all()
audio = session.decode(latent)
assert torch.isfinite(audio.waveform).all()
print("XL TRT validation OK")
print("latent", tuple(latent.tensor.shape), latent.tensor.dtype, latent.tensor.device)
print("audio", tuple(audio.waveform.shape), audio.waveform.dtype, audio.waveform.device, audio.sample_rate)
'@ | uv run python -
```

Validated output shape from that run:

```text
latent (1, 125, 64) torch.bfloat16 cuda:0
audio (1, 2, 240000) torch.float32 cuda:0 48000
```

## ACE-Step XL Notes

Supported now:

- `acestep-v15-xl-turbo`
- Decoder-only TRT builds at 60s and 120s.
- Runtime `Session` execution with TRT decoder and existing VAE TRT engines.

Not yet fully supported:

- `acestep-v15-xl-base`
- `acestep-v15-xl-sft`

Those checkpoints point to different upstream modeling files in `auto_map`.
DEMON currently vendors and dispatches the XL turbo modeling path. Base and SFT
need their modeling files vendored and registered in
`ModelContext._VENDORED_DIT_CLASSES` before they are first-class supported.

## Refit And LoRA Caveat

Decoder engines are built with TensorRT `REFIT` enabled so LoRA weight updates
can be applied without rebuilding engines.

For the validated XL engines, runtime logs currently show:

```text
No engine weights matched
```

That means XL TRT generation works, but dynamic LoRA refit for XL is not yet
validated. The likely follow-up is to inspect XL ONNX initializer names and TRT
engine weight names, then extend the LoRA refit name mapping for XL/dynamo
exported graphs.

## Troubleshooting

TensorRT version rejected:

```text
DEMON TensorRT builds target TensorRT >=10.16,<10.17
```

Fix with:

```powershell
uv sync --upgrade-package tensorrt
```

Missing `onnxscript` during XL export:

```text
ModuleNotFoundError: No module named 'onnxscript'
```

Fix with:

```powershell
uv add onnxscript
```

Windows Unicode failure from `torch.onnx`:

```text
UnicodeEncodeError: 'charmap' codec can't encode character
```

The exporter now reconfigures stdout/stderr to UTF-8 for the dynamo ONNX path.
If this reappears in external scripts, set:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

XL out of memory:

- Use `--decoder-only`.
- Use `--batch-max 4 --batch-opt 4 --builder-optimization-level 5` for local
  experiments, then register only validated dynamic-batch profiles.
- Start with `--duration 60`.
- Use `vae_decode_fp16_3to30s` with `vae_window > 0`.
- Avoid loading PyTorch decoder weights at runtime by setting
  `decoder_backend="tensorrt"`.

Unexpected rebuilds:

- Check `<engine>.metadata.json`.
- The build script rebuilds when schema, component, TensorRT version, GPU
  compute capability, config, or ONNX sha256 changes.

## Engineering Checklist

Before relying on a TRT engine in a demo or benchmark:

1. Confirm the TensorRT preflight shows the expected 10.16.x stack.
2. Build the smallest duration profile that fits the target audio.
3. Prefer windowed VAE decode for streaming or short decode windows.
4. Check `build_report.csv` for status, build time, and size.
5. Load the engine once through `TRTDecoder` or `Session`.
6. Assert finite latent and audio tensors on a short generation.
7. For web-demo runs, verify the selected `--checkpoint` has registered TRT
   profiles in `acestep.paths`.
