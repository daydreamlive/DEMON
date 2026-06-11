# Vocal And Instrumental Stem Extraction

This document describes how uploaded-track stem extraction works in the
`realtime_motion_graph_web` backend.

The Mel-Band RoFormer integration lives in
`acestep/streaming/stems.py`. The demo UI and backend
now resolve every user-uploaded audio source to a stem mode (`full` by default),
so uploads are stemmed automatically. Built-in fixtures still omit
`stem_source_mode` and skip the stem path.

## When Stem Extraction Runs

The frontend can send one of three source modes:

- `full`: keep the original upload as the inference source, but still generate
  vocal and instrumental overlay assets.
- `vocals`: generate stems, then use the vocal stem as the inference source.
- `instruments`: generate stems, then use the instrumental bed as the inference
  source.

Mode validation is handled by `normalize_stem_source_mode()`. The frontend
uses `full` as the fallback for custom uploads, so the selector controls only
which waveform feeds inference; it does not gate whether stems are generated.

For initial session setup, the extraction happens after the upload has been
decoded, trimmed/profile-aligned, and prepared with `Session.prepare_source()`.
For source swaps, the same extraction path runs inside `apply_swap_if_pending()`
after the new uploaded waveform has been decoded and prepared.

## Stem Extraction

Stem extraction is performed with Mel-Band RoFormer through
`extract_upload_stems()`. The helper caches the RoFormer model separately from
the active ACE-Step `Session`; it no longer uses ACE-Step's native `extract`
task for uploaded-track separation.

The RoFormer checkpoint runs at 44.1 kHz, while the realtime backend and client
protocol run at 48 kHz. The extraction path therefore:

1. Takes the backend upload waveform as `[channels, frames]` at 48 kHz.
2. Runs Mel-Band RoFormer separation, internally resampling to 44.1 kHz.
3. Receives `vocals` and `instruments` from the separator.
4. Resamples both stems back to 48 kHz.
5. Normalizes each stem back to the upload shape in the helper,
   which fixes batch/channel/length differences and replaces non-finite values.

The model checkpoint defaults to `daydreamlive/MelBandRoFormer` /
`MelBandRoformer_fp16.safetensors`. The downloader materializes it at
`ACESTEP_MODELS_DIR/MelBandRoFormer/MelBandRoformer_fp16.safetensors`
(for example, `/workspace/.daydream-scope/models/MelBandRoFormer/...` when
`ACESTEP_MODELS_DIR=/workspace/.daydream-scope/models`). Operators can override
the checkpoint with:

```text
MELBAND_ROFORMER_MODEL_PATH
```

The instrumental bed is the RoFormer instrumental output, not ACE-guided
spectral suppression.

## VRAM Management

The RoFormer always loads on top of a resident ACE-Step model stack — the
streaming session at create/swap time, or the shared eager upload-encoder
session in the demo's `upload_track` path. On VRAM-constrained pods that
stack is a memory-pressure spike, so `extract_upload_stems()` accepts the
resident `ModelContext` as `model_context` and runs a park/restore cycle
around separation:

1. **Decide** (`should_park_for_melband()`): park only when the
   realistically-claimable VRAM (driver-free + torch's cached slack) is
   below the reserve the separator needs.
   `DEMON_MELBAND_VRAM_RESERVE_GB` overrides the default (6.0); `0`
   disables parking, a large value forces it.
2. **Park** (`ModelContext.vram_parked()`): the eager modules (DiT, VAE,
   text encoder) and the silence latent move to CPU and the freed pages
   return to CUDA. TRT engines are untouched — their device memory
   belongs to TensorRT execution contexts and cannot be offloaded.
3. **Separate**: the RoFormer loads into the vacated VRAM and runs.
4. **Release**: the RoFormer is dropped and its cache emptied BEFORE the
   restore, so ACE-Step returns into the space the separator vacated.
5. **Restore**: parked modules move back to the device with their
   canonical dtypes.

Concurrency: `vram_parked()` holds the ModelContext's placement lock for
the whole cycle, and every eager-module consumer routes through
`ModelContext._load_model_context()`, which takes the same lock. A
concurrent operation (prompt re-encode, timbre/structure set) issued
while the models are parked therefore blocks until the restore instead
of running GPU inputs against CPU weights. The session create and swap
paths run the extraction on the runner thread (or before the runner
exists), so streaming ticks never overlap a park; in the upload path the
parked encoder is a separate `ModelContext` from any live session, which
keeps streaming untouched.

Beyond the per-separation cycle, two standing policies keep steady-state
VRAM flat:

- **The shared upload encoder lives on GPU only while an upload is in
  flight.** `_handle_upload_track` calls
  `ModelContext.offload_eager_to_cpu()` when each upload finishes
  (persistent park — nothing auto-restores), and
  `_load_model_context()` lazily restores exactly the modules the next
  upload touches (`model` for semantic extract; `vae` only when no TRT
  VAE engine is cached). Without this the first upload would
  permanently pin ~6 GB next to the live streaming session.
- **Shape-aware TRT VAE engine selection** (`acestep/nodes/vae_nodes.py`):
  the process-wide TRT VAE cache can hand the upload encoder an engine
  belonging to the live streaming session, whose optimization profile
  may not cover the upload's length (a 120 s upload vs the session's
  60 s `vae_encode` engine). `_trt_vae_profile_fits()` checks the input
  shape against the cached engine's profile first; on a mismatch, a
  handler that carries an eager VAE falls back to it instead of letting
  TRT reject the shape and fail the upload.
- **One streaming session per pod, enforced** (`ws_adapter`):
  `StreamingSession.create` calls are serialized, and a new main-session
  connection preempts the active session — stops its runner, closes its
  socket with close code 4001 (`PREEMPTED_CLOSE_CODE`, which the web
  client treats as final rather than reconnecting), and waits on
  `StreamingSession.closed` for its GPU teardown before building the new
  stack. This prevents the dual-create OOM and the cascade where a dying
  session's cleanup evicts shared TRT VAE cache entries out from under a
  live one.

Every phase emits a structured `stems_vram` log line
(free/available/allocated/reserved GiB), plus `vram_parked` /
`vram_unparked` from the context. Verify against real models with
`scripts/verify_melband_vram.py`; unit coverage lives in
`tests/unit/test_melband_vram_management.py`.

## Returned Stem Assets

`extract_upload_stems()` returns:

```python
{
    "vocals": vocals,
    "instruments": instruments,
}
```

If the user selected `vocals` or `instruments` as `stem_source_mode`, the
backend prepares that selected waveform as a new `Audio` source and reruns
`Session.prepare_source()` so inference uses the selected stem.

The stem overlay assets are sent to the client with `_send_stem_payload()`:

1. A JSON message of type `stem_assets` with:
   - `fixture_name`
   - `sample_rate`
   - `channels`
   - `frames`
   - `stems`: `["vocals", "instruments"]`
   - `source_mode`
2. Two binary payloads, one per stem, in the same order.

The binary payloads are interleaved `float16` PCM buffers shaped as
`[frames, channels]` on the wire.

If extraction fails and the requested inference source depends on the failed
stem, the backend fails the session or swap. If extraction fails while the full
track is still usable, the backend sends a `stem_failed` message and continues
with the original source.

## Known Limitations

The instrumental stem is still model-separated, not a perfect studio
instrumental. Strong vocal reverb, doubled vocals, backing vocals, or vocal-like
synths can still leak or be over-suppressed.
