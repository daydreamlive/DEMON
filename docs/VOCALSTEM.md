# Vocal And Instrumental Stem Extraction

This document describes how uploaded-track stem extraction works in the
`realtime_motion_graph_web` backend.

The Mel-Band RoFormer integration lives in
`demos/realtime_motion_graph_web/melband_reformer.py`. The demo UI and backend
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
