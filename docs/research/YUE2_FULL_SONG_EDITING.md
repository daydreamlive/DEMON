**YuE2 full-song generation and cached editing — 2026-09-18**

The relevant DEMON workflow is to prepare a song once, then keep its score, semantic tokens, conditioning cache, and acoustic model resident while editing. This investigation measures both fresh text-to-audio and that cached loop. It uses the original 32-step midpoint sampler (64 acoustic evaluations), BF16 acoustic TensorRT, CUDA graphs, and the local RTX 5090. No step reduction or distillation is claimed.

**Fresh generation: 59 seconds of audio in 17.32 seconds**

This is one continuously timed request from style/lyrics to a complete CPU stereo waveform, with a naturally ending score and semantic sequence. Checkpoint loading, ONNX export, engine builds, and initial engine loading are excluded. Per-request AR setup and acoustic graph capture are included. The result is 3.41 audio seconds per wall second; its sequential rate is equivalent to about 3.46 one-minute songs per minute, not a measured multi-request serving throughput.

| Stage | Measured time | Reuse during acoustic-only editing |
|---|---:|---|
| Generate symbolic score | 3.668 s | Entirely reused while the composition is fixed |
| Generate semantic music tokens | 10.586 s | Entirely reused while the encoded performance is fixed |
| Draw noise | 0.001 s | Cheap; retain or regenerate according to the edit |
| Prepare acoustic conditioning KV | 0.139 s | Reused while prefix and semantic contents are fixed |
| Copy noise to GPU | <0.001 s | Small per-generation work |
| Bind/capture new acoustic geometry | 0.429 s | Reused at fixed geometry; required here for a fresh length |
| Acoustic synthesis, all 32 midpoint steps | 2.181 s | Still required for each full acoustic revision |
| Decode full song and return CPU audio | 0.289 s | Replace with window decoding during playback |
| **Measured total** | **17.320 s** | |

The small difference between the stage sum and the total is host orchestration. The VAE is FP32 PyTorch for this full-song row; the acoustic transformer is TensorRT. This is accelerated full generation, not an all-TensorRT claim. Full VAE TensorRT would only target the ~0.29-second component here.

Raw data: [edit_profile_full.json](../../out/yue2/20260918/edit_profile_full.json). Review: [complete generated audio](../../out/yue2/20260918/upstream_song_full_trt.wav), [exact fresh request, score, and semantics](../../out/yue2/20260918/full_request_semantic.json).

**Cached acoustic loop**

Three repeated solves of the same 59-second context took a median **2.187 s**. A complete solve followed by a TensorRT window decode and CPU audio copy took **2.196 s**. This returns a 200 ms interior audio window while denoising the entire song context. Decoding that window alone took **3.05 ms**; decoding the full song took **290 ms**.

Caching removes about 14.3 seconds of planning/semantic generation from each acoustic revision. Cached prefix KV avoids another ~137 ms prefill; retaining the execution graph avoids request setup. These savings are real, but the remaining ~2.2-second acoustic solve is still the main obstacle to fast interaction. A 3 ms VAE window does not make acoustic denoising local.

The window output matched the corresponding full PyTorch decode with maximum waveform difference `6.26e-7`. The TensorRT acoustic latents differed from a same-context upstream solve by relative L2 **2.11%**. That upstream solve took **8.43 s**. These are numerical checks; no listening evaluation of the fresh TensorRT output was performed.

**Actual ring, including window decoding and an in-flight control change**

The fixed-profile ring uses the separately prepared complete **63.16-second** song. Each completion denoises its entire acoustic context and returns a 200 ms interior PCM window to CPU. The harness uses DEMON's real `StreamPipeline` queue, admission, mixed-age slots, and completion lifecycle, with the explicit YuE2 midpoint adaptation. These numbers are not an extrapolation from an isolated forward.

| Ring depth | Median tick | Completed revisions/s | Slot age | First changed PCM after control | First fully updated PCM after control |
|---|---:|---:|---:|---:|---:|
| 1 | 70.0 ms | 0.446 | 2.24 s | 1.68 s | 3.92 s |
| 4 | 255.8 ms | 0.489 | 8.15 s | 2.03 s | 8.14 s |

The four-slot completion intervals were 2.030–2.059 seconds. Batching adds only about 10% throughput over depth one for this full-song context, while substantially increasing convergence time. This BF16 route has not demonstrated the responsiveness needed for DEMON's live editing experience.

The probe changes a velocity multiplier from 1.0 to 1.05 after 40 warmup ticks and records 64 more ticks. It applies to already active slots, without replanning, resampling semantics, or rebuilding KV. The first change is detected numerically in PCM, not by listening. The first fully updated output is from a slot admitted after the control change and matches a separate complete solve with the new multiplier exactly. Every subsequent fully updated result also matches exactly. The timings describe this one change at a specified ring phase, not best/worst-case or perceptual response across edits. Playback-device buffering and network/UI latency are excluded; the measurement ends at CPU PCM.

Depth one completes two outputs in the measured interval; depth four completes eight. Full control records, latent comparisons, and waveform differences are in [depth one](../../out/yue2/20260918/edit_profile_live_d1.json) and [depth four](../../out/yue2/20260918/edit_profile_live_d4.json). This validates a research solver-control path, not all DEMON controls or hot prompt/semantic replacement.

**Which stages can be omitted from the editing loop?**

| Edit | Work that can be reused or bypassed | Work still required / uncertainty |
|---|---|---|
| Acoustic solver controls with a fixed musical performance | Score, semantic tokens, conditioning KV, fixed-shape execution graph | Acoustic steps and a playback-window decode; usefulness of individual controls needs listening |
| Provide or directly edit a score | Model-based score generation | Semantic generation, new conditioning, acoustic synthesis, decoding |
| Change melody, lyrics, musical timing, or form | An unchanged portion of the score may remain authored/cached | The reference path regenerates semantic tokens; no validated semantic infill path is demonstrated here |
| Change style/instrumentation while preserving the song | The score can remain fixed | Reusing semantic tokens may resist the desired change; NAR-only style edits are an experiment, not an established capability |
| Revisit an already prepared variation | Its score, semantics, and conditioning can all remain cached | Switching/blending alignment and continuity need validation; new acoustic rendering may still be desired |
| Play an existing rendered result | Every generative stage | Ordinary audio playback/mixing |

This cache plan is strongest for a YuE2 song whose intermediate representations are already available. Arbitrary imported audio requires a separately validated source-preparation path. Transcribing its score and generating new semantic tokens is a reinterpretation, not a guarantee of preserving the original performance.

The harness also retains both the transformer and VAE on the GPU. The stock high-level `decode()` moves the transformer to CPU before decoding; that migration is unnecessary in this tested resident-model configuration and should not be repeated per edit.

YuE2's public staged API supports separate planning, semantic generation, synthesis, and decoding. Its CFG setting acts during semantic generation in the inspected implementation; an ACE-Step-style acoustic guidance knob cannot be assumed to map directly onto it. [Official generation guide](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/docs/generation.md)

**Engineering findings and measurement limits**

Fresh requests with the same prompt/seed produced different plans and lengths in this runtime. The original fixed 63.16-second TensorRT profile therefore rejected a fresh 59.92-second request. A second engine now accepts variable frame count, conditioning length, rotary values, and positional embeddings. Its bounded profiles cover 1,000–2,500 acoustic frames (40–100 seconds), batch 1–4, and up to 4,000 conditioning tokens. The successful end-to-end run produced 1,475 frames / 58.999 seconds. No cold build was hidden inside the reported generation time, and no fixed-profile result was substituted for a differently shaped fresh request.

The fixed profile built in 49.6 s; the flexible profile in 50.8 s. These are offline preparation costs. Both retain the original FP32 normalization reductions. The checkpoint/source identities and Windows runtime match the earlier investigation. GPU builds and inference ran serially. Saved timings use synchronized wall time; repeated measurements are short bounded probes rather than endurance or tail-latency claims.

For longer songs, [WaveCut's W4A4 runtime](https://huggingface.co/WaveCut/YuE2-3B-OrbitQuant-W4A4) reports complete warm requests on a 4090: 140.7 seconds of audio in 12.26 seconds (`fast`) and 178.2 seconds in 15.57 seconds (`turbo`). Those are author-reported full-generation results, not local DEMON measurements or interactive-edit latency. The runtime quantizes AR and NAR rather than distilling steps; it remains a relevant acceleration candidate. Its Linux/SM120 path has not been run in this investigation. Do not extrapolate the local one-minute result linearly to a three-minute song.

**Next experiments, in order**

1. Keep score, semantic tokens, prefix KV, models, and fixed-geometry graphs resident. This is now a measured baseline, including PCM output. Prefer depth one over four when prioritizing convergence at this context length; depth four brought little additional throughput.
2. Apply a NAR-capable low-bit runtime to that same prepared song and repeat the control-to-PCM probe. AR acceleration improves fresh generation but does not reduce this cached loop's cost. Preserve the original steps unless an actual distilled checkpoint is obtained.
3. Test whether fixed semantics permit useful timbre/style changes or constrain them too strongly. Compare musical identity, intended instrument changes, and continuity with fresh semantic generation. An unchanged score does not by itself justify freezing semantics.
4. Evaluate local acoustic editing with surrounding musical context and preserved outside-region audio. This changes the transformer computation and needs quality validation; it is a separate experiment from the already proven local VAE decode. If effective, it could avoid repeatedly denoising the entire song for each local edit.
5. Consider preparing aligned semantic alternatives in advance for expensive musical changes, then switching between prepared states. Validate alignment, transitions, and response instead of hiding regeneration behind a nominally live control.

**Reproduction**

The reusable [worker](../../scripts/spikes/yue2_worker.py) loads checkpoints once. The new [profiling harness](../../scripts/spikes/yue2_edit_profile.py) runs `edit_profile_prepare`, `edit_profile_export_flexible`, `edit_profile_full`, and `edit_profile_live_d1` / `edit_profile_live_d4` (with the respective `depth` field). The existing `export` phase builds the fixed-song ONNX graph. The [builder](../../scripts/spikes/yue2_nar_trt_build.py) accepts fixed or flexible exports' optimization profiles. Outputs are under `out/yue2/20260918`; engines remain outside the repository under `D:/codex-yue2-runtime-20260914/nar_trt/`. The initial shape-rejection traceback is retained in `error_phase.json`; the flexible-profile end-to-end run resolved it.

Production DEMON code, dependencies, and the git index are unchanged. These are reviewable research scripts and measurements.
