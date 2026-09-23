**YuE2: musical duration, acoustic TensorRT, and DEMON ring measurements — 2026-09-14**

**Follow-up:** [Full-song generation and cached editing measurements](YUE2_FULL_SONG_EDITING.md) add a continuously timed text-to-audio request, one-minute acoustic TensorRT profiles, and actual control-change-to-PCM measurements through the ring.

**The BF16 route is technically integrable, but these results do not establish the responsiveness DEMON needs.** Acoustic TensorRT substantially improves actual ring throughput. With a complete, musically checked 10-second generation, the staggered four-slot ring produces 3.35 generations/s, one completion about every 298 ms, and each generation occupies a slot for 1.19 seconds. Those are measured values, not an extrapolation from a single forward. Hot-control response has not been measured and must not be equated with slot age.

This supersedes the initial feasibility recommendation. A frame's duration says nothing useful about the minimum musical phrase. The earlier eight-second benchmark came from a semantic sampler default, not musical validation. The earlier raw-checkpoint step reductions are not an accepted acceleration route and have been removed from the default spike. **Every acoustic solve in this round uses the released 32-step midpoint solver: 64 velocity evaluations.** Short clips and isolated instruments remain capability questions, not admission gates.

**Distilled/turbo model search**

The search covered the official release/branch lists, the author's Hugging Face models, and 23 YuE2-related community model cards and file inventories. No released checkpoint advertising fewer acoustic steps through distillation was verified. The pinned upstream head remains `0edaf2f4053ef4731334b8329834b107977f9637`. The raw inventory is saved in [variant_audit.json](../../out/yue2/20260914/round2/variant_audit.json).

There is a potentially confusing **`turbo` runtime profile** in [WaveCut's OrbitQuant W4A4 release](https://huggingface.co/WaveCut/YuE2-3B-OrbitQuant-W4A4). Its card explicitly says there is no distillation: it quantizes weights/activations, including transformer projections, and changes output-head precision. It is a candidate for further NAR acceleration, not a few-step student. Its documented runtime requires Linux/Python 3.12/Torch 2.10; SM120 kernels are built but not validated there. It was not run in this Windows investigation.

The [DKmode22 NVFP4 checkpoint](https://huggingface.co/DKmode22/YuE2-3B-NVFP4) targets AR planning and semantic decoding; it retains the original NAR and VAE. It therefore does not address the live acoustic-loop bottleneck. An independent [real-audio tokenizer/NAR LoRA](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4) also exists; its stated purpose is source-audio tokenization and adaptation, with the stock sampler retained. It is not a step-distillation release. This updates the earlier assessment: official source-audio semantic tokenization is missing, but community tooling now deserves separate evaluation.

**What is the shortest amount of music demonstrated here?**

The shortest generated musical passage in these probes is **6.4 seconds**, with a natural semantic ending and recognizable melody/lyrics, but extra/repeated words. The shortest example that passed both independent lyric recovery and the supplied melodic-pattern check is **10.0 seconds**. This is an observed useful example, not a proven architectural minimum or a reliable duration setting.

| Request | Actual full output | Content evidence |
|---|---:|---|
| Supplied one-bar vocal score, seed 381 | 6.4 s, natural end | ASR recovered the requested line plus extra/repeated phrases; note estimates contain C–E–G–E–C. It does not follow the requested one-pass structure. |
| Same one-bar request, seed 382 | 16.2 s, natural end | Requested line recovered; substantially longer duration. |
| Supplied two-bar vocal score | 22.6 s, natural end | Lyrics repeated/partially restarted. |
| Supplied four-bar vocal score, no intro/outro requested | **10.0 s, natural end** | Both lyric lines recovered; both C–E–G–E–C patterns recovered independently, modulo octave. |
| Two-line jingle with model-generated planning | 23.6 s, natural end | Complete plan; both lyric lines recovered. The generated score adds intro/outro material. |
| Same two-line request, planning off | 37.0 s, natural end | Lyrics repeated and vocal ad-libs recovered. |
| Supplied one-bar instrumental piano score | 80.0 s, budget exhausted | Failed to stop at the intended two-second score duration. |
| Supplied four-bar instrumental piano score | 72.2 s, natural end | Failed to follow the intended eight-second duration. |
| Unmodified upstream `examples/song.json` | 60.9 s, natural end | Complete plan; all eight lyric lines recovered, with minor ASR word substitutions. |

These are whole generated outputs, not cropped long audio or shortened acoustic contexts. The shorter successes use **provided symbolic scores**: the semantic generator and acoustic model rendered them, but they do not establish that the symbolic planner autonomously creates a useful eight-second composition. In the tested automatic-planning request it generated a 23.6-second song. No eight-second useful minimum was established.

For the ten-second case, the requested melody is C–E–G–E–C in each of two phrases. Independent MuScriptor note estimates recover those pitch classes in the 0–3 s and 4–8 s intervals; ASR recovers “Hear the morning light” and “Everything is bright.” Neither tool was supplied the score or lyrics. This is evidence of organized musical output rather than a merely valid tensor or nonzero waveform. MuScriptor labels the melody as `flutes`, illustrating why its instrument labels should not be treated as ground truth. The ASR results contain implausible text on piano-only requests, another reason not to use ASR alone as a vocal detector.

Review: [10-second score-conditioned output](../../out/yue2/20260914/round2/concise_4bar_s381/audio.wav), [6.4-second output with repetitions](../../out/yue2/20260914/round2/concise_1bar_s381/audio.wav), [23.6-second planned output](../../out/yue2/20260914/round2/jingle_full/audio.wav), [60.9-second upstream control](../../out/yue2/20260914/round2/upstream_song/audio.wav). Evidence: [ASR words/timestamps](../../out/yue2/20260914/round2/asr.json), [note estimates](../../out/yue2/20260914/round2/notes.json), [score-pattern comparison](../../out/yue2/20260914/round2/score_audit.json). Automated content checks do not replace a listening evaluation of timbre, artifacts, endings, and usefulness in performance.

**Individual instruments**

The first ten seconds of the two piano requests yielded 42 and 44 note estimates respectively, all labeled acoustic piano. This supports investigating piano-only generation; it is not a verified general guarantee of isolated stems. In the original solo-drum request, the same unrestricted transcriber estimated 30 drum notes **and 39 bass notes**. That request did not pass an automated drum-isolation check. These observations concern excerpts of longer generated tracks; they do not establish equally reliable short instrumental requests. The public generator returns a stereo mix.

**What was accelerated and how**

The new engines cover the **acoustic/NAR transformer**, not just the VAE. The exported module contains NAR attention/MLP projections, normalization, latent input/output projections, time embedding, and positional state. AR prefix keys/values are mutable runtime inputs, so changing their contents does not require rebuilding weights. Sequence lengths remain profile-specific.

Two BF16 engines were built from actual complete outputs: 250 frames for the ten-second case and 589 frames for the 23.6-second case, with dynamic batch 1–4 and optimization batch 4. The original FP32 normalization reductions are retained. TensorRT 10.16 builds were strongly typed, TF32 disabled, optimization level 2, 1 GiB tactic workspace, one timing iteration, and no auxiliary streams. The ten-second engine built in 32.3 seconds; both are about 2.84 GB. The first build successfully wrote its engine before a metadata-reporting exception; that reporting bug was corrected without rebuilding it.

Engine inspection shows fused Q/K/V projections, fused gate/up projections, fused elementwise operations, and `_gemm_mha_v2` attention kernels. Thus the measured TensorRT path does include attention fusion. Execution contexts/buffers are cached per batch size and graph-captured; prefix KV is prepared once and reused. No model distillation, raw step reduction, NAR FP8/INT8/INT4, or changed acoustic context was used in these comparisons.

**Actual DEMON ring results**

The harness inherits DEMON's actual [`StreamPipeline.tick`](../../acestep/engine/stream.py), queue, slot admission, completion delivery, and request ownership. An explicitly named research subclass overrides noise initialization and the solver tick to preserve YuE2's frame-major noise and two-pass midpoint updates. This is real batched denoising through DEMON's ring, with mixed diffusion ages, rather than a batch-forward throughput estimate. It is not a fully registered production backend: the stock Euler tick, other knob operations, playback runner, WebSocket server, and audio output are not exercised.

| Musical context | Acoustic backend | Depth | Tick median | Completed generations/s | Slot age median |
|---|---|---:|---:|---:|---:|
| 10.0 s | PyTorch CUDA graphs | 1 | 27.8 ms | 1.13 | 893 ms |
| 10.0 s | PyTorch CUDA graphs | 4 | 72.0 ms | 1.73 | 2,312 ms |
| 10.0 s | TensorRT + graphs | 1 | 21.6 ms | 1.44 | 696 ms |
| 10.0 s | TensorRT + graphs | 4 | 37.5 ms | 3.32 | 1,199 ms |
| 10.0 s | TensorRT + graphs, staggered admission | 4 | **37.3 ms** | **3.35** | **1,190 ms** |
| 23.6 s | PyTorch CUDA graphs | 1 | 60.0 ms | 0.52 | 1,921 ms |
| 23.6 s | PyTorch CUDA graphs | 4 | 234.9 ms | 0.53 | 7,514 ms |
| 23.6 s | TensorRT + graphs | 1 | 27.1 ms | 1.15 | 867 ms |
| 23.6 s | TensorRT + graphs | 4 | 86.7 ms | 1.44 | 2,776 ms |
| 60.9 s control | PyTorch CUDA graphs | 1 | 252.6 ms | 0.12 | 8,095 ms |

Timing uses synchronized wall time, including submit/tick bookkeeping for throughput. Most rows have 40 warmup ticks and 64 measured ticks, giving two completed generations at depth one and eight at depth four. The long control has 32 warmup and 32 measured ticks, only one completed generation: it is a bounded scaling observation, not a tail-latency estimate. GPU builds and GPU inference benchmarks ran serially. Independent audio audits ran on CPU. The VAE and audio rendering are excluded from these ring timings; the separately established ~3 ms TensorRT window decode remains additional work.

The default constant-submit driver admits its initial slots on consecutive ticks, causing clustered completions. Admitting the first four slots eight ticks apart through the normal queue produced completion intervals of **296.8–300.5 ms**, with similar throughput and slot age. Its tick p95 was 38.5 ms. Staggering fixes that driver's completion cadence; it does not remove the 32 sequential midpoint steps.

All measured ring outputs matched the corresponding **same-backend, same-batch-size** 32-step batch solve exactly. Tiny-model CPU tests also passed at depths one and four, with 140 velocity calls over 70 ticks and the expected completion counts. This resolves the ring-math question independently of cross-backend floating-point differences. TensorRT's final latents differ from upstream batch-one PyTorch by relative L2 **1.06%** for the ten-second case and **1.34%** for the 23.6-second case; these are not bit-identical upstream renders. The earlier batched discrepancy must not be described as a demonstrated ring bug. Cross-backend audio comparison still needs listening.

Raw measurements: [ring_summary.json](../../out/yue2/20260914/round2/ring_summary.json) and the linked per-run files in that directory. For comparison, the repository README reports ACE-Step turbo at 11.3 generations/s and ~248 ms parameter convergence for a 60-second source on a 5090. That is a published repository baseline, not a fresh side-by-side run in this investigation. The YuE2 results have not demonstrated comparable performance, even on the shorter musical region.

**Decision and next concrete work**

Do not declare the current BF16 YuE2 backend ready for DEMON's live experience. Keep the working acoustic engine, window codec evidence, and exact ring/batch validation as a research foundation. The most relevant next acceleration experiment is a NAR-capable low-bit runtime such as OrbitQuant W4A4 on Linux/SM120, preserving the sampler and comparing actual ring throughput and audio. AR-only quantization will not solve the refinement-loop cost. A true few-step version requires a released distilled checkpoint or a separate distillation project; reducing the current model's step count is not the plan.

Before production integration, measure live modulation with fixed semantics, then prompt/style changes that may require new semantic generation. Report time to first affected output and convergence separately from slot age and average completion rate. The provided-score ten-second success does not settle the automatic planner's useful minimum, nor does one good short result establish reliable duration control. Those capability findings should be recorded without turning them into blockers for supporting a slower preparation/composition mode.

**Reproduction and artifacts**

The runtime, model/weight revisions, and external checkpoint locations match [the first GPU report](YUE2_GPU_RESULTS.md). The round-two output directory is [`out/yue2/20260914/round2`](../../out/yue2/20260914/round2); source/semantic requests, complete plans, raw unclipped audio arrays, and latents are retained. Acoustic ONNX external weights, TensorRT engines, profiles, and layer inspection are under `D:/codex-yue2-runtime-20260914/nar_trt/{case}/`.

[`yue2_worker.py`](../../scripts/spikes/yue2_worker.py) provides a reusable resident-weight worker. Set `PYTHONPATH` to the pinned upstream source plus the external tiktoken directory, then run:

```powershell
.venv/Scripts/python.exe -u scripts/spikes/yue2_worker.py --model-root Z:/codex-yue2-research/models/demon/yue2/checkpoints --runtime D:/codex-yue2-runtime-20260914 --output out/yue2/20260914/round2
```

Write commands to `<runtime>/round2-command.json`, waiting for `round2-status.json` to report `ready` between dependent phases. Example: `{"action":"run","phase":"trt_ring_staggered","case":"concise_4bar_s381","depths":[4],"stagger_start":true}`. `{"action":"stop"}` releases the weights. Other implemented phases are `duration`, `duration_concise` (with a `cases` filter), `export`, and `ring_graph`. Use unique phase suffixes for ring runs to retain separate result files. Build exported engines serially with [`yue2_nar_trt_build.py`](../../scripts/spikes/yue2_nar_trt_build.py) and `--artifacts <runtime>/nar_trt/<case>`.

Audio audits are in [`yue2_audio_audit.py`](../../scripts/spikes/yue2_audio_audit.py) and [`yue2_note_audit.py`](../../scripts/spikes/yue2_note_audit.py). ASR uses the existing local faster-whisper-small.en checkpoint and an external dependency directory; note transcription uses the existing local MuScriptor-small checkpoint on CPU. DEMON's environment/lockfile, production engine code, protocol registries, and git index were not changed.
