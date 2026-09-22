**YuE2 GPU feasibility experiments — 2026-09-14**

This is the measured follow-up to the [integration plan](YUE2_FEASIBILITY_PLAN.md). Short-duration and single-instrument behavior are capability measurements, not requirements for admitting the model into DEMON.

**Superseded by the [actual ring and musical-duration investigation](YUE2_RING_RESULTS.md).** This first round did not establish a useful eight-second musical context or DEMON throughput. Its recommendation to proceed was premature. Raw-checkpoint step reduction is not the accepted acceleration strategy; the later tests retain 32 midpoint steps, build acoustic TensorRT engines, verify musical content, and measure the real ring. The original numerical observations below are retained as historical evidence.

| Measured operation | Result on RTX 5090 | Interpretation |
|---|---:|---|
| One acoustic velocity evaluation, 200 frames, eager | 30.43 ms median | Reference path with cached conditioning. |
| Same velocity, CUDA graph, batch one | 11.82 ms median | 2.57× faster; exactly equal output in this probe. |
| Full 32-step midpoint solve, eager | 1,894.8 ms | Eight-second acoustic context, 64 velocity evaluations. |
| Same 32-step solve, CUDA graph | 758.3 ms | 2.50× faster; exactly equal final latents. |
| Four rows at distinct timesteps, one graph forward | 26.53 ms median | Shared conditioning; 1.70% relative velocity L2 difference from independent rows. |
| VAE: 200 ms core, eager / CUDA graph | 11.21 / 8.39 ms median | Actual generated piano latents, 16-frame margins; graph matches eager exactly. |
| VAE: same window shape, FP32 TensorRT / TensorRT graph | 3.03 / 2.94 ms median | Gaussian-latent probe; about 3.7× faster than its eager reference, small numerical differences. |
| VAE: full 16-second region | 81.97 ms median | Same generated piano latents. |

The four-row result implies roughly 1.70 seconds for 32 midpoint ticks before rendering and host work, or about 2.36 completed generations/second once the ring is full. This is an extrapolation from velocity measurements, not a measured DEMON session. Increasing ring depth improves throughput while increasing the time each generation spends in the ring; dividing batch time by four would misstate control latency. The batch-one graph captures took 122 ms for NAR and 51 ms for the VAE, separate from warm execution.

**Scope and reproducibility**

The experiments use the released YuE2-3B and YuE2-Vae weights on the local RTX 5090 (32 GB). Checkpoint bytes were verified against their SHA256 manifests before loading. Source revision: `0edaf2f4053ef4731334b8329834b107977f9637`; model revision: `29b3558dd46954a0cd9021dc76d5c91864a0f1c7`; VAE revision: `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a`.

Runtime: DEMON's Python 3.11.13, Torch 2.9.1+cu128, Transformers 4.57.6, and TensorRT 10.16.1.11 on Windows. YuE2 pins Torch 2.10; these measurements establish a working experimental path in the installed DEMON runtime, not complete compatibility with every upstream backend. An isolated Torch installation was abandoned because local disk space could not accommodate its wheel and extraction. Only `tiktoken==0.12.0` was installed in an external dependency directory; DEMON's environment and dependency lock were not changed.

Windows Torch exposes the native FlashAttention operator schema but raises `USE_FLASH_ATTENTION was not enabled for build` when upstream's AR graph auto-selector chooses it. The benchmark explicitly selects upstream's cuDNN attention path. A tiny GPU model passed both cuDNN and SDPA AR graph execution; the batched acoustic wrapper matched four independent tiny GPU forwards exactly with different timesteps. A production runtime needs a real backend capability probe or an explicit setting, rather than checking the operator schema alone.

Models remain outside the repository under `Z:/codex-yue2-research/models/demon/yue2/checkpoints/`. This is a network drive: checkpoint read/verification/loading is measured separately and should not be interpreted as inference latency. Small generated WAVs, latents, semantic tokens, timing JSON, and request/revision fingerprints are in [`out/yue2/20260914/demon_torch_2_9`](../../out/yue2/20260914/demon_torch_2_9).

Benchmarks use synchronized wall time and CUDA events, normally two warmups and five samples. Full solves are single bounded observations; five samples are too few to characterize sustained tail latency. NAR remains BF16; the VAE remains FP32 with TF32 disabled. These are short probes, without an endurance run or a live DEMON session.

The first TensorRT process built its engine while initial semantic/baseline probes were running, then was stopped to avoid further interference. Treat those initial preparation/full-clip baseline timings as rough observations. It was no longer running during the short-context, acoustic-graph, and generated-latent VAE comparisons used in the table above. The final TensorRT inference benchmark ran separately using that cached engine. Model-phase peak Torch allocation was 9.12 GiB, reserved memory 10.72 GiB, excluding desktop/driver/other-process allocations; the VAE-only phase reused the same resident model worker, so its peak is not standalone decoder memory.

**VAE geometry and window decoding**

The actual trained decoder accepts one latent frame and returns 1,856 finite stereo samples: approximately 38.7 ms at 48 kHz. Two, five, and 25 frames also decode successfully. The general output length is `1920*T - 64`; internal windows retain their complete requested cores, while the last song boundary is 64 samples short. This establishes accepted tensor geometry, not a musically useful minimum generation length.

Window comparisons cover the beginning, interior, and end, with output cores of 5/10/25/50 frames and context margins of 12/16 frames on each side. The initial 800-frame Gaussian-latent run produced maximum absolute waveform differences below `1e-5`. Weight-normalization folding and CUDA graph capture each reproduced their direct reference exactly. Sixteen frames per side remains the conservative integration default; reducing it to 12 has a numerical basis but still needs broader audio coverage.

Repeating these comparisons on the generated 400-frame piano latents gave a worst absolute waveform difference of `3.88e-7` across all core sizes, margins, and positions. A one-second core took 13.99 ms eager and 12.48 ms with graphs. This is strong numerical evidence for window rendering, while encoder-produced latents and diverse high-energy examples remain untested.

The first 32-second full decode took 155.5 ms warm median. A 200 ms core with 16-frame margins took 11.3 ms eager and 9.1 ms with CUDA graphs. A one-second core took 13.9 ms eager and 12.1 ms with graphs. Window cost therefore depends on the local context, rather than the duration of the entire song. All these initial values use trained weights with seeded Gaussian latents.

The fixed 37-frame FP32 ONNX export passed validation and contains standard convolution, transposed-convolution, and elementwise operators. No attention or custom plugin appears in this decoder export. Its 37 input frames cover a 200 ms interior core plus the conservative margins; the raw output contains 70,976 samples per channel and must still be cropped.

The TensorRT engine is 267.8 MB and completed its initial build in 20.16 seconds while other GPU work was present. The separate warm inference comparison measured 11.27 ms eager, 3.03 ms TensorRT, and 2.94 ms TensorRT with a graph. Against the FP32 eager decoder, maximum absolute waveform error was `6.03e-5`, RMSE `1.86e-6`, and relative L2 `1.32e-5`; all outputs were finite. This engine was tested on Gaussian latents, while generated-latent parity above applies to PyTorch window decoding and graphs. The simple direct TensorRT call uses the default CUDA stream; use a dedicated stream in an eventual backend. The captured version avoids that direct-call synchronization warning. Full boundary-profile coverage, generated-latent TensorRT listening, and reduced precision remain future validation.

**What the acoustic experiment proves and leaves open**

The benchmark caches semantic generation, prefills invariant AR conditioning once per acoustic context, and evaluates the released 32-step midpoint solver. It also tests a graph-captured batched velocity wrapper with four distinct timesteps and shared conditioning, exercising the operation DEMON's ring needs for a fixed musical region. Different prompt/semantic caches per row, changing context lengths, and cancellation are not implemented by this spike.

Batch one reproduces both the eager velocity and complete reference solver exactly in the measured case. Batch four does **not** pass that same parity criterion: maximum velocity difference is 0.189, relative L2 1.70%. BF16 kernel changes with batch shape are a possible explanation, not an established cause. Isolate batched eager versus graph replay, projection/time-embedding/attention differences, and accumulated solver error before calling the batched backend validated. The tiny GPU test's exact equality is insufficient evidence for trained-weight batching.

The historical 16/8-step renders are not accepted acceleration candidates for the undistilled checkpoint. They do not substitute for a few-step student model. Acoustically synthesizing a short semantic excerpt also changes bidirectional acoustic attention context; it is a different operation from decoding a short window of full-context latents.

Solo-piano, solo-drum, and vocal-control requests are bounded single-seed probes. Generated files make these cases reviewable, but numerical finiteness, RMS, and prompt text do not establish instrument isolation or musical quality. Natural EOS and token-budget truncation are recorded separately. Reliable single-instrument claims still need listening over more examples. The public model returns a stereo mix, not separate instrument stems.

| Capability probe | Observed result | Review artifact |
|---|---|---|
| Supplied four-bar piano score, empty tagged lyrics | 400 semantic tokens / 15.999 s; token budget reached | [Piano request audio](../../out/yue2/20260914/demon_torch_2_9/piano_score/audio.wav) |
| Solo-drum prompt, planning off | 400 tokens / 15.999 s; token budget reached | [Drum request audio](../../out/yue2/20260914/demon_torch_2_9/drums_off/audio.wav) |
| Vocal-song control | Score reached 512-token cap; semantics reached 800-token cap / 31.999 s | [Vocal control audio](../../out/yue2/20260914/demon_torch_2_9/vocal_control/audio.wav) |
| Acoustic semantic excerpts | 1/5/25/50/100/200 frames all solve and decode to finite audio | [~1 second](../../out/yue2/20260914/demon_torch_2_9/piano_excerpt_0025f.wav), [~8 seconds](../../out/yue2/20260914/demon_torch_2_9/piano_excerpt_0200f.wav) |
| Reduced midpoint steps | 16 / 8 steps render finite audio; quality not adjudicated | [16 steps](../../out/yue2/20260914/demon_torch_2_9/piano_8s_graph_16steps.wav), [8 steps](../../out/yue2/20260914/demon_torch_2_9/piano_8s_graph_8steps.wav) |

**Smallest accepted acoustic generation:** one frame, approximately 38.7 ms after decoding, now verified with trained weights. **Smallest natural complete musical generation:** unknown; all direct semantic probes hit their budgets even with `min_tokens=25`, and the vocal score itself was truncated. These are review excerpts, not a natural-ending full-song control. The supplied four-bar score did not force an eight-second duration under this request. No claim of reliable isolated instruments or useful one-shot sounds follows from these probes; audio has been saved but has not received a listening evaluation in this investigation.

The [codec-dropout probe](../../out/yue2/20260914/demon_torch_2_9/piano_text_conditioning_only.wav) hides semantic conditioning keys from the NAR stage while retaining the original sequence layout. Its name does not imply a validated direct-text generation mode. Its supplied piano score also remains part of conditioning. It rendered finite audio but reached an unclipped peak of 1.233 (about 0.011% of samples exceed ±1); WAV exports are clipped to PCM range, while JSON records the original signal statistics.

**Next integration work**

1. Add an experimental YuE2 context/adapter and persistent window codec using DEMON's existing family seams. Prepare a fixed semantic sequence outside the refinement loop and keep model/conditioning state resident.
2. Add explicit midpoint scheduling and exact initial-noise injection. DEMON currently uses Euler updates; an adapter-only substitution would silently change the reference solver. Preserve YuE2's frame-major random draw rather than assuming equal seeds reproduce DEMON's channel-major noise.
3. Resolve the trained-weight batch-four discrepancy, then validate rows with distinct conditioning and mutable graph input buffers. Measure actual ring tick time, generation age, playback lead, and cancellation. Isolated velocity timing is not a live-session latency measurement. If more speed is needed, projection fusion and a fixed-profile NAR TensorRT build are the next parity-preserving candidates; they were not benchmarked here.
4. Test audible control with fixed semantics: source/noise strength first, then changed style/score conditioning. Decide which edits require semantic regeneration and expose that preparation delay honestly.
5. Use the canonical family/knob/protocol registries for the eventual backend, with unsupported capabilities disabled. Keep short-duration and instrument-isolation findings visible without treating them as blockers.

The source/model license distinction from the original plan remains: source Apache-2.0, released weights CC BY-NC 4.0. Deployment suitability is a separate decision from technical feasibility.

**Running the probes**

The scripts do not download checkpoints or start DEMON. Supply the pinned upstream package through its environment or `PYTHONPATH`, plus the external checkpoint root. On this machine:

```powershell
$env:PYTHONPATH='D:/codex-yue2-runtime-20260914/extra-deps;D:/codex-yue2-runtime-20260914/vendor/YuE-0edaf2f4053ef4731334b8329834b107977f9637/src'
.venv/Scripts/python.exe -u scripts/spikes/yue2_feasibility.py --phase all --ar-attention cudnn --model-root Z:/codex-yue2-research/models/demon/yue2/checkpoints --output out/yue2/20260914/demon_torch_2_9

.venv/Scripts/python.exe -u scripts/spikes/yue2_vae_trt.py --stage export --checkpoint Z:/codex-yue2-research/models/demon/yue2/checkpoints/YuE2-Vae --artifacts Z:/codex-yue2-research/models/demon/yue2/trt_engines/rtx5090_fp32_t37 --output out/yue2/20260914/demon_torch_2_9/vae_trt.json
.venv/Scripts/python.exe -u scripts/spikes/yue2_vae_trt.py --stage benchmark --checkpoint Z:/codex-yue2-research/models/demon/yue2/checkpoints/YuE2-Vae --artifacts Z:/codex-yue2-research/models/demon/yue2/trt_engines/rtx5090_fp32_t37 --output out/yue2/20260914/demon_torch_2_9/vae_trt.json
```

[`yue2_feasibility.py`](../../scripts/spikes/yue2_feasibility.py) saves semantic results so subsequent runs reuse them. [`yue2_vae_trt.py`](../../scripts/spikes/yue2_vae_trt.py) separates CPU ONNX export from a single-profile GPU build and benchmark. Run GPU benchmarks serially; overlapping builds or other GPU work invalidate comparisons. Scripts and documentation are research artifacts, ready for review; production DEMON code and the git index are unchanged by this work.
