**YuE2 feasibility plan for DEMON — 2026-09-14**

**Status:** the source-inspection plan below is historical. Read the [actual ring, acoustic TensorRT, distilled-model search, and musical-duration results](YUE2_RING_RESULTS.md) for the current assessment. The measured BF16 route has not established DEMON's required live responsiveness. Short-clip and individual-instrument behavior are capability measurements, not blockers for model support.

This assessment targets **YuE2**, specifically `m-a-p/YuE2-3B` with `m-a-p/YuE2-Vae`. This initial plan covers source inspection and CPU structural checks. The subsequently authorized local-GPU experiments are documented separately in [GPU results](YUE2_GPU_RESULTS.md); the initial evidence section below describes the state before those experiments.

**What DEMON actually needs**

DEMON's ring holds generations of the same musical timeline at different diffusion steps. It advances them together, adopts completed latents, and decodes around the playhead. Playback can keep moving through cached audio while generation changes the upcoming waveform. Therefore, whole-song generation faster than playback is insufficient evidence of suitability: repeated refinement latency, parameter response, and window decode cost determine whether the model works as a live instrument.

The current checkout already supports ACE-Step and Stable Audio 3. Its reusable boundaries are:

| Existing component | What a YuE2 implementation would use |
|---|---|
| [`ModelAdapter`](../../acestep/engine/model_adapter.py) | Family-specific velocity evaluation, timestep interpretation, latent shape, and conditioning. |
| [`StreamPipeline`](../../acestep/engine/stream.py) | Ring slots, batching, source/noise initialization, curves, and completed-generation handling. |
| [`GeneratorBackend`](../../acestep/streaming/generator_backend.py) and [`DiffusionBackend`](../../acestep/streaming/diffusion_backend.py) | Separate `produce()` and `render_window()` operations, capabilities, geometry, and timing. |
| [`families.py`](../../acestep/streaming/families.py) | Backend factories, checkpoint aliases, session creation, warmup policy, and knob discovery. |
| [`SA3Adapter`](../../acestep/engine/sa3_adapter.py), [`SA3Context`](../../acestep/engine/sa3_context.py), [`sa3_session.py`](../../acestep/streaming/sa3_session.py) | The existing second-family implementation and persistent codec/conditioning lifecycle. |
| [`PipelineRunner`](../../acestep/streaming/pipeline_runner.py) | Playhead targeting, adaptive lead, cached rendering, and audio emission. |

**Findings that determine the experiments**

| Question | Evidence and current answer |
|---|---|
| Does YuE2 have an appropriate diffusion stage? | Yes. It exposes acoustic flow matching and a cached velocity evaluator. Adapting that stage is plausible; preserving its solver and proving useful steering remain work. |
| Can its heavy compute be accelerated? | There are concrete targets in both the autoregressive (AR) and acoustic/non-autoregressive (NAR) stages. Which dominates depends on clip length and whether semantics are cached. Measure them separately. |
| Can the VAE decode windows? | Yes, upstream already implements context-and-crop tiling. An efficient random-access decoder for DEMON is a smaller extension than inventing a streaming codec. |
| What is the smallest supported musical duration? | Structural frame counts and the default semantic token minimum do not answer this. The subsequent musical-content tests are reported in [round two](YUE2_RING_RESULTS.md). |
| Can it generate individual instruments? | Instrumental arrangements are worth testing, but dependable isolated piano/drums/bass/guitar output is unproven. The public generation API returns a stereo mix. |

The staged API is `plan()` → `generate_semantic()` → `synthesize()` → `decode()`. Supplying an ABC score avoids score generation; `cot="off"` avoids symbolic planning but still generates semantic tokens. Existing stages can be cached independently. [Generation guide](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/docs/generation.md)

The acoustic path computes AR conditioning keys/values once per chunk, then reuses them during denoising. Its reference integrator uses **32 midpoint steps, two velocity evaluations per step**. Acoustic attention sees all acoustic positions within the chunk. Shortening that attention context changes the computation; query tiling only reduces temporary storage. [Acoustic implementation](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/src/yue2/nar.py)

Our meta-device inspection of the released configuration counted approximately **3.58 billion total parameters**, including **1.409 billion in NAR layer modules**. Those counts describe size, not latency. The published 4090 result is 71.04 seconds for 214.85 seconds of audio, averaged over 32 warm full-planning requests. That is useful baseline context, but does not establish DEMON's response time. [Official model card](https://huggingface.co/m-a-p/YuE2-3B#-speed-and-resources)

Upstream already uses CUDA graphs for AR decoding and has an optional vLLM path. Experimental FP8 targets AR projections and disables the current AR graph path; original AR weights are restored for acoustic prefill. These are alternatives to benchmark, not optimizations whose benefits can be assumed to add together. [AR graphs](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/src/yue2/cuda_graph.py), [vLLM implementation](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/src/yue2/fast.py), [FP8 implementation](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/src/yue2/quantization.py)

The VAE is a convolutional Oobleck decoder with SnakeBeta activations, 64 latent channels, a 1,920-sample stride, and 48 kHz stereo output. Our structural probe counted 66,374,466 decoder parameters and a required symmetric context margin of **12 latent frames** for all tested core sizes. Upstream conservatively supplies **16 frames per side**, or 640 ms. A 200 ms core would therefore decode approximately 1.48 seconds of latent context in an interior window. The source requires FP32 decoder weights; lower precision is a separate quality experiment. [VAE implementation](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/src/yue2/modeling_vae.py)

The natural full-decode length is `1920*T - 64` samples. Internal windows retain their requested cores; the final song boundary is 64 samples shorter than `T/25` seconds. This needs explicit handling in audio geometry and boundary tests. Matching ACE-Step's latent dimensions and rate does not make the two models' latent values or VAE weights interchangeable. [Upstream VAE topology tests](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/tests/test_vae.py)

The default minimum of 200 semantic tokens is a **sampling setting**, not a hard eight-second architecture limit. `Sampling` permits a lower minimum. Setting a maximum token budget limits duration but can truncate music before a natural ending. There is no direct duration-in-seconds field in the released request schema. [Request and generation configuration](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/src/yue2/protocol.py), [EOS handling](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/src/yue2/sampling.py)

There is a particularly valuable exploratory path: `nar_cond_end` can restrict acoustic conditioning to the text prefix; the model source describes this as codec dropout. Test whether this supports useful generation without sampled semantic tokens, including score-conditioned and text-only variants. The branch's existence does **not** establish checkpoint quality in that mode, and token layout, positions, and boundary handling must remain valid. [Model attention and conditioning](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/src/yue2/modeling_yue2.py)

Current instrumental-generation reports are mixed: users report persistent vocals and inconsistent success with empty lyric sections. Treat these as observations that motivate testing, not an authoritative capability guarantee. Also distinguish an instrumental mix from a single instrument, a generated stem from a separated stem, and a sustained musical phrase from a one-shot note. [YuE2 instrumental issue](https://github.com/multimodal-art-projection/YuE/issues/172)

**Investigation sequence and decision gates**

**1. Establish a reproducible GPU baseline and locate the actual bottleneck.**

Use an isolated uv-managed environment initially. DEMON pins Torch 2.9.1+cu128; YuE2 pins Torch 2.10.0. YuE2 documents Python 3.12 as its starting point, while its package permits Python >=3.10, so Python 3.11 alone is not the incompatibility. Start on the intended NVIDIA deployment GPU, using Linux/WSL if testing the supported vLLM path. Avoid changing DEMON's dependency stack to perform this experiment. [YuE2 dependencies](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/pyproject.toml), [DEMON dependencies](../../pyproject.toml)

Use one normal song as an upstream correctness baseline and a small set of shorter requests. Save exact prompt/score tokens, semantic IDs, initial noise, latents, audio, configuration, and revisions. Measure cold load, score generation, semantic prefill/decode, NAR conditioning prefill, one velocity evaluation, the full solver, VAE decode, and CPU/GPU transfers separately. Report synchronized wall time, GPU event time, peak allocated/reserved memory, and p50/p95 warm latency. Compare complete generation with repeated acoustic synthesis from cached semantics.

**Exit evidence:** a stage-by-stage timing table and a trace identifying the dominant work for both first audio and repeated refinement. A whole-song real-time factor alone does not pass this gate.

**2. Find the minimum useful region and establish instrument/control behavior.**

Use frame counts `1, 2, 5, 10, 25, 50, 100, 200, 400, 800, 1500`, corresponding to nominal durations from 40 ms through 60 seconds. Run cheap shape checks first; concentrate listening and GPU measurements on 0.4, 1, 2, 4, 8, and 16 seconds, with 32/60 seconds as controls.

Compare three distinct operations: generate a short sequence directly; synthesize an excerpt of a longer semantic sequence; decode a short window from full-context acoustic latents. The latter only establishes window rendering. It does not prove short-context acoustic generation. For semantic excerpts, compare against the matching interval from the full-context render and score boundary resets, missing attacks, altered instrumentation, and loss of musical structure.

For direct short generation, lower the minimum token count, record whether EOS ended the sequence, and keep token-budget truncations separate. Test supplied one-, two-, and four-bar ABC scores at known tempos, with full/melody/off planning where applicable. Evaluate actual rendered timing rather than assuming notation forces exact duration.

Start with solo piano, drum groove, bass line, and guitar riff, plus an instrumental ensemble and a vocal-song control. Compare empty lyrics, empty tagged sections, and supplied instrumental scores. Screen with three fixed seeds, then validate promising cases over at least ten. Record instrument adherence, vocal leakage, extra instruments, attack/decay integrity, loop seams, and musical usefulness. Include human listening; an ASR transcript alone will miss humming and vocal chops.

On a fixed semantic sequence, test source/noise strength and gentle velocity/x0 controls. Then change the style prefix while keeping semantics fixed, rebuild conditioning, and measure how much audible control remains. Finally test the codec-dropout path as a separate experiment. These results decide whether semantics can stay outside the live loop or must be regenerated on important edits.

**Exit evidence:** independently reported minimum accepted shape, shortest natural generation, shortest useful excerpt, and minimum reliable instrument/loop duration; a control-response comparison; and a decision on whether direct NAR deserves further work. Weak short-clip or single-instrument results do not stop the acceleration investigation or disqualify the backend.

**3. Accelerate the VAE and prove a playhead-window decoder.**

Keep the decoder resident. Build a thin window wrapper around the audited context-and-crop geometry. Compare output cores of 5, 10, 25, and 50 frames, initially retaining the upstream 16-frame margin. Test the measured 12-frame margin only after parity validation with actual weights.

Benchmark FP32 eager, `torch.compile`/CUDA graphs, and ONNX→TensorRT on fixed window shapes. Fold weight normalization into convolution weights for export and precompute invariant Snake parameters. Retain FP32 as the reference; evaluate TF32 or selective reduced precision independently if needed. Export success, runtime speed, and audible equivalence are three separate results.

Compare full decode and arbitrary window decode at the beginning, interior, end, and very short clips. Measure waveform error before clipping, spectral error, and clicks around joins, using both real generated latents and encoder-produced latents. Test high-energy material and silence. Validate any padding used to fit a TensorRT profile, especially at real song boundaries. Do not use a crossfade to conceal decoder errors.

**Exit evidence:** measured window latency independent of total song length, acceptable numerical/listening parity, correct final sample count, and persistent decoder memory usage. This gate can succeed even if short acoustic generation fails.

**4. Accelerate acoustic velocity evaluation and preserve the solver.**

Extract a persistent inference unit with latent state, per-row timestep, positional state, and cached conditioning K/V as explicit inputs. Reuse conditioning while the prefix and semantics are unchanged. Build a benchmark for short durations and ring depths 1/2/4; proceed to 8 only if memory and latency justify it. Measure batch-one looping as well as actual slot batching.

Compare upstream cached BF16/SDPA, compiled execution with preallocated buffers and CUDA graphs, and TensorRT. Start TensorRT with one short profile and batch one before generalizing shapes. Inspect whether grouped attention remains fused; measure cache concatenation/copy cost, projection fusion, and host synchronization. Carry real prefix lengths and masks when padding conditioning. Cache sharing across identical conditions should avoid duplicating every layer's prefix K/V for every slot.

First reproduce 32-step midpoint with the same initial noise and boundary embeddings. Keep solver time, logit conversion/clamp, and model time shifting distinct. Search for released distilled/turbo checkpoints before considering a lower evaluation count. Without a validated few-step checkpoint, retain the released solver and step count. Evaluate NAR quantization as a separate acceleration route with explicit numerical and audio comparisons; it does not turn the base model into a few-step student.

DEMON needs two explicit integration changes for a faithful baseline: midpoint scheduling and a way to preserve YuE2's initial noise. Current DEMON CPU noise is drawn in channel-major layout and transposed; YuE2 draws frame-major noise. An equal seed therefore does not establish equal inputs. Use an explicit initial-noise hook or a narrowly scoped family initialization policy. Include mixed-timestep and mixed-conditioning slot tests, and ensure persistent accelerator output buffers cannot overwrite results still held by other slots.

For `S` midpoint steps and ring depth `D`, let `t_tick` include both velocity passes plus the per-tick rendering/host work. Approximate completed generations/second as `D / (S * t_tick)` and a slot's denoising time as `S * t_tick`. At 32 steps, a 500 ms denoising budget requires a tick below 15.6 ms; playback lead and conditioning updates add further response latency. Increasing depth improves throughput but does not remove a slot's sequential denoising steps.

**Exit evidence:** a quality/latency frontier for clip length, depth, precision, and evaluation count. Identify useful configurations at the durations the model handles well. If only multi-second refinement is achievable, document that product behavior explicitly.

**5. Prove DEMON behavior before expanding model support.**

Prototype `YuE2Context`, `YuE2Adapter`, a persistent window codec, and `YuE2Backend` following the SA3 boundaries. The initial context should prepare or load a fixed semantic sequence once, then let the ring repeatedly synthesize that musical region. Introduce only the controls that passed the preceding experiments. Use the direct-NAR route only if its results warrant it.

The upstream convenience pipeline moves models between CPU and GPU around decode and adjusts process-wide Torch settings. A live integration needs explicit ownership of residency and settings rather than repeated calls to the full pipeline. Decide between a compatible in-process runtime and a dedicated worker after measuring dependency compatibility and per-tick IPC cost. [Pipeline lifecycle](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/src/yue2/pipeline.py)

Add family registration, checkpoint resolution, session construction, and setup/preflight entries after the spike passes. Keep YuE2 assets in a family namespace under DEMON's external model root, respecting `ACESTEP_MODELS_DIR`; do not route its weights through the ACE-Step loader. Record artifact precision, GPU architecture, shape profiles, source revisions, and weight identity for engine discovery.

Use the canonical knob/protocol registries and capability flags. Semantic sampling guidance and acoustic diffusion guidance have different meanings and must not share a misleading control. Unsupported stems, LoRAs, reference conditioning, or per-frame controls should stay disabled. Regenerate wire types after registry changes; rebuild the shared SDK bundle if its source changes. A new frontend is unnecessary for the first feasibility proof.

Run focused adapter/solver/codec tests and existing ACE/SA3 parity tests, then the required unit/contract checks. Validate the live session with headless lead and staleness measurements, followed by listening. A headless primary session preempts another active client, so run this on an isolated test server. Exercise prompt changes, parameter sweeps, cached rendering during preparation, loop wrapping, cancellation, source replacement if supported, and repeated session teardown.

Suggested initial streaming targets are a 30-minute run without underruns or growing audio staleness, warm acoustic-control response below 500 ms, and a 200 ms window decode below 10 ms on the selected deployment GPU. Measure short regions and full arrangements separately; short-clip success is not required for model support. Cached playback alone does not pass: verify that fresh generations reach upcoming audio throughout the run. These are investigation targets, not measured promises. Report first-audio latency and structural prompt-change latency separately, and compare the final result with DEMON's existing ACE path on the same hardware. Start with bounded GPU probes before a long endurance run to minimize time and energy spent on an unsuitable configuration.

**Additional constraints that affect the decision**

The released acoustic VAE encoder cannot produce YuE2's semantic codec IDs. The semantic audio tokenizer was not available in the inspected release, so source-audio conditioning and training cannot assume that conversion exists. Covers through a transcribed score, acoustic latent anchoring, and source-audio semantic conditioning are different capabilities. [Tokenizer release question](https://github.com/multimodal-art-projection/YuE/issues/165)

The source is Apache-2.0, while the current model weights are labeled CC BY-NC 4.0. Record those separately and resolve suitability for DEMON's intended hosted/commercial use before a deployment decision. [Repository license](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/LICENSE), [model license](https://github.com/multimodal-art-projection/YuE/blob/0edaf2f4053ef4731334b8329834b107977f9637/MODEL_LICENSE)

**Evidence collected in this assessment**

Pinned source revision: `0edaf2f4053ef4731334b8329834b107977f9637`.

Observed model revisions: YuE2-3B `29b3558dd46954a0cd9021dc76d5c91864a0f1c7`; YuE2-Vae `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a`. Only configuration metadata was fetched from these model repositories.

CPU/meta inspection used DEMON's existing Python 3.11.13, Torch 2.9.1+cu128, and Transformers 4.57.6. A tiny random acoustic model solved inputs of 1, 2, 5, and 25 frames with finite outputs. Eleven selected upstream tests passed: real VAE topology/length checks, tiled-vs-full tiny-VAE checks, cached-vs-dense acoustic velocity, midpoint solver equivalence, and the 64-evaluation default. These establish structural behavior, not trained-model audio quality or compatibility of the accelerated GPU runtime.

The next concrete deliverable should be the GPU stage timing table plus the short-duration/instrument listening matrix. Together they will tell us whether to pursue cached-semantics streaming, direct acoustic generation, or retain YuE2 as a slower composition/preparation option.
