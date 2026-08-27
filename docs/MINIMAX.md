# MiniMax-Music3 in DEMON

**Verdict: it works, in real time, on a single 5090 — 9.5x realtime
headroom in eager bf16, 15.2x with the TensorRT fp16 engine, at output
that matches the reference model's own trajectory to latent cosine
0.9993.** What is integrated is the model's *renderer*, not the whole
model, and that distinction is the entire design. This document says
why, what the numbers are, and what is not done yet.

Throughput, stated the way this project states it (`depth/(steps*tick)`,
with the 60 s-normalized figure beside it because a "generation" is one
whole song and the families' songs differ in length by 7.5x):

| | gens/s | 60 s-gens/s | realtime |
|---|---|---|---|
| minimax, 8.0 s song, eager bf16 | 1.19 | 0.158 | 9.5x |
| minimax, 8.0 s song, TRT fp16 | 1.90 | **0.253** | 15.2x |
| minimax, 14.4 s song, eager bf16 | 0.80 | 0.192 | 11.5x |
| *sa3 medium, 54 s song, TRT* | *~6.0* | *~5.4* | *~324x* |
| *acestep turbo, 60 s song, TRT* | *11.3* | *11.3* | *~678x* |

**This family is roughly 20-70x slower than its siblings in normalized
terms, and that is structural rather than a missing optimization.** It
spends 32 forwards per generation (16 steps x guidance) where ACE-Step
turbo spends 8 and needs none, and its DiT is 2.43B undistilled. §3a is
why the 32 cannot come down much without the output visibly degrading.

> **Revised twice on 2026-08-27.** The first version claimed 38x/61x —
> real measurements of a sampler configuration that rendered badly (8
> steps, guidance off, against a reference running 30 steps at guidance
> 1.7); §3a is that diagnosis. The second revision fixed a decode path
> that was O(song length) rather than O(window), a duration that was
> hard-coded on a false premise, and instrumentation that reported a
> flat 0.0 ms for the cost it was hiding; §3b is that one.

---

## 1. What MiniMax-Music3 actually is

Three stages, ~11.8B parameters total:

| Stage | Params | Role |
|---|---|---|
| Global LLM (Qwen3-derived) | 8.58B | autoregressive, 25 frames/s, emits semantic RVQ code `c0` |
| RVQ depth decoder | 646M | the 7 residual codebooks within each frame |
| **Flow-matching DiT** | **2.43B** | **renders a continuous 128-ch latent at 86.133 Hz** |
| DAV decoder | 54M | 512x upsample to 44.1 kHz stereo, deterministic |

The AR stage's fused per-frame hidden states (8 x 4096 = 32768 per
frame) pass through a 25M condition encoder to become the DiT's
`encoder_hidden_states` `[B, T, 2048]`.

**The DiT has no cross-attention and no text input.** Its only
conditioning is that tensor. There is no path from a prompt to a
denoise step that does not traverse the full 8.58B LM. This is the
fact that shapes everything below.

## 2. Why only the renderer is streamed

Measured on this machine (RTX 5090, Windows), end to end for 25.9 s of
audio: **RTF 0.436x** — slower than realtime. The breakdown says why:

| stage | share | s of GPU per s of audio |
|---|---|---|
| AR loop | **79.3%** | 0.330 (**0.55x realtime**) |
| denoise loop | 20.2% | — |
| vocoder | 0.4% | 0.018 |

The autoregressive stage alone cannot keep up with a listener, and it
is append-only: a committed frame can never be revised. Streaming the
whole model is not possible.

The renderer is a different animal. One DiT forward at L=689 is
**35.3 ms** in bf16. DEMON's loop does one batched forward per tick
over `depth` slots and finishes a whole generation every `steps/depth`
ticks — so at the shipped 16 steps and depth 4 the entire 8-second song
is regenerated every ~0.85 s eager, ~0.53 s on TensorRT.

So: **run the AR stage once to fix a musical idea, then stream the
renderer over it forever.** The conditioning becomes a captured
artifact rather than a per-tick computation. This is also the only
audio-conditioning path this checkpoint actually supports — upstream
ships no converted audio encoder, and the community's measurements
found arbitrary-WAV continuation losing to a trivial baseline, while
"continue from your own generation" works.

## 3. Measured results

### Streaming (`scripts/minimax/minimax_stream_smoke.py`)

Real backend, real `StreamPipeline`, real weights, depth 4 / steps 16 /
shift 2.0 / guidance 1.7, covering at denoise 0.6:

```
                     eager bf16   TRT fp16    eager, 14.4 s song
tick median            210.9 ms   131.6 ms              312.7 ms
render median            6.1 ms     6.2 ms                6.1 ms
rms                  -18.4 dBFS -18.2 dBFS            -20.0 dBFS
gens/s                   1.186      1.900                 0.799
60 s-gens/s              0.158      0.253                 0.192
realtime                  9.5x      15.2x                 11.5x
```

Two things to read off the third column. Render cost is **identical at
14.4 s and at 8 s**, which is the whole point of §3b — decode is now
O(window), not O(song). And normalized throughput goes *up* slightly
with length, consistent with the length sweep below: the short case is
dispatch-bound.

Guidance is what costs the headroom: it doubles the forwards per step.
It is worth every one of them (§3a).

## 3b. Song length, and the decode that assumed it was fixed

Three claims in the first version of this file were wrong. All three
came from treating 8 s as a property of the model.

**"The DiT is trained on 200-AR-frame windows" — unsupported.** The
number 200 is upstream's `_CHUNK_FRAMES`, an *inference* constant: it
renders in 200-AR-frame windows on a 100-frame hop, splicing 172 latent
frames of conditioning and carrying 172 latent frames between windows.
`transformer/config.json` carries no `max_position_embeddings` (the RVQ
depth decoder and the LM both do, so the absence is meaningful), RoPE is
computed for whatever `seq_len` arrives, and there are no length
assertions in either the diffusers or the reference implementation. No
upstream source states a trained span.

Measured here instead: a real 1240-frame capture renders in **one** DiT
pass at 9.0x realtime, with RMS, high-frequency ratio and spectral
centroid consistent across the song's first, middle and last thirds. It
does not fall apart. Whether it is *as good* as chunk-and-carry at much
longer spans is still unmeasured — that is an honest open question, not
a limit.

**"8.011 s" — arithmetic error.** 689 latent frames x 512 / 44100 =
**7.99927 s**.

**"The decoder is ~2% of the render budget" — wrong by an order of
magnitude, and self-concealing.** It was a *whole-song* decode on every
fresh latent: 44.5 ms at 8 s, and linear, so 346 ms at 60 s. Against a
212 ms tick that is 21% on the tick it lands on and ~11 ms amortized —
but it reported as `render median 0.0 ms`, because it fires on one tick
in four and the median hides it. The backend also never assigned
`last_dec_ms`, so the latency trace read a flat zero for the one cost
that was growing.

### What duration actually is

**The autoregressive stage decides it.** `audio_duration` is a frame
budget for the LM, which stops early on an end-of-audio token: asking
this integration for 30 s returned a 14.4 s piece. So duration is
requested (`SessionConfig.minimax_duration_s`) and then *adopted* from
whatever the conditioning turns out to be. The previous code raised
unless the capture was exactly 689 frames, which made every length but
one unreachable.

Bounds that are real: `MAX_AUDIO_FRAMES = 9000` (360 s) upstream, and
the AR stage's 0.54x realtime — a 60 s composition costs ~111 s of
one-time capture before any streaming starts.

### Windowed decode

Each `render_window` now decodes a fixed **58 latent frames** around the
requested slice and keeps the middle. Two measurements size it, both in
`scripts/minimax/minimax_decode_profile.py`:

| guard (latent frames) | ms | mean rel err | peak rel err |
|---|---|---|---|
| 0 | 0 | 1.1e-2 | 1.8e0 |
| 4 | 46 | 1.7e-3 | 2.0e-1 |
| 8 | 93 | 5.6e-4 | 1.5e-2 |
| **9** | **105** | **5.5e-4** | **4.4e-3** |
| 16 | 186 | 5.5e-4 | 4.4e-3 |
| 32 | 372 | 5.4e-4 | 3.9e-3 |

Peak error reaches the fp32 floor at **9 frames** and is flat after. An
analytic walk of the conv stack puts the one-sided field at 10. The
shipped guard is **12**. (The control — decoding the full 689 frames and
comparing against itself — is 0.0e0 everywhere, so the probe is sound.)

The second measurement is a trap rather than a size. **44100:48000
reduces to 147:160, and a latent frame is 512 native samples with
gcd(512, 147) = 1** — so a frame boundary is *not* an integer delivery
sample. Resampling a block that starts on one lands on a different
sample phase than resampling the whole song does, and the window
disagrees with the full decode by **1.8e-1 relative RMS**. Not a
rounding detail; a misalignment. The block is trimmed forward to a
multiple of 147 native samples before resampling, which makes the
offset back into it exact integer arithmetic. SA3 hit this too and
solves it identically with a 588-sample (4 x 147) guard.

The decode span is 58 and not 56 because the kept span converts with
floor on one end and ceil on the other: 0.36 s is 31.008 latent frames
of audio but touches **33** of them at some start positions. Spot checks
passed. A dense positional sweep did not, and is now a test.

Result, verified across 59 windows spanning the song against a
whole-song decode:

| | before | after |
|---|---|---|
| per render | 0 ms (cached) + 44.5 ms spike per generation | **6.0 ms**, every render |
| at 14.4 s | ~80 ms per generation | **6.1 ms** — identical |
| at 60 s | ~346 ms per generation | 6 ms (unchanged) |
| agreement with full decode | n/a | worst 1.4e-2 (-37 dB), bf16 decode noise |

The guard wraps cyclically: at the song head the leading context comes
from the tail, which is not an approximation because the ring loops and
the tail is literally what plays into the head.

### Two other defects the same audit turned up

- **Depth was silently ignored.** `minimax_session` read
  `pipeline_depth` off `SessionConfig`, which names the field `depth`.
  `getattr` returned its literal default forever, so a client asking for
  depth 8 got 4. SA3 reads the right field.
- **The ring buffer was 34 samples too long.** It was sized
  `round(duration * 48000)` while the resampler yields
  `ceil(n * 160/147)`, so the last ~0.7 ms of every song was never
  written. Both now come from `minimax_delivery_samples`.

### Throughput, and what a 60 s generation would cost

The family's unit is one whole song, and the default is 7.999 s, so
"generations per second" here counts 8 s of audio each. `scripts/minimax/minimax_length_bench.py`
times one DiT forward across lengths and normalizes, so the number is
comparable against a family with a different fixed duration.

Eager bf16, 16 steps, guidance on (32 forwards per generation):

| song | frames | B | ms/forward | ms/sample | vs linear | gen/s | 60 s-gen/s | xRT |
|---|---|---|---|---|---|---|---|---|
| 8 s | 689 | 1 | 36.1 | 36.1 | 1.00 | 0.86 | 0.115 | 6.9x |
| 8 s | 689 | 4 | 103.6 | 25.9 | 0.72 | 1.21 | 0.161 | 9.7x |
| 16 s | 1378 | 4 | 198.2 | 49.5 | 0.69 | 0.63 | 0.168 | 10.1x |
| 30 s | 2583 | 4 | 353.3 | 88.3 | 0.65 | 0.35 | 0.177 | 10.6x |
| 60 s | 5167 | 4 | 862.8 | 215.7 | 0.80 | 0.14 | 0.145 | 8.7x |

**Sequence length is close to free.** `vs linear` is cost per frame
normalized to the 8 s B=1 point; it stays at or *below* 1.0 out to 5167
frames, so the attention term never takes over in this range — the
short case is dispatch-bound and gets *more* efficient per frame as it
grows. Normalized throughput is therefore roughly flat at **0.14-0.18
sixty-second generations per second** across an 7.5x span of lengths,
which makes it a compute roof rather than a length effect.

The B=4 column is the one that matters: it is the ring at depth 4, and
103.6 ms x 2 passes reproduces the measured 212.1 ms tick almost
exactly, which says the tick is essentially pure DiT with no streaming
overhead worth naming.

Two things that number does **not** say:

- **It is affordability, not quality.** The DiT is trained at 689
  frames. Everything past that is out of distribution, and upstream has
  an open long-horizon coherence bug (drift after ~15 s) on top.
- **There is no TensorRT engine past 689.** Engines are built per length
  profile and the shipped one is `l2_689_689`, so every row above 8 s is
  eager. TRT is worth 1.6x at 8 s; whether it holds at 5167 frames is
  unmeasured, because no such engine exists yet.

A 60 s *composition* also costs an AR capture of 1500 frames at 0.54x
realtime — about 111 s, once, before any of this runs.

## 3a. Sampler settings: the measurement that fixed the output

The first version of this integration streamed at 8 steps with no
guidance, and the result was the same song as the reference at
noticeably worse quality. Nothing in the parity suite moved, because
every gate there measures a single forward or a single trajectory
against a stored latent — none of them measure the settings the ring
buffer actually runs.

`scripts/minimax/minimax_quality_ablation.py` is the instrument that
does. It walks one variable at a time from ground truth outward (decode
-> sampler -> step count -> guidance -> pipeline -> ring buffer) and
scores each rung in both the latent and audio domains.

**The A/B, at matched conditioning and matched noise:**

| | latent cos | log-mel | left/right corr | RMS | >8 kHz energy |
|---|---|---|---|---|---|
| reference (diffusers) | 1.0 | 0 | +0.096 | -21.2 dB | 0.0041 |
| 8 steps, no guidance | 0.744 | 0.244 | **-0.128** | **-30.7 dB** | **0.0193** |
| **16 steps, shift 2.0, guidance 1.7** | **0.9993** | **0.032** | +0.101 | -21.1 dB | 0.0040 |

The mechanism is legible in the last column: 4.7x the reference's
above-8 kHz energy is undenoised residual, still sitting on the latent
when the schedule runs out of steps. Because that residual is
uncorrelated between channels it also inverts left/right correlation,
which is why it read as phasey and hollow rather than simply quiet.

**What the grid says, in order of size:**

1. **Guidance is not optional, and it is worth more than steps.**
   Unguided sampling plateaus at ~0.11 log-mel from the reference and
   *stays there* — 40 unguided steps score worse than 8 guided ones.
2. **Step count trades against schedule warp nearly one for one.** The
   unwarped schedule needs the reference's 30 steps; warping toward the
   noise end buys most of that back. The measured pairing is 30/1.0,
   20/1.5, 16/2.0, 12/3.0. Lowering steps without raising shift gives
   up most of what the steps were buying.
3. **RCFG is unusable on this model.** Both StreamDiffusion modes were
   measured: `initialize` (cache the uncond velocity from step 0)
   scores 0.45-0.70 log-mel and `self` (uncond ~ initial noise) scores
   0.52-0.92, against 0.03-0.12 for a real uncond pass. The uncond
   velocity moves too much along the trajectory here to be approximated
   away.
4. **Stock APG is the wrong combine operator here** — ~4x worse than
   textbook CFG (0.125 vs 0.032 log-mel at 16/2.0). Its `norm_threshold`
   is calibrated for ACE's latent scale and throttles a 689-frame
   guidance delta nearly to nothing. The backend therefore asks for
   `apg_eta=1.0`, `apg_norm_threshold=0.0`, `apg_momentum=0.0`, which
   reduces APG exactly to `v_u + w*(v_c - v_u)`.

**16 steps / shift 2.0 / guidance 1.7 is statistically indistinguishable
from the reference's own 30 / 1.0 / 1.7.** Over 8 independent noise
draws, log-mel distance to the reference is 0.1631 for both (sd 0.005),
as are RMS, high-frequency ratio and stereo width. The shipped setting
costs nothing but 32 forwards instead of 60.

### What was ruled out, with numbers

Worth recording, because each was a plausible suspect:

| suspect | measurement | verdict |
|---|---|---|
| TensorRT engine | the degraded files were eager | not involved |
| DAV decode / 44.1->48 kHz resample | decoding the reference's own final latent reproduces it exactly | clean |
| ring buffer, crossfade, window placement | ring vs the same backend's whole-song render: rel RMS **0.0093**, log-mel 0.0011, and 0.000 in every second after the first | clean |
| generations drifting apart across the ring | consecutive-generation cosine median **1.0000**, min 0.9824 over 43 covers | stable |
| `StreamPipeline` solver vs a hand-written Euler loop | same noise, same settings: cos **0.9962** | equivalent |
| the noise source | the reference's own `initial_noise` is iid N(0,1) to within a fresh `randn` control | correct |
| the partial-denoise cover path | at denoise 0.6 a cover holds the anchor's balance to +0.24 dB RMS and 1.05x HF | clean |

The one honest open observation: our takes average left/right
correlation 0.43 (sd 0.15) while the single reference take sits at
0.096. On the reference's *own* noise we reproduce its width exactly
(0.098), so this is either take-to-take variance on a high-variance
statistic or an upstream difference invisible from one reference
sample. It needs more than one upstream render to call, and it is not
the degradation that was being chased.

### Parity

Module level, fp32, against the diffusers reference on real inputs
(real AR hiddens -> real conditioning, not random tensors):

| | cosine |
|---|---|
| DiT, B=1, four timesteps | **bit-identical** (rel_rms 0.0) |
| DiT, B=4 with per-row `t` | 0.999999999998 |
| DAV on a real latent | 1.000000000000 |
| condition encoder | 1.000000000000 |

Chain level (`scripts/minimax/minimax_chain_parity.py`) drives the
adapter from a reference run's `initial_noise` and lands on its
`final_latent`: **cos 0.999868**, rel RMS 1.7e-2 (bf16).

### DiT latency, L=689, this GPU

| B | bf16 | ms/sample | fp32 |
|---|---|---|---|
| 1 | 35.3 ms | 35.3 | 90.2 ms |
| 2 | 56.0 | 28.0 | 167.1 |
| 4 | **103.1** | 25.8 | 319.9 |
| 8 | 183.6 | 23.0 | 596.0 |

Batching is a weak lever (B=8 is only 1.54x the per-sample throughput
of B=1) but nearly free in memory: 5.03 -> 5.47 GB.

### VRAM

Renderer only (DiT + DAV + condition encoder), no LM: **5.03 GB** bf16,
7.18 GB peak during a 20 s decode. The full pipeline including the AR
stage peaks at 24.5 GB.

## 4. The design, and what it costs

Implemented as a **Tier-2 `ModelAdapter` plus a thin `DiffusionBackend`
subclass**, the SA3 shape — the DiT refines a whole fixed-length latent
and can answer "give me the audio at song second X", so `refines_audio`
is true and the append-only path does not apply.

The adapter is **two conversions, not one**:

- **Layout.** MiniMax is native `[B, C, T]`; the pipeline is `[B, T, C]`.
- **Time direction.** MiniMax runs `t` from 0 (noise) to 1 (data) and
  steps Euler *forward*. DEMON runs `s` from 1 down to 0 with
  `x0 = xt - v*s`. Substituting `s = 1-t` makes the interpolants
  identical, so only two scalars convert: `t = 1 - s`, and the velocity
  **negates**.

That negation is load-bearing and silent when wrong — a sign-flipped
control run produces output *uncorrelated* with the reference
(r ~ 0.08) rather than an obvious error, which is why there is a
numeric gate for it rather than a listening test.

**What you give up:** `set_prompt` is seconds, not one pipeline flush,
because it re-runs the 8.58B LM (measured **14.8 s** for an 8 s span,
17.3 GB peak, 0.54x realtime). Duration is fixed for the session's
lifetime — requested via `minimax_duration_s`, decided by the AR stage,
adopted at create. It is not fixed at 8 s and never was a model limit
(§3b).

**What survives:** everything DEMON steers solver-side lands in one
tick via the shared-curve override — denoise, seed, the source lock,
the feedback delay tap. Three knobs fall out of the architecture rather
than being invented for it:

- `minimax_guidance` is the reference pipeline's own CFG scale. 1.0
  disables the negative pass and halves the compute; §3a says why that
  is a bad trade on this model.
- `minimax_cond_strength` interpolates the capture toward zeros, which
  is literally the model's own unconditional CFG branch, so 0.0 is a
  defined operating point rather than an extrapolation. At 0.0 the
  positive and negative bundles coincide and guidance becomes a no-op —
  coherent, since the guidance direction around a point is zero.
- `minimax_shift` warps the schedule. It is **not** an independent
  quality dial: it is matched to the step count (30/1.0, 20/1.5,
  16/2.0, 12/3.0), and moving one without the other gives up most of
  what the other was buying.

Prompt blending slerps between two captures per frame.

### The seam change guidance needed

`SlotRequest.neg_conditions` is ACE-shaped — a list of `SlotCondition`
carrying `encoder_hidden_states` and a mask. A Tier-2 family keeps all
of its conditioning in the opaque `aux_cond` bundle and so could not
populate it, which meant **no `aux_cond` family could run CFG at all**,
silently: `has_cfg` returned False, no negative pass was scheduled,
nothing raised. That is how the first version shipped unguided.

Two additions to the shared pipeline fix it, both defaulted so ACE and
SA3 are byte-identical:

- `SlotRequest.neg_aux_cond` — the negative branch of the bundle. The
  negative forward pass now swaps it in. Sending `aux_cond` on both
  passes would be the worse bug: `v_neg == v_pos`, APG returns `v_pos`,
  and guidance costs a full extra forward per step while doing nothing.
- `SlotRequest.apg_eta` / `apg_norm_threshold` — APG's two shape
  parameters, per request, so a family can select the guidance operator
  its model was actually trained with.

`tests/unit/test_stream_aux_cfg.py` covers both, including the near
miss and the "ACE is untouched" direction.

SA3 is the other `aux_cond` family and also streams with no negative
bundle, but that is deliberate rather than the same bug: its
post-trained checkpoints are guidance-distilled and run `cfg_scale=1.0`
by design (see the note in `sa3_adapter.py`). The distinction is worth
keeping straight — the seam was missing a capability, and only one of
the two families needed it.

## 5. TensorRT — built and measured

Built, gated, and streaming. **2.25x over eager bf16 at B=1.**

### Latency, L=689, RTX 5090 (median ms/forward)

| | B=1 | B=4 |
|---|---|---|
| eager fp32 (TF32 off) | 89.3 | 314.7 |
| eager bf16 | 35.6 | 103.0 |
| eager fp16 | 30.4 | 90.5 |
| TRT fp32 | 45.1 | 182.1 (looped) |
| **TRT fp16** | **15.7** | **54.4** batched / 63.3 looped |

Real depth-4 tick through the adapter: **64.0 ms vs 103.0 ms** eager
bf16. End-to-end streaming at the shipped settings (guidance on, so two
passes per step): **132.2 ms median tick, 15.1x realtime headroom**
against eager bf16's 212.1 ms / 9.4x.

Known lever, not taken: `find_dit_engine` pins `batch=1` and the
adapter loops the ring, so the b2-4 engine that is already built never
gets used. Batched is 54.4 ms at B=4 against 63.3 looped, so this is
worth roughly 15% of a tick. It needs `make_dit` to know the tick batch
size, which is a seam change, and 15.1x realtime did not justify one.

### Parity — real fixture conditioning, never random

Per-step vs eager fp32, bar cos >= 0.9998:

| t | TRT fp16 | rel RMS | TRT fp32 | rel RMS |
|---|---|---|---|---|
| 0.05 | 0.999988 | 4.81e-3 | 0.999999 | 1.64e-3 |
| 0.30 | 0.999981 | 6.14e-3 | 0.999999 | 1.73e-3 |
| 0.60 | 0.999981 | 6.19e-3 | 0.999998 | 1.77e-3 |
| 0.95 | 0.999988 | 4.83e-3 | 0.999999 | 1.71e-3 |

Both pass. Full 30-step CFG-1.7 trajectory vs the fixture's
`final_latent`: TRT fp16 **0.999677**, TRT fp32 0.999641, eager fp32
0.999642, eager bf16 0.999867.

That bf16 number is a trap worth naming: the fixture is *itself* a bf16
reference run, so eager bf16 scoring highest is agreement with its own
quantization, not with ground truth. The gate grades against eager
fp32 and prints bf16 for context only. Relatedly, **eager bf16 is not a
usable parity reference on this model** — it only reaches cos
0.998-0.9997/step against fp32, worst at t=0.30.

### Engines

fp16 4.88 GB / 44 s build / 5.08 GB VRAM / 2.0 s load; fp32 9.73 GB /
51 s. ONNX export 76-78 s each. Built with
`acestep/engine/trt/minimax_build.py`, reusing
`sa3_build.py::_build_strongly_typed_engine` unchanged. No plugins.

```bash
.venv/Scripts/python.exe -m acestep.engine.trt.minimax_build --precision fp16
.venv/Scripts/python.exe scripts/minimax/minimax_trt_parity.py
```

**Build on an idle GPU.** Engines built under contention succeed and
then segfault on load; nothing in the codebase enforces this.

### Things that cost a build attempt each

- **Batch cannot be dynamic from 1.** `torch.export` 0/1-specializes,
  and the `matmul` decomposition emits a `batch != 1` guard at
  `proj_out`. A batch-dynamic export is only legal with `min_batch >=
  2`, so production engines are batch-1 like SA3's and the adapter's
  `trt_batch1` branch loops slots. The `b2_4` engine exists only to
  measure the B=4 line and is excluded from discovery.
- **Same trap on length**; profile min is 2.
- **The "fp32" engine is really TF32** — TRT sets `BuilderFlag.TF32` by
  default. Proven rather than assumed: eager TF32 vs strict eager fp32
  deviates by exactly the amount the engine does (1.64-1.77e-3 across
  the four t values). So its apparent 2x over eager fp32 is 1.23x over
  an eager TF32 baseline. It is a control, not a candidate.

### Precision, and the RoPE question

fp16 is both faster *and* more accurate than bf16 on this DiT in eager
(60.7 ms/step at 48.7 dB SNR vs 70.4 at 29.3), so fp16-mixed
STRONGLY_TYPED was the target and it cleared the bar.

The SA3 failure mode — bf16 RoPE rotation angles reaching thousands of
radians where the spacing is ~32 rad — does not occur here.
`_rope_tables` already pins `arange`/`inv_freq` to fp32 regardless of
module dtype, and the TRT path keeps the **rotation itself** in fp32
rather than casting the table down. The exported graph confirms it:
fp32 `Range`/`Cos`/`Sin`, 147 fp32 `Mul` + 72 fp32 `Neg` (the rotation
island), 72 fp32 `LayerNormalization`, fp32 time-embed `Gemm`s, 290
fp16 trunk `MatMul`, fp32 IO. The builder asserts the fp32 tables so an
edit that drops the `.float()` fails the build rather than shipping
half-radian rotations.

The fp16 error is **flat in t** (4.8e-3 at the ends, 6.2e-3 mid), which
is trunk quantization rather than an angle problem. Attention softmax
was left in fp16 and cleared the bar, so no island was added there —
forcing it fp32 cost SA3 4.3x for nothing.

**Do not try fp16 on the DAV vocoder** — measured all-NaN. The decoder
stays fp32/bf16.

## 6. The capture stage

`acestep/engine/minimax_ar.py` turns a prompt into the only thing the
DiT accepts. Qwen3 global LM through `transformers`, plus the RVQ depth
decoder reimplemented in pure torch from its 47 safetensors keys
(**bit-exact** against the reference at fp32 and bf16). Paged
CPU<->CUDA around each capture so 17 GB does not sit on the card
between compositions.

Measured, 200 frames (8.0 s), seed 7: **14.8 s wall, 13.5 LM tok/s,
0.54x realtime, 17.3 GB peak**, bit-identical across runs at a fixed
seed. Profiled: LM decode 48.5 ms/frame (67%), depth decoder 21.1 ms
(29%). The `lm_head` GEMV already runs near peak bandwidth, so the
remaining ~37 ms is per-kernel dispatch across 36 layers of small ops.
It is **dispatch-bound, not FLOP-bound** — CUDA graphs are the lever
(worth roughly 2-3x upstream), not quantization.

### The transformers trap

`AutoConfig.from_pretrained` **succeeds** on this checkpoint under the
pinned 4.57.6. That is the problem. The v5-style `rope_parameters` dict
falls into `**kwargs`, is stashed as an inert attribute, and
`rope_theta` silently keeps the 4.x default of **10000 instead of the
checkpoint's 1000000**. Every position encoding in a 36-layer model
would be wrong, with nothing raised — the model loads, runs, and
produces confidently wrong audio.

`load_qwen3_config` reads the JSON directly, remaps, and then *asserts*
the value took. Each remap is gated on the installed signature so the
shim retires itself when the pin moves.
`tests/unit/test_minimax_ar_config.py` guards it, including a test that
demonstrates the naive loader getting it wrong.

    .venv/Scripts/python.exe scripts/minimax/minimax_capture.py         --prompt "driving instrumental darkwave, analog synth bass,                   tight gated drums, minor key, 124 bpm"         --lyrics "[instrumental]" --seconds 8 --seed 7         --out <models>/minimax/captures/darkwave_8s.safetensors

Lyrics may not be empty; use `[instrumental]`. Note the reference
normalizes both fields rather than inserting them verbatim, and
**silently drops any lyric sharing a line with a leading `[tag]`**.

## 7. Not done
- **Not driven through `PipelineRunner` or the WS server.** The proof is
  at the backend level: the real `StreamPipeline`, the real produce /
  render loop, the real crossfade. Session creation, `families.py`
  registration and the knob manifest are wired, but a browser session
  has not been run.
- **No web panel**, no boot preflight in `server.py`.
- `swap`, `write_audio`, `timbre`, `structure` and LoRA are
  capability-gated **off**. Most need an audio encoder this checkpoint
  does not ship converted. (One exists unconverted inside `dav.pth` —
  186 `encoder.*` keys plus `mean_proj`/`logs_proj` — so audio-to-audio
  is a real follow-up, not a dead end.)
- Upstream has an open **long-horizon coherence bug** (coherent for
  ~15 s, then genre/timbre drift) and unreliable instrumental control.
  Neither affects the 8 s window this integration uses.
- **Longer single-pass windows are unmeasured, not unavailable.** One
  1240-frame render came out clean; upstream would have rendered the
  same span as two 689-frame windows with a 172-frame carried overlap.
  Which sounds better past ~15 s is an open question, and porting
  upstream's chunk-and-carry is the fallback if single-pass degrades.
- **No TensorRT decoder engine.** Windowed decode is ~6 ms of a 132 ms
  TRT tick, of which roughly 3.5 ms is launch overhead an engine would
  cut. The fixed 58-frame window is exactly the shape to build at. Note
  fp16 is not an option — it produces all-NaN on this vocoder.
- **The DiT engine is length-pinned.** `l2_689_689` serves only the
  default song, so any other duration silently drops to eager. The
  builder already takes `--min-latents/--max-latents`, so a ranged
  engine is a build-flag change, not a code change.
- **Stereo width is one observation short of a conclusion.** Our takes
  average left/right correlation 0.43 (sd 0.15 over 8 draws) against a
  single reference take at 0.096 — which we reproduce exactly on that
  take's own noise. Settling it needs several upstream renders, which
  means the AR stage and a diffusers install, not just the renderer.
- **The b2-4 TensorRT engine is built but never selected** (§5). Worth
  ~15% of a tick; needs `make_dit` to learn the tick batch size.

## 8. Running it

```bash
# Streaming proof from a saved capture. The step / shift / guidance
# defaults are the measured ones; add --accel tensorrt for the engine.
.venv/Scripts/python.exe scripts/minimax/minimax_stream_smoke.py \
    --capture <models>/minimax/fixtures/minimax_music3_8s_seed7.safetensors \
    --seconds 24 --depth 4 --out out/v2_stream_eager.wav

# Where the quality is and what it costs: the ablation ladder, the grid
# that picked the defaults, then the cover path the ring actually runs
.venv/Scripts/python.exe scripts/minimax/minimax_quality_ablation.py \
    --fixture <models>/minimax/fixtures/minimax_music3_8s_seed7.safetensors \
    --out-dir out/ablation
... --out-dir out/ablation --sweep
... --out-dir out/cover --cover-sweep

# Prove a knob steers the stream live
... --sweep minimax_denoise --sweep-from 0.15 --sweep-to 0.95
... --sweep minimax_guidance --sweep-from 1.0 --sweep-to 2.5

# No capture needed: the model's own unconditional branch
... --uncond

# Parity gate
.venv/Scripts/python.exe scripts/minimax/minimax_chain_parity.py \
    --fixture <models>/minimax/fixtures/minimax_music3_8s_seed7.safetensors
```

Weights: `MiniMaxAI/MiniMax-Music3`. The repo ships **two** complete
copies (57.35 GB); only the diffusers layout is needed (**28.52 GB**).
Skip `qwen_7B/`, `flowmatching_vae.pth`, `dav.pth` unless you are doing
encoder work. Set `HF_HUB_DISABLE_SYMLINKS=1` on Windows.

## 9. Licence

**MiniMax-Music3 Community License** — custom, not OSI. Commercial use
is permitted with two conditions: display "MiniMax-Music3" prominently
in the product UI, and obtain written authorisation if aggregate yearly
revenue exceeds **US$20M**. There is no territorial carve-out. Third
party lineage per the licence: Qwen3-8B (Apache-2.0), Stable Audio
tools (MIT) for the DiT, DAC (MIT) for the VAE.

Widely repeated claims that this is CC BY-SA 4.0 are **wrong** — that
comes from a Creative Commons *logo* in the GitHub README badge, which
links to the custom licence.
