# MiniMax-Music3 in DEMON

**Verdict: it works, in real time, on a single 5090, in eager bf16 —
without TensorRT.** The measured headroom is ~38x realtime. What is
integrated is the model's *renderer*, not the whole model, and that
distinction is the entire design. This document says why, what the
numbers are, and what is not done yet.

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
ticks — so the entire 8-second song is regenerated every ~0.21 s.

So: **run the AR stage once to fix a musical idea, then stream the
renderer over it forever.** The conditioning becomes a captured
artifact rather than a per-tick computation. This is also the only
audio-conditioning path this checkpoint actually supports — upstream
ships no converted audio encoder, and the community's measurements
found arbitrary-WAV continuation losing to a trivial baseline, while
"continue from your own generation" works.

## 3. Measured results

### Streaming (`scripts/minimax/minimax_stream_smoke.py`)

Real backend, real `StreamPipeline`, real weights, depth 4 / steps 8:

```
ticks              188  (92 fresh generations)
tick   median      104.4 ms   (p90 105.9)
render median      0.1 ms
peak / rms         0.6097 / 0.07551 (-22.4 dBFS)
full generation every ~0.21s of compute for 8.0s of audio
                                          = 38.3x realtime headroom
```

Spectrally the output matches the reference model, and is nothing like
noise:

| | centroid | flatness |
|---|---|---|
| reference (diffusers) | 4061 Hz | 0.305 |
| **DEMON streamed** | **4249 Hz** | **0.281** |
| white noise | 11989 Hz | 0.843 |

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
because it re-runs the 8.58B LM (measured **14.7 s** for an 8 s span,
23.7 GB peak). Duration is fixed per session at 8.011 s (689 latent
frames), the span the DiT was trained on.

**What survives:** everything DEMON steers solver-side lands in one
tick via the shared-curve override — denoise, seed, the source lock,
the feedback delay tap. Two knobs fall out of the architecture rather
than being invented for it:

- `minimax_cond_strength` interpolates the capture toward zeros, which
  is literally the model's own unconditional CFG branch, so 0.0 is a
  defined operating point rather than an extrapolation.
- Prompt blending slerps between two captures per frame.

## 5. TensorRT

The renderer is unusually well-shaped for it, and none of the SA3
plugin machinery is needed.

- **Export is already proven.** All three modules export via
  `torch.onnx.export(dynamo=True)` with dynamic batch *and* dynamic
  length.
- **Static shapes.** A session is one 689-frame window, so a single
  profile `{"hidden_states": (1,128,689), "timestep": (1,),
  "encoder_hidden_states": (1,689,2048)}` covers it. No bank system, no
  duration matrix.
- **Reuse the generic builder verbatim.**
  `acestep/engine/trt/sa3_build.py::_build_strongly_typed_engine`
  has zero model knowledge — it needs only that profile dict.
- **No plugins.** Self-attention only, no exotic ops, nothing like
  SA3's `samel::diff_attn_swa`.

**Precision — and this is a genuine correction to the SA3 lesson.**
Measured per Euler step at L=689:

| | ms/step | SNR vs fp32 | Pearson r |
|---|---|---|---|
| fp32 | 178.7 | — | — |
| bf16 | 70.4 | 29.3 dB | 0.999476 |
| **fp16** | **60.7** | **48.7 dB** | **0.999993** |

fp16 is both faster *and* more accurate than bf16 here, so the
fp16-mixed STRONGLY_TYPED recipe applies directly. **But the DAV
vocoder in fp16 produces all-NaN output** — the decoder must stay
fp32/bf16. Scope the "no fp16" rule to the decoder, not the DiT.

Expected gain is ~1.5-2x on the DiT, taking the tick from ~105 ms to
roughly 55-70 ms. Build it against a parity harness first, modelled on
`scripts/sa3/sa3_trt_dit_cond_parity.py`, with the same bar
(cos >= 0.9998/step on *real* conditioning — random-input cosine is a
known false positive).

The AR stage is not this toolchain's problem: it is a Qwen3 decode loop
and belongs to TensorRT-LLM. It is also dispatch-bound rather than
FLOP-bound (CUDA graphs bought 3.3-4.1x upstream), so graph capture is
the lever there, not quantization. Note a measured negative result:
torchao fp8 weight-only was **2.1x slower** on Windows.

## 6. Not done

- **The autoregressive capture stage** (`acestep/engine/minimax_ar.py`)
  is in progress. Until it lands, sessions stream from a saved capture
  (`DEMON_MINIMAX_CAPTURE`) or unconditioned. Free-text `set_prompt`
  needs it.
- **No TensorRT engine built** — the plan above is validated as far as
  ONNX export, not compiled.
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

## 7. Running it

```bash
# Streaming proof from a saved capture
.venv/Scripts/python.exe scripts/minimax/minimax_stream_smoke.py \
    --capture <models>/minimax/fixtures/minimax_music3_8s_seed7.safetensors \
    --seconds 24 --steps 8 --depth 4 --out out/minimax_stream.wav

# Prove a knob steers the stream live
... --sweep minimax_denoise --sweep-from 0.15 --sweep-to 0.95

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

## 8. Licence

**MiniMax-Music3 Community License** — custom, not OSI. Commercial use
is permitted with two conditions: display "MiniMax-Music3" prominently
in the product UI, and obtain written authorisation if aggregate yearly
revenue exceeds **US$20M**. There is no territorial carve-out. Third
party lineage per the licence: Qwen3-8B (Apache-2.0), Stable Audio
tools (MIT) for the DiT, DAC (MIT) for the VAE.

Widely repeated claims that this is CC BY-SA 4.0 are **wrong** — that
comes from a Creative Commons *logo* in the GitHub README badge, which
links to the custom licence.
