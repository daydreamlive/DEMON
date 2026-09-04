# MiniMax-Music3 in DEMON

**Verdict: MiniMax-Music3 is an autoregressive model that already
streams natively, and DEMON drives its own loop, as one CUDA graph per
frame over a static KV cache. On a 5090 the combined pipeline runs
above realtime: the 8.58B language model at 1.54x in-session, the
renderer at 7.8x, steady state ~1.3x at the default hop, first audio
~6 s after create. Live steering works, at a knob-to-ear of seconds
rather than the ~60-230 ms the diffusion families reach.**

This document replaces an earlier one that reported 9.5-16.7x realtime
and a 20-70x deficit against the other families. Both were wrong, in
opposite directions, and for the same reason: they measured the
*renderer* and never the *model*. A second version then measured the
model through a plain torch loop and reported 0.54x, called it "flat in
context length", and attributed a co-residency penalty to cache
eviction; §3 corrects all three. §5 is the accounting.

> **Scope.** This family is a backend-generality demonstration. Stable
> Audio 3 is the model that matters. No distillation and no quantization
> is proposed. The one speed change made (§3, the AR loop as a CUDA
> graph) is what the model's own serving stack does and is what it takes
> for the family to keep up with a playhead at all.

---

## 1. What MiniMax-Music3 actually is

Three stages, ~11.8B parameters, and the first stage is the whole story:

| Stage | Params | Role |
|---|---|---|
| **Global LM** (Qwen3-derived) | **8.58B** | **autoregressive, emits one acoustic frame at 25 Hz over a KV cache until an end-of-audio token** |
| RVQ depth decoder | 646M | the 7 residual codebooks within each frame |
| Flow-matching DiT | 2.43B | renders a continuous 128-ch latent at 86.133 Hz |
| DAV decoder | 54M | 512x upsample to 44.1 kHz stereo, deterministic |

**The DiT has no cross-attention and no text input.** Its only
conditioning is `encoder_hidden_states [B, T, 2048]`, which a 25M
condition encoder produces from the LM's fused per-frame hidden states
(8 x 4096 = 32768 per frame). There is no path from a prompt to a
denoise step that does not traverse the LM.

### Two frame rates, and they are not the same number

This is the error that produced every wrong figure in the previous
version of this document, so it gets its own heading:

| | rate | one frame is | what counts it |
|---|---|---|---|
| **AR acoustic frame** | **25 Hz** | 40 ms | the LM's emission; `MAX_AUDIO_FRAMES = 9000` = **360 s** |
| **DiT latent frame** | **86.133 Hz** | 11.6 ms | 44100/512; the renderer's sequence length |

They differ by exactly 441/128. A 200-AR-frame conditioning window is
689 latent frames. **Neither is comparable to another model family's
frame rate as a throughput figure**, and the earlier document's
`MINIMAX_LATENT_RATE_HZ` (86.13 Hz) was put next to ACE-Step's 25 Hz in
a table as though it were.

Both constants now live in one place
(`acestep/engine/minimax_render.py`) with the ratio stated as an exact
integer, and `tests/unit/test_minimax_backend.py` pins the geometry the
backend declares to the AR rate specifically.

## 2. The architecture, and why it changed

### What was there

The previous integration ran the AR stage **once** at session create,
froze its output into a static conditioning tensor, and then streamed
that tensor through DEMON's ring buffer: every tick submitted a
partial-denoise "cover" of the same frozen composition, on the batch-axis
staircase, at pipeline depth 4.

The ring buffer and the staircase exist to make a **one-shot, whole-song
diffusion model** behave like a stream. Applying them here converted a
streaming model into a one-shot model in order to have something for the
streaming machinery to do. It also made `set_prompt` cost a full 8.58B
regeneration (tens of seconds), fixed the song length at session create,
and made the only live controls re-renders of a composition that could
never change.

### What is there now

The backend drives the model's own loop.

```
MiniMaxARStream          25 Hz frames, one at a time, over a live KV cache
        |                (steerable per frame; re-promptable without losing the music)
        v
MiniMaxLatentStream      200-frame conditioning window -> 689 latent frames,
        |                172 frames of committed carry locked at every sampler
        |                step, 344 frames committed, 173 discarded as lookahead
        v
MiniMaxLatentDecoder     guarded windowed decode, 12 latent frames each side
        v
_DeliveryResampler       44.1 -> 48 kHz, phase-exact across block boundaries
        v
MiniMaxBackend           append-only frontier behind GeneratorBackend
```

`MiniMaxBackend` is the **second append-only family** behind the Tier-1
seam, after MRT2, and it takes MRT2's shape: `render_window` ignores the
runner's position hint and returns the next frontier chunk, the song is
a rolling window the frontier overwrites and the player loops, and each
emission re-issues the previous one's last 1200 samples so the runner's
leading-edge crossfade blends identical audio.

`Capabilities` is all-False, and honestly so: an autoregressive stage
cannot revise a frame it has emitted, so `refines_audio` is not a
missing feature.

### Why the ring is gone rather than optional

Chunk *k*'s left context is chunk *k-1*'s committed output. Consecutive
renders are strictly dependent, so there is nothing to put on a batch
axis and nothing to pipeline. A staircase over this model would be a
staircase of one.

### The chunk geometry

Upstream's inference constants, read as a streaming loop:

```
|<-- carry 172 -->|<---- commit 344 ---->|<-- lookahead 173 -->|
|<--------------------- chunk 689 ----------------------------->|
   already committed    the only part      rendered and thrown
   audio, locked at     that is kept       away
   every sampler step
```

* **Carry.** At every sampler step the first 172 latent frames are
  overwritten with the noised committed latent at that step's `t`
  (MiniMax's own forward interpolant, `x_t = (1-t)*noise + t*data`), and
  restored exactly at `t = 1`. This is what makes the stream continuous
  rather than a sequence of independent 8-second renders. Measured at a
  chunk seam: sample-to-sample delta 0.006 against a p99.9 of 0.18 for
  the signal as a whole — the seam is quieter than the music.
* **Commit.** Exactly the region the next chunk's carry will need, so
  consecutive commits abut and nothing is written twice.
* **Lookahead.** Discarded so the committed region is never generated at
  the edge of the model's window. It is not waste, it is **latency** —
  see §4.
* **No drift.** Each chunk's latent origin is derived from its absolute
  AR index (`ar_index * 441 // 128`), not from a constant hop. A
  constant 344-frame hop would slip half a latent frame per chunk, which
  is 0.6 s of conditioning-to-latent skew over a six-minute piece. The
  hop alternates 344/345 instead. `tests/unit/test_minimax_render.py`
  pins this over 400 chunks.

## 3. What it costs

Measured on an RTX 5090, TensorRT fp16 renderer, bf16 AR stage.

### Each stage alone

| stage | rate | realtime | script |
|---|---|---|---|
| **AR emission, graphed session** (the backend's path) | **25.7 ms/frame** at the full 9600-slot cache, 23.2 ms at 820 | **1.56x** (1.72x) | `minimax_ar_bench.py --graph` |
| AR emission, plain loop | 52.1 ms/frame at the start of a piece | 0.77x, falling with length | `minimax_ar_bench.py` |
| chunk render, TRT fp16 | 513 ms / 4.0 s commit | 7.8x | `minimax_stream_bench.py` |
| chunk render, eager bf16 | 841 ms / 4.0 s commit | 4.8x | `minimax_stream_bench.py` |
| guarded decode | ~6 ms/window | — | `minimax_decode_profile.py` |

The plain loop is **not flat in context length**, which an earlier
version of this document claimed from a 300-frame run. Its single-token
LM forward alone measures **36.7 / 40.0 / 58.6 / 77.7 ms** at 0 / 3000 /
6000 / 9000 frames of history (`minimax_graph_spike.py --long-context`),
so a piece that starts at 0.77x is near 0.43x by its sixth minute. The
graphed session attends its whole preallocated cache on every frame, so
its cost is the same at frame 1 and frame 9000 by construction: 25.7 ms
with 9600 slots, 22.7 ms with 1024.

### Both together, which is not what the parts predict

Session means, TensorRT, graphed AR, eager DiT parked, `--lead 1000` so
pacing never throttles the worker:

| | hop=100 (default) | hop=50 |
|---|---|---|
| **end-to-end, steady state** (from the means below) | **1.29x realtime** | **1.10x realtime** |
| end-to-end over the run, first-window fill included | 1.23x (41.8 s in 34.0 s; 73.8 s in 60.0 s) | 1.03x (41.8 s in 40.5 s) |
| AR, co-resident | 25.9-26.0 ms/frame (**1.54x**) | 26.0 ms/frame (1.54x) |
| chunk render, co-resident | 510-515 ms per 4.0 s | 514 ms per 2.0 s |
| first audio | 5.9-6.0 s | 6.0 s |
| VRAM in use, whole session | 28.7 GB of 32.6 | 28.4 GB |

The AR stage runs at its isolated speed and so does the render: with the
memory headroom restored there is no co-residency penalty left to
report. The run-level figures are below steady state because a piece
starts with a 200-frame fill (~5 s) before anything renders, and the
bench's default prompt and seed end their piece at frame 1174, 47 s of
audio, so the fill is a large share of a short run. Under the old
default `--lead 8.0` the worker idles once the frontier is 8 s ahead of
the playhead, which it now reaches; the pacing, not the model, then
sets the rate.

Earlier versions of this table (0.54x at hop 100, 0.48x at hop 25) were
the plain loop with the eager DiT resident: paging, described below.

**The co-residency penalty was VRAM paging, and an earlier version of
this document misattributed it.** It said the DiT's weights were "gone
from cache" between renders; a 5090's L2 is 96 MB and the DiT is 4.9 GB,
so that was never a mechanism. What was happening: the context loaded
the eager bf16 DiT onto the card and *kept it there* while the TensorRT
engine did the rendering, so the LM (17.2 GB), the KV cache, both copies
of the DiT (4.9 GB each), the codec and the desktop sat at **32.1 GB of
32.6** for the whole session (`nvidia-smi`, polled every 4 s), and WDDM
paged weights across PCIe under both stages with nothing in the logs to
say so. With the graphed AR's 2.8 GB static cache on top it got worse,
not better: 69 ms per AR frame in-session against 23 ms alone, 1106 ms
per chunk against 513, first audio at 27.9 s. Shrinking the cache to
2000 frames alone brought first audio to 5.5 s. The eager DiT is now
parked on the host whenever the engine serves.

So benchmarking either stage alone overstates the pipeline, and a
co-residency number is meaningless without the card's memory headroom
next to it. That is the same trap the previous document fell into, at a
larger scale.

These are session means. The per-sample values wander by ~15% run to
run, which is enough to move an end-to-end figure — the backend's params
echo reports means for that reason.

**Below 1.0 means the frontier cannot keep ahead of a playhead.** The
rolling window laps and the listener hears earlier material repeat; a
60 s window is ~90% freshly written after 100 s. This is reported rather
than hidden: `frontier_lead_s`, `ar_realtime`, `chunk_render_ms` and
`ar_finished` ride the params echo on every generation.

### Whose limitation this is, measured

`minimax_ar_bench.py --profile` splits the frame. On the 5090, stable
across three runs:

| | |
|---|---|
| wall clock | 52.6-57.0 ms/frame |
| GPU kernel time | **22.2-22.4 ms/frame** (3895 kernels) |
| **GPU busy fraction** | **39-42%** |
| of which GEMM | 17.0 ms (76% of GPU time) |
| dispatch gap (GPU idle) | 30-35 ms/frame |
| **bandwidth achieved during GEMM** | **1.55 TB/s of 1.79 peak — 86%** |

So the stage is **dispatch-bound, not bandwidth-bound**, and the two
halves of that say different things:

* The kernels are already at **86% of the card's memory roof.** There is
  nothing left to win inside them on this hardware.
* The GPU nevertheless **idles ~60% of every frame**, waiting on Python
  to launch the next of ~3900 kernels — one LM forward at batch 2 with a
  sequence length of one, ~460 GEMMs, plus seven depth-decoder forwards
  and eight sampled codes, all per 40 ms of audio.

**0.77x is a property of the plain torch loop, not of the model.**
Upstream serves this checkpoint through SGLang-Omni, which replays the
decode step as a CUDA graph; the checkpoint ships only an HTTP client
for it.

### Would a bigger card have fixed the plain loop? No.

A faster GPU scales only the busy 22 ms; the dispatch gap is CPU-side
and does not move. Scaling by memory bandwidth alone: H100 PCIe
(2.0 TB/s) 0.73-0.79x, H100 SXM (3.35 TB/s) 0.86-0.95x for the AR stage
alone, before the renderer's share and before co-residency, so still
short end to end — and generous, since the projection assumes 86% of
peak holds on HBM3 and ignores that H100 SXM clocks lower than a 5090.
Removing the gap instead was worth 1.79x on the card in hand. So that is
what was done.

### The lever, taken: one CUDA graph per frame

`acestep/engine/minimax_ar_graph.py` captures the whole frame — feedback
embedding, LM step, head, guided sampling, seven depth-decoder forwards,
seven more samples, next feedback — as one graph and replays it. The
frame's ~3900 launches become one. Two things had to be true for that to
pay, and both were measured rather than assumed
(`scripts/minimax/minimax_graph_spike.py`):

**The static cache must not cost what the graph saves.** A graph needs
fixed addresses, so the KV cache is preallocated to the session's length
(`StaticCache`, 9600 slots for the 9000-frame ceiling). Through HF's
own attention that cache is ruinous:

| static cache slots | HF sdpa, graphed | grouped-query `bmm`, graphed |
|---|---|---|
| 512 | 24.0 ms | — |
| 1024 | 26.0 ms | **22.7 ms** |
| 2048 | 29.9 ms | — |
| 4096 | 36.8 ms | 32.4 ms (efficient kernel) |
| 9600 | 55.6 ms | **25.7 ms** |

HF's sdpa path spends ~3.7 ms per 1024 padded slots per frame: with a
mask present it cannot use SDPA's native GQA, so `repeat_kv` materialises
the padded K/V four times over in every layer, then a mask-bearing kernel
reads all of it. SDPA's memory-efficient kernel on the un-expanded K/V is
no better at a query length of one (0.64 ms per layer at 9600 slots
against a 0.044 ms bandwidth floor): one CTA per (batch, head) walks
every slot. The decode attention that ships folds the four query heads
sharing a KV head into the query's row axis and does the two matmuls as
`bmm` with an fp32 result — 0.072 ms per layer, so the whole 9600-slot
cache costs ~3 ms per frame rather than ~23.

**The result must match the plain path.** Teacher-forcing the graphed
session's codes back through the dynamic cache and comparing LM hidden
states frame by frame: worst cos **0.99983** over 48 frames. HF's own
eager attention against its sdpa, on the same dynamic path: 0.99967. The
graphed path is inside the noise of a kernel swap.

What changes for an operator: nothing in the interface. Sampling
controls become device scalars, so a knob move is a buffer write and
never a recapture; `top_k` derives its threshold from a sort for the
same reason. What is not preserved is the exact random draws — a captured
`multinomial` runs on graph-owned philox offsets, so a seed reproduces a
graphed session against another graphed session, not against the plain
loop. Captures and parity fixtures come from `generate_frame_hiddens`,
which does not use the graph, so they stay valid. The plain loop remains
reachable (`minimax_ar_graph: false`, `--no-ar-graph`) for parity work.

Two measurement traps, handled in the scripts because both give
confident wrong answers: `key_averages()` reports device time on both an
`aten::` op and the kernel it launched, so summing everything double-
counts and turns a 42%-busy frame into 84%; and the profiler costs more
than the work here (a 53 ms frame became 265 ms under it), so wall clock
is taken with profiling off. Also: a graph replay past the end of a
static cache is an out-of-bounds `index_copy_` that surfaces as an
access violation, not an exception.

### VRAM

| | |
|---|---|
| AR stage, resident | 17.4 GB |
| static KV cache, batch 2, 9520 slots | 2.8 GB |
| TRT fp16 engine | 4.88 GB |
| eager DiT, bf16 | 4.9 GB — **parked on the host while the engine serves** |
| DAV + condition encoder | ~0.2 GB |
| **whole streaming session, `nvidia-smi`** | **28.7 GB** (32.1 with the eager DiT resident, which pages) |

Streaming requires the AR stage **resident**, not paged: it runs
continuously rather than once per composition, and moving 18 GB across
PCIe between chunks would cost more than the chunks do. That is the real
deployment cost this architecture adds — the renderer alone fits
comfortably on a 24 GB card and the pair does not. On a 32 GB card the
pair fits with ~3 GB to spare *only* because the eager DiT leaves;
anything else on the card (a desktop was 1.7-4.6 GB during these runs)
eats into that, and once the card is full nothing reports it — the
frame times simply triple.

## 4. Knob-to-ear, measured

Measured knob-to-**frontier** (the wall time until audio produced under
the new setting reaches the delivery frontier). Add the playback lead
for knob-to-ear.

Graphed AR, eager DiT parked, probe at 12 s into the piece:

| knob | stage | hop=100 | hop=50 | plain loop, hop=100 (old) |
|---|---|---|---|---|
| `minimax_guidance` | renderer | **3.30 s** | **1.21 s** | 6.7-7.1 s |
| `minimax_temperature` | AR | **3.33 s** | — | 8.6-9.8 s |
| `set_prompt` (re-prefill, 0.28 s over 400 frames) | AR | **3.57 s** | — | 7.9 s |
| end-to-end throughput, steady state | | 1.29x | 1.10x | 0.54x |

Two different floors, for two different reasons, and both halved with
the AR stage because both are paid in AR frames.

**A renderer knob waits for the next chunk render to begin**, which
means waiting for the AR stage to fill the next hop. `minimax_hop` is
the lever and it trades against throughput: a smaller hop commits less
audio per render, so more of the same audio is re-rendered. 3.3 s down
to 1.2 s costs 1.29x down to 1.10x, both still above realtime.

**An AR knob does not benefit from the hop**, and the reason is
geometric rather than budgetary. A frame written now sits inside a
200-frame conditioning window whose commit region ends 150 frames before
the window does, so it cannot be committed until the LM has written up
to 150 more frames. That is 6 s of audio whatever the hop is; in wall
clock it is 150 frame times, ~4 s at 26 ms and ~8 s at the plain loop's
53. Shrinking it further means shrinking the chunk or the lookahead,
which is a quality question — the committed region would then sit at
the edge of the model's window — and it is unmeasured.

### What live steering actually buys here

Weaker than the diffusion families on latency, and stronger on kind:

* **AR sampling steers the composition as it is being written.**
  `minimax_temperature`, `minimax_top_k` and `minimax_ar_guidance` are
  read fresh at every 40 ms frame the LM emits. The cover architecture
  could only re-render a composition that was already fixed.
* **`set_prompt` is live.** `MiniMaxARStream.reprompt` swaps the text
  prefix and rebuilds the KV cache against the audio history rather than
  regenerating it: the frames are already decided, so they replay as one
  batched prefill. **Measured 127 ms over 150 frames (6 s) of history,
  and 270 ms over 600 frames (24 s) mid-stream**, against tens of
  seconds to regenerate the same span. The piece keeps its history and
  its phase, and the new caption steers what comes next; the swap itself
  is far below the ~8 s it then takes to reach the frontier, so the AR
  geometry is the cost, not the re-prefill.
  How hard the swap steers is a balance-of-context question: a minute of
  the model's own frames outweighs a 30-token caption, so with the full
  history kept a new caption lands as a gradual morph (~20 s to fully
  arrive, measured over a lo-fi -> drum-and-bass swap).
  `minimax_reprompt_history_s` crops the replayed history at the swap to
  the last N seconds, inverting the balance for a fast pivot while the
  kept frames carry tempo and phase across it. 0 keeps everything.
  Measured on the same swap pair: 10 s pivots up in energy in ~1-6 s but
  a pivot *down* never lands (10 s of aggressive material outweighs a
  mellow caption for at least 38 s); 2.5 s pivots both directions in
  ~1-4 s.
* **`set_prompt_blend` is refused, loudly.** The other families
  interpolate two conditioning tensors. MiniMax's conditioning is a KV
  prefix inside an 8.58B LM; there is no second one to interpolate
  toward without running a second LM. The backend raises
  `UnsupportedOperation` rather than accepting a knob that does nothing.

## 5. What was published before, and what is true

Every row below was in `docs/MINIMAX.md` or `out/README.md` and is
wrong. Treat the git history of those files, and the current body of
draft PR #332, as superseded.

| claimed | actual |
|---|---|
| "9.5x realtime eager, 15.2x TRT" | Renderer only, with the AR stage excluded. End to end is **1.29x** steady state with the graphed AR stage (§3), and was 0.54x through the plain loop. |
| "16.7x realtime at a 14.4 s song" — the headline | Same omission, and the best case of it. |
| "roughly 20-70x slower than its siblings in normalized terms" | Also renderer-only, in a `60 s-gens/s` unit that has no meaning for a model with no fixed song length. The real gap is different in size and in kind: **the AR stage is under realtime and the renderer is not the problem**. |
| `gens/s` and `60 s-gens/s` tables | A "generation" was one cover of a frozen composition. There is no such object now: audio is committed once and never re-rendered. Both units are withdrawn. |
| `chunk_rate_hz` = 86.133 Hz beside ACE-Step's 25 Hz | Different quantities (§1). The backend now declares **25 Hz**, the AR acoustic rate, with the distinction pinned by a test. |
| "the AR stage's 0.54x realtime" (§6, in passing) | Directionally right, and never connected to the architecture. Measured cleanly through the plain loop it is **0.77x** at the start of a piece and falls with length; the earlier figure was taken under per-stage profiling. Below realtime either way, which is the fact the design should have turned on — and which the graphed session (**1.54x**) then removed. |
| "a 60 s composition costs ~111 s of one-time capture before any streaming starts" | True of the capture architecture, and no longer how the family works: the stream starts after 200 AR frames plus one render, **6 s**, and extends until the model ends the piece. |
| "Treating 200 as a model limit was an error" | Correct, and still correct — but the conclusion drawn from it (render the whole song in one pass) was the wrong one. 200/100/172 is upstream's *streaming* contract, and it is what the backend now implements. |
| "8.011 s" | 689 x 512 / 44100 = **7.99927 s**. Carried over; it was already corrected once. |
| "render cost is identical at 14.4 s and 8 s" | True of the windowed decode, and now moot: there is no whole-song render to compare against. |

The second version of this document, which introduced the native
streaming architecture, carried three claims of its own that §3 has
since corrected:

| claimed (second version) | what is true |
|---|---|
| "0.54x realtime end to end; it does not keep up" | The plain torch loop, measured with the eager DiT resident beside the TensorRT engine, i.e. under VRAM paging. Graphed and with the DiT parked: **1.29x** steady state at the default hop. |
| "AR cost is flat in context length" | Measured over 300 frames. The plain loop's LM forward is 36.7 ms at frame 0 and **77.7 ms at frame 9000**. The graphed session is flat, by construction. |
| "the chunk render doubles because the DiT's weights are gone from cache" | A 96 MB L2 cannot hold a 4.9 GB model; nothing was evicted. The card was at 32.1 of 32.6 GB and WDDM was paging. |
| "an H100 would not fix it; CUDA graphs would reach 1.79x" | Half right: the projection for a bigger card stands, and the graph reached **1.56-1.72x** for the AR stage once the static cache's attention was made to cost what it should. |

Two earlier claims **survive** and are unchanged:

* **The sampler operating point** (§6 below): 16 steps / shift 2.0 /
  guidance 1.7, at latent cosine 0.9993 and log-mel 0.032 against the
  reference. Re-derived nothing; the measurement was sound and the
  shipping sampler uses it.
* **The TensorRT recipe** (§7): fp16 trunk with fp32 islands,
  STRONGLY_TYPED, 15.7 ms/forward at L=689 against eager bf16's 35.3.
  The engine is built at exactly the chunk shape the streaming renderer
  uses, so it transferred without a rebuild.

## 6. Sampler settings

Unchanged from the previous measurement, and reproduced here because it
is the one part of the earlier work that the audit confirmed rather than
contradicted. `scripts/minimax/minimax_quality_ablation.py` walks one
variable at a time from ground truth outward and scores each rung in
both the latent and audio domains.

**The A/B, at matched conditioning and matched noise:**

| | latent cos | log-mel | L/R corr | RMS | >8 kHz energy |
|---|---|---|---|---|---|
| reference (diffusers) | 1.0 | 0 | +0.096 | -21.2 dB | 0.0041 |
| 8 steps, no guidance | 0.744 | 0.244 | **-0.128** | **-30.7 dB** | **0.0193** |
| **16 steps, shift 2.0, guidance 1.7** | **0.9993** | **0.032** | +0.101 | -21.1 dB | 0.0040 |

4.7x the reference's above-8 kHz energy is undenoised residual still
sitting on the latent when the schedule runs out of steps; being
uncorrelated between channels it also inverts left/right correlation,
which is why it read as phasey rather than merely quiet.

1. **Guidance is not optional and is worth more than steps.** Unguided
   sampling plateaus at ~0.11 log-mel and stays there — 40 unguided
   steps score worse than 8 guided ones.
2. **Step count trades against schedule warp nearly one for one.**
   Measured pairing: 30/1.0, 20/1.5, 16/2.0, 12/3.0.
3. **RCFG is unusable here.** `initialize` scores 0.45-0.70 log-mel and
   `self` 0.52-0.92, against 0.03-0.12 for a real uncond pass.
4. **Stock APG is the wrong combine operator** — ~4x worse than textbook
   CFG (0.125 vs 0.032 at 16/2.0); its norm cap is calibrated for ACE's
   latent scale. The streaming renderer therefore computes CFG directly
   (`v_neg + (v_pos - v_neg) * w`) rather than routing through the
   shared solver's guidance path.

16/2.0/1.7 is statistically indistinguishable from the reference's own
30/1.0/1.7: over 8 independent noise draws, log-mel distance to the
reference is 0.1631 for both (sd 0.005), as are RMS, HF ratio and stereo
width. It costs 32 forwards instead of 60.

The ablation now runs this rung on **the code a session executes**
(`MiniMaxChunkRenderer.render_cond`), not on the harness's own Euler
loop: `L4_shipping_sampler` scores cos **0.99931** / log-mel **0.0322**
on the fixture, reproducing the row above, and agrees with the reference
loop to cos 0.999878 (rel RMS 1.6e-2, bf16 accumulation). Before the
rewrite that rung measured a sampler nobody shipped.

### The step count needed its own knob

`SessionConfig.steps` defaults to ACE's 8, so the create path takes the
family floor of 16. That was not enough on its own: the shared
`steps_override` knob **also** defaults to 8 and caps at 16, so the
first tick read it back and reset every session to the broken setting —
found by instrumenting a real session create, which rendered 16 forwards
where it should have run 32. The family now declares `minimax_steps`
(default 16, range 8-40) instead, and the create path publishes the
resolved value into the bank. `seed` is still taken from the shared
registry, because it means exactly the same thing everywhere.

**Parity** (fp32, against the diffusers reference on real inputs):
DiT B=1 bit-identical; DiT B=4 with per-row `t` cos 0.999999999998; DAV
1.000000000000; condition encoder 1.000000000000. Chain level
(`minimax_chain_parity.py`) cos 0.999868, rel RMS 1.7e-2 (bf16).

## 7. TensorRT

Built, gated, and streaming. **2.25x over eager bf16 at B=1.**

Median ms/forward at L=689, RTX 5090:

| | B=1 | B=4 |
|---|---|---|
| eager fp32 (TF32 off) | 89.3 | 314.7 |
| eager bf16 | 35.6 | 103.0 |
| eager fp16 | 30.4 | 90.5 |
| TRT fp32 | 45.1 | 182.1 (looped) |
| **TRT fp16** | **15.7** | 54.4 batched / 63.3 looped |

Per-step parity vs eager fp32, bar cos >= 0.9998: 0.999981-0.999988
across t, rel RMS 4.8-6.2e-3, flat in t (trunk quantization, not an
angle problem). Full 30-step CFG trajectory vs the fixture's
`final_latent`: TRT fp16 0.999677.

> Eager **bf16 is not a usable parity reference** on this model — it
> only reaches 0.998-0.9997/step against fp32. The fixture is itself a
> bf16 run, so bf16 scoring highest is agreement with its own
> quantization.

The canonical engine is `l2_689_1400` (ranged, tuned at 689). The
streaming renderer always asks for exactly 689, so the pinned engine is
selected when both are built; the range exists for the parity harness.
fp16 is 4.88 GB / 33-44 s build / 2.0 s load.

Notes that each cost a build attempt: batch cannot be dynamic from 1
(`torch.export` 0/1-specializes; production engines are batch-1 and the
renderer issues cond and uncond as two forwards); same trap on length,
profile min 2; the "fp32" engine is really TF32. **Build on an idle
GPU** — engines built under contention succeed and then segfault on
load. **Do not try fp16 on the DAV vocoder**: measured all-NaN.

fp16 is both faster and more accurate than bf16 on this DiT in eager
(60.7 ms/step at 48.7 dB SNR vs 70.4 at 29.3). The SA3 RoPE failure mode
does not occur: `_rope_tables` pins `arange`/`inv_freq` to fp32 and the
TRT path keeps the rotation itself in fp32; the builder asserts it, so
an edit that drops the `.float()` fails the build rather than shipping
half-radian rotations.

```bash
.venv/Scripts/python.exe -m acestep.engine.trt.minimax_build --precision fp16
.venv/Scripts/python.exe scripts/minimax/minimax_trt_parity.py
```

## 8. The capture stage, now a development path

`scripts/minimax/minimax_capture.py` still runs the AR stage to
completion and saves the fused per-frame hidden states. It is no longer
how a session is conditioned — a session opens a live `MiniMaxARStream`
— but the artifact is more useful than before:
`ReplayARStream` serves a saved capture through the same interface, so
the renderer, the chunk geometry and the whole frontier path can be
exercised **without 21 GB of language model resident**. That is how the
streaming gates run on a machine that cannot hold the LM, and it is what
`DEMON_MINIMAX_CAPTURE` selects.

A capture must carry `frame_hiddens` (the raw 25 Hz output). One holding
only `encoder_hidden_states` has already been projected to the latent
rate at one fixed window length and cannot be re-chunked; `load_replay_stream`
refuses it rather than mis-windowing it silently.

`MiniMaxAR.generate_frame_hiddens` is now a thin wrapper over
`MiniMaxARStream` and is **bit-identical** to the version that predated
it (verified against the pre-refactor implementation on shared weights),
so existing captures and the parity fixtures remain valid.

### The transformers trap

`AutoConfig.from_pretrained` **succeeds** on this checkpoint under the
pinned 4.57.6, which is the problem. The v5-style `rope_parameters` dict
falls into `**kwargs`, is stashed as an inert attribute, and `rope_theta`
silently keeps the 4.x default of **10000 instead of the checkpoint's
1000000**. Every position encoding in a 36-layer model would be wrong,
with nothing raised. `load_qwen3_config` reads the JSON directly, remaps,
and then *asserts* the value took; each remap is gated on the installed
signature so the shim retires itself when the pin moves.
`tests/unit/test_minimax_ar_config.py` guards it, including a test that
demonstrates the naive loader getting it wrong.

Lyrics may not be empty; upstream's convention for "no singing" is the
`[instrumental]` tag, which is `SessionConfig.minimax_lyrics`'s default.
The reference normalizes both prompt and lyrics rather than inserting
them verbatim, and **silently drops any lyric sharing a line with a
leading `[tag]`**.

## 9. Not done

* **Not driven through `PipelineRunner` or the WS server.**
  `StreamingSession.create(backend="minimax")` is verified end to end —
  family dispatch, create path, backend assembly, geometry/capability/
  knob payloads, produce and render ticks, params echo, clean close —
  and `scripts/minimax/minimax_stream_bench.py` runs the real backend
  through the loop the runner runs, with the real crossfade. What has
  not been run is the runner and a browser session. **No web panel**, no
  boot preflight in `server.py`.
* **Headroom is thin on a 32 GB card.** 28.7 GB in use for a session,
  with the desktop's share on top. The next 1.1 GB is the LM head: it
  scores all 151,936 tokens every frame and the sampler admits 16,385
  of them, so a sliced head would save that memory and ~0.6 ms of
  bandwidth per frame. Not done; the plain loop shares the weight.
* **A piece ends when the model says so**, and the window then repeats.
  For one prompt: 47 s in the stream bench (seed 1528), 29 s through the
  plain loop and 115 s through the graphed one in `minimax_ar_bench.py`
  with the same seed (the two draw different random streams), 22 s at
  temperature 2.0. `minimax_endless` (default off) masks the end token
  out of the vocabulary so the stream runs to the frame cap instead;
  what the model composes past its trained horizon is a matter for the
  ear, which is why it is a knob and not the default.
* **The AR knob-to-ear floor (~8 s) is geometric and unexplored.** It
  comes from the 150-frame lookahead in a 200-frame conditioning window.
  Shrinking the window or the lookahead would shrink it, at an unknown
  quality cost — the committed region would sit nearer the edge of the
  model's context. Worth one ablation if this family ever matters.
* **`swap`, `write_audio`, `timbre`, `structure` and LoRA are gated
  off.** Most need an audio encoder this checkpoint does not ship
  converted. One exists unconverted inside `dav.pth` (186 `encoder.*`
  keys plus `mean_proj`/`logs_proj`), so audio-to-audio is a real
  follow-up rather than a dead end — but note the AR stage has no
  audio-prefix path either, so an encoder would condition the renderer
  only.
* **Long-horizon coherence is upstream's open bug**, not ours: coherent
  for ~15 s, then genre/timbre drift. The streaming architecture makes
  this *more* exposed than the old 8 s cover did, and it is unmeasured
  past ~60 s here.
* **No TensorRT decoder engine.** Guarded decode is ~6 ms against a
  ~880 ms chunk render — 0.7%, and not worth building.
* **The b2-4 TensorRT engine is built but never selected.** It would
  take cond and uncond in one forward instead of two; worth measuring
  now that the renderer issues exactly that pair, which it did not when
  the engine was excluded from discovery.
* **Stereo width is one observation short of a conclusion.** Our takes
  average L/R correlation 0.43 (sd 0.15 over 8 draws) against a single
  reference take at 0.096 — which we reproduce exactly on that take's
  own noise. Settling it needs several upstream renders.

## 10. Running it

```bash
# End-to-end stream with the live language model. Writes audio and a
# JSON report: AR rate, render rate, end-to-end realtime, and a
# measured knob-to-frontier latency.
.venv/Scripts/python.exe scripts/minimax/minimax_stream_bench.py \
    --seconds 100 --accel tensorrt --window 60 \
    --sweep minimax_temperature --sweep-at 30 \
    --out out/native_stream.wav --json out/stream.json

# The same, replaying a saved capture: exercises the renderer, the
# chunk geometry and the frontier without 21 GB of LM resident.
.venv/Scripts/python.exe scripts/minimax/minimax_stream_bench.py \
    --capture <models>/minimax/captures/darkwave_30s.safetensors \
    --seconds 14 --accel tensorrt

# The bottleneck on its own, including a live re-prompt measurement.
.venv/Scripts/python.exe scripts/minimax/minimax_ar_bench.py \
    --frames 300 --reprompt --json out/ar_bench.json

# Where the quality is and what it costs.
.venv/Scripts/python.exe scripts/minimax/minimax_quality_ablation.py \
    --fixture <models>/minimax/fixtures/minimax_music3_8s_seed7.safetensors \
    --out-dir out/ablation

# Parity gates.
.venv/Scripts/python.exe scripts/minimax/minimax_chain_parity.py \
    --fixture <models>/minimax/fixtures/minimax_music3_8s_seed7.safetensors
.venv/Scripts/python.exe scripts/minimax/minimax_trt_parity.py

# Save a capture (the development path; a session no longer needs one).
.venv/Scripts/python.exe scripts/minimax/minimax_capture.py \
    --prompt "driving instrumental darkwave, analog synth bass, 124 bpm" \
    --lyrics "[instrumental]" --seconds 30 --seed 7 \
    --out <models>/minimax/captures/darkwave_30s.safetensors
```

Weights: `MiniMaxAI/MiniMax-Music3`. The repo ships **two** complete
copies (57.35 GB); only the diffusers layout is needed (**28.52 GB**).
Skip `qwen_7B/`, `flowmatching_vae.pth`, `dav.pth` unless doing encoder
work. Set `HF_HUB_DISABLE_SYMLINKS=1` on Windows.

## 11. Licence

**MiniMax-Music3 Community License** — custom, not OSI. Commercial use
is permitted with two conditions: display "MiniMax-Music3" prominently
in the product UI, and obtain written authorisation if aggregate yearly
revenue exceeds **US$20M**. There is no territorial carve-out. Third
party lineage per the licence: Qwen3-8B (Apache-2.0), Stable Audio tools
(MIT) for the DiT, DAC (MIT) for the VAE.

Widely repeated claims that this is CC BY-SA 4.0 are **wrong** — that
comes from a Creative Commons *logo* in the GitHub README badge, which
links to the custom licence.
