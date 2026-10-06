# Adding a model family

A *family* is one generative model behind DEMON's streaming runner: ACE-Step,
Stable Audio 3, Magenta RealTime 2, MiniMax-Music3 and YuE2 today. This
document is the contract for adding one and the map of what still branches on
a family name in the core. Read it before `acestep/streaming/families.py`.

## The one thing you register

Every family is a single `FamilySpec` in `acestep/streaming/families.py`,
registered in `FAMILY_SPECS`. The spec is the source of truth; the dicts the
rest of the code imports (`FAMILIES`, `CHECKPOINT_ALIASES`, `WARMUP_POLICIES`,
`FAMILY_KNOB_UNIVERSES`, `SESSION_CREATORS`) are derived from it and checked
against it by `tests/unit/test_family_conformance.py`.

| Field | What it declares | Read by |
| --- | --- | --- |
| `name` | the `SessionConfig.backend` value; also the family half of a pod's pool identity | everything |
| `display_name` | human label | diagnostics |
| `make_backend(session)` | builds the family's `GeneratorBackend` from a constructed `StreamingSession` | `StreamingSession.__init__` |
| `knob_universe()` | every `KnobSpec` the family can ever expose, without a GPU | the homonym guard, the conformance test |
| `checkpoint_aliases` | `--checkpoint` alias → model id (unique across families) | `server.py` at CLI parse |
| `create_session` | the family's per-connect create path (`acestep/streaming/ace_session.py`, `sa3_session.py`); `StreamingSession.create` only dispatches to it | `StreamingSession.create` |
| `warmup_policy` | `"ace_trt"` or `"none"` | `server.py` boot |
| `preflight(request)` | the family's boot check, a pure verdict (`acestep/streaming/preflight.py`); the server prints and exits on failure | `server.py` boot |
| `prompt_policy` | which prompt-tooling policy `/api/enhance` infers (`"acestep"` or `"sa3"`; a policy name, not a family name) | `server.py` |
| `text_only` | a `TextOnlySpec` (default and max anchor length, and the config key that sets it) or `None` when the family cannot run without a source; the adapter advertises `supports_text_only` from it | `ws_adapter.py` handshake |
| `config_fields` | session-config keys the family adds to the handshake (`FamilyConfigField`: name, wire type, description); parsed into `config.family_config[name]` and projected flat into `/api/protocol` `config` and the generated TS/C++ types | `SessionConfig.from_dict`, `protocol.config_catalog` |
| `shutdown()` | releases process-wide state the family holds (a cached model, an installed extension) when the server exits | `server.py` shutdown |
| `accepts_checkpoint_dir` | whether `--sa3-base-checkpoint` may point the family at a non-catalog directory | `server.py` CLI |
| `supports_extensions` | whether `--model-extension` may target the family | `acestep.plugins.selection` |

Both callables do their own lazy imports, so registering a family adds no
model import to the registry. (The registry itself still imports the
engine logger, which pulls torch; that predates the spec.)

## The two seams a family implements

**Tier 1, `GeneratorBackend`** (`acestep/streaming/generator_backend.py`) is
what the runner drives every tick: capabilities, geometry, the knob manifest,
the LoRA facade, and the hot loop `sync_source → read_knobs → produce(mode) →
render_window`. Every family implements it. `DiffusionBackend`
(`acestep/streaming/diffusion_backend.py`) is a shared skeleton for diffusion
families.

**Tier 2, `ModelAdapter`** (`acestep/engine/model_adapter.py`) is optional. A
rectified-flow model plugs in here and inherits the ring buffer, slot
batching, shared curves and CFG from `StreamPipeline`. ACE and SA3 both use
it; a token or autoregressive model implements Tier 1 directly.

A family whose model cannot run in the DEMON process (a different framework
or torch pin) implements Tier 1 as a client of a sidecar process. The
credit-paced TCP frame protocol from the Magenta RT2 branch (#230) is the
template.

## Steps

1. Write the backend module (Tier 1, plus a Tier 2 adapter if the model is
   rectified-flow). Declare only the `Capabilities` bits you honour; the
   session turns every other command into a `command_failed`.
2. Write the create path. `acestep/streaming/ace_session.py` and
   `sa3_session.py` are the two references: load or reuse a process-cached
   context, encode the source, stash the construction payload on
   `backend_init` for `make_backend`, return `cls(...)`.
3. Write `knob_universe()`. Knob names shared with another family must have
   byte-identical semantics or be renamed with a family prefix
   (`tests/unit/test_knob_homonyms.py`).
4. Add the `FamilySpec` and register it in `FAMILY_SPECS`.
5. Run `tests/unit/test_family_conformance.py` and `test_knob_homonyms.py`.
6. Regenerate the wire types if you added a capability bit or a command
   (`AGENTS.md`, repo-wide rules).

## What still branches on a family name

The spec removes the registry's five hand-written dicts, every family
branch in `server.py` (preflight, warmup, enhancer policy, the base-checkpoint
flag) and the text-only path in `ws_adapter.py`. The prompt tooling under `demos/realtime_motion_graph_web/prompt_*.py`
branches on the *policy* name a spec selects, which is legitimate. These
places still branch on `"acestep"` / `"sa3"` and are the remaining work of
phase 1 of the platform plan; a third family today would have to edit each.

| Where | Branch | Planned home |
| --- | --- | --- |
| `acestep/lora_metadata.py`, `acestep/engine/lora.py` | weight-format sniff returns `"sa3"` / `"ace"` | `FamilySpec.lora_format` |

The `sa3_duration_s` handshake key is now declared by the SA3 spec and reaches
the wire contract from there; `SessionConfig` carries it in `family_config`.

`tests/unit/test_family_boundary.py` walks the AST of the frozen core files
(`pipeline_runner.py`, `session.py`, `generator_backend.py`,
`diffusion_backend.py`, `knobs.py`, `config.py`, `ws_adapter.py`, `server.py`,
`protocol.py`) and fails on a family name used as a literal, an identifier or
an import. The rows above are its allow-list; an entry the code no longer
needs fails the test too, so the list only shrinks.

## Runtime

One family per pod. The pod's engine family is `DEMON_MODEL`; its routing
identity is `RTMG_POOL_MODEL`, which defaults to the family and may carry a
variant (`sa3-controlnet`). Warmup and preflight are family policy, read from
the spec by the server at boot.

## Stable Audio 3 (`sa3`)

A refining diffusion family on the `ModelAdapter` seam: `SA3Adapter` plugs
the SA3 DiT into `StreamPipeline`'s ring buffer, and every emit is an
8-step pingpong audio-to-audio cover of the uploaded source.

- **Boot:** `--checkpoint sa3-small` (catalog id `small-music`) or
  `--checkpoint sa3-medium`. Weights are a manual
  `huggingface-cli download` (see `docs/INSTALL.md`); preflight fails the
  boot with the command when they are missing.
- **Engines:** `python -m acestep.engine.trt.sa3_build --model small-music
  --all` builds the small DiT engines (`sa3_sm_dit_l1_{324,646,1292}`, from
  upstream's `onnx/sa3-sm-music/dit_fp16.onnx`, fp16mixed
  STRONGLY_TYPED); `--all` without `--model` builds medium's DiT engines
  plus the SAME-L window decoder. Discovery (`acestep/engine/sa3_trt.py`
  `find_dit_engine`) picks the smallest engine covering the session's
  latent window; with none, the DiT runs eager and preflight logs the build
  command. Small decodes with SAME-S eager (full decode, ~60 ms per
  render tick at 54 s on a 5090, now its largest per-tick cost), medium with
  the SAME-L window engine (~10 ms).
- **Speed (RTX 5090, 54 s session = 614 latent frames, 8 steps,
  `scripts/sa3/sa3_throughput_probe.py`):** small TRT ~21 gens/s flat
  across depth 1-8 (tick p50 5.8 ms at depth 1, 23 ms at depth 4); small
  eager 2.9 gens/s at depth 1 rising to 19.5 at depth 8; medium TRT 11.7
  gens/s (fp8 engine) / 9.6 (fp16mixed). Real server, small TRT depth 4:
  session ready in 6.6 s, ~2.1 GB torch VRAM (medium 5.4 GB).

## Magenta RealTime 2 (`mrt2`)

The first token/autoregressive family and the first sidecar-hosted one, and
the reference for both. `acestep/streaming/mrt2/backend.py` is a Tier 1
`GeneratorBackend` implemented directly (no `ModelAdapter`); generation runs
in `scripts/mrt2_sidecar.py`, because JAX has no CUDA on native Windows.

- **Boot:** `--checkpoint mrt2-sidecar`. The alias only selects the family;
  the sidecar picks the model variant at its own launch.
- **Prerequisite:** a running sidecar in a Linux/WSL venv with `magenta_rt`,
  JAX (CUDA) and numpy (no torch):
  `python scripts/mrt2_sidecar.py --model mrt2_small`. It listens on
  `127.0.0.1:7531` once its JIT warmup (~30 s) is done; WSL2 forwards
  localhost to the Windows-side server. Override with
  `DEMON_MRT2_SIDECAR=host:port`. Preflight is a TCP connect and fails the
  boot with "MRT2 sidecar not running" when nothing listens.
- **Protocol:** `acestep/streaming/mrt2/protocol.py` (stdlib only, loaded by
  file path in the sidecar venv): `u32 len | u8 kind | payload`, JSON control
  (hello/meta, prompt, blend, knobs, credit, ping) and 48 kHz stereo f32 audio
  in 40 ms frames. The backend grants credit so the frontier stays `mrt2_lead`
  seconds ahead of the playhead; that lead is the knob-to-ear latency.
- **Shape:** append-only on a 60 s rolling window the player loops;
  `render_window` ignores the position hint and returns the next frontier
  chunk. Uploaded audio is ignored, so the family is text-only
  (`text_only` 60 s, no duration field). Capabilities all False; LoRA off.
  Because `refines_audio` is False, `PipelineRunner` writes each chunk
  verbatim: no edge crossfades and no wrap-spill re-render (both are for
  refining families).
- **Failure handling:** the client pings every 2 s from its own thread and
  declares the link lost after 8 s of silence; a lost link (sidecar died,
  deadline, send error) ends the session with a `pipeline_error`
  SessionError. The sidecar drops a backend that has sent nothing for 20 s,
  and the client shuts its socket down on close, so a stopped session frees
  the one-session sidecar at once. Credit the sidecar discards on a
  `generate` error is reported in its `err` and refunded. When the frontier
  falls behind the playhead it restarts one `mrt2_lead` ahead of it (a hard
  seam at that point).
- **Known gaps:** one session per sidecar; a second concurrent session hangs
  5 s and is told the sidecar is unreachable (no busy reply). No reconnect
  after a lost link. The playhead unwrap counts at most one lap between
  ticks, so after the runner's idle pause spans more than one lap the
  frontier can sit up to a lap "ahead" and generation waits for the playhead
  to catch up. A sidecar slower than real time (`mrt2_base`) re-anchors
  repeatedly: fresh audio arrives in bursts between stretches of the previous
  lap.
- **Knobs:** `mrt2_temperature`, `mrt2_top_k`, `mrt2_cfg_musiccoca`,
  `mrt2_cfg_notes`, `mrt2_cfg_drums` (forwarded to the sidecar),
  `mrt2_lead` (backend-local). `set_prompt` / `set_prompt_blend` go to the
  sidecar, which embeds tags with MusicCoCa and lerps A/B.
- **Speed (RTX 5090):** `mrt2_small` ~1.7x real time, `mrt2_base` ~0.93x
  (below real time; expect underruns).
- **Frontend:** `demos/mrt2/` (static plain-canvas page, route `/mrt2` on the
  backend port) and `demos/realtime_motion_graph_web/web/app/magenta` (route
  `/magenta`); both send `backend: "mrt2"`.

## MiniMax-Music3 (`minimax`)

An append-only autoregressive family: an 8.58B Qwen3 language model plus a
646M RVQ depth decoder write 25 Hz acoustic frames, a 2.43B flow-matching DiT
renders them in chunks, and a 54M DAV decoder turns latents into 44.1 kHz
stereo (delivered at 48 kHz). Integration report: `docs/MINIMAX.md`.

**Boot.** `--checkpoint minimax-music3` (model id
`MiniMaxAI/MiniMax-Music3`). Warmup policy `"none"`: the first session pays
the model load, later sessions reuse the process-cached context.

**Prerequisites.**
- Weights: the diffusers layout of the Hugging Face repo
  `MiniMaxAI/MiniMax-Music3` (`transformer`, `vocoder`, `condition_encoder`,
  `scheduler`, plus `language_model` and `tokenizer` for the AR stage):
  ~18 GB bf16 LM + 2.4B DiT + DAV decoder. Found under `DEMON_MINIMAX_DIR`,
  `<models>/minimax/checkpoints/MiniMax-Music3` (`ACESTEP_MODELS_DIR`
  overrides the models root) or the local HF cache. bf16/fp32 only.
- VRAM: the AR stage stays resident (~21 GB with its KV cache) on top of the
  renderer; measured on a 32 GB 5090.
- Optional: a fp16 TensorRT engine for the DiT renderer, built by
  `acestep/engine/trt/minimax_build.py` against tensorrt 10.16 (main's pin),
  in `DEMON_MINIMAX_TRT_DIR` or `<models>/minimax/trt_engines`. Without it
  the renderer runs eager.
- Preflight (`minimax_preflight`) checks the renderer components offline and
  fails if `DEMON_MINIMAX_TRT_DIR` is set to a missing directory. An absent
  AR stage fails it too, unless `DEMON_MINIMAX_CAPTURE` names a saved capture
  to stream (logged at warning level: every user gets that one composition).

**Config fields.** `minimax_duration_s` (rolling-window length, default 60 s,
capped at the AR ceiling of 360 s), `minimax_lyrics` (default
`"[instrumental]"`), `minimax_ar_graph` (default true: one CUDA graph per AR
frame over a static KV cache; false = plain loop for parity work),
`minimax_seed` (the AR seed, i.e. the composition; absent = a random seed per
session, echoed as `minimax_ar_seed` in params; give one to replay a piece).
The shared `seed` knob seeds only the renderer's noise; zero is a valid value
for it and for `minimax_cond_strength`.

**Behaviour.** Append-only: `refines_audio=False` and every other capability
off (no swap, write_audio, timbre, structure, stems, depth, LoRA). Uploaded
audio is ignored (the checkpoint ships no audio encoder); every session
starts from a silent window, and `text_only` advertises the same window.
`set_prompt` re-prefills the LM against the audio already written;
`prompt_b` / prompt blend are refused. The piece ends when the LM emits
end-of-audio. Shutdown evicts the cached contexts; each backend frees its
CUDA graphs and static cache on close, unless its worker is still running
after the 5 s join (then the state is left to the worker and logged). The
runner writes chunks verbatim (no edge crossfades or wrap-spill re-render
for `refines_audio=False`). If the generation worker dies (for example an
OOM), the session ends with a `pipeline_error` SessionError.

**Known gaps.** `set_prompt` after the piece has ended (or on a replayed
capture) is logged and dropped with no client-visible error. Caption plus
lyrics share a 512-token prompt budget; an over-long prompt fails session
create, and on a live reprompt it is dropped. A backward seek is counted as a
window lap, which breaks pacing until the session ends (no client seeks a
MiniMax session today). The generation worker starts inside the backend
constructor. The ws_adapter shedding bypass is gated on `refines_audio`, so
it applies to MRT2 as well.

**Measured (RTX 5090).** Steady ~1.3x realtime (AR stage 1.54x in session,
renderer 7.8x). With the model already loaded, first audio comes ~6 s after
generation starts (200 AR frames plus one render). The first session in a
freshly started server also loads the language model and captures its CUDA
graphs: session create takes ~27-28 s on a cold pod and first audio arrives
21-34 s after Start, so expect roughly half a minute. Later sessions in the
same server reuse the process-cached context and skip the load.

**Hot loop.** The family leaves `acestep/engine/stream.py` and
`acestep/engine/ode_steps.py` as they are on main: `MiniMaxChunkRenderer`
runs its own CFG.

## YuE2

`--checkpoint yue2-3b` (family `yue2`). YuE2 composes a whole song from a
style prompt and lyrics; DEMON runs its acoustic stage in the ring.

**What runs in the ring.** The acoustic NAR: a 32-step uniform midpoint
flow-matching solve of the WHOLE song, one step per tick, behind the Tier 2
seam (`acestep/engine/yue2_adapter.py`). The adapter returns the midpoint
velocity, so the pipeline's Euler step is upstream's midpoint update exactly.
`yue2_denoise` truncates the released grid (the last `ceil(32*d)` steps from
the re-noised song anchor); it is never rescaled. Ring depth is 1 (see
Measured).

**What is conditioning.** Score plan, semantic tokens and the AR-prefix KV
prefill (`acestep/engine/yue2_context.py`). They run at create, before
`ready` (the page shows "composing"), and again for a prompt change on the
context's worker thread (`acestep/streaming/yue2_recompose.py`). The KV bundle
rides `SlotRequest.aux_cond`. Lyrics and song length are fixed for the
session.

**Prompt changes re-compose.** A YuE2 style lives in the semantic tokens:
with the semantics frozen, swapping only the `[Tags]` prefix moved the audio
less than a seed change does (long-term spectrum distance 0.04-0.06 dB vs
0.70-1.20 dB for a seed change and 3.3-4.2 dB for a fresh composition, two
style pairs). So `set_prompt` composes the session's lyrics under the new tags
at the session's length (semantic `min_tokens = max_tokens = T`) while the
current song keeps playing, publishes the new song atomically, and the ring
drops the old song's in-flight slots and renders the new anchor with one full
solve. Latest wins per slot: going back to the playing prompt cancels the job
in flight, re-sending the tags a slot is already composing keeps that job (so
does an unchanged A sent alongside a new B), and a song finishing after the
session closes releases its KV bundle. A replaced song's KV bundle is freed by
the runner as soon as no slot plays it and no in-flight solve uses it. The
params telemetry carries `yue2_recomposing` (the slots with a job in flight)
and `yue2_nar` (the NAR path of the song that last emerged). A failed re-compose is logged at WARNING and sent as the session's
runtime error event (wire `error`, the SDK's `server_error`); the current song
keeps playing. A distinct `prompt_b` is a second song composed at create;
`set_prompt_blend` is a hard switch at 0.5.

**Settled ring.** With fixed conditioning, seed and knobs and no feedback,
the ring stops after one generation and the renderer keeps playing that latent
(repeat windows come from a cache), so the GPU is free until something moves.
Slots still in flight when it settles (depth >= 2) are dropped, so the next
change does not first finish a stale one.

**Knobs.** `yue2_denoise` (prefixed: it is not ACE's `denoise`), `yue2_steps`,
`x0_target`, `feedback`, `feedback_depth`, `seed` (new acoustic noise, same
composition). No `steps_override` (its shared default of 8 would change the
released behaviour), no LoRA, no CFG, no per-frame curves.

**`yue2_steps`.** Midpoint steps per acoustic solve: 32 (the default) is
upstream's released grid; 24, 16, 12, 8, 6 or 4 answer faster (one step per
tick, so update latency scales with the step count) at lower quality. Other
values snap to the nearest choice; a change restarts the ring on the new grid,
and `yue2_denoise` truncates whichever grid is active.

**Caps.** `yue2_duration_s` is the longest song YuE2 may compose: 2-100 s,
clamped (absent or null = 100 s). The ceiling is the flexible NAR engine's
2500 frames; the 2 s floor keeps every song longer than the window decoder's
37 frames, and a composition that still comes out shorter fails create with a
clear error. With engines present the semantic stage is held to the floor of
the largest engine profile the budget reaches (1000 frames = 40 s, or 250
frames = 10 s for shorter budgets on an engine with the short profile), so
songs land on TensorRT. (The family's
`TextOnlySpec` default of 60 s only sizes the silent stub source, which YuE2
ignores.)

**Environment.**

| Variable | Meaning |
| --- | --- |
| `DEMON_YUE2_ROOT` | weights root holding `YuE2-3B/` and `YuE2-Vae/` (checked against a SHA256 manifest) |
| `DEMON_YUE2_YUE_SRC` | upstream YuE `src/` at revision `0edaf2f4` |
| `DEMON_YUE2_EXTRA_PATH` | optional extra import paths (`os.pathsep`-separated) |
| `DEMON_YUE2_TRT_DIR` | optional; holds `flexible_song/velocity.trt` (NAR) and `vae_fp32_t37.trt` (window VAE). Without it everything runs eager |

**Engines.** `python -m acestep.engine.trt.yue2_build nar --out $DEMON_YUE2_TRT_DIR`
and `python -m acestep.engine.trt.yue2_build vae --out $DEMON_YUE2_TRT_DIR`.
The NAR engine has two optimization profiles: 1000-2500 frames (40-100 s) and
250-1000 frames (10-40 s, short songs), both batch 1-4 and up to 4000
conditioning tokens (`yue2_build nar --long-only` writes the first only, as
older engines have). At run time the profiles are read from the loaded engine
and each song binds on the first that holds it; a song outside every profile
(or one whose bind the engine refuses) runs eager instead of failing. The VAE engine decodes 37-frame windows and keeps 5 core frames. The
numbers below used the engines built during the feasibility spike; this
builder is a port of that build and was not re-run here.

**Measured** (RTX 5090, Windows, one process; raw data in the PR notes):

| | 30 s budget | 60 s budget |
| --- | --- | --- |
| NAR path | eager (750 frames < engine floor) | TensorRT |
| create: plan / semantic / prefill / anchor solve / decode | 5.7 / 4.9 / 0.06 / 3.8 / 0.2 s | 5.7 / 9.5 / 0.1 / 2.5-2.7 / 0.3-0.4 s |
| ring tick p50 / p95 (depth 1) | 118.7 / 120.2 ms | 70.2 / 71.5 ms |
| knob change on a settled ring to new audio: `seed`, `x0_target` | 4.0 s | 2.4-2.5 s |
| same, `yue2_denoise` 0.5 | 2.0 s | 1.2 s |
| change landing mid-solve: first changed / fully updated (`seed`) | 1.9 / 5.9 s | 1.15 / 3.6 s |
| `set_prompt` to the new song heard | | 21-42 s idle ring, 36-58 s busy ring |
| nvidia-smi peak (incl. 3.9 GB desktop) | 18.4 GB | 21-22 GB |

Knob-to-ear is measured to CPU PCM (produce + window render), excluding
transport and the client buffer. Parity: the ring is bit-exact against
upstream's eager midpoint solve at depth 1 and 4, and against an explicit
TensorRT solve on TensorRT; TensorRT vs eager is 0.79-1.19% relative L2 (two
compositions). Depth 1 is a performance cap: a settled ring runs one solve per
change, and depth 2 ran a 102-124 ms tick (60 s songs) and slowed every update
by 40-75% (settled latents identical to depth 1).

Browser smoke (headless Chromium on `/yue2/`, 57.8 s song, TensorRT): `ready`
34.1 s after Start in a fresh server (model load included), first audio 0.1 s
later; slices kept arriving through `yue2_denoise` and `x0_target` moves (about
290 per 8 s); a prompt change was heard 20.9 s later (re-compose 18.0 s plus
the anchor solve 2.7 s, ring idle); no console errors.

**Known gaps.**

- Songs under 10 s, and songs under 40 s on an engine built without the short
  profile, run the NAR eager.
- A composition whose conditioning exceeds 4000 tokens runs eager (seen
  for re-composed songs: 4181 and 4839 tokens); `yue2_nar` reports it and the
  server logs `yue2_nar_path_changed`.
- The AR is about 2x slower inside a live session than at create (and up to
  3.7x while the ring is busy): a re-compose takes about 30 s at 60 s. The
  ring never underran during one. A tokens-only conditioning subprocess would
  isolate it.
- Songs end where the length budget cuts them unless the model ends earlier
  (a re-compose forces the session's length).
- Ear verdicts on `yue2_denoise` < 1 and on re-composed songs are pending.

**Licence.** The YuE2 weights are CC BY-NC 4.0: non-commercial use only.

**Turbo.** There is no YuE2-Turbo checkpoint: "Turbo" is NoizAI's vLLM
serving layer for the AR stages (Linux), so it could only shorten create and
re-compose, and it is out of v1.
