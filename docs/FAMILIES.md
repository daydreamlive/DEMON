# Adding a model family

A *family* is one generative model behind DEMON's streaming runner: ACE-Step,
Stable Audio 3, Magenta RealTime 2 and MiniMax-Music3 today. This document is
the contract for adding one and
the map of what still branches on a family name in the core. Read it before
`acestep/streaming/families.py`.

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
  fails if `DEMON_MINIMAX_TRT_DIR` is set to a missing directory; an absent
  AR stage only warns (a saved capture via `DEMON_MINIMAX_CAPTURE` still
  streams).

**Config fields.** `minimax_duration_s` (rolling-window length, default 60 s,
capped at the AR ceiling of 360 s), `minimax_lyrics` (default
`"[instrumental]"`), `minimax_ar_graph` (default true: one CUDA graph per AR
frame over a static KV cache; false = plain loop for parity work).

**Behaviour.** Append-only: `refines_audio=False` and every other capability
off (no swap, write_audio, timbre, structure, stems, depth, LoRA). Uploaded
audio is ignored (the checkpoint ships no audio encoder); every session
starts from a silent window, and `text_only` advertises the same window.
`set_prompt` re-prefills the LM against the audio already written;
`prompt_b` / prompt blend are refused. The piece ends when the LM emits
end-of-audio. Shutdown evicts the cached contexts; each backend frees its
CUDA graphs and static cache on close.

**Measured (RTX 5090).** Steady ~1.3x realtime (AR stage 1.54x in session,
renderer 7.8x), first audio ~6 s after connect.

**Hot-loop note.** This branch also adds an aux-cond CFG path to
`acestep/engine/stream.py` and an exact-CFG shortcut to
`acestep/engine/ode_steps.py` (ACE/SA3 defaults unchanged). The live family
does not use them: `MiniMaxChunkRenderer` runs its own CFG, and only
`tests/unit/test_stream_aux_cfg.py` exercises the new path.
