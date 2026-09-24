# Adding a model family

A *family* is one generative model behind DEMON's streaming runner: ACE-Step
and Stable Audio 3 today. This document is the contract for adding one and
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
