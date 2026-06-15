# AGENTS.md — repo guide for coding agents

DEMON is a streaming diffusion engine for ACE-Step v1.5 (see
[README.md](./README.md) for the full tour). This file is a router:
find your task below, then read the linked doc — it carries the
specifics this one deliberately doesn't duplicate.

## Setting up this repo from scratch

If you are helping a user install or run DEMON, follow this exactly —
do not improvise model sources, versions, or paths.

1. `uv sync` (installs Python 3.11 + all deps; never pip-install into
   a system Python).
2. `uv run demon-setup` — downloads the model checkpoints AND builds
   the minimal TensorRT engine set. Idempotent; re-run to resume.
3. `uv run python -u -m demos.realtime_motion_graph_web.run`, then
   open http://localhost:6660.

Hard rules:

- **Models are the ACE-Step v1.5 weights from the Hugging Face repo
  `ACE-Step/Ace-Step1.5`**, fetched by `demon-setup` or
  `uv run acestep-download`. Do NOT substitute other ACE-Step releases,
  other checkpoints, or hand-built paths — the loaders check for exact
  component directories and fail on anything else.
- **Everything lives under `~/.daydream-scope/models/demon/`**
  (override: `ACESTEP_MODELS_DIR` env var), NOT in the repository.
  Checkpoints in `<models dir>/checkpoints/`, TensorRT engines in
  `<models dir>/trt_engines/`.
- The demo's default `--accel tensorrt` requires built engines; the
  server preflight at boot prints the exact fix when they're missing.
  The no-engines fallback is `-- --accel compile`, not eager hacks.
- Full walkthrough + troubleshooting table: [docs/INSTALL.md](docs/INSTALL.md).

## Where to go by task

| Task | Read first |
|---|---|
| **Build / re-skin / vibecode a frontend** against a DEMON pod | [`packages/demon-client/AGENTS.md`](packages/demon-client/AGENTS.md) — the demon-client SDK recipe: discovery-first workflow, the invariants the manifests can't express, reference implementations. Start there, NOT in the demo app's component code. |
| Work on the bundled web demo app itself | [`demos/realtime_motion_graph_web/README.md`](demos/realtime_motion_graph_web/README.md), then the SDK AGENTS.md above for the protocol layer. |
| Add a standalone static (no-build) demo | [`demos/arp/README.md`](demos/arp/README.md) — the pattern: a plain directory with a `demo.static.json` mount manifest, served by the backend via `demos/common/static_site.py`, loading the shared SDK bundle from `/sdk/` (built from `packages/demon-client`, see its `build.mjs`). |
| Drive a live session from an agent (no browser) | The MCP server (`demos/realtime_motion_graph_web/mcp_server.py`): `describe_protocol` / `list_knobs` return the same manifests the HTTP API serves; `set_knob(s)` / `set_prompt` / `swap_to_fixture` / etc. control a running session. |
| Reproduce frontend/realtime bugs headlessly (e.g. generation lagging the playhead) | `headless_start` on the same MCP server spawns a full PRIMARY client (`demos/realtime_motion_graph_web/headless_client.py`) with a simulated audio clock — no browser needed. `headless_lag_report` (and the snapshot in `headless_status`) measure slice lead vs the playhead and audio staleness. Preempts any live browser session (one session per pod). |
| Engine / pipeline / nodes work | [README.md](./README.md) — Session API (`acestep/engine/session.py`), StreamPipeline (`acestep/engine/stream.py`), typed node graph (`acestep/nodes/`). |

## Repo-wide rules

- **The control surface is contract-first.** Two registries are the
  single source of truth: `acestep/streaming/knobs.py` (knobs, served at
  `GET /api/knobs`) and `demos/realtime_motion_graph_web/protocol.py`
  (WS commands/events/config, served at `GET /api/protocol`). Never
  hand-declare a knob, command shape, or enum option list anywhere else
  — clients build from the manifests, and adding a control is an edit to
  exactly one registry.
- **After any registry change, regenerate the TS types:**
  `python demos/realtime_motion_graph_web/scripts/gen_wire_types.py`
  (output: `packages/demon-client/types/wireContract.gen.ts`). A stale
  copy fails the drift-guard tests.
- **The browser SDK is a shared package.** `packages/demon-client` is
  the single client implementation: the rtmg Next app imports its TS
  source via the `@demon/client` tsconfig alias; static demos load the
  committed `packages/demon-client/dist/` bundle from the backend's
  `/sdk/` mount. Never copy SDK/worker/worklet code into a demo. After
  editing SDK source, rebuild the bundle (`npm run build` in the
  package) so static demos pick the change up.
- **Tests:** `.venv/Scripts/python.exe -m pytest tests/unit` (the system
  Python lacks the deps; always use the repo venv). The contract drift
  guards live in `tests/unit/test_wire_contract.py` and
  `tests/unit/test_client_sdk.py` and run without a GPU.
- **Web checks:** `npm run typecheck` and `npm run build` in
  `demos/realtime_motion_graph_web/web/`.
- **sys.path gotcha:** standalone scripts must force the repo root to
  the FRONT of `sys.path` before importing `acestep`, or a sibling
  ACE-Step checkout can shadow it (see `scripts/gen_wire_types.py` for
  the pattern).
