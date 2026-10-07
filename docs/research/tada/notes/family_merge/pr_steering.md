# feat: generic activation steering slot, steering packs, SA3 steering engine

Base: `main` (b4409206). Branch: `ryanontheinside/feat/steering-generic`.

## What

- **Generic steering slot** (`acestep/engine/stream.py`, `acestep/steering/layout.py`). `StreamPipeline` steering no longer assumes ACE. An adapter declares `steering_layout()` (block count, hidden size, hook point, engine input or not) and `steering_blocks()` (eager modules). The pipeline installs removable hooks or passes a `steering=` tensor `[B, blocks, hidden]` to adapters with `accepts_steering`. Per-row step gating supports the ACE one-hot step and per-step weight curves. Hooks are removed on close, on engine swap and on SA3 pipeline rebuild, because the SA3 model is cached for the whole process.
- **Steering packs** (`acestep/steering/packs.py`). Each vector is a `.safetensors` file (unit vector plus JSON metadata: family, checkpoint, hook, block, hidden size, magnitude, step policy, provenance) under `DEMON_STEERING_PACKS_DIR` (default `<models>/steering_packs`). The family factory loads the packs that match the live layout and registers each one as a `steer_<name>` knob, with the same semantics in every family (homonym guard). ACE and SA3 share the same steering slot.
- **SA3 steering DiT engine** (`acestep/engine/trt/sa3_steering_onnx.py`, `sa3_build.py --steer`, `sa3_trt.py`). An edit to the ONNX graph (proto only) adds one fp32 `steering [1, 24, 1536]` input, added after each block's residual. The engine is selected only when packs exist for the checkpoint. Without packs, the fp8 default is unchanged, and LoRA refit keeps precedence.
- **Tools** (`scripts/steering/`):
  - `discover.py` builds prompt-pair difference-of-means vectors on the generic slot, picking the block by a residual patching sweep. Drivers exist for SA3 and ACE.
  - `sanity_sa3_stream.py` checks the production TRT backend.
  - `proxies.py` holds the spectral concept proxies.
- **Demo**: `/sa3` shows a Steer section with one slider per `steer_*` knob in the manifest. The page has no family or pack knowledge.
- **Docs**: `docs/STEERING.md` (seam, pack format, discovery, TADA reference) and a Steering section in `docs/FAMILIES.md`.

## Numbers (SA3 medium, RTX 5090)

**Engine parity at L=646** (cos, for t = 1.0 / 0.7 / 0.4 / 0.1):

| Comparison | t = 1.0 | t = 0.7 | t = 0.4 | t = 0.1 |
| --- | --- | --- | --- | --- |
| Zero steering vs the previous fp16mixed engine | 0.999968 | 0.999921 | 0.999953 | 0.999960 |
| Zero steering vs eager | 0.999962 | 0.999853 | 0.999957 | 0.999968 |
| Previous engine vs eager (reference) | 0.999968 | 0.999877 | 0.999967 | 0.999981 |
| Nonzero steering, TRT vs eager | 0.999966 | 0.999899 | 0.999960 | 0.999967 |

Re-zeroing after steering is bit-exact.

**Timing:**

| Measurement | Before | After |
| --- | --- | --- |
| Per step | 12.73 ms | 12.88 ms (+1.2%) |
| Streaming tick, no packs (default fp8 engine) | 43.2 ms | |
| Streaming tick, steer engine at knob 0 | | 52.1 ms |
| Streaming tick, steer engine with a knob active | | 52.8 ms |

**Packs:** five prompt-pair packs, each from 32 pairs with 8 patch pairs:

| Pack | Block | Norm |
| --- | --- | --- |
| bright | 23 | 18.6 |
| warm | 15 | 29.5 |
| percussive | 15 | 40.2 |
| rough | 2 | 15.7 |
| density | 1 | 12.9 |

**Live backend sanity:**
- Knob 0 is bit-identical to running without packs.
- Bright (centroid), warm (low/high dB) and percussive (HPSS ratio) move the right way on 3/3 prompts at ±1 and ±10.
- The proxies for rough (flatness) and density (onset rate) do not track those concepts, so their direction is unverified.

**Real server** (headless client on the page's wire), knob 0 / +10 / -10:

| Pack | Proxy | Knob 0 | +10 | -10 |
| --- | --- | --- | --- | --- |
| bright | centroid | 589 Hz | 1860 Hz | 169 Hz |
| warm | low/high | 12.8 dB | 58.0 dB | -8.8 dB |

## Tests

- New: `test_steering_seam.py` (10), `test_steering_packs.py` (8), `test_sa3_steering_engine.py` (8).
- Unit suite: 874 passed, 4 skipped, and 4 failures in `test_lora_facade_session.py` drain tests. Those 4 also fail on `main`.
- The golden harness was not run.

## Not in this PR

- Packs (data, outside the repo).
- A 324-latent steer engine profile.
- An ACE discovery run: the ACE driver is written but untested.
- A replication of TADA. These vectors are prompt-pair difference of means; TADA is planned separately on top of this seam.
