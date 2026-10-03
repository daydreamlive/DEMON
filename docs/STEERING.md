# Activation steering

Steering adds a fixed vector to the output of one transformer block while
the model denoises. Turning a steering knob up pushes the music toward a
concept, such as "bright", and turning it down pushes it the other way. Knob 0
does nothing at all. The core implements steering once, and each family only
says where its blocks are, so the same knobs work on ACE-Step and on Stable
Audio 3.

The packs shipped with this seam are prompt-pair difference-of-means
vectors on the generic steering slot. For each concept, paired prompts with
and without it are generated, the post-block residuals are averaged, and the
mean difference becomes the vector. One block is then chosen by a patching
sweep. These packs are not a replication of TADA (arXiv
[2602.11910](https://arxiv.org/abs/2602.11910), MIT licence, code at
[github.com/luk-st/steer-audio](https://github.com/luk-st/steer-audio)),
which is the reference work for activation steering in audio diffusion. The
seam and the pack format do not depend on how a vector was found, so
vectors from TADA or any other method load the same way.

## The seam

`StreamPipeline` (`acestep/engine/stream.py`) has one steering slot. Each
entry is `(block, vector, scale, gate)`, and the gate is either the ACE
one-hot denoise step or a per-step weight curve. An adapter declares where
the slot lands with two optional methods on `ModelAdapter`:

| Method | Returns | Meaning |
| --- | --- | --- |
| `steering_layout()` | `SteeringLayout(num_blocks, hidden_size, hook, engine_input)` or `None` | the block count and width the vectors must match; `None` = no steering |
| `steering_blocks()` | the eager block modules, or `None` | the pipeline installs removable forward hooks on them |

How the shift is delivered depends on what the adapter declares:

* **Eager blocks.** The pipeline registers hooks that add the shift for each
  row's current step. The hooks are removed when the pipeline closes, when
  the engine is swapped, and when SA3 rebuilds its pipeline. This matters
  because the SA3 model is cached for the whole process and shared.
* **TensorRT engine input.** `adapter.accepts_steering = True` makes the
  pipeline pass `steering=` as an fp32 tensor `[B, num_blocks, hidden]` (or
  `None`) to `batched_forward`. The adapter then binds it to the engine.
  ACE fills its own engine buffer inside the pipeline instead.

The hook point is the post-block residual stream (`post_block_residual`):
the vector is added to every token of the block output.

SA3 runs on TensorRT through a separate engine,
`sa3_m_dit_steer_l{min}_{opt}_{max}`, built with
`python -m acestep.engine.trt.sa3_build --steer`. The build adds the steering
input by editing the upstream fp16mixed ONNX graph
(`acestep/engine/trt/sa3_steering_onnx.py`): one fp32 input
`steering [1, 24, 1536]`, a single cast to fp16, and one `Gather` and `Add`
after the residual `Add` of each block. No weights change, so a zero input
reproduces the original graph exactly. A session uses this engine only when
packs exist for its checkpoint. Without packs, the default engine choice
(fp8 when built) is unchanged, and a LoRA refit engine still takes
precedence.

## Packs

A pack is one `.safetensors` file. It holds a tensor `vector` (a unit
direction, fp32, `[hidden]`) and a metadata key `steering_pack` containing
this JSON:

```json
{
  "format": 1, "family": "sa3", "checkpoint": "medium",
  "hook": "post_block_residual", "block": 23, "hidden_size": 1536,
  "name": "bright", "label": "bright", "blurb": "...",
  "method": "caa_diff_means", "norm": 18.6, "magnitude": 1.86,
  "policy": {"kind": "range", "start": 0.0, "end": 1.0},
  "provenance": {"pos": "...", "neg": "...", "pairs": 32, "proxy": "centroid", "...": "..."}
}
```

The applied shift is `knob × magnitude × policy_weight(step) × vector`. The
policy can be `range` (a fraction of the denoise steps), `step` (a single
step) or `curve` (explicit weights). Discovery sets `magnitude` to
`norm × knob_unit`, so with the default unit of 0.1, knob 10 adds one full
mean difference.

Packs live in `DEMON_STEERING_PACKS_DIR`, which defaults to
`<models>/steering_packs`. Files are found by a recursive scan; the
convention is `<family>/<checkpoint>/<name>.safetensors`. When a session is
created, the family factory keeps the packs whose family, checkpoint, hook,
block range and hidden size match the live layout, and registers each one as
a knob `steer_<name>` (range ±30, group `steering`). Because the knob name
and semantics are the same in every family, the homonym guard accepts them.
A pack that does not match is logged and skipped.

Clients need no family-specific code. The `/sa3` demo renders any
`steer_*` knob in its manifest as a slider in the Steer section.

## Discovering packs for a family

`scripts/steering/discover.py` generates audio for pairs of prompts that
differ only in the concept. For each pair it records the post-block
residuals and chooses a block by patching. It then writes one pack per
concept and a JSON report.

```bash
python scripts/steering/discover.py --family sa3 --checkpoint medium \
    --concept bright --concept warm --concept rough --concept density \
    --report out/discover_sa3.json
# custom concept
python scripts/steering/discover.py --family sa3 --checkpoint medium \
    --concept airy --pos "airy, breathy" --neg "dry, close" --proxy centroid
```

1. **Pairs.** Each of `--pairs` (default 32) seeds runs a positive and a
   negative prompt over 8 neutral base styles. A spectral proxy
   (`scripts/steering/proxies.py`: centroid, low/high dB, flatness, onset
   rate, percussive ratio) checks that the prompts actually differ.
2. **Vector.** The direction is the CAA difference of means of the
   token-mean residual after each block, averaged over denoise steps and
   pairs.
3. **Block (residual patching sweep).** On `--patch-pairs`
   pairs, the block's contribution (output minus input) in the negative run
   is replaced with the positive run's. The block's effect is
   `(proxy_patched − proxy_neg) / pooled_gap`. The block with the largest
   median effect wins.
4. **Calibration.** At the chosen block, the knob is swept over
   `--calibrate` on held-out seeds, and the proxy response is stored in
   provenance.

A new family needs a driver in `DRIVERS`: an object that renders a prompt
to audio with a hook on a given block list. `SA3Driver` (a depth-1
`StreamPipeline` with the eager model) and `ACEDriver` (through
`Session.stream`) are the two references. If the adapter implements
`steering_layout()` and `steering_blocks()`, the packs then load and work
with no further code.

`scripts/steering/sanity_sa3_stream.py` checks the packs on the production
TensorRT backend. Knob 0 must be bit-identical to running with no packs,
±K must move the proxy the right way, and it reports tick time.
