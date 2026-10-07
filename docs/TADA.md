# TADA: activation steering for model families

DEMON replicates TADA, "Tuning Audio Diffusion Models through Activation
Steering" (Staniszewski, Zaleska, Modrzejewski and Deja, arXiv
[2602.11910](https://arxiv.org/abs/2602.11910)). The reference code is at
[github.com/luk-st/steer-audio](https://github.com/luk-st/steer-audio)
(MIT, Copyright (c) 2026 Lukasz Staniszewski). The method, the concept sets
and the evaluation protocol are theirs. DEMON's part is running the method on
its own model families through the pipeline's steering slot. The port is
family-generic: a family gives it its cross-attention modules, and nothing
in `acestep/tada/` names a particular model.

## Method

1. **Localisation by activation patching.** For a counterfactual prompt pair
   (a clean prompt that contains the concept, and a corrupted one that does
   not), generate from the corrupted prompt with the same initial noise,
   but let one cross-attention layer `l` see the clean prompt's conditioning
   (its keys and values). Score the audio against the concept text. The
   impact is

   `I(l, c) = (s(l <- c) - s(none)) / (s(all) - s(none))`,

   floored at 0 and not capped. The impacts are averaged over concepts, and
   the functional layers are those with `I(l) >= tau`, where `tau = 0.10`.
2. **Contrastive activation addition (CAA).** For each denoise step `t` and
   each localised layer `l`, run 50 positive and 50 negative prompts. Average
   each run's cross-attention output over time frames, take the difference
   of the means and normalise it to unit L2 norm. During generation, add
   `alpha * v_{t,l}` to that layer's cross-attention output, before the
   residual add, on the conditional CFG pass only. The released evaluation
   configs also rescale each frame back to its original L2 norm (`renorm`).
   A negative `alpha` removes the concept.
3. **AUSteer (Feng et al.), as TADA adapts it.** From the same paired
   conditional-pass activations, every time frame of every pair is a sample
   of the momentum `m_j = h_pos[j] - h_neg[j]`. Each dimension gets the
   signed sign-consistency score `beta_j = +-max(r+_j, r-_j)` (the fractions
   of samples with `m_j > 0` and `m_j < 0`; positive on a tie, as in the
   reference code). At each step a global top-`s` over `|beta|`, across every
   active layer, keeps `s` dimensions. The steering vector is that sparse
   `beta`, added as in CAA (`h <- h + alpha * v`, conditional pass, renorm).
   The reference default is `s = 256`. The released per-concept budgets
   (256 to 8192) are in the data. The original multiplicative form gave no
   gain in the paper and is not ported.
4. **Multi-concept steering (Sec. 5.5).** The combined vector is the
   unit-weight sum of single-concept vectors of one method, with
   concept-suppressing directions sign-flipped (`caa.combine(vectors,
   negate=...)`).
5. **Evaluation.** The alignment gain is sign-corrected
   (`sign(alpha) * (sim(alpha) - sim(0))`, using MuQ-MuLan and CLAP) and is
   plotted against LPAPS distance from the `alpha = 0` audio. The area under
   that curve is cut off at the LPAPS of a full prompt swap (the PCI cutoff).
   It is reported per direction and averaged. Smoothness is the standard
   deviation of the gaps between consecutive points of the peak-normalised
   gain curve. Quality is Audiobox Aesthetics, interpolated at 15 LPAPS
   points per direction.

Block numbering: the paper counts layers from 1, and DEMON and the
reference code count blocks from 0. The paper's ACE-Step layers {7, 8} are
blocks 6 and 7.

## Code

| Module | What it holds |
| --- | --- |
| `acestep/tada/target.py` | `ActivationTarget` protocol, `ModuleTarget` (a family's cross-attention modules and conditioning argument names), `ActivationRecorder` (per step and block, time-averaged outputs), `ConditioningPatcher` (record the clean run's conditioning, substitute it into the chosen layers), `forward_counter` (which forward is the conditional pass of which step) |
| `acestep/tada/patching.py` | impact score, the `L + 2` patch sets, `patching_sweep`, `localize` (mean over concepts, `tau`) |
| `acestep/tada/caa.py` | `caa_vectors`, `stack_vectors` (pack layout `[blocks, steps, hidden]`), `steer_activation` (the reference intervention), `combine` (multi-concept sum with optional negation), `alphas_from_range` |
| `acestep/tada/austeer.py` | `austeer_scores` (signed sign-consistency `beta` per step and block, from recordings made with `reduce=frames`), `select_top_s` (global top-`s` per step, optionally restricted to the localised blocks), `austeer_vectors`, `DEFAULT_TOP_S = 256` |
| `acestep/tada/packs.py` | `caa_pack` (method `tada_caa`), `austeer_pack` (method `auscore`), `write_caa_pack`: a TADA vector set as a steering pack (`hook = cross_attn_output`, `blocks`, `cond_only`, `renorm`), one `steer_<concept>` knob |
| `acestep/tada/metrics.py` | sign-corrected gain, AUC with cutoff, smoothness, LPAPS cutoff, quality at LPAPS, extra preservation axes; the CLAP, MuQ and mir_eval scorers are imported lazily |
| `acestep/tada/concepts.py`, `acestep/tada/data/` | the reference concept sets, prompt pairs, alignment queries, benchmark prompts, calibrated ranges and patching data, verbatim (regenerate with `scripts/tada/import_upstream_data.py --repo <steer-audio checkout>`) |

`tests/unit/test_tada_core.py` checks the following on CPU. The sweep picks
a planted block. The CAA vector equals a planted difference. A pack
round-trips. AUSteer selects planted sign-consistent dimensions, matches
the reference score formula, and an AUSteer pack is a no-op at zero
strength. The multi-concept sum negates as asked. Zero strength is bit-identical to no steering through the real
`StreamPipeline`, and a nonzero strength moves only the conditional pass at
the steered cross-attention output. The AUC and smoothness port reproduces
the paper's published piano numbers from the released per-alpha table.

## Running TADA on a new family

1. **Expose the hook point.** The family's adapter returns a
   `SteeringLayout` whose `extra_hooks` include `cross_attn_output`, and it
   implements `steering_hook_modules("cross_attn_output")`, which returns one
   cross-attention module per block, in block order. For an accelerated
   forward, the engine takes the cross-attention steering rows as an input
   instead (see the ACE-Step engine's `steering_xattn` and
   `steering_xattn_renorm`).
2. **Build a target.** `ModuleTarget({HOOK_CROSS_ATTN_OUTPUT: modules},
   cond_keys=(...))`, where `cond_keys` names the arguments that carry the
   text conditioning into those modules. The patcher substitutes exactly
   these.
3. **Localise.** For each localisation concept
   (`concepts.localization_concepts`), run `patching_sweep` with a
   `generate(patched_blocks)` that uses `ConditioningPatcher.record()` on the
   clean prompt, then `.patch()` on the corrupted prompt with the same seeds.
   Pass a `similarity` built from `metrics.muq_similarity` or
   `metrics.clap_similarity`. Then call `localize(sweeps)`.
4. **Compute vectors.** For each steering concept (`concepts.prompt_pairs`),
   generate every positive and negative prompt under an `ActivationRecorder`
   with `forward_counter(passes_per_step, cond_pass)` matching the family's
   CFG layout. Then call `caa_vectors` and `stack_vectors` over the localised
   blocks.
5. **Ship.** Pass the stacked vectors to `caa_pack(...)` and `write_caa_pack`
   into the family's pack directory. The pack loads as a `steer_<concept>`
   knob with no further code.
6. **Evaluate.** Run the 100 benchmark prompts (`concepts.benchmark_prompts`)
   over the `alphas_from_range` grid. Then compute LPAPS against `alpha = 0`,
   the sign-corrected gains, `both_directions` AUC up to the PCI cutoff,
   smoothness and quality.

## Results

All GPU runs and proofs are on Stable Audio 3. ACE-Step is wired through the
same code but is not run here.

### Localisation

| Family | Checkpoint | Concepts | Impact per block (top) | Functional set (tau 0.10) | Paper |
| --- | --- | --- | --- | --- | --- |
| Stable Audio 3 | TBD | TBD | TBD | TBD | Stable Audio Open {12, 13, 14} (blocks 11 to 13) |
| ACE-Step | v1.5 | not run | not run | not run | {7, 8} (blocks 6, 7) |

### CAA steering benchmark

| Family | Variant | AUC MuQ (pos / neg / avg) | AUC CLAP (pos / neg / avg) | Smoothness | Quality | Paper (ACE-Step) |
| --- | --- | --- | --- | --- | --- | --- |
| Stable Audio 3 | localised | TBD | TBD | TBD | TBD | Table 1 CAA loc: 0.104 MuQ / 0.065 CLAP |
| Stable Audio 3 | all blocks | TBD | TBD | TBD | TBD | Table 1 CAA all: 0.075 MuQ / 0.046 CLAP |
| Stable Audio 3 | ablated | TBD | TBD | TBD | TBD | App. L, 4 concepts: 0.055 / 0.083 / 0.021 MuQ (all / loc / ablated) |
| Stable Audio 3 | AUSteer localised | TBD | TBD | TBD | TBD | Table 1 AUSteer loc: 0.096 MuQ / 0.063 CLAP |
| Stable Audio 3 | AUSteer all blocks | TBD | TBD | TBD | TBD | Table 1 AUSteer: 0.081 MuQ / 0.058 CLAP |

### Per concept (Stable Audio 3)

| Concept | Range | AUC MuQ | AUC CLAP | Smoothness | Notes |
| --- | --- | --- | --- | --- | --- |
| piano | TBD | TBD | TBD | TBD | |
| mood | TBD | TBD | TBD | TBD | |
| tempo | TBD | TBD | TBD | TBD | |
| vocal_gender | TBD | TBD | TBD | TBD | |
| vocal_style | TBD | TBD | TBD | TBD | |
| guitar_electronic | TBD | TBD | TBD | TBD | |
| violin | TBD | TBD | TBD | TBD | |
| rock_genre | TBD | TBD | TBD | TBD | |
| electronic_music | TBD | TBD | TBD | TBD | |

## TODO

- ACE-Step checks are not run. These are the paper's own ACE-Step results:
  the localisation of blocks 6 and 7, the Table 1 CAA numbers, and the
  calibrated ACE ranges. The paper used ACE-Step v1, and DEMON ships v1.5,
  so the blocks may differ.
- Neither paper checkpoint is a DEMON family. The paper's Stable Audio
  results are for Stable Audio Open 1.0, not Stable Audio 3.
- The reference code and the paper disagree on two points. The released
  Stable Audio Open localised set is blocks 4, 11, 12, 13 and 18, while the
  paper reports {12, 13, 14}. The patching config uses 9 s, while the paper
  says 10 s. Results here follow the paper and note the config.
- SAE and Concept Sliders: scope and status to be filled from the SAE lane.

## Citation

```
@article{staniszewski2026tada,
  title={TADA! Tuning Audio Diffusion Models through Activation Steering},
  author={Staniszewski, {\L}ukasz and Zaleska, Katarzyna and Modrzejewski, Mateusz and Deja, Kamil},
  journal={arXiv preprint arXiv:2602.11910},
  year={2026}
}
```
