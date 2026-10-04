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

All GPU runs and proofs are on Stable Audio 3 medium (RTX 5090). ACE-Step is
wired through the same code but is not run here.

### Stable Audio 3 setup

- Hook point: the output of `blocks[i].cross_attn` (before the residual
  add), the module the reference Stable Audio Open controller hooks. K/V
  patching substitutes the `context` argument of the same modules, which
  is exactly a K/V patch on SA3 (padding is masked inside the context).
- SA3 medium is post-trained for `cfg_scale = 1` with an 8-step pingpong
  sampler, so every forward is the conditional pass and TADA's `cond_only`
  covers every row. The paper's Stable Audio Open settings (100 steps,
  guidance 7) do not transfer; SA3's native 8 steps are used throughout.
- Live path: a versioned cross-attention steering DiT engine
  (`sa3_m_dit_steerxa_l1_646_646`, input `steering_xattn`), built with
  `python -m acestep.engine.trt.sa3_build --dit --steer-cross-attn`; the
  previous engines are kept. Parity at the existing bar (0.9998): zero
  steering vs eager min cos 0.999863 (plain engine 0.999877), steered TRT vs
  eager hooks min cos 0.999863, re-zero bit-exact; 12.77 ms per DiT step
  (plain 12.68, fp8 10.45). A session selects it only when a pack targets
  `cross_attn_output`.
- Drivers: `scripts/tada/sa3_tada_run.py` (generation, vectors, packs) and
  `scripts/tada/sa3_tada_score.py` (scoring with the reference
  `steer-audio` evaluation code, run in its own environment).

### Localisation

| Family | Checkpoint | Concepts | Impact per block (top) | Functional set (tau 0.10) | Paper |
| --- | --- | --- | --- | --- | --- |
| Stable Audio 3 | medium | 10 Stable Audio Open concepts (female, male, fast, slow, happy, sad, violin, flute, maracas, reggae) | block 7: 0.253, block 3: 0.179, block 6: 0.164, block 0: 0.073, block 5: 0.073; all others below 0.07 | blocks 3, 6, 7 (paper numbering 4, 7, 8) | Stable Audio Open {12, 13, 14} (blocks 11 to 13): 0.51, 0.22, 0.12 |
| ACE-Step | v1.5 | not run | not run | not run | {7, 8} (blocks 6, 7) |

Protocol as the paper: patch one block's cross-attention K/V with the clean
prompt's, score with CLAP ("This is a music of {p}", the concept's eval
prompt), Eq. 2 impact floored at 0, mean over concepts, tau 0.10. 32 prompt
pairs x 8 seeds per concept (the paper uses every pair x 8 latents), seed
222, 10 s. The aggregate is strongly peaked like the paper's, but SA3
concentrates its concept routing earlier (block 7) than Stable Audio Open
(block 12). Per concept top block: female 7 (0.64), male 7 (0.45), fast 6
(0.40), slow 6 (0.48), happy 3 (0.58), sad 6 (0.45), violin 7 (0.30), flute
7 (0.60), maracas 15 (0.21), reggae 5 (0.21). Guitar and techno, in the
paper's set, have no prompt pairs in the released data and are not run.

### CAA steering benchmark

Alignment-preservation AUC up to the PCI cutoff, `pos / neg / avg`;
smoothness is MuQ / CLAP (lower is better); quality is Audiobox Aesthetics
production quality and content enjoyment at matched LPAPS. Nine concepts
(ablated: the paper's four Stable Audio Open table concepts).

| Family | Variant | AUC MuQ (pos / neg / avg) | AUC CLAP (pos / neg / avg) | Smoothness | Quality | Paper (MuQ / CLAP avg) |
| --- | --- | --- | --- | --- | --- | --- |
| Stable Audio 3 | localised (blocks 3, 6, 7) | 0.006 / 0.015 / 0.011 | 0.004 / -0.001 / 0.001 | 0.712 / 0.595 | PQ 7.91, CE 7.21 | Table 1 CAA loc: 0.104 / 0.065; SAO Table 23 loc: 0.334 / 0.195 |
| Stable Audio 3 | all blocks | 0.012 / 0.017 / 0.014 | 0.007 / -0.001 / 0.003 | 0.472 / 0.992 | PQ 7.92, CE 7.22 | Table 1 CAA all: 0.075 / 0.046; SAO Table 23 all: 0.310 / 0.175 |
| Stable Audio 3 | ablated (4 concepts) | 0.008 / 0.027 / 0.017 | 0.005 / -0.004 / 0.001 | 0.172 / 0.497 | PQ 7.92, CE 7.21 | App. L ACE: 0.021 MuQ; SAO Table 23 ablated: 0.158 / 0.094 |
| Stable Audio 3 | AUSteer localised | 0.006 / 0.012 / 0.009 | 0.001 / -0.002 / -0.000 | 0.509 / 0.867 | PQ 7.92, CE 7.22 | Table 1 AUSteer loc: 0.096 / 0.063 |
| Stable Audio 3 | AUSteer all blocks | 0.009 / 0.011 / 0.010 | 0.005 / -0.003 / 0.001 | 0.638 / 1.053 | PQ 7.92, CE 7.22 | Table 1 AUSteer: 0.081 / 0.058 |
| Stable Audio 3 | CAA at the K/V site, localised (negative-result variant) | -0.007 / 0.003 / -0.002 | 0.002 / -0.004 / -0.001 | 1.172 / 2.216 | PQ 7.91, CE 7.21 | not in the paper |

Paper scales: ACE-Step Table 1 and App. Table 24 (localised 0.083 / 0.061
average); Stable Audio Open Table 23 (localised 0.334 / 0.195, all 0.310 /
0.175, ablated 0.158 / 0.094).

On the four Stable Audio Open table concepts (piano, tempo, mood, female
vocal), AUC averaged over directions (MuQ / CLAP):

| Variant | MuQ | CLAP |
| --- | --- | --- |
| localised (blocks 3, 6, 7) | 0.016 | -0.000 |
| all blocks | 0.019 | 0.000 |
| ablated (4 concepts) | 0.017 | 0.001 |

**Finding.** On SA3 medium, TADA's steering reproduces only weakly. Tempo
steers clearly (MuQ "fast track" 0.024 at the negative end, 0.083 at zero,
0.122 at the positive end; AUC 0.050), rock genre and vocal style weakly,
and the other concepts sit within about 0.01 of zero. CLAP gains are near
zero for every variant. The localised blocks do not beat all blocks or
the ablated set on SA3, unlike the paper. AUSteer lands slightly below CAA
(a result, not a bug: it is above zero, so no sign check was needed).
Audio quality is flat across variants.

Negative-result follow-ups, all run through the same protocol: (a) all
blocks (above, no better); (b) per-step vectors from the cached per-prompt
activations (that is already how the vectors are built: 8 steps, one per
SA3 step, from per-prompt caches, so it is the main run); (c) CAA at the
K/V site instead of the output (the conditioning-token mean difference,
added to the non-padding tokens at blocks 3, 6, 7): worse, MuQ -0.002 /
CLAP -0.001.

Zero strength is a no-op: offline, every CAA and AUSteer vector at 0 is
bit-identical to no steering (renorm on and off); live, knob 0 builds no
steering configs and the engine's re-zero is bit-exact.

### Per concept (Stable Audio 3, localised)

| Concept | CAA range | CAA AUC MuQ | CAA AUC CLAP | CAA smoothness (MuQ) | AUSteer AUC MuQ / CLAP | PCI cutoff (pos / neg) |
| --- | --- | --- | --- | --- | --- | --- |
| electronic_music | -8 to 16 | 0.006 | 0.001 | 2.442 | 0.005 / -0.001 | 2.40 / 2.27 |
| guitar_electronic | -8 to 8 | -0.001 | 0.001 | 0.897 | 0.004 / -0.001 | 1.85 / 1.90 |
| mood | -16 to 8 | 0.002 | -0.004 | 0.239 | -0.002 / -0.003 | 2.28 / 2.41 |
| piano | -16 to 16 | 0.007 | -0.003 | 0.610 | 0.006 / 0.001 | 2.44 / 2.41 |
| rock_genre | -16 to 16 | 0.013 | 0.008 | 0.197 | 0.006 / 0.006 | 2.55 / 2.49 |
| tempo | -32 to 16 | 0.050 | -0.002 | 0.084 | 0.049 / -0.007 | 3.21 / 3.40 |
| violin | -16 to 16 | 0.004 | 0.002 | 0.751 | 0.002 / 0.002 | 2.70 / 2.41 |
| vocal_gender | -16 to 32 | 0.006 | 0.008 | 0.936 | 0.006 / 0.005 | 3.10 / 2.78 |
| vocal_style | -16 to 32 | 0.011 | 0.003 | 0.249 | 0.007 / -0.005 | 3.17 / 3.09 |

### Evaluation scale and deviations

- Benchmark prompts: the first 50 of the 100 test prompts; 10 strengths per
  side plus 0 (the paper uses 15). Seed 2115, 10 s. Scaled down for GPU
  time: the reference protocol scores at about 0.45 s per clip here.
- Strength ranges: calibrated on the 20 held-out prompts, as the smallest
  power-of-two probe whose mean LPAPS reaches the PCI cutoff in each
  direction (the paper releases ranges for ACE-Step and Stable Audio Open,
  which do not transfer to SA3).
- PCI: switch lengths 1 to 8 of 8 steps both ways, at every block (PCI-all)
  and at blocks 3, 6, 7 (PCI-loc); scored at the full-swap endpoints, which
  is where the cutoff is taken.
- CAA vectors: 50 pairs, seed 10, no renorm (the released Stable Audio Open
  eval configs leave renorm off). AUSteer: the same 50 pairs, every
  (pair, frame) a sample, global top-s of 1024 over blocks 3, 6, 7 and 2048
  over all blocks (the released budgets are per ACE-Step concept and width).
- Packs: written by `sa3_tada_run.py pack` under `<packs>/sa3/medium/` and
  loaded through `DEMON_STEERING_PACKS_DIR`; 9 CAA
  (`steer_<concept>`) and 9 AUSteer (`steer_<concept>_austeer`), blocks
  3, 6, 7, per-step `[3, 8, 1536]`; magnitude maps the knob's full scale (30)
  to the calibrated localised strength.
- Ear package: `scripts/tada/sa3_tada_listen.py tempo rock_genre` (zero,
  positive and negative strength of one prompt each, with a README).

## TODO

- ACE-Step checks are not run. These are the paper's own ACE-Step results:
  the localisation of blocks 6 and 7, the Table 1 CAA numbers, and the
  calibrated ACE ranges. The paper used ACE-Step v1, and DEMON ships v1.5,
  so the blocks may differ.
- Neither paper checkpoint is a DEMON family. The paper's Stable Audio
  results are for Stable Audio Open 1.0, not Stable Audio 3.
- ACE-Step: localisation sweep, CAA and AUSteer packs, and the benchmark on
  the ACE-Step engine (`steering_xattn`) are wired but not run.
- SA3 at the full benchmark scale (100 prompts, 15 strengths per side) and
  with finer strength calibration; SA3 at more sampler steps, to test
  whether the weak steering comes from the 8-step distilled sampler.
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
