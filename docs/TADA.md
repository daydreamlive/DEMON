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
  is exactly a K/V patch on SA3.
- Text reaches the SA3 medium trunk by one path only: the T5Gemma prompt
  embedding, through a shared MLP, is the `context` of every block's
  cross-attention. The 64 tokens prepended to the latent sequence are
  learned memory tokens, not text; the global (AdaLN) conditioning carries
  `seconds_total` and the timestep only.
- SA3 medium is the served ARC checkpoint: `cfg_scale = 1`, an 8-step
  pingpong sampler, sigma per step 1.000, 0.994, 0.984, 0.958, 0.891,
  0.746, 0.513, 0.274. Every forward is the conditional pass, so TADA's
  `cond_only` covers every row.
- SA3 medium generates no vocals, so the paper's vocal concepts (vocal
  gender, vocal style; female and male in localisation) are not generable
  by SA3 and are excluded (paper App. A: a concept the model cannot
  generate cannot be localised or steered). The SA3 benchmark has 7
  concepts: tempo, piano, mood, violin, acoustic/electric guitar, rock
  genre, electronic music.
- Live path: a versioned cross-attention steering DiT engine
  (`sa3_m_dit_steerxa_l1_646_646`, input `steering_xattn`), built with
  `python -m acestep.engine.trt.sa3_build --dit --steer-cross-attn`; the
  previous engines are kept. Parity at the existing bar (0.9998): zero
  steering vs eager min cos 0.999863 (plain engine 0.999877), steered TRT vs
  eager hooks min cos 0.999863, re-zero bit-exact; 12.77 ms per DiT step
  (plain 12.68). A session selects it only when a pack targets
  `cross_attn_output`.
- Drivers: `scripts/tada/sa3_tada_run.py` (generation, vectors, packs) and
  `scripts/tada/sa3_tada_score.py` (scoring with the reference
  `steer-audio` evaluation code, run in its own environment).

### Scoring instrument on SA3

The paper scores localisation with MuQ for mood, tempo, instruments and
genres and with CLAP for vocal gender (Sec. 4), and reports both MuQ and
CLAP for steering. On SA3, CLAP barely separates the real positive and
negative prompts: rendering the full positive versus the full negative
prompt moves the CLAP concept score by 0.008 to 0.025, against MuQ +0.104
(tempo), +0.065 (piano) and +0.024 (mood). Every decision on SA3
(localisation, calibration, the headline) therefore uses MuQ; CLAP is
reported beside it.

### Localisation

Sec. 4 as written: patch one block's cross-attention K/V with the clean
prompt's, eq. 2 impact floored at 0, `I(l)` averaged over concepts,
`tau = 0.10` on the average. Stable Audio Open concepts without the two
vocal ones (8 concepts, all MuQ-scored), 32 prompt pairs x 8 seeds each,
seed 222, 10 s, 8 steps.

| Concepts | Functional set (tau 0.10) | Top impacts |
| --- | --- | --- |
| 8 (vocals excluded), MuQ | blocks 0, 1, 3, 5, 6, 7 | 7: 0.251, 6: 0.238, 3: 0.127, 5: 0.124, 1: 0.115, 0: 0.101, 11: 0.096 |
| 10 (paper mix, vocal gender on CLAP) | blocks 3, 5, 6, 7 | 7: 0.310, 6: 0.191, 3: 0.107, 5: 0.101 |
| 10, CLAP only (earliest runs) | blocks 3, 6, 7 | 7: 0.253, 3: 0.179, 6: 0.164 |

Paper: Stable Audio Open layers {12, 13, 14} (blocks 11 to 13).

Tau sensitivity (8 concepts, MuQ, Table 7 style):

| tau | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
| --- | --- | --- | --- | --- | --- | --- |
| blocks | 0, 1, 3, 5, 6, 7, 8, 9, 10, 11 | 0, 1, 3, 5, 6, 7 | 6, 7 | 6, 7 | 7 | none |

Per concept (App. F.1 style): peak block and impact, and the blocks at or
above tau.

| Concept | Peak | Blocks >= 0.10 |
| --- | --- | --- |
| fast | 6: 0.47 | 3, 5, 6, 7 |
| slow | 6: 0.60 | 3, 6, 11 |
| flute | 7: 0.56 | 7 |
| violin | 7: 0.25 | 7 |
| happy | 5: 0.48 | 0, 1, 2, 3, 5, 6, 8, 12 |
| sad | 5: 0.29 | 3, 5, 6, 7 |
| maracas | 7: 0.84 | 15 blocks |
| reggae | 11: 0.10 | none |

Blocks 6 and 7 are robust to every choice; blocks 0, 1, 3, 5 and 11 sit at
0.10 +- 0.03. The steering results below use blocks 3, 5, 6, 7 (the set
measured before the vocal exclusion; blocks 0 and 1 enter only at 0.101
and 0.115). For `fast`, the MuQ clean reference scores below the
corrupted one (0.171 vs 0.235 on "fast song"), so its eq. 2 ratio has a
negative denominator; the method is followed literally. A concept-free
control sweep is flat (max 0.07), so the localisation is
concept-specific.

### Vectors

Per diffusion step (Sec. 5.2, I.1.4), 8 steps, unit norm, no renorm (H.4:
the following norm layer renormalises). The time average runs over audio
tokens only: SA3 prepends 64 learned memory tokens and pads the latent
past the requested duration plus 6 s headroom to its chunk alignment (174
frames, 172 valid for 10 s); the paper's models have neither, so both are
excluded from the mean (the vector is still added to every token of the
hooked output, `h <- h + alpha v`).

Two estimators are reported:

- **CAA, paper sample size** (eq. 6): the mean of 50 positive/negative
  pair differences of the time-averaged cross-attention output.
- **CAA, data-scale** (the deviation that closes most of the gap): the
  same eq. 6 difference of means, taken over every caption of a 5521
  caption cache that carries the concept's keyword, balanced per class
  (positive minus negative pole: tempo 332 per class, piano 751, mood
  184, violin 245, acoustic/electric guitar 444, electronic 375, rock
  173). Cosine to the 50-pair vectors at blocks 3, 5, 6, 7: tempo 0.48,
  piano 0.13, mood 0.29, violin 0.55, guitar 0.39, electronic 0.43, rock
  0.49. A linear probe on the same cache separates the classes at every
  block and step (mean-level accuracy 0.96 piano, 0.98 tempo, 0.97 mood;
  shuffled labels 0.5), so the direction exists at this site; the 50-pair
  estimate of it is noisy on SA3.
- Control: the unit-norm probe weight direction itself (`caa_e5p`, tempo,
  piano, mood).

### Steering benchmark against the real prompt

The paper's reference point on one model is PCI, the real prompt pushed
through the same protocol. Ratios on ACE-Step, Table 1 (MuQ, PCI-all
0.084): CAA localised 0.104 (1.24), CAA all layers 0.075 (0.89), AUSteer
localised 0.096 (1.14), SAE localised 0.118 (1.40). Localisation gain for
CAA: ACE-Step +39 percent MuQ (Table 2), Stable Audio Open 0.310 to 0.334
(+8 percent, Table 23), AudioLDM2 0.149 to 0.406 (+172 percent, Table 22).
The Stable Audio Open table carries no PCI row, so no ratio exists for it.

SA3 medium (ARC, 8 steps), the first 50 benchmark prompts, seed 2115,
blocks 3, 5, 6, 7. MuQ alignment-preservation AUC averaged over the two
directions, up to SA3's own PCI cutoff; the ratio column divides by the
real prompt's own AUC (PCI-all) on the same prompts.

| Concept | PCI-all AUC | CAA loc, data-scale | loc / PCI | CAA all, data-scale | all / PCI | CAA loc, 50 pairs, / PCI | CAA all, 50 pairs, / PCI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tempo | 0.060 | 0.062 | 1.03 | 0.069 | 1.15 | 0.92 | 0.81 |
| piano | 0.035 | 0.017 | 0.47 | 0.015 | 0.44 | 0.39 | 0.55 |
| mood | 0.009 | 0.018 | 1.96 | 0.019 | 2.14 | 0.76 | 0.27 |
| violin | 0.036 | 0.009 | 0.24 | 0.016 | 0.44 | 0.30 | 0.35 |
| acoustic/electric guitar | 0.024 | 0.006 | 0.27 | 0.010 | 0.43 | 0.21 | 0.24 |
| rock genre | 0.053 | 0.024 | 0.45 | 0.025 | 0.46 | 0.43 | 0.41 |
| electronic music | 0.024 | 0.012 | 0.51 | 0.016 | 0.66 | 0.47 | 0.77 |
| mean of 7 (ratio of means) | 0.034 | 0.021 | 0.61 | 0.024 | 0.71 | 0.52 | 0.53 |
| paper, ACE-Step Table 1 | 0.084 | 0.104 | 1.24 | 0.075 | 0.89 | | |

- Localised versus all blocks: data-scale CAA localised is 13 percent
  below all blocks on SA3 (paper ACE-Step +39 percent).
- Probe-direction control (localised, tempo / piano / mood): 0.72 / 0.51
  / 1.49 of PCI; ratio of the 3-concept means 0.72, against 0.93 for the
  data-scale CAA and 0.73 for the 50-pair CAA on the same three.
- Mood's real-prompt AUC is small (0.009: 0.015 one way, 0.003 the
  other), so its ratios above 1 rest on a tiny denominator.
- CLAP, mean of 7: PCI-all 0.009, data-scale CAA localised 0.006, all
  blocks 0.006.
- Every sweep reaches its cutoff in both directions, so no ratio is a
  range artifact.
- Matched check on 20 held-out prompts (tempo, piano, mood): data-scale
  CAA localised 1.15 / 0.41 / 0.66 of PCI against 0.87 / 0.15 / 0.61 for
  the 50-pair vectors on the same prompts.

**Finding.** With the paper's 50-pair estimator, CAA on SA3 reaches about
half of what the real prompt reaches (0.52). The same difference of means
estimated over the full caption cache raises that to 0.61 localised and
0.71 on all blocks, and matches the real prompt on tempo (1.03) and mood.
Piano, violin, guitar, rock and electronic stay at 0.24 to 0.66 of PCI.
The paper's ratios on ACE-Step (1.24 localised, 0.89 all) are not reached,
and localisation does not help on SA3.

A distillation control rules out the ARC sampler as the cause: the same
50-pair CAA on the non-distilled SA3 medium base checkpoint (50 steps,
guidance 7, conditional-pass steering, 20 held-out prompts) gives tempo
1.07, piano -1.08, mood 0.05 of PCI.

### Fixed vectors versus the state-dependent effect

Patching the clean prompt's K/V into the localised blocks recovers 0.47 to
0.65 of the concept on eq. 2 (MuQ); substituting the clean run's exact
cross-attention output recovers 0.30; adding a same-pair, same-step
difference vector (positive minus negative output for that prompt pair)
recovers 0.16. A fixed additive vector therefore carries about a third of
what the patch carries; the rest depends on the state of the run.

Retraction: an earlier run labelled "oracle" added the CAA training-pair
difference of pair i to benchmark prompt i. That was not a same-pair
oracle, and its conclusion, that the cross-attention output is not a lever
on SA3, is withdrawn.

Earlier negative results that stand, with the earlier vectors: guidance on
the steering direction (v0 + g (v1 - v0), g = 3, 5, 7) and 30 sampler steps
did not raise CAA above MuQ AUC 0.025 averaged over tempo, piano and mood;
renorm on at 4x strength lifted tempo from 0.050 to 0.069.

Zero strength is a no-op: offline, every CAA and AUSteer vector at 0 is
bit-identical to no steering (renorm on and off); live, knob 0 builds no
steering configs and the engine's re-zero is bit-exact.

### Deviations from the paper (SA3)

- Estimator: the headline vectors are eq. 6 over the full balanced
  keyword classes of a 5521-caption cache (the data-scale estimator), not
  50 prompt pairs; the 50-pair rows are reported beside them.
- Concepts: 7; vocal gender and vocal style are not generable by SA3 and
  are excluded from every table, average and localisation.
- Sampler: SA3's 8-step CFG-free ARC sampler (paper 30 Euler steps,
  guidance 5). Vectors are per step on that schedule.
- Blocks: 3, 5, 6, 7 (see Localisation; the 8-concept tau 0.10 set adds
  blocks 0 and 1 at the threshold).
- Benchmark: the first 50 of the 100 test prompts, seed 2115, 10 s clips
  (paper 30 s on ACE-Step, 10 s on Stable Audio Open and AudioLDM2 in Sec.
  4). Data-scale rows use 14 strengths per side, the 50-pair rows 10 per
  side (paper 15).
- Strength ranges: the largest strength reaches SA3's own PCI maximum
  distortion (the Sec. 5.2 rule), calibrated on 20 held-out prompts (as
  I.2) as the smallest power-of-two probe whose mean LPAPS reaches the PCI
  cutoff in each direction.
- MuQ is primary for every decision on SA3; CLAP is reported beside it.
- Localisation: 32 pairs x 8 seeds (paper up to 256 pairs; its Table 6
  shows 34 pairs recover ACE-Step's set exactly).
- Audio-token means: SA3's memory tokens and latent padding are excluded
  from every time average.
- PCI: switch lengths 1 to 8 of 8 steps both ways, at every block
  (PCI-all, the ratio denominator) and at blocks 3, 5, 6, 7 (PCI-loc).
  PCI renders start from the reference's neutral wrapper ("a song, {p}"),
  steering sweeps from the raw prompt, as in the reference.
- AUSteer and the K/V-site variant were measured only with the 50-pair
  data (AUSteer slightly below CAA, the K/V site below zero) and are not
  in the final table.
- SAE: not reproduced on SA3. BatchTopK dictionaries on the paper's grid
  (blocks 3 and 5 trained, held-out FVU 0.24 to 0.41) fail an absolute
  per-sigma check at the last step (sigma 0.27, FVU 0.8 to 1.3), because
  token variance falls about 67x across the 8 steps; training stopped
  there, and no v_SAE row exists.
- Packs under `<packs>/sa3/medium/` (live demo) were written from the
  earliest 50-pair vectors at blocks 3, 6, 7; the data-scale vectors are
  not packed yet.
- Ear package: `E:/Projects/tada-replication/listen_sa3/`
  (`sa3_tada_listen.py tempo piano --sub eval_e5 --with-pci`): zero, the
  real prompt, data-scale CAA at the strongest admitted strength and at
  the largest grid strength, all from the same seed and batch layout.

## TODO

- ACE-Step checks are not run. These are the paper's own ACE-Step results:
  the localisation of blocks 6 and 7, the Table 1 CAA numbers, and the
  calibrated ACE ranges. The paper used ACE-Step v1, and DEMON ships v1.5,
  so the blocks may differ.
- Neither paper checkpoint is a DEMON family. The paper's Stable Audio
  results are for Stable Audio Open 1.0, not Stable Audio 3.
- ACE-Step: localisation sweep, CAA and AUSteer packs, and the benchmark on
  the ACE-Step engine (`steering_xattn`) are wired but not run.
- SA3 at the full benchmark scale (100 prompts, 15 strengths per side);
  packs from the data-scale vectors.
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
