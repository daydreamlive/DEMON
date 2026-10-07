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
(localisation, calibration choices, the headline) therefore uses MuQ;
CLAP is reported in a secondary column. Vocal gender localisation keeps
CLAP, as in the paper.

### Localisation

Sec. 4 as written: patch one block's cross-attention K/V with the clean
prompt's, eq. 2 impact floored at 0, `I(l)` averaged over concepts,
`tau = 0.10` on the average. Ten Stable Audio Open concepts, 32 prompt
pairs x 8 seeds each, seed 222, 10 s, 8 steps.

| Scoring | Functional set (tau 0.10) | Top impacts |
| --- | --- | --- |
| Paper mix (MuQ; CLAP for vocal gender), used | blocks 3, 5, 6, 7 | 7: 0.310, 6: 0.191, 3: 0.107, 5: 0.101, 11: 0.097, 1: 0.096 |
| MuQ only | blocks 1, 3, 5, 6, 7 | 7: 0.323, 6: 0.204, 1: 0.122, 3: 0.113, 5: 0.103 |
| CLAP only (earlier runs, superseded) | blocks 3, 6, 7 | 7: 0.253, 3: 0.179, 6: 0.164 |

Paper: Stable Audio Open layers {12, 13, 14} (blocks 11 to 13).

Tau sensitivity (paper mix, Table 7 style):

| tau | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
| --- | --- | --- | --- | --- | --- | --- |
| blocks | 0, 1, 3, 5, 6, 7, 8, 9, 10, 11 | 3, 5, 6, 7 | 6, 7 | 7 | 7 | 7 |

Per concept (App. F.1 style, paper mix): peak block and impact, and the
blocks at or above tau.

| Concept | Metric | Peak | Blocks >= 0.10 |
| --- | --- | --- | --- |
| fast | MuQ | 6: 0.47 | 3, 5, 6, 7 |
| slow | MuQ | 6: 0.60 | 3, 6, 11 |
| female | CLAP | 7: 0.64 | 7, 8, 11 |
| male | CLAP | 7: 0.45 | 7 |
| flute | MuQ | 7: 0.56 | 7 |
| violin | MuQ | 7: 0.25 | 7 |
| happy | MuQ | 5: 0.48 | 0, 1, 2, 3, 5, 6, 8, 12 |
| sad | MuQ | 5: 0.29 | 3, 5, 6, 7 |
| maracas | MuQ | 7: 0.84 | 15 blocks |
| reggae | MuQ | 11: 0.10 | none |

Blocks 6 and 7 are robust to every choice; blocks 1, 3, 5 and 11 sit at
0.10 +- 0.01. For `fast`, the MuQ clean reference scores below the
corrupted one (0.171 vs 0.235 on "fast song"), so its eq. 2 ratio has a
negative denominator; the method is followed literally, and dropping that
concept moves the tau 0.10 set to {1, 6, 7}. A concept-free control sweep
is flat (max 0.07), so the localisation is concept-specific.

### Vectors

Per diffusion step (Sec. 5.2, I.1.4), 8 steps, at blocks 3, 5, 6, 7. CAA
(eq. 6): per prompt, the cross-attention output averaged over time, the
mean of the 50 pair differences, unit norm, no renorm (H.4: the following
norm layer renormalises). The time average runs over audio tokens only:
SA3 prepends 64 learned memory tokens and pads the latent past the
requested duration plus 6 s headroom to its chunk alignment (174 frames,
172 valid for 10 s); the paper's models have neither, so both are excluded
from the mean (the vector is still added to every token of the hooked
output, `h <- h + alpha v`). AUSteer uses the same 50 pairs, every (pair,
audio frame) a sample, global top-s of 1024 over the localised blocks. The
K/V site variant takes the mean difference over the real prompt tokens of
the conditioning (the conditioner's own mask; padding and the
`seconds_total` token excluded).

The correction moves the CAA vectors by a few degrees. Cosine between the
corrected and the earlier all-token vectors at blocks 3, 5, 6, 7 (mean over
steps and blocks, minimum in brackets): tempo 0.963 (0.88), piano 0.943
(0.84), mood 0.951 (0.82), vocal gender 0.939 (0.64), vocal style 0.976,
violin 0.971, acoustic/electric guitar 0.959, electronic 0.953, rock 0.954
(0.68). The nine vectors stay close to orthogonal: cosine to tempo is piano
-0.03, mood 0.14 (earlier vectors -0.02 and 0.15); the largest pair is
piano and violin at 0.24.

### Steering benchmark against the real prompt

The paper's reference point on one model is PCI, the real prompt pushed
through the same protocol. Ratios on ACE-Step, Table 1 (MuQ, PCI-all
0.084): CAA localised 0.104 (1.24), CAA all layers 0.075 (0.89), AUSteer
localised 0.096 (1.14), SAE localised 0.118 (1.40). Localisation gain for
CAA: ACE-Step +39 percent MuQ (Table 2), Stable Audio Open 0.310 to 0.334
(+8 percent, Table 23), AudioLDM2 0.149 to 0.406 (+172 percent, Table 22).
The Stable Audio Open table carries no PCI row, so no ratio exists for it.

<<TABLE_MAIN>>

<<TABLE_CONCEPT>>

<<FINDING>>

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
on SA3, is withdrawn. The correct same-pair numbers are the ones above.

Earlier negative results that stand, with the earlier vectors: guidance on
the steering direction (v0 + g (v1 - v0), g = 3, 5, 7) and 30 sampler steps
did not raise CAA above MuQ AUC 0.025 averaged over tempo, piano and mood;
renorm on at 4x strength lifted tempo from 0.050 to 0.069.

Zero strength is a no-op: offline, every CAA and AUSteer vector at 0 is
bit-identical to no steering (renorm on and off); live, knob 0 builds no
steering configs and the engine's re-zero is bit-exact.

### Evaluation scale and deviations

- Benchmark: the first 50 of the 100 test prompts, 21 strengths (10 per
  side plus 0; paper 31), seed 2115, 10 s clips (paper 30 s on ACE-Step,
  10 s on Stable Audio Open and AudioLDM2 in Sec. 4).
- Sampler: SA3's 8-step CFG-free ARC sampler (paper 30 Euler steps,
  guidance 5). Vectors are per step on that schedule.
- Strength ranges: the largest strength reaches SA3's own PCI maximum
  distortion (the Sec. 5.2 rule), calibrated on 20 held-out prompts (as
  I.2) as the smallest power-of-two probe whose mean LPAPS reaches the PCI
  cutoff in each direction.
- MuQ is primary for every decision on SA3; CLAP is reported beside it
  (see "Scoring instrument on SA3").
- Localisation: 32 pairs x 8 seeds (paper up to 256 pairs; its Table 6
  shows 34 pairs recover ACE-Step's set exactly).
- Audio-token means: SA3's memory tokens and latent padding are excluded
  from the CAA and AUSteer time average (the paper's models have neither).
- PCI: switch lengths 1 to 8 of 8 steps both ways, at every block
  (PCI-all, the ratio denominator) and at blocks 3, 5, 6, 7 (PCI-loc). PCI
  renders start from the reference's neutral wrapper ("a song, {p}"),
  steering sweeps from the raw prompt, as in the reference.
- The 30-step row of the earlier runs reused 8-step vectors and is dropped.
- Packs under `<packs>/sa3/medium/` (live demo) were written from the
  earlier vectors at blocks 3, 6, 7.
- Ear package: `E:/Projects/tada-replication/listen_sa3/`
  (`sa3_tada_listen.py piano mood tempo --sub eval_e3 --with-pci`): zero,
  the real prompt, CAA at the strongest admitted strength and at the
  largest grid strength, all from the same seed and batch layout.
