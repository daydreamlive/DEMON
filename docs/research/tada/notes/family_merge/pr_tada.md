# PR draft: TADA activation steering for model families

Branch: ryanontheinside/feat/tada (core, then ace, sa3, sae), base: steering-generic.
Title: feat(tada): replicate TADA activation steering for model families

## Summary

This PR replicates TADA (Staniszewski, Zaleska, Modrzejewski and Deja,
"TADA! Tuning Audio Diffusion Models through Activation Steering", arXiv
2602.11910; reference code github.com/luk-st/steer-audio, MIT) in DEMON. It
is family-generic and runs through the pipeline's one steering slot.

- **Hook point.** The slot gains a `cross_attn_output` hook: the output of a
  block's cross-attention, before the residual add. It has conditional-pass
  only (`cond_only`), per-token renorm, per-step vectors and multi-block
  packs. The hook is delivered by eager hooks or by the ACE-Step TRT engine
  inputs (`steering_xattn`, `steering_xattn_renorm`).
- **Core (`acestep/tada/`).** Activation capture and conditioning patching
  over any family's cross-attention modules, the impact-score patching sweep
  and localisation (tau 0.10), per-step per-layer unit-norm CAA vectors
  written as packs (one `steer_<concept>` knob each), and the evaluation
  metrics (AUC over LPAPS with the PCI cutoff, smoothness, quality). The
  reference concept sets and prompts are included verbatim as data, under
  their MIT licence.
- **Stable Audio 3.** TBD (SA3 lane): hook wiring, localisation result,
  packs, benchmark numbers, demo.
- **SAE / Concept Sliders.** TBD (SAE lane): what was ported versus scoped.

## Results

Stable Audio 3 only; see docs/TADA.md. ACE-Step is wired but not run, and
its checks are listed as TODO there.

| Family | Functional blocks | CAA loc AUC MuQ / CLAP | Paper reference |
| --- | --- | --- | --- |
| Stable Audio 3 | TBD | TBD | SAO {12, 13, 14}; ACE CAA loc 0.104 / 0.065 |

## Tests

- tests/unit/test_tada_core.py (CPU): the sweep picks a planted block; the CAA
  vector equals a planted difference; a pack round-trips; zero strength is
  bit-identical through the real StreamPipeline, and nonzero strength moves
  only the conditional pass at the steered cross-attention output; the AUC
  and smoothness port reproduces the published piano Table 1 numbers.
- tests/unit/test_steering_xattn_hook.py: hook delivery (eager, TRT buffer
  fill, export island, pack format).
- SA3 and SAE lane tests: TBD.

## Notes

- Block indices are 0-based; the paper's layers are 1-based (ACE {7, 8} =
  blocks 6, 7).
- The reference code's released Stable Audio Open localised set (blocks 4,
  11, 12, 13, 18) differs from the paper's {12, 13, 14}. The paper is
  followed.
