# TADA on Stable Audio 3: replication record

This directory records DEMON's replication of TADA, "TADA! Tuning Audio
Diffusion Models through Activation Steering" (Staniszewski, Zaleska,
Modrzejewski and Deja, arXiv 2602.11910; reference code
github.com/luk-st/steer-audio, MIT). It is a record of experiments, not a
feature to merge as-is. The method write-up and the full SA3 results section
live in [`docs/TADA.md`](../../TADA.md); this README is the summary and the
index.

## What TADA is

TADA steers a text-to-audio diffusion model by editing activations instead of
the prompt. It first localises the layers that carry a concept by activation
patching: generate from a prompt without the concept, let one layer's
cross-attention see the conditioning of a prompt with the concept, and score
how much of the concept comes back. It then adds a fixed direction to the
cross-attention output of those layers at every denoising step: a
contrastive activation addition (CAA) vector (unit-norm difference of means
over 50 positive and 50 negative prompts), a sparse AUSteer vector, or a
sparse-autoencoder (SAE) feature direction. Evaluation compares each
method's alignment-versus-distortion curve (MuQ or CLAP gain against LPAPS
distance) with the same curve for prompt interpolation (PCI), the real
prompt pushed through the same protocol.

## What was replicated, and on which models

- **Stable Audio 3 medium** (the served ARC checkpoint: 8-step pingpong,
  cfg 1). Localisation, CAA (paper 50-pair estimator and a data-scale
  estimator), AUSteer, the PCI-referenced benchmark, a distillation control
  on the non-distilled SA3 medium base checkpoint, and an SAE attempt. A
  cross-attention steering TensorRT engine (`sa3_m_dit_steerxa_l1_646_646`)
  drives the live demo.
- **ACE-Step v1.5**: the hook point, the TensorRT engine input
  (`steering_xattn`, `steering_xattn_renorm`) and the build/parity scripts are
  wired; the engine build, localisation sweep, packs and benchmark were never
  run. No ACE-Step numbers in this record are ours; the ACE-Step figures below
  are the paper's.

## Headline

7-concept CAA/PCI ratio (MuQ alignment-preservation AUC, ratio of means;
first 50 benchmark prompts, seed 2115, blocks 3, 5, 6, 7 for "localised").
Source: [`results/e3/e5_table.md`](results/e3/e5_table.md), verified against
the file.

| | Localised | All blocks |
| --- | --- | --- |
| SA3 medium, data-scale CAA (5521-caption estimator) | 0.61 | 0.71 |
| SA3 medium, 50-pair CAA (paper estimator) | 0.52 | 0.53 |
| Paper, ACE-Step Table 1 (CAA) | 1.24 | 0.89 |

Localised versus all blocks on SA3 (data-scale): 13 percent lower
(paper ACE-Step: +39 percent).

### Per concept (data-scale CAA, ratio to PCI)

| Concept | PCI-all AUC | Localised / PCI | All / PCI | Outcome |
| --- | --- | --- | --- | --- |
| tempo | 0.060 | 1.03 | 1.15 | works: matches or beats the real prompt |
| mood | 0.009 | 1.96 | 2.14 | unreliable: the PCI denominator is tiny (0.015 one way, 0.003 the other) |
| piano | 0.035 | 0.47 | 0.44 | about half of PCI |
| violin | 0.036 | 0.24 | 0.44 | weak |
| acoustic/electric guitar | 0.024 | 0.27 | 0.43 | weak |
| rock genre | 0.053 | 0.45 | 0.46 | weak |
| electronic music | 0.024 | 0.51 | 0.66 | weak |
| vocal gender, vocal style | | | | not generable on SA3 (the model sings no vocals); excluded from every table, average and localisation |

## What was established

- The site is right: a linear probe on the time-averaged cross-attention
  output separates piano, tempo and mood at 0.96 to 0.98 accuracy.
- The estimator was the fixable part: the 50-pair CAA directions have cosine
  0.13 to 0.55 to the data-scale ones; the data-scale estimator raised the
  ratio from 0.52 to 0.61 (localised) and 0.53 to 0.71 (all blocks).
- Distillation is not the gap: the same 50-pair CAA on the SA3 medium base
  checkpoint (50 steps, guidance 7) gave tempo 1.07, piano -1.08, mood 0.05
  of PCI.
- A fixed additive vector carries about a third of what a K/V patch
  carries (same-pair difference vector recovers 0.16 of the concept, exact
  output substitution 0.30, K/V patch 0.47 to 0.65 on eq. 2).
- Localisation does not help on SA3.
- Zero strength is bit-identical to no steering, offline and live.

## SAE attempts

SAE steering on SA3 medium was attempted twice and never scored:

1. August 2026 (SAE2 lane): TopK 12288 x k32 dictionaries at blocks 17 and 22
   on SAME-L latent caches, held-out FVU 0.41, failed the quality gate.
2. October 3 to 4, 2026 (this record, `feat/tada-sae-sa3`): TADA's BatchTopK
   recipe on the paper grid at the cross-attention output, 5521 MusicCaps
   captions x 8 steps cached, blocks 3 and 5 trained. Held-out FVU 0.24 to
   0.41 (paper 0.22 to 0.24), but every configuration failed the per-sigma
   reconstruction gate at the last step (FVU 0.68 to 1.31 at sigma 0.27),
   because per-step token variance falls about 67x across the 8 steps.
   Training stopped there; blocks 6 and 7, the k=128 sweep and the SAE
   steering benchmark never ran. Sweep data: [`results/sae/sweep.json`](results/sae/sweep.json).

So: the dictionaries failed the reconstruction gate, and no SAE row exists.

## Verdict

On SA3 medium, activation knobs at the cross-attention output lose to prompt
interpolation for every concept except tempo. The metric is CLAP/MuQ-based
(MuQ primary, CLAP reported beside it, because CLAP barely separates SA3's
positive and negative prompts); there has been no listening study.

## Reproduce

All drivers run from the branch root with the DEMON venv (`P`) and the
reference evaluation environment (`E`, a separate venv with the
`steer-audio` evaluation code):

```bash
export PYTHONPATH=$PWD PYTHONUTF8=1 HF_HOME=D:/huggingface_cache TADA_ROOT=E:/Projects/tada-replication
P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe
E=/e/Projects/tada-replication/evalenv/Scripts/python.exe
R=scripts/tada/sa3_tada_run.py
S=scripts/tada/sa3_tada_score.py

# TensorRT cross-attention steering engine (live path only)
$P -m acestep.engine.trt.sa3_build --dit --steer-cross-attn

# Localisation (K/V patch sweep, MuQ eq. 2, tau 0.10)
$P $R patch --patch-site xattn_cond --pairs 34 --seeds 8 --batch 16
$E $S patch --metric muq --patch-dir patch
$P $R localize --metric muq --patch-site xattn_cond

# Data-scale CAA vectors (E5): 24-block per-step means over the caption cache, then vectors
$P scripts/tada/sa3_e5_capture.py --root E:/Projects/tada-replication --out D:/tada-replication/e5_means
$P scripts/tada/sa3_e5_vectors.py --means D:/tada-replication/e5_means --loc 3 5 6 7

# Calibration, sweeps, scoring and AUC: the exact E5 chain
bash docs/research/tada/results/steer_logs/sa3_e5_chain.sh

# SAE: cache, then train (paper grid)
$P scripts/tada/sa3_sae.py cache --blocks 3 5 6 7 --steps 8 --every 1 --batch 16 --work D:/tada-replication
$P scripts/tada/sa3_sae.py train --blocks 3 --epochs 10 --lr 3e-5 --work D:/tada-replication

# Listening package
$P scripts/tada/sa3_tada_listen.py tempo piano --sub eval_e5 --with-pci
```

Every chain that produced a number is copied verbatim under
[`results/steer_logs/`](results/steer_logs/) (`sa3_e3_*.sh`, `sa3_e4_chain.sh`,
`sa3_e5_chain.sh`, `sa3_base_chain.sh`, `sae_e3_*.sh`). They hard-code the
worktree path (`DEMON-tada-sa3`) and the data roots below; adjust `cd` to this
branch.

## Where the data lives (not in this branch)

| What | Where |
| --- | --- |
| Rendered audio, per-run protocol CSVs, vectors (`caa*`), calibration and eval sweeps | `E:\Projects\tada-replication\sa3\` (`eval_e3`, `eval_e5`, `eval_e5p`, `caa_e5`, ...) |
| Base-checkpoint control | `E:\Projects\tada-replication\sa3_base\` |
| Listening package | `E:\Projects\tada-replication\listen_sa3\` |
| Steering packs (live demo; 50-pair vectors at blocks 3, 6, 7) | `E:\Projects\tada-replication\packs\sa3\medium\` (load via `DEMON_STEERING_PACKS_DIR`) |
| E5 caption-cache means (3.3 GB), SAE activation cache (89 GB), SAE dictionaries | `D:\tada-replication\` (`e5_means`, `sae_cache`, `sae\sa3\block_*`) |
| Reference code, evaluation venv, paper PDF | `E:\Projects\tada-replication\steer-audio`, `evalenv`, `tada_2602.11910.pdf` |

## Results in this directory

- `results/e3/`: the E3/E5 table builders and outputs (`e5_table.md`,
  `e5_table.py`, `e3_table.py`), the SA3 results section drafts
  (`sa3_section.md`, `sa3_section_final.md`), vector cosine helpers.
- `results/e1/`, `results/e2/`: guidance and probe helpers from E1/E2.
- `results/sa3/`: every small JSON/CSV/TXT from the run root: `auc.json` per
  eval sub, `e3_table.json`, `e5_table.json`, calibrated ranges,
  localisation JSONs (MuQ, CLAP, paper mix, with and without vocals), vector
  metadata and cosines, skeptic checks, the zero check, E4 probe results.
- `results/sa3_base/`: base-checkpoint control AUC and ranges.
- `results/sae/`: SAE sweep (per-step held-out FVU, gate), cache and timing reports.
- `results/steer_logs/`: engine parity and discovery JSONs, and the run chains.

## Touches outside scripts/experiments

The five TADA branches sit on an early `feat/steering-generic` base
(4beb08d4), so this branch also carries that steering seam. Besides
`acestep/tada/` (new), `scripts/`, `tests/unit/` and `docs/`, it modifies:

- `acestep/engine/`: `diffusion.py`, `model_adapter.py`, `sa3_adapter.py`,
  `sa3_context.py`, `sa3_internals.py`, `sa3_trt.py`, `stream.py`,
  `trt/export.py`, `trt/sa3_build.py`; adds `sa3_tada.py`,
  `sa3_tada_tokens.py`, `trt/sa3_steering_onnx.py`.
- `acestep/paths.py`; `acestep/steering/layout.py`, `acestep/steering/packs.py`.
- `acestep/streaming/`: `ace_backend.py`, `diffusion_backend.py`,
  `families.py`, `knobs.py`, `sa3_backend.py`.
- `demos/sa3/` (`index.html`, `sa3.js`, `styles.css`): the STEER pedal.
- `docs/FAMILIES.md` (Steering section), `docs/STEERING.md`, `docs/TADA.md`.

Merge notes: `acestep/engine/trt/sa3_build.py` and `docs/FAMILIES.md`
conflicted with main's small-music/SAME-S work (#372); both sides were kept
(steer and steer-cross-attn builds stay medium-only; the SAME-L window build
keeps main's `args.model == "medium"` guard).

## Related work not in this branch

- `ryanontheinside/spike/steer-bench` (DEMON-steer-bench): the many-knobs
  steering bench built on this stack (`scripts/steering_bench/`: residual
  capture, ridge directions, catalogue, ship packs), with uncommitted work in
  that worktree. Its results are in `notes/steering_pr/` (git-ignored).

## Open

- Rebuild packs from the all-blocks data-scale vectors (`caa_e5`); the live
  packs still hold the earliest 50-pair vectors at blocks 3, 6, 7.
- Listening session via the STEER pedal on `demos/sa3` in
  `feat/steering-generic`.
