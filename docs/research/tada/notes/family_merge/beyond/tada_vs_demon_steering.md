# TADA! (arXiv 2602.11910) vs DEMON activation steering: timeline and comparison

Compiled 2026-10-03. Read-only fact finding. No verdict on intent; facts only.

Sources: DEMON git history (local and origin), `gh pr list` on daydreamlive/DEMON, arXiv abstract and HTML pages for 2602.11910 v1 and v3 and for 2605.28657 v1, GitHub API for luk-st/steer-audio. The arXiv HTML pages were read through a summarising fetch tool, so the exact layer numbers quoted from the paper should be rechecked against the PDF before they are repeated anywhere.

## 1. Timeline (both sides interleaved)

| Date (UTC) | Side | Event | Public? |
|---|---|---|---|
| 2023-08 / 2023-12 | prior art | ActAdd (Turner et al., arXiv 2308.10248); CAA, "Steering Llama 2 via Contrastive Activation Addition" (Rimsky/Panickssery et al., arXiv 2312.06681). Difference-of-means steering vectors from contrastive prompt pairs. | yes |
| 2023-11 | prior art | Concept Sliders (Gandikota et al., arXiv 2311.12092), LoRA sliders for image diffusion. | yes |
| 2024 to 2025 | prior art | Localising knowledge in diffusion layers (Basu et al. 2024a/b); SAEs on diffusion (Staniszewski et al. 2025, same first author as TADA); CASteer (Gaintseva et al. 2025, activation steering in image diffusion); activation patching in music models (Facchiano et al. 2025). Years from TADA's own citations; check exact months before quoting. | yes |
| 2026-02-09 | TADA | GitHub repo luk-st/steer-audio created (GitHub API creation date; no code commits until May). | repo only |
| **2026-02-12** | **TADA** | **arXiv v1 posted.** Already contains activation patching for layer localisation, CAA difference-of-means vectors, SAEs, a LoRA concept-slider baseline; models AudioLDM2, Stable Audio Open, ACE-Step. | yes |
| 2026-05-18 | TADA | arXiv v2 posted (7.9 MB revision). | yes |
| 2026-05-21 | DEMON | First steering commits of any kind, local only: `502b05c1 exp: spectral experiment results`, `6a112ec2 exp: spectral experiment UI spike` (experiments/spectral_control: build_vectors, alpha_sweep, channel_baseline, SUMMARY.md). Exists only on the local branch `ryanontheinside/archive/spectral-control`, on no remote. | **no** |
| 2026-05-23 | DEMON | `c662a966 feat: fractional step formula and manual steering` (same local archive branch). | no |
| 2026-05-27 | TADA | First code commit to steer-audio: `8994048 code release`. | yes |
| 2026-05-27 | DEMON | DEMON paper arXiv 2605.28657 v1 posted. Does not describe activation steering (no steering vectors, probes or spectral axes; no CAA or TADA citation). | yes |
| **2026-05-28** | **DEMON** | **First public steering code: PR #162 "feat(engine): spectral control backend" and PR #163 "feat(rtmg): spectral steering"** opened on daydreamlive/DEMON (commits c0e7c7b0, 5a3c9cef). These two were never merged themselves. | yes |
| 2026-06-09 | DEMON | Steering lands on main via PR #232 (engine, `7f92e8ff`) and PR #233 (web UI, `07f87594`). `acestep/steering/` on main from this date. | yes |
| 2026-06-11 | DEMON | PR #246 (setup experience); steering vector bundles fetched from HF repo daydreamlive/demon-onnx under `steering_vectors/`. | yes |
| 2026-06-18 | DEMON | `67258cc1` moves knob copy into the client SDK; public tooltip text names vectors `brightness_l09_t3`, `warmth_l15_t0`, `density_l18_t3`. | yes |
| 2026-09-10 to 09-12 | TADA | Repo refactor ("code refactor with full reproduction") and README passes. | yes |
| 2026-09-27 | TADA | arXiv v3 posted. | yes |

Key ordering facts:

- TADA v1 (2026-02-12) predates DEMON's earliest steering commit of any kind (2026-05-21, local) by about 14 weeks. TADA v2 (2026-05-18) also predates it, by 3 days.
- DEMON's first public steering code (2026-05-28) came 15 weeks after TADA v1 and one day after TADA's code release.
- Only TADA v3 (2026-09-27) and the September repo refactor postdate DEMON's public steering.

## 2. Side-by-side technical comparison

| | DEMON (public main: acestep/steering, engine/stream.py) | TADA (v1 and v3) |
|---|---|---|
| Model | ACE-Step **v1.5**, 2B turbo DiT, 24 blocks (`v15-turbo`) | ACE-Step **v1** (github ace-step/ACE-Step), plus Stable Audio Open and AudioLDM2 |
| Hook point | Additive shift on the **post-block residual** of each `AceStepDiTLayer`, gated per row by denoise step | Hooks on attention, mainly **cross-attention** outputs (README: `transformer_blocks.7.cross_attn`) |
| Vector method | Difference of means over paired prompts per (layer, timestep) cell. The local SUMMARY.md says "Mean-diff" and notes that a linear probe or CAA was "not attempted". Code calls the cells "probe" cells. | CAA difference of means; also AUSteer (v3), SAEs with TopK and TF-IDF feature scoring, concept-slider LoRAs, prompt-level and score-space baselines (FreeSliders) |
| Layer selection | Grid over (layer, denoise step) ranked by mean-diff SNR, then alpha sweeps scored with DSP metrics (centroid, flatness, spectral tilt) and cross-axis drift. A "Phase-3 transfer" finding injects density 3 layers shallower than its probe layer. | Activation patching sweep: cache K/V from a target-concept prompt, patch per layer, score concept presence with CLAP/MuQ; impact threshold 0.10 |
| Layers | brightness probe l9, step 3; warmth l15, step 0 (sign flipped); roughness l9, step 3; density probe l18, step 3, injected at l15. Early local research also used l3 and l18. | ACE-Step: 2 of 24 blocks. v1 text reports {6, 7}, v3 text reports {7, 8} (possibly zero vs one indexing; README uses tf6/tf7). SAO {11, 12} in v1, {12, 13, 14} in v3. AudioLDM2 decoder layers. |
| Timestep handling | Per-step gating is central (each vector tied to a denoise step; fractional step formula for other step counts) | Not a headline feature in what was read |
| Concepts | Low-level timbre/spectral axes: brightness, warmth, roughness, density (manual-only: attack, tonality, punch, bass_emphasis) | Semantic musical concepts: vocal gender, tempo, mood, instruments (drums, flute, guitar, maracas, trumpet, violin, piano), genres (jazz, techno, reggae); v3 has a 99-concept benchmark |
| Evaluation | DSP metrics (centroid Hz, flatness, tilt dB/oct), monotonicity (Pearson/Spearman), cross-axis drift, comparison against per-channel latent guidance; listening by ear | CLAP/MuQ alignment delta vs LPAPS/FAD preservation (AUC), smoothness, Audiobox Aesthetics, user study |
| Real-time | Yes: live knobs (`steer_*`, manual slots `man_src/layer/step/alpha_N`) on the streaming ring-buffer engine, changeable mid-stream | Offline generation of 10 to 30 s clips; no real-time or streaming discussion |
| Artifacts | Vectors on HF daydreamlive/demon-onnx; code in the public repo | Vectors and SAEs on HF (lukasz-staniszewski collections); code at luk-st/steer-audio |

## 3. What each side cites

- TADA v1 and v3: no mention of DEMON, Daydream, StreamDiffusion, RyanOnTheInside, Fosdick or arXiv 2605.28657. No real-time or streaming steering work cited. None of the words brightness, warmth, roughness, density or spectral centroid appear. Cited steering and audio-control work includes Basu et al., Staniszewski et al. 2025, CASteer, Concept Sliders, Facchiano et al. 2025, FreeSliders (Ezra et al. 2025), Koo et al. 2025, Singh et al. 2026 and SteerMusic (Niu et al. 2026).
- TADA README: no mention of DEMON. It credits ACE-Step, DDPM inversion for audio, CASteer and Universal DiffSAE.
- DEMON paper 2605.28657: does not describe activation steering and does not cite TADA, CAA or any activation-steering work.
- DEMON README and main: steering appears as "spectral-control sliders" in an Experimental tab. No citation of TADA or CAA found.

## 4. Overlaps

- Both apply activation steering to ACE-Step-family diffusion transformers with 24 blocks.
- Both build steering vectors as a difference of means over contrastive prompt pairs (the CAA recipe, public since 2023).
- Both localise steering to a few layers and expose a scalar strength alpha.
- Both publish vectors on Hugging Face.

## 5. Differences

- Order: TADA's method, models and layer findings were on arXiv 14 weeks before DEMON's first steering commit, even counting the local one.
- Checkpoint: ACE-Step v1 (TADA) vs ACE-Step v1.5 turbo (DEMON).
- Hook point: cross-attention (TADA) vs post-block residual (DEMON).
- Localisation: activation patching scored by CLAP/MuQ (TADA) vs SNR grid over (layer, timestep) plus DSP-metric alpha sweeps (DEMON).
- Layers: TADA blocks 6/7 (or 7/8) vs DEMON 9, 15, 18 (density injected at 15).
- Concepts: semantic (instruments, genre, mood, tempo, vocal gender) vs low-level timbre (bright, warm, rough, dense). No axis names are shared.
- Scope: TADA adds SAEs, AUSteer, sliders and a benchmark. DEMON adds per-denoise-step gating and real-time, mid-stream control.

## 6. What is public vs private on the DEMON side

- Public: PRs #162/#163 (opened 2026-05-28), #232/#233 (merged 2026-06-09), `acestep/steering/`, stream.py hooks, web UI knobs, client-SDK copy naming probe cells, HF vector bundles.
- Local only: `experiments/spectral_control/` (SUMMARY.md, build_vectors, sweeps) on local branch `ryanontheinside/archive/spectral-control`. PROMPT_BASELINE.md and PHASE3_ANALYSIS.md, which policy.py comments reference, were not found on any remote branch checked. Private plugin work is not visible to outsiders.

## 7. Unknowns

- What exactly changed between TADA v2 (2026-05-18) and v3 (2026-09-27). v3 has no change log. Per the summary read, v3 adds AUSteer and a 99-concept benchmark, but the versions were not diffed line by line.
- Whether the v1 vs v3 layer numbers ({6, 7} vs {7, 8}) reflect an indexing change or a changed result.
- Whether any TADA author ever looked at the DEMON repo. There is no evidence either way: no citation and no shared naming.
- Exact publication months of the prior-art papers in row 3 of the timeline (taken from TADA's citations, not checked one by one).
- Public posts about DEMON steering before 2026-05-28: the repo references none.
