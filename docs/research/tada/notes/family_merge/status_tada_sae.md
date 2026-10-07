# Lane SAE status (TADA SAE features + Concept Slider LoRAs)

- 2026-10-03 worktree C:\_dev\projects\DEMON-tada-sae, branch ryanontheinside/feat/tada-sae at steering-generic 7bf6e260. No commits (nothing to port).
- Scoped directly from the source of truth (arXiv 2602.11910 PDF, github.com/luk-st/steer-audio, HF org lukasz-staniszewski) because DIGEST READY had not appeared yet; the paper and repo win over the digest anyway.
- STATE: SCOPED, TRAINING REQUIRED, NOTHING BUILT. GPU not used (lock held by steering-generic during this run).

## Verdict: released artifacts do NOT apply to our checkpoints
- Released SAEs: lukasz-staniszewski/ace-step-sae-tf6-cross-attn (m=2, k=32) and ace-step-sae-tf7-cross-attn (m=4, k=64), d_in = 2560, ACE-Step v1 only (repo tf6/tf7 are 0-indexed = paper blocks {7, 8}). Per-concept score tables ace-step-sae-scores-<concept> (tf6/tf7_scores.pkl) for all 9 concepts: piano, mood, tempo, vocal-gender, violin, vocal-style, guitar-electronic, rock-genre, electronic-music.
- Released Concept Sliders: ace-step-cs-<concept>-r{4,8,16}-{all,tf6tf7}, peft LoRA on ACE-Step v1 attention modules (to_q/to_k/to_v/to_out.0/add_*), Linear 2560.
- Our ACE is v1.5 (acestep-v15-*/config.json: hidden_size 2048, 24 layers, Qwen3-style DiT with AdaLN). Different width, different module names, different weights: v1 SAEs and LoRAs cannot load, and even a width match would not transfer (features are weight-specific).
- Stable Audio: no SAE and no Concept Slider artifact is released for Stable Audio Open (only stable-audio-caa-{piano,mood,tempo,vocal-gender}); SA3 medium is embed_dim 1536, depth 24. Nothing to port.
- Licences: repo MIT; HF model cards carry no licence field; MusicCaps captions CC BY-SA 4.0.

## What replication needs (per family, ACE v1.5 and SA3 medium)
Hook point (both methods for SAE): TADA's SAE reads and steers the CROSS-ATTENTION OUTPUT of the localized blocks (h_l = h_l + alpha * v_SAE added to the cross_attn module output, paper eq. 14), not the post-block residual. ACE v1.5: decoder.layers[i].cross_attn output, added ungated to the residual before the MLP (modeling_acestep_v15_turbo.py AceStepDiTLayer), so the shift is NOT equivalent to a post-block add (the MLP sees it). The family lanes must add a `cross_attn_output` hook to SteeringLayout and the TRT exports; SAE vectors then ride the existing [B, num_blocks, H] slot. Repo applies per-timestep vectors (top-k chosen per diffusion step from [num_timesteps, num_latents] score tables); the paper's eq. 13 is one vector. A per-step variant needs a pack-format extension (vector [T, H]); the single-vector form fits format 1 as is.
Block choice depends on the family lanes' patching sweep (SAEs are trained only on localized blocks).

SAE pipeline (paper I.1.4 + I.2, repo train_ace.py / scorer.py):
1. Cache cross_attn outputs at the localized blocks over MusicCaps captions (~5.5k), 30 Euler steps, CFG 5, 10 s, cache every 6th step (paper: every 5 of 30), cond pass only for training. Disk: ~5.5k x 5 steps x ~250 tokens (ACE 25 Hz x 10 s) x 2048 x fp16 = ~28 GB per hookpoint for ACE (SA3 smaller width, similar order). E: has 334 GB free.
2. Train BatchTopK SAE per block (lr 3e-5 linear, warmup 1000, auxk_alpha 1/32, effective batch 4096, 10-15 epochs), sweep m in {2,4,8,16}, k in {16,32,64}; pick by FVU/dead%/fire% (paper got FVU 0.216/0.243).
3. Score features per concept: TF-IDF eq. 12 over 50 pos / 50 neg prompts from the Table 8 templates (repo steer_prompts.py), 30 s, 30 steps, CFG 5, seed 10. Top-k_c per concept from {5,10,20,50,100,500} on a 20-prompt held-out set (paper winners for ACE v1 listed in I.2; must be re-swept for our models).
4. v_SAE = sum of W_dec rows of the top-k features (unit weights), write packs with hook cross_attn_output.
GPU estimate per family (5090, eager bf16, UNMEASURED, verify with one timed batch first): caching ~2-3 h (5.5k prompts x 60 forwards), SAE training sweep ~1-2 h, scoring 900 generations x 30 s ~1 h. About 4-6 GPU h per family.

Concept Sliders (paper I.2, repo train_concept_slider.py): LoRA on all attention projections of every block (and a localized variant), 500 iters AdamW lr 1e-4, eta 7, loss MSE(v_lora(x_t,c_pos), v(x_t,c_neut) + eta (v(x_t,c_pos) - v(x_t,c_uncond))), x_t from a random partial denoise (up to 50 steps, CFG 5), 10 s audio, 50 prompt pairs, ranks {4,8,16} swept per concept. Data-free (prompts only). Needs a checkpoint whose unconditional prediction is meaningful (v15-base, or confirm turbo handles empty text). Application is a weight LoRA, not the activation slot: it goes through DEMON's existing LoRA refit paths (ACE refit, SA3 refit) with LoRA scale as alpha, including negative scales (eval range about -0.32..0.35).
GPU estimate per family (UNMEASURED): ~55 forwards/iter x 500 iters = ~27k forwards per concept, ~15-25 min per concept, 9 concepts ~3 h at one rank, ~9 h with the rank sweep.

Total for both methods on both families: roughly 15-30 GPU hours, plus they block on the family lanes' localized blocks and on the cross_attn_output hook. That exceeds "a few GPU hours", so per the runbook nothing further is built.

## Cheapest first step if the user wants it
SAE on ACE v1.5 only, at the ACE lane's localized blocks, single (m,k) = paper's (2,32)/(4,64), 4 concepts with released HF CAA counterparts (piano, mood, tempo, vocal gender): ~3 GPU h after the ACE lane lands the cross_attn_output hook.
