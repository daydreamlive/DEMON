# Cross-check: instrument-controlnet synesthesia SAE lane (Aug-Sep 2026) vs TADA's SAE recipe

Sources: beyond/prior_sae_work.md (inventory of the prior lane, with file:line evidence) and
status_tada_sae.md (TADA recipe from the paper and repo). Both on SA3 medium is the comparison that
matters; the prior lane used sa3-medium-base, TADA ran on Stable Audio Open and ACE-Step v1.

## Where the two differ (these are the reasons to run it again)
| | prior lane | TADA |
|---|---|---|
| Hook site | residual-stream OUTPUT of trunk blocks 17 and 22 | cross-attention OUTPUT of the localized blocks (before the residual add) |
| Block choice | 17/22, picked from the activation-transplant layer profile (which layers carry content for grafting) | per-concept patching sweep for causal effect of text concepts; SAO gave blocks 11/12 of 24 (mid-trunk) |
| Training distribution | noised REAL audio (SAME-L caches) pushed through once under a NEUTRAL prompt, cfg 1; on-policy captures used only for the eval gate | activations cached DURING GENERATION from ~5.5k varied captions, conditional pass, every 5th or 6th of 30 steps: on-policy and concept-varied by construction |
| SAE | plain TopK, MSE, no AuxK, lr 3e-4, dict 8x/16x, k 32-128 | BatchTopK, AuxK 1/32, lr 3e-5 with warmup, expansion m in {2,4,8,16}, k in {16,32,64} |
| Dictionary quality | FVU 0.16-0.19 at k=128 (good) | FVU 0.216/0.243 (good). Not the differentiator |
| Concepts | timbral adjectives (boomy, warm, mellow, hard, heavy) and signal descriptors | semantic concepts carried by text (piano, mood, tempo, vocal gender, violin, guitar, genre) |
| Feature selection | word-vs-pool effect-size contrast, or ridge to descriptors ("correlates, not causes") | TF-IDF over 50 pos / 50 neg concept prompts, then k_c swept on a 20-prompt HELD-OUT set by the downstream steering metric |
| Steering | single decoder row scaled by p90 (sub-audible), pooled top-16 signed sum (0.35-0.62 std), or multiplicative edit-where-active (sign-inverted, destructive) | v = sum of W_dec rows of the top-k_c features (k_c up to 500), added at the cross-attn output with strength alpha, cond pass only, renorm; per-step feature choice in the repo |
| Metric | descriptor z-scores with a 3.0-std audibility floor | concept presence (CLAP / classifier based) against CAA as the baseline |

## Prior lane's dead-end list: which items TADA's recipe would repeat
Items 1-5 and 12 (sigma grid, training longer, relative-only gate, R2 labelling, browse coherence,
ops traps) are process lessons; TADA's recipe does not hit them, but the SAE lane must still add
an ABSOLUTE per-sigma-bucket FVU check (item 3, the k=192 blow-up) and avoid the fork-after-CUDA and
numpy/librosa traps (item 12). Items 6 (single decoder row), 8 (gain rescue sweeps on
word-linked sets) and 9 (word-vs-pool contrast selection) are not part of TADA. Item 7 (pooled
directions consistent but weak) is the closest to TADA's v_SAE; TADA's answer is a different site,
a concept-varied training distribution, semantic concepts, far larger k_c and alpha tuned on held-out
prompts. Item 10 (quote only audible effects) and item 11 (early steps carry authority) apply as
evaluation discipline: report their metric AND an ear check, and record which steps the vector was
active on.

## What is reusable from the prior lane
Lessons and gates, not code: the prior scripts hook the vendor SA3 trunk path and harvest real audio,
neither of which TADA does. DEMON's own generation path (the tada-ace lane's runtime capture through
StreamPipeline, generalized to SA3) is the natural base for caching during generation. The 63 GB of
residual-stream shards and the k128 L17/L22 dictionaries on B2 are the wrong site and the wrong
distribution for TADA; do not pull them. The SAME-L caches are not needed (TADA is prompt-driven).

## Verdict
Run TADA's SAE pipeline on SA3 medium exactly as published, at the blocks the SA3 patching sweep
localizes, trained on generation-time cross-attn outputs over varied captions, with held-out-validated
top-k_c selection and their metric, judged against the CAA vectors from the same sweep. Expected GPU
spend 4-6 hours on the 5090 (unmeasured; time one batch first). The user accepted this spend on
2026-10-03 on the condition that it follows TADA explicitly. If TADA's SAE does not beat CAA on
SA3, that is a result, not a failure of the lane: record it and ship CAA.
