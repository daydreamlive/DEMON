**The numerical result is credible; “the site is not a steering lever” is not.** The evidence supports a narrower statement: *these fixed additive vectors give small alignment gains at the chosen preservation budget, and this localization has not improved their performance.*

Several stronger claims in both the replication status and the skeptic’s correction overreach.

I reviewed the local implementation, reference clone, paper v3, and saved results read-only. I did not rerun GPU generation or modify files.

1. **Tempo already contradicts a blanket “not a lever” conclusion.** Its CAA MuQ AUC is about 0.050; the renormalized runs reach 0.069–0.075. Thus “all interventions stay below 0.04” also needs qualification.
2. **The E2 oracle is not an oracle for the evaluated prompts.** Its differences come from CAA training pairs; its additions are applied to benchmark prompts.
3. **Even the corrected, same-pair frozen difference is not a trajectory-adaptive oracle.** Earlier interventions change the inputs to subsequent attention calls.
4. **PCI’s AUC is a baseline, not a ceiling.** The prompt swap establishes the preservation budget; it does not mathematically upper-bound steering alignment.
5. **Thirty steps with remapped eight-step vectors does not rule out sampler effects.**
6. **The published 0.334 comparison is not matched.** Table 23 averages four concepts, whereas 0.011 averages nine. The reference SAO configs also expose discrepancies discussed below. [Paper v3, Table 23](https://arxiv.org/html/2602.11910v3#A12.T23)

The most decisive conceptual check is this. Write a cross-attention output as \(A(q,c)\). At the **current intervened query** \(q\), define

\[
d(q)=A(q,c_+)-A(q,c_0).
\]

Adding this at the output with strength one gives

\[
A(q,c_0)+d(q)=A(q,c_+).
\]

That is exactly the output produced by replacing the context at that call. With the same random draws, applying this identity throughout generation must reproduce the corresponding K/V-patched trajectory, up to numerical error. **A successful K/V intervention therefore implies an equivalent state-dependent additive intervention at its output.** What can fail is compressing that intervention into fixed vectors—not the output site’s causal accessibility.

The following is the failure-mode audit. “Checked” means supported by the inspected code or saved artifacts; “open” means the supplied experiments do not settle it.

**First, the measurements and comparisons need these distinctions.**

| Possible failure or overinterpretation | Evidence and the single decisive check |
|---|---|
| **Wrong AUC sign, baseline subtraction, integration axis, or cutoff interpolation** | **Checked for `eval50`.** I independently reconstructed every available MuQ/CLAP AUC from its CSV means and recorded cutoff; maximum discrepancy was \(1.84\times10^{-16}\). The reference integrates sign-corrected delta over LPAPS in increasing \(|\alpha|\): [auc.py:87](E:/Projects/tada-replication/steer-audio/src/steering/eval/auc.py:87), [auc.py:201](E:/Projects/tada-replication/steer-audio/src/steering/eval/auc.py:201). This rules out arithmetic disagreement, not invalid inputs. |
| **Missing strengths, duplicated alphas, incomplete scoring, or different sample counts** | **Checked for reviewed `eval50` sweep cells:** 21 distinct strengths, zero present, 50 alignment scores per strength. **Check for E1/E2:** run the same manifest-versus-CSV completeness audit on every reported cell. |
| **AUC curves backtrack and cancel area** | **Ruled out for reviewed `eval50` curves:** their LPAPS means are monotone in \(|\alpha|\). **Check elsewhere:** enumerate decreasing LPAPS segments before interpreting a small path integral. |
| **The sweep never reaches its cutoff** | **Ruled out for reviewed `eval50` cells.** Every curve crosses its recorded cutoff. However, some directions retain only three nonzero measured points below it. **Check grid adequacy:** insert midpoints below the cutoff and measure AUC convergence. |
| **Endpoint-only PCI scoring understates the cutoff** | **Open, and the assumed identity is already false in one inspected curve.** PCI-all piano negative peaks at \(-7\): 3.37530, versus 3.36829 at \(-8\). This small difference does not determine the current piano cutoff because PCI-loc is lower. **Check:** score every PCI switch length for both sites and recompute their directional maxima. The shortcut is explicit at [sa3_tada_score.py:172](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_score.py:172). |
| **The cutoff is inappropriate despite being computed correctly** | A preservation budget is a scientific choice. Correct integration does not prove that a larger useful effect is absent beyond it. **Check:** inspect alignment, preservation, and quality curves across the full range, with blind listening at several matched distortion levels. Tempo already improves beyond the cutoff. |
| **Thirty-step results use the wrong preservation reference** | Their eight-step cutoff answers “performance under the old budget,” not the native thirty-step protocol. **Check:** compute thirty-step PCI-all/loc curves using thirty-step baselines and rerun the comparison. |
| **The metric barely responds to an actual concept change** | A establishes useful MuQ sensitivity for tempo/piano and weak sensitivity for mood; CLAP responses are small. But weak prompt separation alone does not distinguish scorer insensitivity from weak generation. **Check:** blindly label the actual positive/negative audio, then measure scorer discrimination on those labels. |
| **“CLAP is uninterpretable” is too categorical** | A small but reproducible response can remain useful; the relevant issue is discrimination and uncertainty. Fusion CLAP used for localization also differs from music CLAP used for steering evaluation. **Check:** score the same human-labeled clips with both exact checkpoints and report paired confidence intervals. |
| **Wrong metric checkpoint, preprocessing, or text template** | Localization uses fusion CLAP; evaluation uses music CLAP and MuQ-MuLan with different templates. **Check:** pass identical PCM and query strings through an independently instantiated reference scorer and compare per-clip scores. See [sa3_tada_score.py:65](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_score.py:65), [editing/eval.py:39](E:/Projects/tada-replication/steer-audio/editing/eval.py:39), [metrics.py:83](E:/Projects/tada-replication/steer-audio/src/metrics/metrics.py:83). |
| **Sample-rate, channel, amplitude, clipping, or windowing errors** | Saving averages stereo channels, quantizes, and clips to int16. Checked baseline files are 44.1 kHz, ten seconds, mono, with a small nonzero clipping fraction. **Check:** score the same generated float audio before and after serialization, including stereo versus mono, across the strength range. [sa3_tada_run.py:93](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_run.py:93). |
| **Decreasing similarity to “fast” is mistaken for increasing “slow”** | Both directions currently use the same positive-concept query; the inspected CSVs confirm this. Reduced similarity can also mean noise, silence, or unrelated music. **Check:** jointly score positive, opposite, and nuisance descriptions on the same clips and validate direction with a concept-specific label. |
| **PCI and CAA start from different baseline prompts** | **Confirmed.** CAA receives the raw benchmark prompt; PCI adds phrases such as “a song” or “with instrument.” Their zero audio differs for tempo, piano, and mood. This need not violate the reference method, but weakens “CAA achieves X% of the prompt effect.” **Check:** compare both interventions starting from exactly the same prompt and noise. [sa3_tada_run.py:418](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_run.py:418), [sa3_tada_run.py:665](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_run.py:665). |
| **PCI is called an attainable ceiling** | **Incorrect inference.** Its path can be inefficient, its wording suboptimal, and its score lower than activation steering. The supplied ACE comparison itself demonstrates this. **Check:** compare CAA against several prompt formulations and a trajectory-adaptive positive control at matched distortion; treat PCI as a reference, never an upper bound. |
| **Cross-model AUC magnitude is treated as a calibrated effect size** | Raw area depends on both alignment response and LPAPS scale. **Check:** compare per-concept curves and within-model positive controls on a common evaluation design before interpreting the ratio \(0.011/0.334\). |
| **Concept/direction averaging conceals the actual comparison** | Recomputing the four SAO-table concepts gives SA3 CAA-loc MuQ **0.0163 averaged over directions**, or **0.00339 positive-only**, versus 0.0108 over nine concepts and both directions. These remain small, but are different estimands. **Check:** reconstruct the published comparator with exactly matched concepts and directions. |
| **Failure to reject is presented as evidence of absence** | There are no reported paired uncertainty intervals establishing a practical upper bound. Shared prompts, reused noise rows, and correlated frames also reduce independence. **Check:** perform a paired, clustered bootstrap that recomputes the entire curve and cutoff, using prompt and noise realization as sampling units. |
| **Flat aesthetics proves preservation** | It does not establish retained melody, identity, rhythm, or successful concept control. **Check:** blind comparisons at matched LPAPS that separately rate intended change and preservation of unrelated content. Onset density alone is not tempo, and a beat tracker alone is not a listening test. |

A and C therefore substantially reduce concern about a gross scoring failure, but they do not support the skeptic’s proposed “ceiling” interpretation.

**Second, localization can be real while the selected steering experiment is wrong.**

The implementation’s all-CLAP localization differs from the paper’s stated use of MuQ for non-vocal concepts and CLAP for vocal gender. The implementation is unambiguous: [sa3_tada_score.py:79](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_score.py:79), [sa3_tada_run.py:794](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_run.py:794). [Paper v3, §4](https://arxiv.org/html/2602.11910v3#S4)

| Possible failure or overinterpretation | Single decisive check |
|---|---|
| **Wrong localization scorer selects the wrong blocks** | Rescore the existing patch audio with the paper’s concept-specific metric assignment and recompute the selected set. No new generation is needed. |
| **Small or negative localization denominators create unstable ratios** | Bootstrap numerator and denominator jointly, report their raw values, and require reliable positive clean–corrupt separation before interpreting the ratio. [patching.py:39](C:/_dev/projects/DEMON-tada-sa3/acestep/tada/patching.py:39) only rejects exactly zero/NaN denominators. |
| **Flooring negative impacts biases aggregate importance upward** | Recompute localization with signed impacts and confidence intervals. The implementation floors at zero and does not cap at one; the ratio itself is not mathematically confined to \([0,1]\). |
| **Threshold 0.10 is unstable under sampling or concept selection** | Bootstrap prompt pairs and concepts, recording each block’s selection frequency across nearby thresholds. Thirty-two first-selected pairs and omitted concepts are not equivalent to full-data localization. |
| **A shared union contains counterproductive blocks for an individual concept** | On holdout prompts, compare each singleton and subset of `{3,6,7}` at matched LPAPS. A peak at block 6 does not prove blocks 3 and 7 are irrelevant, either. |
| **Single-block patching misses distributed interactions or redundancy** | Measure joint sufficiency and joint removal of the selected set on the same pairs. Single-block impacts cannot be added or interpreted as fractions of a causal pathway. |
| **K/V localization does not identify the best fixed-output-addition site** | Compare holdout steering performance using K/V-derived, output-patch-derived, and directly optimized layer selections. These are different interventions. |
| **Localization concepts do not cover evaluation concepts** | Localize the actual evaluated concepts, especially piano and the genre axes, using disjoint pairs, then test the resulting selection. |
| **The neutral control fully establishes concept specificity** | D is encouraging, but changes both the target metric and perturbation magnitude. **Check:** compare concept and neutral perturbations matched for baseline output distance, with the same scoring rule and held-out captions. |
| **Indexing or runtime block mapping is off by one** | Log the actual module names receiving a sentinel intervention and reconcile them with zero-based artifact indices. Do not compare displayed paper numbering directly to `tf` indices. |

The saved [output-patching localization](E:/Projects/tada-replication/sa3/localization_xattn_out.json) also selects `{3,6,7}`, with impacts approximately **0.135, 0.120, 0.127**. That supports output-site sensitivity; it is not evidence for zero causal influence.

**Third, hook correctness, cache correctness, and oracle correctness are separate questions.**

The inspected eager hook does target `cross_attn` output. The vendor adds that output into the residual through `cross_attn_scale`, which is identity for the stated configuration. The ONNX surgery finds the tensor leaving cross-attention and rewires its consumers through an addition. These observations make a gross wrong-site bug unlikely. [sa3_tada.py:85](C:/_dev/projects/DEMON-tada-sa3/acestep/engine/sa3_tada.py:85), [vendor transformer.py:1033](C:/Users/ryanf/.daydream-scope/models/demon/sa3/vendor/stable-audio-3/stable_audio_3/models/transformer.py:1033), [sa3_steering_onnx.py:122](C:/_dev/projects/DEMON-tada-sa3/acestep/engine/trt/sa3_steering_onnx.py:122).

| Possible failure or overinterpretation | Single decisive check |
|---|---|
| **Hook is before projection, after the residual, gated, scaled, or overwritten** | Capture the module output and actual residual-add operand in the same forward; verify that the injected difference arrives unchanged. Source inspection strongly supports this already. |
| **Only some rows are steered because CFG slicing was copied incorrectly** | Inject distinguishable row sentinels and verify every conditional row changes once. SA3’s offline path currently applies to all rows, unlike SAO’s conditional half-batch logic. |
| **Forward-call index is mistaken for sampler time** | Log `(generation, sampler step, actual t/sigma, model call, block, vector key)` and assert the intended mapping. The current counter ticks at block 0; repeated solver evaluations or auxiliary passes require different semantics. |
| **Cached activations refer to the wrong prompt, seed, batch row, duration, or checkpoint** | Regenerate a small cache from its full provenance and compare every selected tensor. B establishes exact cache reproduction for its tested tempo batch, not all artifacts. |
| **A resume silently mixes configurations** | Compare artifact hashes and complete configuration manifests for every alpha, not just directory names. Generation skips existing NPZs; scoring skips a directory when `lpaps.csv` exists. [sa3_tada_run.py:687](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_run.py:687), [sa3_tada_score.py:164](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_score.py:164). |
| **Matching integer seeds fails to match all random draws** | Record and replay initial noise **and every PINGPONG renoising tensor**, verifying row identity across conditions. Batch size changes alter the stream; the driver restarts the same seed for each batch. |
| **Oracle row number matches but prompt identity does not** | **Confirmed E2 flaw.** Construct positive/base/negative variants of each evaluated benchmark prompt itself. Training-pair extraction is at [sa3_tada_run.py:487](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_run.py:487); application is at [sa3_tada_run.py:638](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_run.py:638). |
| **Positive-minus-negative is applied to neutral and called exact** | Compute positive-minus-**actual-base** at the same query, and negative-minus-actual-base separately. A full positive–negative contrast added to a third condition has no reconstruction identity. |
| **Frozen oracle differences are invalidated by previous interventions** | Run the adaptive identity test \(A(q,c_0)+[A(q,c_+)-A(q,c_0)]\) at the current query and compare against context patching. B’s increasing divergence at later blocks/steps is expected for frozen differences. |
| **Same token index is mistaken for same semantic event** | Compare same-query contrasts with contrasts between independently evolved trajectories. Identical noise and tensor shapes do not align beats, instruments, or events after trajectories diverge. |
| **Memory tokens dilute or redirect the vector** | Recompute and apply vectors in a factorial memory-only/audio-only/all-token ablation. With 64 of 238 rows being memory, about **27%** of the mean comes from non-audio tokens; this cannot be dismissed as necessarily minor. The recorder averages every row: [target.py:123](C:/_dev/projects/DEMON-tada-sa3/acestep/tada/target.py:123). |
| **Padding or duration tokens corrupt K/V-site CAA** | Use the conditioner’s true token mask and explicit token-type boundaries, then recompute the K/V vector. The current `abs().sum()>0` mask is invalid for learned padding and includes the seconds token: [sa3_tada_run.py:599](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_run.py:599), [conditioners.py:38](C:/Users/ryanf/.daydream-scope/models/demon/sa3/vendor/stable-audio-3/stable_audio_3/models/conditioners.py:38). |
| **CAA/AUSteer reduction, axes, sign, or normalization is wrong** | Feed one saved activation batch to both implementations and compare per-step/per-layer vectors, norms, and selected coordinates. CAA normalizes each step/block separately; AUSteer globally selects coordinates across active blocks. [caa.py:55](C:/_dev/projects/DEMON-tada-sa3/acestep/tada/caa.py:55), [austeer.py:79](C:/_dev/projects/DEMON-tada-sa3/acestep/tada/austeer.py:79). |

The **corrected frozen oracle in B remains informative**, but only as a test of frozen trajectory transport. Its failure does not show that information is “lost” at the output site. Exact output replacement is also a transplantation from another trajectory; its poorer result than K/V patching can reflect incompatibility with the receiving residual state.

**Fourth, PINGPONG and the guidance experiments leave substantial gaps.**

The actual sampler computes

\[
\widehat{x}_0=x_t-t\,v(x_t,t),\qquad
x_{\text{next}}=(1-t_{\text{next}})\widehat{x}_0+
t_{\text{next}}\epsilon.
\]

At the same input and with shared noise, the immediate effect of a velocity perturbation is

\[
\delta x_{\text{next}}
=-(1-t_{\text{next}})t\,\delta v.
\]

Thus steps are not interchangeable, fresh noise can obscure an early perturbation, and later model calls may undo it. This is not the same accumulation process as taking more Euler steps. [sampling.py:312](C:/Users/ryanf/.daydream-scope/models/demon/sa3/vendor/stable-audio-3/stable_audio_3/inference/sampling.py:312)

| Possible failure or overinterpretation | Single decisive check |
|---|---|
| **Early steering is attenuated by renoising or corrected away later** | Apply one-step steering pulses while replaying noise; measure the effect immediately before renoising, immediately after, and at the final latent/audio. |
| **A useful intervention requires a particular time window** | Sweep actual-noise-level windows on holdout prompts, including late-only steering, at matched distortion. “All steps” can combine helpful and harmful effects. |
| **Eight-to-thirty mapping uses the wrong notion of time** | Recapture vectors at the exact thirty-step schedule and compare against the existing ordinal remapping. Current mapping is `floor(s*8/30)`, not sigma matching: [sa3_tada_run.py:575](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_run.py:575). |
| **More steps are outside the useful regime of a distilled model** | First establish that unsteered thirty-step generation and true-prompt controls retain quality and concept responsiveness. A degraded base sampler is not a valid steering control. |
| **PINGPONG specifically disfavors fixed additions** | Compare against a supported deterministic sampler with sampler-native vectors and successful unsteered/prompt controls. Merely changing the sampler flag without validating the base model would introduce another confound. |
| **Guidance merely rescales the same weak perturbation** | At fixed inputs, compare `g·[v(α/g)−v(0)]` across \(g\). If these agree, the experiment mostly reparameterizes alpha. The reported inverse-\(g\) range scaling makes this particularly plausible. |
| **Steering guidance is mistaken for restoring ordinary CFG** | Compare actual conditional-versus-null CFG, where supported, with the steered-versus-unsteered same-prompt guidance. Their base fields and trajectories differ. The implemented latter operation is at [sa3_tada.py:98](C:/_dev/projects/DEMON-tada-sa3/acestep/engine/sa3_tada.py:98). |
| **Renormalization was insufficiently tested** | Compare calibrated renorm-on/off curves on failing concepts, not only tempo, with identical held-out selection. Renorm changes direction as well as magnitude; expanding range fourfold does not exhaust that choice. |

**Fifth, numerical and export checks need to measure the intervention, not merely the large baseline output.**

| Possible failure | Single decisive check |
|---|---|
| **FP16 erases small additions or cancellation damages activation differences** | Compare native FP16 against an FP32 control for vector extraction, hook addition, and the resulting **steering delta** at representative strengths. Exact reproduction of an FP16 cache does not establish its adequacy. |
| **Renorm creates nonfinite values or inaccurate norms** | Record finite-value checks and pre/post token norms through the full strength range. The local renorm denominator lacks the reference’s clamp: [caa.py:92](C:/_dev/projects/DEMON-tada-sa3/acestep/tada/caa.py:92), [reference controller.py:257](E:/Projects/tada-replication/steer-audio/src/models/stable_audio/stable_audio_steering/controller.py:257). |
| **TRT ignores, miscasts, misindexes, or overwrites the steering input** | Compare \((v_{\rm steer}-v_0)_{\rm TRT}\) against the eager difference using per-block sentinels and several magnitudes. A 0.99986 cosine between total outputs can hide an inaccurate small delta. |
| **TRT cannot represent the tested intervention** | Check runtime input shape against the intended batch/token structure. The export uses `[1, blocks, hidden]`, broadcast across tokens and batch; it cannot represent the oracle’s per-pair/per-token tensor. [sa3_steering_onnx.py:202](C:/_dev/projects/DEMON-tada-sa3/acestep/engine/trt/sa3_steering_onnx.py:202). |
| **Wrong engine, stale bindings, CUDA-graph buffers, or live step state** | Alternate zero/sentinel/zero inputs and inspect actual bindings and output deltas for a complete session. Existing re-zero parity helps, but is not a full runtime-state audit. |
| **TRT is blamed for these offline AUCs** | **Ruled out by the inspected generation path:** `_load_sam` loads eager FP16. TRT concerns apply to deployment parity, not these reported offline tables. [sa3_tada_run.py:85](C:/_dev/projects/DEMON-tada-sa3/scripts/tada/sa3_tada_run.py:85). |

**Finally, several protocol and experimental-design differences remain.**

| Deviation or missing control | Single decisive check |
|---|---|
| **Paper/repository version mismatch** | Pin paper version, reference commit, artifact revisions, and configurations, then reconstruct one published cell. Earlier paper versions have materially different evaluation descriptions. |
| **Released SAO “localized” configuration disagrees with the claimed set** | Reconcile the artifact/config actually used for Table 23. The inspected config selects **`tf4tf11tf12tf13tf18`**, uses **100 steps, CFG 7**, and positive-only alphas: [eval_loc_tempo.yaml:1](E:/Projects/tada-replication/steer-audio/configs/steering/stable_audio/stable_audio_caa/eval_loc_tempo.yaml:1). Do not silently equate this with three-block localization. |
| **Renorm-off is incorrectly classified as a SAO replication error** | Inspect the instantiated SAO controller configuration. Its default is **false**, so renorm-off is consistent with the released SAO path inspected here. ACE settings and older paper equations must not be substituted indiscriminately. [method.py:52](E:/Projects/tada-replication/steer-audio/src/steering/methods/stable_audio_caa/method.py:52). |
| **Fifty first-listed prompts give a biased subset** | Evaluate the remaining fifty, reporting the two halves separately before pooling. This tests subset sensitivity without selecting a favorable replacement subset. |
| **AUSteer budgets were arbitrary rather than tuned for SA3** | Select budget per concept/site on holdout prompts, then evaluate once on test prompts. Fixed 1024/2048 budgets can suppress useful dimensions or admit noise. |
| **Training-pair wording poorly elicits the target on SA3** | Measure actual concept separation in the generated positive/negative extraction pairs, then compare vectors from SA3-effective, held-out prompt formulations. A vector can faithfully encode a poor contrast. |
| **Mean vectors cancel heterogeneous directions** | Compare split-half vector consistency and holdout steering from prompt-cluster-specific vectors. A failed global average does not establish absence of usable directions. |
| **Unit normalization gives noisy blocks equal influence** | Compare unit-normalized against confidence/norm-weighted vectors, selecting weights on holdout prompts. The existing normalization discards raw contrast magnitude. |
| **All/loc/ablated comparisons confound location with intervention count and energy** | Compare random matched-size block sets and matched-norm random directions at the same LPAPS budget. “All blocks also failed” does not exhaust layer weighting or interactions. |
| **CAA and AUSteer are treated as exhaustive tests of additive steering** | Fit a small regularized additive controller at the same output site on training prompts and test held-out transfer. Success would implicate vector estimation, not site accessibility. |
| **The entire replication pipeline lacks an end-to-end positive model control** | Reproduce a published SAO cell with its released model, vectors, inference settings, audio serialization, and scoring pipeline. CPU reconstruction of ACE result tables checks arithmetic, not generation or extraction. |
| **Further interventions were tried only on selected concepts** | Restrict the negative claim to those concepts, or replicate the chosen final configuration across all nine. Three-concept E1/E2 results cannot establish nine-concept universality. |

The paper’s shared benchmark uses 100 prompts and 15 strengths per direction; it also tunes AUSteer budgets per concept and configuration. Those are substantive differences from this reduced experiment, although none automatically explains its result. [Paper v3, Appendix I.2](https://arxiv.org/html/2602.11910v3#A9.SS2)

I would run these **three checks first**, incorporating rather than repeating the completed A–F work:

1. **Complete the matched positive-control evaluation.** Use identical base prompts and replayed noise for CAA, PCI, and direct positive/negative prompting; score every PCI switch point. Include human or validated concept labels. This resolves the remaining cutoff issue, the baseline mismatch, and whether low scorer response reflects the metric or the audio. AUC arithmetic itself is already settled.

2. **Run the adaptive output-difference identity test.** On the actual benchmark prompts, evaluate both contexts at each current query, add their difference, and require trajectory agreement with K/V patching. Alongside it, test correctly prompt-matched frozen differences. Adaptive success plus frozen failure would directly support “fixed-vector transport is weak.” Adaptive disagreement would expose an implementation, state, or RNG error.

3. **Run a sampler-native timing experiment.** Recapture vectors for eight and thirty steps, use each schedule’s own PCI calibration, and compare all-step versus late-window steering with replayed noise. Track the intervention before/after renoising and through subsequent denoising. This is substantially more diagnostic than increasing steps while reusing ordinally mapped vectors.

What would convince me is a **bounded negative**: successful metric and generation positive controls; adaptive-output/KV identity passing; implementation and precision checks passing; and, across held-out prompts and multiple noise realizations, well-calibrated fixed-vector methods showing no practically useful benefit over matched random controls or no localization advantage, with confidence intervals excluding a prespecified useful effect.

The current data can support **“small absolute gains and no demonstrated advantage from this localization under the tested protocol.”** They cannot support **“cross-attention output is not a steering lever,” “PINGPONG is ruled out,” “the oracle rules out better vectors,”** or **“PCI proves the attainable ceiling is 0.03.”**