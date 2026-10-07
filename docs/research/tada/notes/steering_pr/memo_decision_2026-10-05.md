# Memo: the SA3 steering knobs, what they are, what the numbers and the papers say, and the options that survive

Date 2026-10-05. Written after reading discover.py, proxies.py, docs/STEERING.md, acestep/steering/policy.py (the
ACE lineage), runbook 16, the status and results files, and the main bodies of TADA (2602.11910), Facchiano
(2504.04479), SMITIN (2404.02252) and Sketch2Sound (2412.08550). Analysis CSVs: E:\Projects\steering-bench\2026-10-05\analysis\.

## 1. How the five knobs came to be

- May 2026, ACE-Step v1.5 "spectral control backend" (commit 7f92e8ff): eight axes defined by spectral metrics
  (brightness, warmth, roughness, density, attack, tonality, punch, bass_emphasis), one probe vector per
  (layer, step) cell from an HF bundle. Only four had a verified prompt-to-metric premise (PROMPT_BASELINE.md),
  so the auto knobs became steer_bright, steer_warm, steer_rough, steer_density (policy.py:62-110). Density's
  positive = sparse comes from there.
- 2026-10-03, runbook 16: the goal was to plug TADA-style steering into SA3 through a family-generic seam and
  "compute SA3 packs for the four axes the ACE demo has so the user can compare like for like, plus up to two
  more if cheap". discover.py reproduced the four ACE axes by prompt-pair difference of means and added
  percussive. Proxies were chosen because CLAP is not a repo dependency (discover.py:35-37), so each concept
  had to have an STFT statistic: centroid, low/high dB, flatness, onset rate, HPSS ratio.
- So the concept set was never chosen for musical coverage. It is the set of things that were cheap to measure
  in May, carried over twice. The knobs are spectral descriptors with prompt labels, not musical concepts.

## 2. What the measurements say (one seed, 50 prompts, ratio noise about 0.1)

- Every current knob beats prompt interpolation per unit LPAPS on its own descriptor (1.8x to 4.5x). Bright and
  warm move every genre the right way; percussive, density, rough do not.
- Effective dimensionality is about three, not five:
  - bright and warm are one axis: descriptor r = -0.88 (Pearson) at alpha 0, and the data-derived directions
    have cosine -0.95 to -0.99 at every block and step. Gram-Schmidt leaves warm with 23% of its norm and
    0.67 sd of effect (from 2.90), so you cannot orthogonalise them without losing the knob.
  - percussive and rough directions have cosine 0.89. Rough's flatness proxy runs opposite to the rough
    prompt, so rough has never been measured; its vector is in practice a second transient knob.
  - density is its own axis (onset rate |r| <= 0.31 with everything else) but sits at the wrong block
    (readout R² 0.04 to 0.19 at b1, 0.42 at b11-13; both estimators put its best block at 11) and is
    non-monotone on the positive side.
- Calibration: every knob overshoots the LPAPS cutoff at +-30 (bright by 5.3x / 4.0x, percussive 2.1x,
  density 1.6x / 2.2x, warm 1.6x, rough 1.1x / 1.5x). Monotone up to the cutoff; per-sign gains are in
  calibration_maps.csv. Asymmetries: percussive + is 0.58 of -, density + is 1.67x -.
- Re-estimation from descriptor-sorted self-generated renders (v1): bright improved on both the descriptor
  (4.45 to 5.74) and the independent MuQ score (1.09 to 1.68). For percussive and density the descriptor rose
  while MuQ fell, which is the circularity of sorting by the scoring descriptor. Seed-2 confirmation of bright
  is running on the box.
- MuQ cannot see "warm" at all (gate fails), so warm has no independent score.

## 3. What the papers say that bears on this (read, not summarised)

- TADA: CAA at cross-attention outputs, per-timestep vectors, localised to the "semantic bottleneck"
  ({7,8} on ACE-Step, {12,13,14} on Stable Audio Open). Localised CAA 0.104 MuQ AUC vs PCI 0.084 (1.24x);
  SAE features best at 0.118; Concept Sliders 0.086 and they LOSE from localisation. Human study: localised
  CAA tops "seamless edit". Our earlier SA3 replication found localisation gave no gain on SA3 and that the
  estimator (thousands of labelled captions) was the lever.
- Facchiano (MusicGen): DiffMean on the EOS hidden state, injected into BOTH CFG branches, one direction into
  ALL layers ("one-to-all") beat per-layer; mid layers 10-18 carry both tempo and timbre; bright +20% and dark
  +40% centroid at lambda 1.25; FAD climbs sharply past 1.5; 10 prompts already give the direction, 25
  saturate. Scored with centroid and BPM, not text similarity.
- SMITIN (MusicGen): per-head logistic probes (drums 94%, piano 75%), direction = probe weights scaled by
  projection std, soft head weights acc^3, intervene every 5 steps, and a self-monitor that switches the
  intervention off once the probes say the trait is present. Result: success 49.9% (plain ITI) to 23.3%, but
  FAD 0.420 to 0.336 and similarity 0.896 to 0.913. It buys fidelity by giving up effect. One minute of
  paired audio is enough to train usable probes.
- Sketch2Sound (Adobe): per-frame loudness, centroid (Hz to MIDI/127) and CREPE pitch at the latent frame
  rate, each through one linear projection added to the noisy latent; the WHOLE text-to-audio DiT is
  fine-tuned for 40k steps with 20% per-control dropout and random median filters. Centroid error 4.4 st vs
  10.4 text-only, but CLAP text adherence drops 0.273 to 0.211 and FAD 2.57 to 2.51 to 2.67. Centroid
  entangles room tone. This is a fine-tune, not an adapter.

## 4. Options that survive

A. Rebuild the knob set as an orthogonal basis of measurable axes, each with a validated scorer, estimated
   from descriptor-sorted self-generated renders at the block the sweep picks, calibrated per sign to the
   LPAPS cutoff. Candidate axes: spectral tilt (replaces bright + warm, or keep both names on one vector with
   opposite signs), transient share (percussive), event rate (density at b11-13), and new axes with scorers
   that exist: loudness/dynamics, attack hardness (AudioCommons), reverb/space, stereo width. Rough is
   dropped or redefined once a scorer exists (harmonic-to-noise ratio is the candidate). Cost: about 2 GPU
   hours on the box, most of the pieces are already running. Killed if the new basis does not beat the
   current packs on an INDEPENDENT score (MuQ or a held-out descriptor) by more than 0.1 ratio.
B. Closed-loop strength (SMITIN) on top of A: a linear readout at the hook block drives alpha to a target.
   Cost about 1 GPU hour. Killed if readout R² < 0.5 at the steerable blocks, or if no fidelity gain at
   matched effect. Expect it to trade effect for quality, as in the paper.
C. Trained time-varying descriptor conditioner (Sketch2Sound recipe) for tilt, loudness and event rate.
   Requires fine-tuning SA3 medium (the paper fine-tunes the whole model), a corpus with computed descriptor
   curves, 6 to 12 GPU hours, a new engine input, and it costs text adherence. Buys physical-unit,
   time-varying control that A cannot. Killed if its held-out centroid error is not below what A reaches at
   the cutoff, or if CLAP drops as in the paper. Only worth it if time-varying control is a product goal.
D. SAE feature steering at the production site (TADA's best method on ACE). Needs SAEs trained at the block
   output; our existing SAEs are at the cross-attention output, blocks 3,5,6,7. Cost: a day. Lower priority:
   on SA3 our replication showed the estimator, not the feature basis, was the gap.

## 5. Recommendation

A now, B as a cheap add-on once A's vectors exist, C only if time-varying control is wanted as a product
feature. Two decisions are the user's: whether bright and warm stay two knobs for the player's sake even
though they are one axis, and whether rough is dropped or redefined.
