# Discriminator roster for the many-knobs corpus (2026-10-05)

Machine-readable copy: discriminators.csv (same rows, plus local paths). Costs marked "estimate" are not
measured; the only measured numbers are proxies (1.5 CPU-s per 10 s clip) and render/capture (0.11 to 0.15
GPU-s per clip, from the 800-clip self-label capture). Column naming follows many_knobs_master_plan.md:
`<scorer>.<label>` in labels/<scorer>.parquet.

## Families and the independence rule

A knob needs two scorers from DIFFERENT families to agree. Families:

| family | scorers | why they are one family |
|---|---|---|
| DSP | proxies, timbral, level, rhythm, tonal, spectral, stereo, basicpitch | signal statistics of the same waveform; many are the same fact (centroid vs timbral.brightness vs spectral.tilt) |
| TAG | passt, panns, ast | all trained on AudioSet labels; PaSST is the one we run, the other two are fallbacks |
| LCLAP | clap_music, clap_general | same LAION architecture and contrastive objective, overlapping training data |
| MUQ | muq | MuQ-MuLan, music-text contrastive, different data and model from LAION |
| MT | musetimbre | CLAP audio tower fine-tuned for timbre; no text side |
| ABX | audiobox | aesthetics regressors |
| SEP | demucs | source separation energy shares |
| ESS | essentia (optional) | Discogs / MTG-Jamendo supervised heads |

Rules: the second scorer of a catalogue row is always in a different family from its primary. Where only
DSP can see the concept (stereo width, pan motion) the row is marked provisional (same family). Text scorers
(LCLAP, MUQ) are "weakest": a knob whose two agreeing scorers are both text gets grade B and needs a human
spot check.

## Roster

| scorer | labels | output | available locally | on the box (box_status.md) | cost / clip |
|---|---|---|---|---|---|
| proxies | centroid, lowhigh_db, flatness, onset_rate, perc_ratio (the five production descriptors) | 5 scalars | DEMON-steer scripts/steering/proxies.py (bench copy) | yes | 1.5 CPU-s measured |
| timbral | AudioCommons hardness, depth, brightness, roughness, warmth, sharpness, boominess, reverb | 8 scalars | no | yes in evalenv, needs /root/sb/tm_shim.py | ~10 CPU-s |
| level | LUFS, RMS, peak, crest, dynamic range (momentary p95-p10), loudness slope, momentary std, silence share, transient level, tremolo AM (4-8 Hz), beat-rate AM (sidechain), peak-to-LUFS, clip share | 13 scalars | pyloudnorm in instrument-controlnet\judging\.venv; definitions in sa3judge level.py / artifact.py | pyloudnorm yes | 0.3 CPU-s |
| rhythm | tempo, beat strength, pulse clarity, tempo stability, off-beat share, swing ratio, quarter-note kick periodicity, >6 kHz onset rate, SuperFlux rate, HPSS-percussive onset rate, tempo slope, bar-lag repetition | 12 scalars | librosa; beat_this 1.1.0 in DEMON\.venv; madmom only in a Windows venv (E:\Projects\instrument-controlnet\judging_runs\_env_madmom) | librosa yes; beat_this no (install); madmom no (skip) | 1.5 CPU-s + 0.05 GPU-s |
| tonal | major/minor score, key clarity, chroma entropy, pitch centroid (register), sensory dissonance, harmonic change rate | 6 scalars | sa3judge tempo_key.py (Krumhansl over CQT top-4) | port, librosa only | 1 CPU-s |
| spectral | tilt, band shares (sub, bass, low-mid, nasal, presence, 8-16 kHz, air), bandwidth, rolloff, flux, ZCR, HF flatness, tone persistence, noise floor, harmonic ratio | 16 scalars | sa3judge tone.py, artifact.py | port | 0.5 CPU-s |
| stereo | side/mid dB, L/R correlation, pan motion, HF side/mid | 4 scalars | sa3judge stereo.py | computed inline in capture (shards are mono) | ~0 |
| basicpitch (opt.) | note rate, polyphony, pitch range, mean note duration, mean pitch | 5 scalars | C:\datasets\musetimbre_spike_2026-10-02\venv | no | 0.5 CPU-s |
| passt | AudioSet 527 class probabilities | 527 probs | hear21passt in instrument-controlnet\.venv-eval | NOT listed: install (box-prep) | 0.02 GPU-s |
| panns (fallback) | AudioSet 527 | 527 probs | Cnn14_mAP=0.431.pth + class_labels_indices.csv in C:\Users\ryanf\panns_data | no | 0.01 GPU-s |
| ast (fallback) | AudioSet 527 | 527 probs | sa3judge identity.ast (HF MIT/ast-finetuned-audioset-10-10-0.4593) | downloadable (HF_TOKEN set) | 0.04 GPU-s |
| clap_music | text-anchor score per concept | 481 scalars | music_audioset_epoch_15_esc_90.14.pt (steer-audio res/, tada evalenv laion_clap) | yes | 0.02 GPU-s |
| clap_general | same anchors | 481 scalars | 630k-audioset-best.pt in instrument-controlnet\.venv-eval (1.86 GB) | NOT on box: 1.86 GB download into shm | 0.02 GPU-s |
| muq | same anchors | 481 scalars | MuQ-MuLan-large in tada evalenv | yes (/root/hf; transformers 4.47.1 evalsite) | 0.04 GPU-s |
| musetimbre | 512-d embedding + cosine to per-instrument solo prototypes | 512 + 103 | private-audio-extension (below) | ckpt not confirmed; upload encoder tensors only (~0.3 GB) | 0.03 GPU-s |
| audiobox | CE, CU, PC, PQ | 4 scalars | tada evalenv audiobox_aesthetics 0.0.4 | yes | 0.05 GPU-s |
| demucs | drums / bass / other / vocals energy share | 4 scalars | htdemucs in torch hub cache; demucs in judging\.venv | unknown, check (84 MB ckpt) | 0.15 GPU-s |
| essentia (opt.) | Discogs-EffNet 400 styles, MTG-Jamendo mood/theme, danceability | ~460 probs | no (only essentia.js in notes/toolpa) | no; pip essentia-tensorflow + ~0.1 GB models | 0.1 s |
| yamnet | AudioSet 521 | | no | no | SKIP (TF, redundant with PaSST) |

Available on the box today: proxies, timbral (with shim), level (pyloudnorm), librosa-based rhythm, tonal and
spectral, clap_music, muq, audiobox, LPAPS. Missing and needed: passt (or ast as the zero-install fallback),
clap_general, musetimbre encoder tensors, beat_this, pyarrow/parquet (check), demucs (check). Optional:
essentia-tensorflow, basic-pitch. Skipped: YAMNet, madmom.

## Text anchors (clap_music, clap_general, muq)

Each catalogue row carries pos_anchor and neg_anchor. Column value = cos(audio, pos) - cos(audio, neg).
Unipolar concepts (instruments, genres, sound effects) use a neutral negative ("music" or "silence"), so the
column is a relative salience score. Anchors are encoded once per scorer. MuQ already failed to see "warm"
(results_2026-10-05.md): every text column has to pass the label-validity gate (runbook phase 3, gate 0)
before any GPU time is spent on that concept.

## Which concepts each scorer can label (catalogue counts, primary label)

| scorer | primary for | rows |
|---|---|---|
| proxies | bright, warm, noisy, percussive, onset_rate (legacy density) | 5 |
| timbral | hard_attack, deep, timbral_brightness, rough, timbral_warmth, sharp, boomy, reverb | 8 |
| spectral, tonal, level, rhythm, stereo, basicpitch | band shares, tilt, register, dissonance, mode, key clarity, loudness, dynamics, tempo, pulse, swing, syncopation, density, width, pan | 42 |
| passt | 61 instruments, 49 genres, 7 moods, 72 sound effects, 7 rooms/spaces, 10 production classes | 206 |
| audiobox | production quality, complexity, enjoyment, usefulness | 4 |
| clap_music | text-only timbre words, rhythm feels, production terms, 42 instruments, 37 genres, 31 moods, 45 abstract | 210 |
| clap_general | 6 designed sound effects (laser, riser, impact, glitch, UI click, sparkle) | 6 |

## MuseTimbre

- Repo (private, read-only): C:\_dev\projects\private-audio-extension, branch feat/reference-demo (HEAD
  72d4b49 "MuseTimbre live demo page"). Second checkout C:\_dev\projects\private-audio-extension-jam.
- Encoder: scripts/musetimbre_precompute.py `fine_tuned_clap(ckpt, clap_ckpt, device)` builds the authors'
  `musetimbre.model.load_clap_timbre_encoder` (package at C:\datasets\musetimbre_spike_2026-10-02\repo,
  venv C:\datasets\musetimbre_spike_2026-10-02\venv) on the LAION-CLAP music checkpoint, then loads the
  `timbre_encoder.*` tensors of musetimbre_v1.pt over it. `forward_cls(wav_44k)` resamples 44.1 to 48 kHz and
  calls laion_clap `get_audio_embedding_from_data`, giving one global [B, 512] embedding. The adapter reads a
  reference as the loudest 5 s at -18 dB (scripts/musetimbre_metrics.py); the live service
  (scripts/musetimbre_condition_service.py, PUT /clap) uses the first 10 s.
- Checkpoints: E:\Projects\instrument-controlnet\musetimbre_spike_2026-10-02\ckpt\musetimbre_v1.pt (1.68 GB)
  and clap_music.pt (2.35 GB, byte-size identical to music_audioset_epoch_15_esc_90.14.pt, which the box
  already has). For the box, extract only the timbre_encoder.* tensors (~73 M params, ~0.3 GB fp32).
- Can it score instrument identity on a mix? Not reliably. It has no class head and no usable text side: the
  text tower was frozen while the audio tower was fine-tuned, so text zero-shot against it is invalid. It was
  trained only on single-instrument material (SoundFont-rendered MIDI stems and solo recordings, two disjoint
  5 s crops of one source as target and reference). On a mix it returns one vector dominated by the most salient
  source, and identity can only be read as cosine to solo-instrument prototypes, which is out of distribution.
  The MuseTimbre paper itself scores timbre match with PaSST family classification, not with this encoder.
- Use in this plan: third scorer for instrument rows only. Prototypes = mean embedding of 8 dedicated solo
  renders per instrument (runbook phase 1b, 824 clips). Score the full mix by default. Optionally score the
  demucs "other" stem (or the "bass" stem for bass instruments), which is closer to its training distribution.
  It is never a primary label and never the only second scorer.
