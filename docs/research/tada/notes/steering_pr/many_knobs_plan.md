# Many knobs for SA3 medium: capture once, label many (plan, 2026-10-05)

Planner deliverables for many_knobs_master_plan.md. Inputs read: memo_decision_2026-10-05.md,
results_2026-10-05.md, box_status.md, directions_status.md, the capture and direction scripts in
DEMON-steer-bench, proxies.py in DEMON-steer, the sa3judge judges in instrument-controlnet\judging, the
MuseTimbre extension, the AudioSet label list and the MusicCaps CSV. Nothing was run on a GPU.

## Files (all in notes/steering_pr/)

| file | what |
|---|---|
| discriminators.md / discriminators.csv | 20 scorers, families, availability local vs box, cost per clip, output type, MuseTimbre notes |
| concept_catalogue.csv | 481 candidate knobs: primary label + sign, independent second scorer, third, label strength, class rule, population, screening set, expected tagged clips, difficulty, text anchors, blurb |
| catalogue_summary.json | counts by category and label strength |
| render_prompts.json | the 5000-prompt capture corpus, {id, prompt, seed, category, tags}; becomes $CAP/prompts.json |
| holdout_sfx_prompts.json | 20 sound-effect screening prompts, not in the corpus (the music holdout is the 20 TADA holdout prompts, also not in the corpus) |
| proto_prompts.json | 824 solo-instrument renders (103 x 8) used only as MuseTimbre prototypes |
| gen_catalogue_and_prompts.py | the deterministic generator for the four files above (seed 20261005); rerun to regenerate |
| many_knobs_runbook.md | phases 0-6 with GPU minutes, disk, acceptance rule, and the code-change checklist |

## Catalogue: 481 rows

| category | rows | descriptor (strongest) | classifier (strong) | text only (weakest, spot check) |
|---|---|---|---|---|
| timbre | 39 | 23 | 0 | 16 |
| dynamics | 14 | 10 | 0 | 4 |
| rhythm | 26 | 13 | 0 | 13 |
| space | 14 | 3 | 7 | 4 |
| production | 36 | 4 | 14 | 18 |
| instrument | 103 | 0 | 61 | 42 |
| genre | 86 | 0 | 49 | 37 |
| sound_effect | 78 | 0 | 72 | 6 |
| mood | 40 | 2 | 7 | 31 |
| abstract | 45 | 0 | 0 | 45 |
| total | 481 | 55 | 210 | 216 |

Sources: AudioSet ontology (instrument, genre, mood, sound-effect, room and production classes, names checked
against class_labels_indices.csv), the MusicCaps aspect vocabulary (234 aspects appear in >= 40 captions;
the musical ones are in the catalogue, vocal ones dropped), instrument lists, mixing vocabulary, moods, seasons,
places, eras and colours. Vocal and speech concepts are excluded: SA3 medium generates no vocals.

The second scorer is always from a different family than the primary (families in discriminators.md). Two
rows are provisional because only DSP can see them (wide_stereo, pan_motion; CLAP and MuQ are mono). Abstract
and other text-only rows need a human spot check before shipping.

## Render prompt set: 5000 prompts

- 2500 music: genre x 2-3 instruments x mood x one modifier (timbre, production, space, dynamics or rhythm),
  40% with a second genre, 70% with an abstract descriptor, 80% with a second modifier. Five phrasings.
- 300 solo-instrument, 1300 sound-effect scenes (2-3 sources + space, sometimes a production term; foley,
  ambience, nature, vehicles, impacts, machines, crowds, UI sounds), 300 hybrid (beat + sound effect), 200
  abstract-led soundscapes.
- 100 TADA test prompts and 300 MusicCaps captions (seeded pick among captions with no vocal words).
- Every slot draws the least-used vocabulary item, so tagged counts are balanced. Seeds: 30000 + id.
- Mean 5.9 tags per prompt.

Expected count per concept (generated tags only; classifier and text labels also score untagged clips):

| category | tagged clips per concept, min / median / max |
|---|---|
| instrument | 67 / 70 / 121 |
| genre | 49 / 52 / 93 |
| sound_effect | 46 / 47 / 50 |
| mood | 82 / 87 / 130 |
| abstract | 47 / 48 / 58 |
| production (sparse rows) | 48 / 49 / 386 |
| rhythm (text rows) | 47 / 50 / 100 |
| space (classifier/text rows) | 134 / 136 / 136 |
| timbre (text rows) | 35 / 39 / 61 (9 rows under 40: marked thin) |
| dynamics (text rows) | 72 / 83 / 85 |

Descriptor rows are dense: every clip has a value, so each quartile holds 925 clips (music population, 3700)
or 325 (sfx, 1300). Sparse rows build their direction from the top 250 by score against the bottom 25%, so
with 35 to 130 tagged clips per concept the positive class is about 15 to 50% prompt-tagged, the rest being
clips the scorer rates high anyway. The v1 self-label directions used 200 per class, and Facchiano found 25
prompts enough for a direction. Gate 0 (runbook phase 3) checks per concept that the label sees the tags.

## Discriminators: available vs missing

- On the box now: proxies, timbral_models (needs the shim), pyloudnorm, librosa (rhythm, tonal, spectral
  ports), CLAP music, MuQ-MuLan, audiobox aesthetics, LPAPS.
- Missing, needed: PaSST (hear21passt; box_status.md does not list it; AST via HF download is the zero-install
  fallback), CLAP general (630k-audioset-best, 1.86 GB), MuseTimbre encoder tensors, beat_this, pyarrow
  (check), demucs (check).
- Optional: essentia-tensorflow (lifts 37 text-only genres and many moods to classifier grade), basic-pitch.
  Skipped: YAMNet (redundant, needs TensorFlow), madmom (Windows-only venv locally; beat_this replaces it).
- MuseTimbre: C:\_dev\projects\private-audio-extension. Global 512-d fine-tuned CLAP audio embedding, no
  class head, no valid text side, trained on solo instruments only. It cannot reliably score instrument identity
  on a mix, so it is a third scorer for instruments via solo prototypes, never primary or sole second.

## Budget

About 1045 GPU-min (17.4 GPU-h): phase 1 13, phase 2 41, phase 5 450 (about 300 candidates after gate 0 and
dedupe, 1.5 GPU-min each), phase 6 540 (about 60 survivors, 9 GPU-min each). About 5 h wall on 4 GPUs.
RAM-disk peak about 15.0 GB of the 16 GB usable (19 GB free minus the 3 GB floor), with the corpus audio still
in trash. Margins and fallbacks are in the runbook.

## Decisions for the user (not made here)

1. Bright and warm are one axis (cos -0.95 to -0.99). Dedupe is sign-agnostic, so it will merge them into one
   cluster. Keep two names on one vector, or ship one knob?
2. Rough: the catalogue redefines it as AudioCommons roughness (timbral.roughness) with dissonance and
   flatness as separate rows. Drop the old flatness-based rough?
3. Corpus audio purge after labelling (frees 5.1 GB on the RAM disk): needed only if the 1 GB margin is lost.

## Notes on the master plan

- Directions at fp16 cost 0.59 MB per concept for all 24 blocks x 8 steps (0.28 GB for 481). The master plan's
  "0.1 GB per concept" is 170x too high, so there is no need to keep only 4 blocks.
- Screening here follows the requested 20 holdout x 5 alphas (about 1.5 GPU-min). The master timeline's
  12 x 3 at about 0.7 GPU-min halves phase 5 if time runs short.
- The 50 TADA test prompts used by phase 6 are in the corpus, as requested. Exclude them from the direction
  classes (runbook phase 3) so the protocol is not scored on prompts that helped build the vector.
