# G. Frontier capabilities DEMON does not have (2026-10-03)

Lane G of `notes/family_merge/15_frontier_gaps.md`. Web research only, no code changed, no GPU.
Judged against origin/main plus the family stack (`DEMON-fam-readme`, top of stack `36289ee`).
Builds on `model_scouting.md`, `transcription_options.md` and `muscriptor_plan.md`; it does not
repeat their picks (Beat This!, stems tap, BRAVE/RAVE/AFTER, DiffRhythm 2, HeartMuLa, JASCO, MuScriptor,
basic-pitch, SheetSage2Kern, LiveBand, 2606.24307 ACE streaming distillation), except to say where new
evidence changes them.

Every external claim carries a URL. "UNVERIFIED" means not confirmed at a primary source in this pass.
Items I re-checked myself at the primary page are marked (checked). Raw per-area notes with more rows
live in the session scratchpad and are not part of the deliverable.

## 0. Verdict

- The research frontier for live music has converged on two scheduling knobs DEMON already owns:
  **future visibility** (how far output leads or trails the input) and **commit size** (how much is
  frozen per step). The controlled study is Wu et al., arXiv 2510.22105 (2025-10-25, rev 2026-09-21):
  best coherence at future visibility of about +-0.2 s with 1 s chunks; truly causal output (leading
  the player) falls apart under supervised training alone. https://arxiv.org/abs/2510.22105
  DEMON's ring is a large-positive-commit system. That makes it great at steering and weak at
  *following a live player*, which is the biggest capability gap.
- The three to do first: **(1) tempo and beat lock** (beat-phase curve, click-in-context, tempo
  follow), **(2) concept steering at scale** (TADA-style localisation, SAE features, per-step
  schedules), **(3) live accompaniment inside the ring** (live input in context, frozen commit
  horizon, re-denoised tail). All three are training-free or nearly so on the first pass, and each
  has a one-to-two day experiment on one 5090.
- Hype check: there is still no open streaming video-to-music model, no spatial *music* generator, no
  open streaming singing synthesiser worth integrating, no browser timing for any music DiT, and the two
  most relevant streaming-diffusion papers (2606.24307 and LMDM 2605.22717) ship no weights.

## 1. What DEMON already approximates (so it is not counted as new)

Read from the family-stack README and code: per-frame `[T]` curves on every solver knob, prompt A/B
blend, LoRA hot-swap with TRT refit, latent-mask inpainting and repaint, x0-target morph, timbre and
structure reference audio, `write_audio` (live audio into the canvas,
`acestep/streaming/session.py:2486`), activation steering with four auto axes (`steer_bright`,
`steer_warm`, `steer_rough`, `steer_density` at probe layers 9/15/18, `acestep/steering/policy.py`),
key detection (`acestep/audio/key_detection.py`), one-shot MIDI transcription
(`acestep/analysis/midi_transcribe.py`), Mel-RoFormer stems on uploads (`docs/VOCALSTEM.md`), MIDI
learn, audio-reactive WebGL from output audio, MCP control bus, a plugin API (`docs/PLUGINS.md`), and
privately, control branches (melody, rhythm, chord chart, accompaniment stem) plus MuseTimbre instrument
transfer on SA3. MRT2 and MiniMax are text-only append-only streams in DEMON (`docs/FAMILIES.md`).

## 2. Survey by area

Seams: **ring** = Tier 2 ModelAdapter / StreamPipeline; **append** = Tier 1 GeneratorBackend;
**tap** = lookahead analysis tap (muscriptor_plan); **post** = post-decode or post-player effect;
**client** = SDK/browser; **bus** = MCP/control bus tool; **plugin** = model-extension plugin;
**steer** = the existing steering-vector slot. Effort: S = days, M = 1-3 weeks, L = a research project.

### 2.1 Live accompaniment and jamming

| Work | Date | Link | Licence, code, weights | Hardware, latency, how | DEMON read |
|---|---|---|---|---|---|
| Karchkhadze and Dubnov, "Towards Real-Time Human-AI Musical Co-Performance" | 2026-04-08 | https://arxiv.org/abs/2604.07612 ; https://github.com/karchkha/musical-accompaniment-ldm | MIT code, checkpoints on Zenodo incl. a consistency-distilled one (checked) | 257M U-Net LDM in Music2Latent space; 6 s window, 1.5 s hop, look-ahead mode w=1; 5-step diffusion 1398 ms per cycle on an RTX 2070, consistency-distilled 1-2 steps 362 ms (2070), 331 ms (M2); Max/MSP over OSC | **The ring with the live mix in the context.** Direct precedent for an "accompany the input" mode |
| ReaLJam (Google DeepMind / Magenta) | 2025-02-28, CHI EA 2025 | https://arxiv.org/abs/2502.21267 | no code or weights; paper CC BY 4.0 | MIDI melody in, chords out (ReaLchords RL agent). **Commit window** (default 4 beats) frozen, tail keeps revising; most responses under 100 ms; falling-chord UI shows the plan | Commit window plus adaptive tail is ring semantics: freeze near-playhead slots, re-denoise far ones |
| ReaLchords | ICML 2024 (arXiv 2025-06-17) | https://arxiv.org/abs/2506.14723 | no code found | RL plus distillation from a teacher that sees future melody, to make an online model behave when it must lead | Recipe if we ever train an accompaniment adapter for negative visibility |
| Aria / Aria-Duet (EleutherAI, QMUL) | 2025-11-03, NeurIPS 2025 Creative AI | https://arxiv.org/abs/2511.01663 ; https://github.com/EleutherAI/aria | Apache-2.0 code and weights (1B piano MIDI) | Turn-taking duet. Naive prefill 1-2 s to first note; **continuous KV prefill while the user plays** brings it to 100-200 ms; Apple Silicon | The AR-family answer to input latency (MRT2, MiniMax) |
| StreamMUSE "Real-Time LM Jamming" | 2026-06-10, RTAS 2026 | https://arxiv.org/abs/2606.11886 ; https://stream-muse-webpage.vercel.app/ | code per site (licence UNVERIFIED) | 0.12B symbolic AR; overlapping responses as jitter backup; client buffers tick-aligned to an external clock; RTT model vs generation length | Transport and clock design for the SDK |
| Magenta RT audio injection (MRT1) and MRT2 | 2025-08 / 2026-06-04 | https://arxiv.org/abs/2508.04651 ; https://magenta.withgoogle.com/magenta-realtime-2 | Apache-2.0 code; MRT2 weights terms UNVERIFIED | MRT1 mixes user audio into the next context (2 s chunks, several seconds lag); "Looper" mode. MRT2: 40 ms frames, about 200 ms control latency, MIDI note control; drums only on/off because direct drum control is infeasible at that latency | MRT2's MIDI note conditioning is unused in DEMON's text-only MRT2 family |
| Lyria RealTime API | current | https://ai.google.dev/gemini-api/docs/realtime-music-generation | closed | No audio input; BPM and scale changes need a context reset (official docs) | Tempo following is a weak spot of chunked AR systems; DEMON curves can beat it |

Seam: **ring** (live input via `write_audio` plus frozen commit horizon) for diffusion families;
**append** with continuous prefill for AR families. Effort M. Risk: coherence when output must lead
the player (2510.22105); input-to-response is bounded by ring depth. Genuinely new: yes as a product
mode; the building blocks (`write_audio`, a2a, private accompaniment branch) exist.

### 2.2 Tempo, beat and clock sync (symbolic and hybrid)

| Work | Date | Link | Licence, code | What | DEMON read |
|---|---|---|---|---|---|
| Silent Metronome (Bretz, Soydaner, Plaat) | 2026-09-07 | https://arxiv.org/abs/2609.07688 (checked) | no code stated | Strictly causal accompaniment conditioned on **beat phase and bar phase as periodic functions** plus tempo and meter; beat alignment 3.2x better than the causal baseline | A beat-phase channel is a per-frame curve. Needs a trained input, so it fits the private plugin path |
| STAGE (Sapienza, Sony) | 2025-04, ISMIR 2025 | https://arxiv.org/abs/2504.05690 ; https://github.com/giorgioskij/stage | CC BY-NC-ND 4.0 | Tempo lock by **summing a metronome click into the conditioning mix**, no tempo module | Zero-architecture lock: try it on ACE cover / SA3 a2a sources and structure refs today |
| Matchmaker (JKU) score following | 2025-10, ISMIR 2025 | https://arxiv.org/abs/2510.10087 ; https://github.com/pymatchmaker/matchmaker | code licence UNVERIFIED | Online DTW and HMM, audio or MIDI in; 0.07-3.6 ms per frame; median error 91-152 ms on piano sets | Position and tempo tracker when a player follows a known piece; feeds a tempo curve |
| LoopGen | 2025-04, ISMIR 2025 | https://arxiv.org/abs/2504.04466 | UNVERIFIED | Training-free circular context in MAGNeT, +55% seam consistency | See 2.6 loop mode |

Known from model_scouting: Beat This! (MIT) for the beat grid on the lookahead. Ableton Link: no
generative-model paper locks to Link directly; StreamMUSE's tick-aligned buffers are the nearest
design. Seam: **tap** (beat grid) plus **client** (Link/clock) plus **ring** (click-in-context
now, beat-phase curve later via **plugin**). Effort S for click-in-context, M for a trained phase
channel. New: yes. Memory records "non-bar-exact canvas kills phase lock" as the DEMON ONE root
cause, so this fixes a known pain.

### 2.3 Real-time control beyond text

| Work | Date | Link | Licence, code, weights | What | DEMON read |
|---|---|---|---|---|---|
| **TADA!** activation steering for audio diffusion | 2026-02-12, v3 2026-09-27 | https://arxiv.org/abs/2602.11910 ; https://github.com/luk-st/steer-audio | MIT (checked); steering vectors, SAEs (e.g. hookpoint `transformer_blocks.7.cross_attn`), concept-slider LoRAs, 21-concept prompt set on HF | Activation patching finds a narrow **semantic bottleneck** (ACE-Step cross-attn layers 7-8 of 24 per the v3 HTML, 6-7 per an earlier reading, so check indexing; SAO 11-12 of 24); CAA and SAE steering beat prompt, score and weight interventions | **steer** slot and LoRA hot-swap. DEMON has 4 axes; TADA gives the method to find the right layer and a library of concepts. Neither the paper nor the HF cards name the ACE checkpoint (v1 3.5B vs 1.5) (checked), so the released vectors may not load on 1.5; the method transfers |
| Learning interpretable features in audio latent spaces via SAEs | 2025-10 | https://arxiv.org/html/2510.23802v1 | UNVERIFIED | Pitch converges early in denoising, then timbre, loudness last | Argues for **per-step steering schedules**, which DEMON's per-step curves can express |
| MusicRFM | 2025-10, ICLR 2026 | https://arxiv.org/abs/2510.19127 | code URL UNVERIFIED | Probe-derived note/chord directions with time-varying injection; target-note accuracy 0.23 to 0.82 on MusicGen | Shows chord-level control via steering without a control branch; AR-only so far |
| SongEcho | 2026-02, ICLR 2026 | https://arxiv.org/abs/2602.19976 ; https://github.com/lsfhuihuiff/SongEcho_ICLR2026 | **Apache-2.0, checkpoints on HF** (checked) | 49M melody adapter on ACE-Step, 100 Hz vocal pitch via element-wise FiLM; base version not stated (likely v1, UNVERIFIED) | Public, permissive ACE melody curve |
| MuseControlLite | 2025-06, ICML 2025 | https://arxiv.org/abs/2506.18729 ; https://github.com/fundwotsai2001/MuseControlLite | MIT code, weights updated 2026-01 | 85M decoupled cross-attn with RoPE on time-varying melody, rhythm, dynamics, in/outpaint on SAO 1.0 | Reference design for SAO-lineage (SA3) control curves |
| Sketch2Sound (Adobe) | 2024-12, ICASSP 2025 | https://arxiv.org/abs/2412.08550 | no code or weights | Pitch, loudness, brightness curves plus text on a latent DiT; **random median filtering in training** so sloppy vocal imitations drive it | Recipe for voice-as-controller (2.5) |
| FreeSliders / MorphFader | 2025-10 / 2024-08 | https://arxiv.org/abs/2511.00103 ; https://arxiv.org/abs/2408.07260 | | Training-free sliders via extra prompt branches; morph via cross-attn Q/K/V interpolation | DEMON's prompt blend and guidance curves already cover most of this |
| Guitar tone morphing | 2025-10 | https://arxiv.org/abs/2510.07908 | | SLERP in latent space beats LoRA fine-tuning for morphs | Consistent with memory: lerp is off-manifold; SLERP is the cheap fix for any latent blend |
| DITTO-2 | 2024 | https://arxiv.org/abs/2405.20289 | | Inference-time noise optimisation for control | Per-request optimisation does not fit a continuously re-rendering ring. Skip |

Seam: **steer**, **ring**, **plugin**. New: partly. Steering exists with 4 axes; scale, layer
localisation, SAE features and step schedules are new. Melody/rhythm control exists privately on SA3;
a public permissive ACE melody path (SongEcho) does not.

### 2.4 Few-step and streaming generation for the ring

| Work | Date | Link | Licence, code, weights | Numbers | DEMON read |
|---|---|---|---|---|---|
| **Live Music Diffusion Models (LMDM), ARC-Forcing** | 2026-05-21 | https://arxiv.org/abs/2605.22717 ; https://github.com/ZacharyNovack/live-music-diffusion-models | MIT code, **no weights** (checked; dev code on request) | SAO / SAO-Small made block-causal with block-wise KV cache, then Self-Forcing rollouts plus the ARC discriminator; 2-8 steps, about 30 ms round trip on an RTX 6000 Pro Blackwell, runs on a gaming laptop; about 8 GPU-hours initial finetune (per the paper HTML, numbers UNVERIFIED by me) | An **append** SA3-lineage family, and a "generative delay" live effect. The cheapest published few-step post-training recipe for a music DiT |
| InfiniteAudio FIFO sampling | 2025-06-03, Interspeech 2025 | https://arxiv.org/abs/2506.03020 (checked) | training-free; no code link | Queue of frames at increasing noise; pop a clean frame each step | Ring with **per-frame** instead of per-slot timesteps; off the fixed canvas. Per-frame t is out of distribution for ACE/SA3 |
| RFLAV rolling flow matching | 2025-03 | https://arxiv.org/abs/2503.08307 | | Trained version of the same diagonal schedule | What a trained ring-without-canvas would look like |
| IntMeanFlow | 2025-10-09 | https://arxiv.org/abs/2510.07979 | code not found | MeanFlow targets from integrating the teacher velocity, no JVP; 1 NFE token-to-spectrogram | **Best fit for YuE2's token-conditioned acoustic stage** |
| MeanAudio | 2025-08, ACL 2026 | https://arxiv.org/abs/2508.06098 ; https://github.com/xiquan-li/MeanAudio | MIT, weights | 1 NFE, RTF 0.013 on a 3090; fine-tunes from an FM checkpoint | Needs JVPs; avoid at YuE2's 3.6B |
| AudioTurbo | 2025-05-28 | https://arxiv.org/abs/2505.22106 | UNVERIFIED | Reflow on cached teacher (noise, sample) pairs, 3 steps | Simplest one-GPU YuE2 recipe |
| Presto! (Adobe) | 2024-10, ICLR 2025 | https://arxiv.org/abs/2410.05167 | no code | DMD plus GAN plus **layer distillation**, 230 ms for 32 s mono | Layer dropping cuts tick cost, orthogonal to steps |
| SmoothCache | 2024-11 | https://arxiv.org/abs/2411.10510 | | Training-free layer-output reuse, shown on SAO | Little gain at 8 steps; worth a probe on YuE2's 32 |
| PTQ for audio DiTs | 2025-09-30 | https://arxiv.org/abs/2510.00313 | no code | SAO W8A8 and W4A8 with timestep-aware smoothing | Matches DEMON's fp8 finding (outlier front-end layers need precision) |

YuE2 facts at source: one 3.58B AR-NAR Mixture-of-Transformers whose experts share one attention
computation, NAR = flow matching on 25 Hz x 64 latents, CC BY-NC 4.0; the paper does not state the
step count or CFG (https://arxiv.org/abs/2609.33757 , https://huggingface.co/m-a-p/yue2-3b). Seam:
**ring** (a distilled student still drops in per step). New: yes; the README itself names this as
YuE2's missing piece.

### 2.5 Voice and singing

| Work | Date | Link | Licence | Numbers | DEMON read |
|---|---|---|---|---|---|
| MeanVC2 | 2026-06-08 | https://github.com/ASLP-lab/MeanVC2 ; https://huggingface.co/ASLP-lab/MeanVC2 | Apache-2.0 | 110 ms first packet, 40 ms chunks, RTF under 0.63 on one CPU core; singing not claimed | Best permissive live VC; **post** or pre-`write_audio` sidecar |
| RVC realtime | ongoing | https://huggingface.co/niel-blue/RVC-Realtime-GUI | MIT (UNVERIFIED) | 90-170 ms | De facto singing VC baseline, per-voice training |
| YingMusic-Singer-Plus | 2026-03-25 | https://arxiv.org/abs/2603.24589 ; https://github.com/ASLP-lab/YingMusic-Singer-Plus | CC BY 4.0 code and weights; VAE under Stability Community | Melody-preserving lyric replacement, 454M DiT on an SA2 VAE | Offline vocal-stem lyric swap |
| Vevo2 (Amphion) | 2025-08-22 | https://arxiv.org/abs/2508.16332 | CC BY-NC-ND 4.0 | Humming-to-singing, instrument-to-singing; AR plus FM, not streaming | Licence blocks it |
| JAM (declare-lab) | 2025-07 | https://arxiv.org/abs/2507.20880 ; https://huggingface.co/declare-lab/JAM-0.5 | non-commercial (Jamify + Stability terms) | 530M rectified-flow song model with **word-level timing JSON** | Ring-sized; word timing is the right primitive for a live lyric track; licence blocks shipping |
| ACE-Step flow-edit lyric editing | v1 README | https://github.com/ace-step/ACE-Step | Apache-2.0 | Training-free lyric edit preserving melody and accompaniment; whether 1.5 keeps it: UNVERIFIED | Live lyric swap as a ring op (two-condition velocity difference over a mask) |
| CSSinger chunkwise streaming SVS | AAAI 2025 | https://arxiv.org/abs/2412.08918 | code UNVERIFIED | no latency number in abstract | Not ready |

Voice-to-instrument: the cheapest route is not a conversion model but Sketch2Sound-style curves
(pitch, loudness, centroid, median-filtered) from the mic driving existing melody/rhythm control.
Seam: **client** (curve extraction) plus **ring** or **plugin**. New: yes.

### 2.6 Structure and editing live

- **Loop mode by canvas roll.** Mobius (video, 2025-02) shifts the cyclic latent every denoising step so
  the seam never settles: https://arxiv.org/abs/2502.20307 ; LoopGen is the token analogue. On a
  bidirectional DiT this is a near-zero-cost "bar-looped canvas" mode. UNTESTED on ACE/SA3; RoPE
  positions may need wraparound. Seam **ring**, effort S, new: yes.
- **Off the fixed canvas.** FIFO / rolling schedules (2.4). DEMON already has a validated free-running
  path (walk-mode chained repaint, memory), so this is partly approximated; FIFO is the per-tick
  alternative. Effort S to probe, L to make in-distribution.
- **Stems.** New evidence for model_scouting's stems pick: a stock offline separator loses under 1 dB
  with at least about 93 ms of lookahead and is within 0.12 dB of offline at 3 s, and chunked offline
  runs are about two orders of magnitude cheaper than per-hop streaming
  (Sechet, Evrard, Kowalski, 2026-09-28, https://arxiv.org/abs/2609.35397 (checked)). So: chunked
  Mel-RoFormer on each lookahead block; no causal separator needed. Causal separators sit near 5 dB SDR
  (RT-STT, https://arxiv.org/abs/2511.13146) against about 10 dB offline. Generating synchronised
  stems in one pass via shared noise (Stemphonic, https://arxiv.org/abs/2602.09891) has no code.
  No open, permissive 2025-2026 model edits one stem inside a mix (LaDA-Band is gated with no licence,
  https://github.com/Duoluoluos/TME-LaDA-Band).
- **Mixing and mastering.** The live-capable pattern is "network predicts console parameters, DSP
  renders": Diff-MST (https://arxiv.org/abs/2407.08889 , code CC BY-NC-SA), Diff2Mix 2026-08
  (https://arxiv.org/abs/2608.05442), AILive Mixer 2026-03 (https://arxiv.org/abs/2603.15995).
  The only open-weight generative mastering model found is SonicMaster (0.9B, Apache-2.0, no speed
  figures; https://huggingface.co/amaai-lab/SonicMaster). Seam **post** on stems. Low priority.
- **Section control.** YuE2's editable score (melody, chords, key, meter, form) re-run through the NAR
  only is section control (https://arxiv.org/abs/2609.33757); per-window prompts (SegTune,
  https://arxiv.org/abs/2510.18416) generalise DEMON's A/B blend to a per-frame prompt curve.

### 2.7 Symbolic and MIDI rendering

| Work | Date | Link | Licence | Numbers | DEMON read |
|---|---|---|---|---|---|
| SpanSynth-Edit (QMUL) | 2026-09-22 | https://arxiv.org/html/2609.25546v1 ; https://mimbres.github.io/spansynth-edit/ | checkpoints and code released, licence UNVERIFIED | 481M DiT on HeartCodec latents, region re-synthesis from edited MIDI, 0.93 s per 20.48 s on a GH200 | The closest MIDI-conditioned ring member; "edit the notes, re-render the bar" |
| MIDI-LLM | 2025-11 | https://arxiv.org/abs/2511.03942 | Llama 3.2 terms (UNVERIFIED) | 3-14x real time via vLLM | Plans bars ahead of the playhead; feeds a MIDI-conditioned renderer |
| Anticipatory Music Transformer | 2023/2024 | https://github.com/jthickstun/anticipation | Apache-2.0 | | Melody-to-accompaniment infilling in token space |

No 2025-2026 work runs a full live audio-to-MIDI-to-neural-audio accompaniment loop end to end; the
parts exist (MuScriptor tap, Aria/MIDI-LLM, SpanSynth-Edit or MRT2 note control). Seam: **append** or
**ring** family plus the **tap**. Effort L. New: yes, but heavy.

### 2.8 Cross-modal live

- **Visuals from model internals.** Nobody publishes a live visual layer driven by a music diffusion
  model's internals. Pieces exist: cross-attention projected onto a mel grid (AIBA,
  https://arxiv.org/abs/2509.20891), SAE concept features (https://arxiv.org/abs/2505.18186), an MIT
  thesis streaming attention weights during performance (https://dspace.mit.edu/handle/1721.1/164526).
  DEMON can emit per-tick numbers (slot progress, per-frame attention or steering-feature energy) as
  events for plain-canvas demos. Pairs naturally with Daydream Scope for video
  (https://huggingface.co/daydreamlive/scope). Seam **ring** hook plus **client**. Effort S. New: yes,
  and unique.
- **Gesture and dance.** Encypher (2026-09-16, https://arxiv.org/abs/2609.18062) drove Lyria RT from a
  depth camera; dancers said the 2 s delay broke cause and effect, and text BPM steering was unreliable.
  Gesture2Music (CVPR 2026, https://arxiv.org/abs/2511.00793): webcam to note events, 60-70 ms loop.
  Smith and Pardo, NIME 2026, generate from *predicted* gestures to hide latency
  (https://nime.org/proc/nime2026_153/). For DEMON: webcam or sensor to curves through the **bus** is
  cheap and needs no model work.
- **Video-to-music.** No streaming audio-level video-to-music model with a latency number was found.
  VidMuse (CVPR 2025) is CC BY-NC (https://github.com/ZeyueT/VidMuse). The practical path is video
  features to prompts and curves. Not now.
- **Spatial.** No spatial music generator found. Stereo upmix after decode is the viable path:
  ImmersiveFlow stereo to 7.1.4 (2026-01, https://arxiv.org/abs/2601.12950, no code stated), neural
  binaural upmix (DAFx24, https://audiolabs-erlangen.com/resources/2024-DAFx-Neural-Binaural-Upmix).
  No speed figures. Seam **post**. Not now.

### 2.9 Interaction, evaluation, on-device

- **Latency perception.** Note triggers: under 10 ms with under 1 ms jitter (Wessel and Wright,
  https://arxiv.org/abs/2010.01570); 20 ms, or 10 ms +- 3 ms jitter, is rated worse (Jack, Stockman,
  McPherson 2016, https://instrumentslab.org/data/andrew/jack_am2016.pdf). Continuous control without
  touch: JND 20-30 ms, and slow passages hid much larger delays (Maki-Patola and Hamalainen,
  https://users.aalto.fi/~hamalap5/publications/icmcarticlefinal10.pdf). A latency *change* on top of
  an existing delay is what people notice (Schmid et al., Audio Mostly 2024,
  https://epub.uni-regensburg.de/59003/). **No study measures acceptable latency for style or timbre
  knobs on a generative model.** Measured system numbers: Lyria RT 263 ms median, 818 ms p95
  key-to-audible (MazzikaAI, https://arxiv.org/html/2608.10360); MRT2 about 200 ms (official page).
  DEMON's 0.2-1.8 s sits inside this band, and the paper's thesis is responsiveness.
- **Evaluation protocols.** ReaLJam's n=6, one-hour, within-subject protocol is a ready template
  (https://arxiv.org/abs/2502.21267); PANEL is an open listening-test platform
  (https://arxiv.org/abs/2609.31392); TTM-Bench reports latency and RTF next to alignment
  (https://arxiv.org/abs/2609.18585).
- **On-device and browser.** ACE-Step 1.5 XL-Turbo has INT4/INT8 WebGPU ONNX exports with **no browser
  timings** (https://huggingface.co/emb1ter/ACE-Step-v1.5-XL-Turbo-ONNX-WebGPU); an MLX port claims 30 s
  in about 4 s on an unspecified M-series Mac
  (https://huggingface.co/ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit); the "aria" C runtime runs SA3
  small-music 10 s in 0.29 s on an RTX 3070 and fits a Raspberry Pi 5 at 4-bit
  (https://arxiv.org/abs/2607.08526 , https://github.com/matteospanio/aria , licence conflicting
  between README and paper). MRT2 is real time on Apple Silicon via MLX. These are platform questions
  for Lane H; none is a streaming ring in a browser.

## 3. Ranked shortlist: the 10 capabilities most worth adding

**1. Tempo and beat lock (click-in-context now, beat-phase curve later, tempo follow).**
Every live use (Ableton, a drummer, a DJ, a dancer) needs the canvas on a grid, and memory records the
non-bar-exact canvas as the root cause of DEMON ONE's lost phase lock. The frontier answers are cheap:
sum a click into the conditioning audio (STAGE) or condition on beat and bar phase as periodic signals
(Silent Metronome, 3.2x better beat alignment). Lyria RT needs a context reset for a BPM change;
DEMON's per-frame curves could move tempo smoothly, which would be a real differentiator. Pairs with
the Beat This! tap already planned. *First experiment (1 day, training-free):* on ACE cover and SA3
a2a, render a 60 s canvas with a click track at 90, 120 and 140 BPM summed into the source / structure
reference at three gains; run Beat This! on the output and score tempo error, beat F1 against the click
grid, and audible click leakage; compare against no click and against a text "120 BPM" prompt.

**2. Concept steering at scale (TADA method, SAE features, per-step schedules).**
DEMON already has a steering slot with four hand-picked axes. TADA (MIT, 2026) shows that a patching
sweep finds a narrow semantic bottleneck where difference-of-means vectors beat every other
intervention, and ships 21 concepts plus SAEs. The SAE paper shows pitch settles early and timbre late,
which DEMON's per-step curves can exploit. Training-free, lands in the existing slot, no wire change
beyond more knobs. Risk: released vectors are for an unnamed ACE checkpoint, so expect to recompute
on 1.5 and SA3. *First experiment (1-2 days):* port TADA's patching sweep to ACE 1.5 turbo and SA3
medium eager; compute CAA vectors for 6 concepts (tempo, piano, female vocal, mood, brightness,
distortion) at the found layers; compare CLAP delta and audio drift against DEMON's current layer
9/15/18 axes; then test an early-step-only versus late-step-only schedule for one pitch-like and one
timbre-like concept.

**3. Live accompaniment inside the ring ("the band follows you").**
The largest capability gap and the one the backing-track spike is circling. The precedent
(Karchkhadze and Dubnov, MIT, checkpoints) is literally the ring with the live mix in the context, at
362 ms per cycle on a 2070 after consistency distillation. ReaLJam's commit window maps onto ring
slots: freeze what is near the playhead, keep re-denoising the tail. DEMON has `write_audio`, SA3 a2a
and a private accompaniment branch. Risk: output must lead the player, which 2510.22105 shows hurts
coherence; expect to need a beat grid (item 1) to plan on. *First experiment (2 days):* play a
recorded guitar or bass stem into `write_audio` in real time as a live-input stand-in; drive the
private accompaniment branch (or ACE cover) with the incoming audio as context; sweep ring depth 1/2/4
and a commit offset of 1, 2 and 4 beats; log input-to-audible latency per bar and beat alignment of the
generated part to the input (Beat This! on both), and listen for chord agreement.

**4. YuE2 acoustic stage distilled from 32 to 4-8 steps.**
The README already says YuE2 "would work much better with a few-step turbo distillation", and its
32-tick solve is the knob-to-ear floor (2.4-4.0 s). IntMeanFlow reached 1 NFE in a token-to-spectrogram
setting of the same shape and needs no JVP; reflow on cached teacher pairs (AudioTurbo) is the simplest
fallback; LMDM shows a short ARC pass is affordable on one GPU. Train only the NAR-expert weights
because attention is shared with the AR stage. CC BY-NC is inherited, so this stays a research family.
*First experiment (2 days):* cache 300-500 32-step teacher solves (noise, 4 intermediate states, final
latent) from the YuE2 ring at a 30 s song; train a LoRA on NAR-only projections with a reflow loss for
a few thousand steps; evaluate 4 and 8 step students on mel distance and CLAP versus the teacher and
listen; measure knob-to-PCM latency in the ring.

**5. Voice and hum as the controller.**
Every musician has a voice; humming a line or beatboxing a groove is the most natural non-text control,
and the private melody and rhythm branches already accept per-frame signals. Sketch2Sound's trick
(median-filter the control curves in training so sloppy imitation still works) says the main risk is
whether those branches tolerate vocal input. MeanVC2 (Apache-2.0, 110 ms on CPU) covers the "re-voice
my vocal" half as a sidecar. *First experiment (1 day):* record 10 hummed melodies and 10 beatbox
grooves; extract pitch (RMVPE or CREPE), loudness and onset curves, median-filter at three widths, and
feed them to the SA3 melody and rhythm branches; score pitch-contour correlation and onset F1 of the
output against the input, and listen.

**6. A latency-perception study for generative control (and a constant-latency policy).**
DEMON's paper thesis is responsiveness, and the literature has no measurement of what latency is
acceptable for style, timbre or density knobs on a generative model; the closest evidence (continuous
control JND 20-30 ms, slow passages hide large delays, *changes* in latency are what people notice)
suggests constancy and preview may matter more than the absolute number. DEMON is uniquely able to run
this because ring depth is a hot dial. A small study would be first-of-kind and directly publishable.
*First experiment (2 days):* build a static demo page that randomises depth (1, 2, 4, 8) and an
artificial extra delay, with a "did that respond in time?" button and a preview-on/off condition (a
ReaLJam-style plan display of the pending curve); pilot with 3-5 people using PANEL-style logging.

**7. A model-internals side channel for visuals and inspection.**
Nobody ships visuals driven by a music diffusion model's internals; DEMON already computes them every
tick. Emitting a few numbers per tick (per-slot denoise progress, per-frame cross-attention energy,
steering-feature activations from item 2) makes demos show *what the model is doing*, which no audio-only
visualiser can, and pairs with Scope for video. Low risk, plain canvas, no wire break (unknown events
already reach hosts as JSON per muscriptor_plan). *First experiment (1 day):* hook the ACE decoder's
cross-attention in eager mode, reduce to a `[T]` energy per tick plus per-slot progress, emit as an
event at tick rate, and plot it in a canvas page next to the waveform; measure the tick-time cost.

**8. Loop mode and off-canvas generation (canvas roll, FIFO schedule).**
Loopers and live producers want bar-exact loops that keep evolving; the ring's fixed canvas is the
README's named limitation. Rolling the canvas by an offset every tick (Mobius / LoopGen idea) makes a
seamless loop on a bidirectional DiT at almost no cost; a FIFO per-frame schedule (InfiniteAudio) is the
training-free probe at a ring without a canvas. Risk: RoPE wraparound, and per-frame timesteps are out
of distribution. *First experiment (1 day):* on ACE turbo, render an 8-bar loop canvas with circular
padding and a per-tick roll of one bar; score seam discontinuity (spectral flux at the wrap) against a
plain render; separately, try a two-zone per-frame timestep (only if the adapter accepts per-token t;
if not, stop and record it).

**9. Public, permissive time-varying control on ACE (SongEcho) and the SAO design (MuseControlLite).**
The private SA3 control branches stay private; the public ACE family has no melody curve. SongEcho is
Apache-2.0 with weights, a 49M FiLM adapter driven by 100 Hz pitch, i.e. a control curve; MuseControlLite
is the MIT reference for melody, rhythm and dynamics on the SAO lineage. Risk: SongEcho likely targets
ACE v1, not 1.5, in which case it is a recipe rather than a checkpoint. *First experiment (1 day):*
load the SongEcho checkpoint, confirm its base model and tensor shapes against ACE 1.5; if it matches,
run it eager in the ring with a melody curve from a vocal stem and score pitch correlation; if it does
not, record the retrain cost (data, steps) from its README.

**10. Live input for the AR families (continuous prefill, MRT2 note control).**
MRT2 already supports MIDI note conditioning at about 200 ms and DEMON exposes it as text-only;
Aria-Duet shows continuous prefill cuts input-to-response from 1-2 s to 100-200 ms; MRT1's Looper mode
is a cheap input trick. This makes the append-only families respond to a player instead of only to
prompts. *First experiment (1 day):* in the MRT2 sidecar, pass a MIDI note stream from a file through
MRT2's note-conditioning input and measure note-on to audible latency and adherence; if the sidecar
protocol lacks a channel, add one locally (no commit) to measure.

Outside the ten, worth watching: LMDM block-causal SA3 (needs a training run, no weights), SpanSynth-Edit
as a MIDI-rendering family (licence unverified), lyric live-edit via ACE flow-edit plus a word-timing
track (licence-clean only via ACE itself), chunked stems on the lookahead (already ranked in
model_scouting; now with numbers), console-parameter mixing on stems.

## 4. Do these three first

1. **Tempo and beat lock (item 1).** Cheapest, training-free first step, fixes a recorded root cause,
   and every later live feature (accompaniment, loops, Ableton) plans on the grid.
2. **Concept steering at scale (item 2).** Training-free, lands in an existing slot, multiplies the
   knob count, and feeds item 7 for free.
3. **Live accompaniment in the ring (item 3).** The biggest product gap and the backing-track spike's
   core; do it after item 1 so the commit window can be expressed in beats.

## 5. Hype or not ready, plainly

- **ACE streaming consistency distillation (2606.24307)** and **LMDM (2605.22717)**: the two most
  relevant streaming papers ship no weights. Treat them as recipes.
- **WanSong v1.0**: report only, 25B, no weights (https://arxiv.org/abs/2607.14749).
- **Stemphonic**, **Sketch2Sound**, **ReaLJam**, **Presto!**: no code; ideas only.
- **Real-time singing synthesis**: no open streaming SVS worth integrating; Vevo2 (NC-ND) and JAM
  (non-commercial) are licence-blocked.
- **Causal source separators**: about 5 dB SDR, half offline quality, and unnecessary given the
  lookahead.
- **Generative mastering as a live stage**: SonicMaster is 0.9B with no speed figures; it would fight
  the generator for the GPU.
- **Video-to-music, spatial music generation, EEG-driven generation**: no streaming, open,
  latency-measured system exists; only control-to-prompt wrappers.
- **Browser or WebGPU music diffusion**: exports exist, timings do not. Do not claim a browser story
  until measured.
- **DITTO-style inference-time optimisation**: per-request optimisation does not fit a ring that
  re-renders continuously.
- **"Truly causal" accompaniment by supervised training alone**: 2510.22105 shows it degrades; RL or
  a planning grid is needed.
