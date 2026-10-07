"""v2 rows for concept_catalogue_v2.csv (run: python gen_catalogue_and_prompts.py --v2).

Design rules (catalogue_v2_notes.md has the evidence):
* text rows use MINIMAL-PAIR anchors: pos and neg name the same instruments / setting and differ only in the
  target attribute, so cos(pos) - cos(neg) cancels genre, instrumentation and overall brightness (v1 text rows
  with a neutral neg such as "plain, neutral sound" scored texture salience and collapsed onto
  wall_of_sound / enjoyment / drum_machine);
* the second scorer is the DSP column that physically measures the attribute (a different family from
  CLAP; screen_knobs drops it if the corpus |corr| is weak and falls through to the MuQ third), never a MuQ
  anchor for a DSP concept MuQ cannot see (bass_heavy, warmth, muted failed that way);
* DSP primaries only where the v1-corpus check (|cos| to every v1 keeper < 0.8, split-half reliability
  >= 0.75) passed: restricted to the solo or sfx population, where the genre/drum-density confound is absent;
* redefinitions (category kept, name <v1>_r2) fix one diagnosed v1 failure each (column `blurb` says which).
"""
import collections
import csv
import json
import re

DSP = {"desc", "dyn", "timbral"}
POP_N = {"music": 3700, "sfx": 1300, "solo": 300, "all": 5000}
V2 = []


def _row(name, cat, primary, sign, second, third, pos, neg, blurb, diff, pop, kw, rule=None):
    V2.append(dict(name=name, category=cat, primary_label=primary, sign=sign, second_scorer=second,
                   third_scorer=third, pos_anchor=pos, neg_anchor=neg, blurb=blurb, difficulty=diff,
                   population=pop, screen_set="sfx" if pop == "sfx" else "music", kw=kw or [], rule=rule))


def T(name, cat, pos, neg, second="M", third="M", diff="medium", blurb="", pop="music", kw=None):
    """Text primary (clap_music; clap_general for sfx) with a minimal-pair anchor."""
    fam = "clap_general" if pop == "sfx" else "clap_music"
    sec = f"muq.{name}" if second == "M" else second
    thr = "" if third == "M" and sec.startswith("muq.") else (f"muq.{name}" if third == "M" else third)
    if pop == "sfx" and thr == f"muq.{name}":
        thr = ""                                     # MuQ is music-only (weak corr on every sfx row in v1)
    _row(name, cat, f"{fam}.{name}", "+", sec, thr, pos, neg,
         blurb or f"Minimal pair: '{pos}' vs '{neg}'.", diff, pop, kw)


def D(name, cat, primary, sign, pos, neg, third="M", diff="medium", blurb="", pop="music"):
    """DSP primary restricted to a population; second = the text anchor of the same concept."""
    fam = "clap_general" if pop == "sfx" else "clap_music"
    thr = f"muq.{name}" if third == "M" else third
    _row(name, cat, primary, sign, f"{fam}.{name}", thr, pos, neg, blurb, diff, pop, None,
         "quartile (top 25% vs bottom 25% of population)")


# ------------------------------------------------------------------ timbre (36)
TI = "timbre"
T("honky_mids", TI, "nasal, honky midrange tone", "smooth, scooped midrange tone", "desc.nasal_db")
T("boxy_tone", TI, "boxy, cardboard-like tone", "open, resonant tone", "desc.lowmid_db")
T("tinny", TI, "tinny sound from a small speaker", "full sound from large speakers", "desc.bass_db")
T("buzzy_reed", TI, "buzzy, reedy tone", "pure, flute-like tone", "timbral.roughness")
T("sine_pure", TI, "pure sine-like synth tones", "rich, harmonic-laden synth tones", "desc.flatness")
T("bell_partials", TI, "bell-like, chiming tones with inharmonic partials", "plain tones with simple harmonics", "dyn.key_strength")
T("brassy_blare", TI, "brassy, blaring tone", "mellow, covered tone", "desc.presence_db")
T("detuned_unison", TI, "detuned, slightly out-of-tune unison", "perfectly in-tune unison", "dyn.chroma_entropy")
T("tube_overdrive", TI, "gently overdriven tube amp tone", "clean, undistorted amp tone", "timbral.roughness", "passt.Distortion")
T("fizzy_highs", TI, "fizzy, harsh distorted top end", "smooth, rounded top end", "desc.hf_db")
T("felt_piano", TI, "soft felt-dampened piano", "bright concert grand piano", "desc.centroid", "passt.Piano")
T("square_hollow", TI, "hollow square-wave synth tone", "buzzy sawtooth synth tone", "desc.nasal_db")
T("fm_metallic", TI, "metallic FM synthesizer tones", "warm analog synthesizer tones", "timbral.sharpness")
T("band_limited", TI, "telephone-like band-limited sound", "full-bandwidth sound", "desc.bandwidth")
T("tape_wobble", TI, "wobbly tape with wow and flutter", "stable, steady pitch", "dyn.am_tremolo")
T("crystalline", TI, "crystalline, glassy high notes", "dull, woolly high notes", "desc.air_db")
T("tight_low_end", TI, "tight, controlled low end", "boomy, uncontrolled low end", "timbral.boominess")
T("thick_unison", TI, "thick, layered unison of many instruments", "a single thin instrument", "desc.bandwidth")
T("ringing_resonance", TI, "ringing, resonant overtones", "damped, controlled tone", "dyn.hp_ratio_db")
T("clicky_attack", TI, "clicky, sharp transient attack", "soft, rounded attack", "timbral.hardness")
T("acoustic_not_synth", TI, "acoustic instruments played by hand", "synthesized electronic instruments", "passt.Synthesizer")
T("rosin_bow", TI, "gritty bow-on-string rosin texture", "smooth, silky bowed tone", "timbral.roughness", "passt.Bowed string instrument")
T("breath_noise", TI, "audible breath noise on a wind instrument", "clean wind instrument tone", "desc.flatness")
T("hollow_body", TI, "hollow-body, airy acoustic resonance", "solid, dense electric tone", "timbral.depth")
T("nylon_soft", TI, "soft nylon-string guitar tone", "bright steel-string guitar tone", "desc.centroid", "passt.Guitar")
T("dark_cello", TI, "dark, low cello register", "bright, high violin register", "dyn.pitch_centroid", "passt.Bowed string instrument")
T("chiptune_pulse", TI, "8-bit chiptune pulse waves", "smooth orchestral strings", "desc.flatness")
T("growl_bass", TI, "growling, modulated synth bass", "clean, round synth bass", "timbral.roughness")
T("glassy_pad", TI, "glassy, digital synth pad", "warm, analog synth pad", "desc.air_db")
T("woody_body", TI, "woody, resonant wooden instrument body", "metallic, ringing metal instrument", "timbral.sharpness")
T("muted_horn", TI, "muted trumpet with a harmon mute", "open, bright trumpet", "desc.centroid", "passt.Brass instrument")
T("saturated_drums", TI, "saturated, crunchy drum sound", "clean, natural drum sound", "timbral.roughness", "passt.Drum kit")

T("hollow_flute_tone", TI, "hollow, breathy wooden flute", "bright, piercing metal flute", "desc.centroid")
T("glossy_sheen", TI, "glossy, polished high-end sheen", "raw, unpolished top end", "desc.air_db")
T("soft_mallets", TI, "soft mallets on a marimba", "hard mallets on a marimba", "timbral.hardness", "passt.Marimba, xylophone")
T("bowed_metal", TI, "bowed metal with an eerie singing resonance", "struck metal with a short clang", "dyn.hp_ratio_db")

# ------------------------------------------------------------------ articulation (31)
AR = "articulation"
T("legato_phrasing", AR, "smooth legato phrasing", "choppy, detached phrasing", "desc.perc_ratio")
T("pizzicato_strings", AR, "plucked pizzicato strings", "bowed arco strings", "desc.perc_ratio", "passt.Pizzicato")
T("tremolo_picking", AR, "rapid tremolo picking", "single sustained picked notes", "dyn.am_tremolo")
T("wide_vibrato", AR, "notes with wide vibrato", "straight notes without vibrato", "dyn.chroma_entropy")
T("palm_muted", AR, "palm-muted guitar chugs", "open, ringing guitar chords", "desc.centroid", "passt.Guitar")
T("glissando", AR, "sliding glissando between notes", "discrete, separate notes", "dyn.chroma_entropy")
T("pitch_bends", AR, "bent notes and pitch bends", "notes held at a fixed pitch", "dyn.key_strength")
T("trills_ornaments", AR, "fast trills and ornaments", "plain, unornamented melody", "dyn.onset_rate")
T("marcato", AR, "marcato, strongly accented notes", "smooth, unaccented notes", "dyn.peak_to_lufs")
T("portamento_lead", AR, "gliding portamento synth lead", "stepped synth lead", "dyn.chroma_entropy")
T("fingerpicked", AR, "fingerpicked guitar", "strummed guitar", "passt.Strum", "passt.Plucked string instrument")
T("string_harmonics", AR, "chiming natural harmonics on strings", "ordinary fretted notes", "desc.air_db")
T("spiccato", AR, "bouncing spiccato bow strokes", "long sustained bow strokes", "desc.perc_ratio", "passt.Bowed string instrument")
T("sustain_pedal", AR, "piano with the sustain pedal down, notes ringing", "dry, damped piano notes", "timbral.reverb", "passt.Piano")
T("rolled_chords", AR, "rolled, spread chords", "block chords struck together", "dyn.onset_rate")
T("octave_doubling", AR, "melody doubled in octaves", "single-octave melody line", "desc.bandwidth")
T("grace_notes", AR, "melody with grace notes and flourishes", "melody of plain notes", "dyn.onset_rate")
T("short_bass_notes", AR, "short, staccato bass notes", "long, sustained bass notes", "dyn.momentary_std")
T("tongued_flute", AR, "crisply tongued flute notes", "breathy, slurred flute notes", "timbral.hardness")
T("choked_cymbal", AR, "choked, short cymbal hits", "long, ringing cymbal crashes", "desc.hf_db", "passt.Cymbal")
T("damped_strings", AR, "damped, muted plucked strings", "open, ringing plucked strings", "desc.centroid")
T("virtuosic_runs", AR, "fast virtuosic runs", "slow, sustained melody", "dyn.onset_rate")
T("plucky_synth", AR, "plucky, short-decay synth notes", "sustained synth pad notes", "desc.perc_ratio")
T("chord_stabs", AR, "short rhythmic chord stabs", "long held chords", "dyn.momentary_std")
T("bowed_swells", AR, "bowed notes swelling in and out", "bowed notes at a steady level", "dyn.momentary_std")
T("hammered_notes", AR, "hammered, percussive piano attack", "gentle, soft piano touch", "timbral.hardness", "passt.Piano")
T("slurred_brass", AR, "slurred, connected brass phrases", "tongued, separated brass notes", "desc.perc_ratio")
T("let_ring", AR, "notes left to ring and overlap", "notes cut short with silence between", "dyn.silence_frac")

T("arco_tremolo", AR, "trembling string tremolo", "calm sustained strings", "dyn.am_tremolo", "passt.Bowed string instrument")
T("prepared_piano", AR, "piano with muted, prepared strings", "normal open piano strings", "desc.centroid", "passt.Piano")
T("pick_attack_bass", AR, "bass played with a sharp pick attack", "bass softly plucked with the fingers", "timbral.hardness", "passt.Bass guitar")

# ------------------------------------------------------------------ dynamics (25)
DY = "dynamics"
T("swelling_notes", DY, "notes swelling in volume", "notes at a steady volume", "dyn.momentary_std")
T("sudden_accents", DY, "sudden loud accents", "even, constant volume", "dyn.peak_to_lufs")
T("pianissimo", DY, "the band playing very softly, pianissimo", "the band playing very loudly, fortissimo", "dyn.lufs")
T("terraced_dynamics", DY, "abrupt switches between loud and soft sections", "one constant dynamic level", "dyn.lra")
T("gentle_touch", DY, "gentle, light touch on the instruments", "heavy, forceful playing", "timbral.hardness")
T("pumping_compression", DY, "pumping, breathing compression", "transparent, uncompressed sound", "dyn.am_tremolo")
T("squashed_drums", DY, "squashed, over-compressed drums", "open, dynamic drums", "dyn.crest_db")
T("long_decay", DY, "long, slowly decaying notes", "notes that stop abruptly", "dyn.silence_frac")
T("gradual_build", DY, "slowly building intensity", "steady, unchanging intensity", "dyn.loudness_slope")
T("expressive_dynamics", DY, "expressive, breathing dynamics", "mechanical, flat dynamics", "dyn.momentary_std")
T("explosive_hits", DY, "explosive, hard-hitting impacts", "soft, cushioned hits", "dyn.peak_to_lufs")
T("fade_away", DY, "music slowly fading away", "music getting steadily louder", "dyn.loudness_slope")
T("whisper_quiet", DY, "barely audible, whisper-quiet playing", "full-volume playing", "dyn.rms_db")
T("loud_dense_master", DY, "loud, dense modern master", "quiet, open vintage master", "dyn.peak_to_lufs")
D("solo_loudness", DY, "dyn.lufs", "+", "a solo instrument played loudly", "a solo instrument played softly", pop="solo",
  blurb="Loudness within solo prompts only (v1-corpus check: max |cos| 0.78 vs 1.00 to loud in music).")
D("solo_build", DY, "dyn.loudness_slope", "+", "a solo instrument getting louder", "a solo instrument getting quieter", pop="solo",
  blurb="Loudness trend within solo prompts (check: max |cos| 0.65, rel 0.85).")
D("solo_crest", DY, "dyn.crest_db", "+", "a solo instrument with sharp peaks", "a solo instrument at an even level", pop="solo",
  blurb="Crest factor within solo prompts (check: max |cos| 0.71, rel 0.80).")
D("solo_range", DY, "dyn.lra", "+", "a solo instrument with wide dynamic range", "a solo instrument at one level", pop="solo",
  blurb="Loudness range within solo prompts (check: max |cos| 0.78, rel 0.92).")
D("sfx_level", DY, "dyn.rms_db", "+", "loud sound effects", "quiet sound effects", "passt.Sound effect", pop="sfx",
  blurb="RMS level within sfx prompts (check: max |cos| 0.69, rel 0.95).")
D("sfx_swell", DY, "dyn.loudness_slope", "+", "a sound growing louder", "a sound dying away", "passt.Whoosh, swoosh, swish", pop="sfx",
  blurb="Loudness trend within sfx prompts (check: max |cos| 0.73, rel 0.82).")
D("sfx_level_motion", DY, "dyn.momentary_std", "+", "sounds with sudden level jumps", "sounds at a steady level", "passt.Bang", pop="sfx",
  blurb="Momentary-loudness variance within sfx (check: max |cos| 0.75, rel 0.96).")
D("sfx_peaky", DY, "dyn.peak_to_lufs", "+", "sharp, peaky impacts", "smooth, continuous sound", "passt.Thump, thud", pop="sfx",
  blurb="Peak-to-loudness within sfx (check: max |cos| 0.67, rel 0.95).")
D("sfx_crest", DY, "dyn.crest_db", "+", "transient-heavy sound effects", "dense, constant sound effects", "passt.Smash, crash", pop="sfx",
  blurb="Crest factor within sfx (check: max |cos| 0.75, rel 0.93).")
D("sfx_range", DY, "dyn.lra", "+", "sound effects alternating loud and quiet", "sound effects at one level", "passt.Sound effect", pop="sfx",
  blurb="Loudness range within sfx (check: max |cos| 0.77, rel 0.94).")

T("restrained_tension", DY, "quiet, restrained tension", "loud, full release", "dyn.lufs")

# ------------------------------------------------------------------ rhythm (32)
RH = "rhythm"
T("shuffle_hats", RH, "shuffled, swung hi-hats", "straight, quantized hi-hats", "dyn.offbeat_share", "passt.Hi-hat")
T("laid_back", RH, "laid-back, behind-the-beat groove", "pushing, ahead-of-the-beat groove", "dyn.offbeat_share")
T("quantized", RH, "tightly quantized, machine-precise timing", "loose, human timing", "dyn.tempo_stability")
T("rubato_free", RH, "free tempo, rubato phrasing", "strict metronomic tempo", "dyn.pulse_clarity")
T("double_time_hats", RH, "double-time hi-hats", "half-time hi-hats", "dyn.hf_onset_rate", "passt.Hi-hat")
T("sparse_kick", RH, "sparse kick drum hits", "a kick drum on every beat", "dyn.beat_strength", "passt.Bass drum")
T("breakbeat_feel", RH, "chopped breakbeat drums", "straight four-on-the-floor drums", "dyn.offbeat_share", "passt.Drum kit")
T("clave_pattern", RH, "clave rhythm pattern", "plain straight rhythm", "dyn.offbeat_share")
T("odd_meter", RH, "odd time signature, seven-eight", "even four-four meter", "dyn.pulse_clarity")
T("hihat_rolls", RH, "rapid hi-hat rolls and triplets", "steady eighth-note hi-hats", "dyn.hf_onset_rate", "passt.Hi-hat")
T("backbeat_snare", RH, "snare backbeat on two and four", "no backbeat, no snare", "dyn.beat_strength", "passt.Snare drum")
T("tom_patterns", RH, "pounding tom-tom patterns", "light cymbal patterns", "desc.lowmid_db", "passt.Drum")
T("brushed_drums_feel", RH, "soft brushed drums", "hard-hit drum sticks", "timbral.hardness", "passt.Snare drum")
T("stop_time", RH, "rhythmic stops and breaks", "a continuous, unbroken groove", "dyn.silence_frac")
T("driving_eighths", RH, "driving eighth-note bass line", "held whole-note bass", "dyn.onset_rate", "passt.Bass guitar")
T("syncopated_bass", RH, "syncopated bass line", "bass on the downbeat", "dyn.offbeat_share", "passt.Bass guitar")
T("locked_groove", RH, "tight, locked-in groove", "loose, disjointed groove", "dyn.tempo_stability")
T("pulsing_synth", RH, "pulsing eighth-note synth", "one sustained synth chord", "dyn.am_tremolo", "passt.Synthesizer")
T("beatless", RH, "beatless, free-floating texture", "the same texture over a steady beat", "dyn.pulse_clarity")
T("downbeat_accent", RH, "heavy downbeat accents", "evenly accented beats", "dyn.peak_to_lufs")
T("ghost_notes", RH, "snare ghost notes between the hits", "plain snare hits only", "dyn.hf_onset_rate", "passt.Snare drum")
T("faster_groove", RH, "the groove played at a fast tempo", "the groove played at a slow tempo", "dyn.tempo")
T("dotted_rhythm", RH, "dotted, long-short rhythms", "even, equal note lengths", "dyn.offbeat_share")
T("call_response", RH, "call-and-response phrases", "one continuous phrase", "dyn.silence_frac")
T("handclap_groove", RH, "groove driven by handclaps", "groove driven by a kick drum", "dyn.hf_onset_rate", "passt.Clapping")
T("shaker_pulse", RH, "steady shaker sixteenth notes", "no shaker, open space", "dyn.hf_onset_rate", "passt.Maraca")
D("solo_note_rate", RH, "desc.onset_rate", "+", "a solo instrument playing many fast notes", "a solo instrument playing few slow notes",
  pop="solo", blurb="Note density within solo prompts (check: max |cos| 0.76 vs 0.89 to drum_machine in music).")
D("sfx_event_rate", RH, "dyn.onset_rate", "+", "many rapid sound events", "a few isolated sound events", "passt.Sound effect", pop="sfx",
  blurb="Event rate within sfx (check: max |cos| 0.68, rel 0.88).")
D("sfx_regular", RH, "dyn.tempo_stability", "+", "regular, repeating sound events", "irregular, random sound events", "passt.Tick", pop="sfx",
  blurb="Regularity of sound events within sfx (check: max |cos| 0.67, rel 0.82).")
D("sfx_flutter", RH, "dyn.am_tremolo", "+", "fluttering, pulsing sound", "steady, smooth sound", "passt.Vibration", pop="sfx",
  blurb="4-8 Hz amplitude modulation within sfx (check: max |cos| 0.69, rel 0.85).")

T("sixteenth_funk", RH, "tight sixteenth-note funk guitar", "slow strummed guitar chords", "dyn.hf_onset_rate", "passt.Guitar")
T("rhythmic_gate", RH, "rhythmically gated, chopped pad", "smooth, continuous pad", "dyn.am_tremolo")

# ------------------------------------------------------------------ space (26)
SP = "space"
T("vocal_booth_dry", SP, "dry sound recorded in a small booth", "the same sound in a large reverberant hall", "timbral.reverb", "passt.Inside, small room")
T("plate_reverb", SP, "bright plate reverb", "dark room reverb", "desc.air_db", "passt.Reverberation")
T("long_tail", SP, "long reverb tail", "short room ambience", "timbral.reverb", "passt.Reverberation")
T("stadium_ambience", SP, "huge stadium ambience", "intimate small room ambience", "timbral.reverb", "passt.Inside, large room or hall")
T("tape_echo", SP, "repeating tape echo", "no echo at all", "passt.Echo")
T("through_wall", SP, "music heard through a wall", "music in the same room", "desc.rolloff85")
T("open_air", SP, "open-air outdoor acoustics", "indoor room acoustics", "passt.Outside, rural or natural")
T("stone_church", SP, "stone church acoustics", "dry studio acoustics", "timbral.reverb", "passt.Inside, large room or hall")
T("foreground_background", SP, "melody in front, accompaniment far behind", "every part at the same distance", "desc.presence_db")
T("bedroom_intimate", SP, "intimate bedroom recording", "big studio production", "timbral.reverb")
T("in_ear_close", SP, "very close, in-your-ear sound", "sound from across the room", "desc.presence_db")
T("roomy_drums", SP, "roomy, ambient drum sound", "tight, close-miked drums", "timbral.reverb", "passt.Drum kit")
T("gated_reverb", SP, "gated reverb on the snare", "natural, decaying reverb", "dyn.silence_frac")
T("reverse_reverb", SP, "reverse reverb swells", "normal reverb", "dyn.loudness_slope")
T("spring_reverb", SP, "twangy spring reverb", "clean, dry sound", "timbral.reverb")
T("shimmer_reverb", SP, "ethereal shimmer reverb", "plain, dry sound", "desc.air_db")
T("dub_space_echo", SP, "dub delay throws and space echo", "tight, dry mix", "passt.Echo")
T("canyon_echo", SP, "echoing across a canyon", "an acoustically dead room", "passt.Echo", pop="all")
T("sfx_far_away", SP, "a sound far away in the distance", "the same sound right next to the microphone", "desc.presence_db",
  "passt.Outside, rural or natural", pop="sfx")
T("sfx_indoor_hall", SP, "sounds echoing in a large hall", "the same sounds outdoors in open air", "timbral.reverb",
  "passt.Inside, large room or hall", pop="sfx")
D("sfx_reverb", SP, "timbral.reverb", "+", "reverberant sound effects", "dry sound effects", "passt.Reverberation", pop="sfx",
  blurb="Reverb within sfx (v1 reverb had too few clips in music; sfx check: max |cos| 0.79, rel 0.83).")
D("sfx_sub_weight", SP, "desc.sub_db", "+", "sound effects with deep sub-bass weight", "light sound effects without low end",
  "passt.Rumble", pop="sfx", blurb="Sub-60 Hz share within sfx (check: max |cos| 0.73, rel 0.95).")

T("car_radio", SP, "music playing on a car radio", "the same music on studio monitors", "desc.bandwidth")
T("room_tone_air", SP, "quiet room tone and air around the instruments", "a silent, sterile background", "desc.floor_db")
T("sfx_tunnel", SP, "sounds in a long concrete tunnel", "the same sounds in an open field", "timbral.reverb", "passt.Echo", pop="sfx")
T("sfx_close_foley", SP, "close-up foley detail", "a distant ambient recording", "desc.presence_db", pop="sfx")

# ------------------------------------------------------------------ redefinitions of v1 failures (category kept)
RD = []


def R(v1, cat, primary, sign, second, third, pos, neg, why, pop="music"):
    name = f"{v1}_r2"
    _row(name, cat, primary, sign, second, third, pos, neg, f"Redefines {v1}: {why}", "medium", pop, None)
    RD.append(name)


# wrong second scorer: own label moved (z >= 3.3) but the MuQ second could not see the concept
R("bass_heavy", "timbre", "desc.bass_db", "+", "clap_music.bass_heavy_r2", "passt.Bass guitar", "heavy, loud bass line",
  "quiet, thin bass line", "z 3.26 but muq.bass_heavy moved the wrong way (MuQ blind to bass level); CLAP minimal pair second.")
R("shimmering", "timbre", "clap_music.shimmering", "+", "desc.air_db", "timbral.brightness", "", "",
  "z 4.61, muq second flat; physical second desc.air_db.")
R("timbral_warmth", "timbre", "timbral.warmth", "+", "clap_music.warm", "desc.lowhigh_db", "", "",
  "z 8.17, muq second weak-corr and flat; second = the v1 clap_music.warm column.")
R("lush", "timbre", "clap_music.lush", "+", "desc.bandwidth", "timbral.reverb", "", "", "z 4.68, muq second flat.")
R("muted", "timbre", "clap_music.muted", "+", "desc.centroid", "desc.rolloff85", "", "", "z 4.29, muq second flat; centroid second (sign from corpus corr).")
R("swing", "rhythm", "clap_music.swing", "+", "dyn.offbeat_share", "passt.Swing music", "", "", "z 6.79, muq second not monotone.")
R("halftime", "rhythm", "clap_music.halftime", "+", "dyn.flux_mean", "dyn.tempo", "", "", "z 4.26, muq and dyn.tempo seconds weak-corr.")
R("rolling_rhythm", "rhythm", "clap_music.rolling_rhythm", "+", "dyn.hf_onset_rate", "dyn.onset_rate", "", "", "z 4.03, muq second flat.")
R("ostinato", "rhythm", "clap_music.ostinato", "+", "dyn.chroma_entropy", "dyn.key_strength", "", "", "z 3.73, muq second weak-corr.")
R("arpeggiated", "rhythm", "clap_music.arpeggiated", "+", "dyn.onset_rate", "dyn.hf_onset_rate", "", "", "z 3.34, muq second flat.")
R("shoegaze", "genre", "clap_music.shoegaze", "+", "timbral.reverb", "desc.flatness", "", "", "z 7.48, muq second flat.")
R("slap_bass", "instrument", "clap_music.slap_bass", "+", "passt.Bass guitar", "desc.perc_ratio", "", "", "z 3.49, muq second flat.")
R("modular_synth", "instrument", "clap_music.modular_synth", "+", "passt.Synthesizer", "dyn.am_tremolo", "", "", "z 3.99, muq second flat.")
R("vaporwave", "genre", "clap_music.vaporwave", "+", "passt.Electronic music", "dyn.tempo", "", "", "z 5.84, muq second weak-corr.")
R("hammond_organ", "instrument", "clap_music.anchor:organ", "+", "passt.Hammond organ", "passt.Electronic organ", "", "",
  "own passt z 3.63 but slope 0.039 and muq second flat; CLAP 'organ' column primary, PaSST second.")
# too-thin tag: PaSST class probability on a 10 s mix barely moves (own z < 2); CLAP column primary, PaSST second
for v1, col, cls in [("trumpet", "clap_music.anchor:trumpet", "Trumpet"), ("sitar", "clap_music.anchor:sitar", "Sitar"),
                     ("harpsichord", "clap_music.anchor:harpsichord", "Harpsichord"), ("double_bass", "clap_music.upright bass", "Double bass"),
                     ("string_section", "clap_music.string section", "String section"), ("electric_piano", "clap_music.electric piano", "Electric piano")]:
    R(v1, "instrument", col, "+", f"passt.{cls}", f"muq.{v1}", "", "", f"own passt.{cls} z < 2 (thin probability on mixes); CLAP primary.")
# anchors synonymous with an existing knob / unipolar neutral neg: absorbed by wall_of_sound or enjoyment in dedupe
R("chorus_fx", "production", "clap_music.chorus_fx_r2", "+", "dyn.chroma_entropy", "passt.Chorus effect", "guitar with a chorus effect",
  "the same guitar without effects", "anchor vs 'clean studio mix' scored effect salience; merged into wall_of_sound.")
R("echo_delay", "space", "clap_music.echo_delay_r2", "+", "passt.Echo", "timbral.reverb", "a melody with rhythmic delay echoes",
  "the same melody without delay", "neg 'studio recording' made it an outdoor/effects axis; merged into wall_of_sound.")
R("staccato", "dynamics", "clap_music.staccato_r2", "+", "desc.perc_ratio", "dyn.momentary_std", "strings playing short staccato notes",
  "strings playing long legato notes", "n_band 1 and no instrument held fixed; instrument-matched pair.")
R("small_room", "space", "clap_music.small_room_r2", "+", "timbral.reverb", "passt.Inside, small room", "a band in a small dry room",
  "the same band in a large hall", "merged into enjoyment (abstract-place axis); minimal pair, reverb second.")
R("deep", "timbre", "clap_music.deep_r2", "+", "timbral.depth", "dyn.pitch_centroid", "deep, low-register melody",
  "the same melody in a high register", "timbral.depth merged into drum_machine; text pair with register held as the only change.")
R("punchy", "timbre", "clap_music.punchy_r2", "+", "dyn.crest_db", "timbral.hardness", "punchy, tight drum hits",
  "soft, loose drum hits", "neg 'plain, neutral sound'; merged into drum_machine.")
R("airy", "timbre", "clap_music.airy_r2", "+", "desc.air_db", "desc.hf_db", "airy, breathy high end on the same instruments",
  "the same instruments with no air on top", "desc.air_db merged into drum_machine (genre confound).")
R("syncopated", "rhythm", "clap_music.syncopated_r2", "+", "dyn.offbeat_share", "dyn.pulse_clarity", "syncopated drum pattern",
  "the same drums playing on the beat", "dyn.offbeat_share merged into drum_machine.")


def write_v2(rows, prompts, cols, AS, out):
    names_v1 = {r["name"] for r in rows}
    texts = [p["prompt"].lower() for p in prompts]
    for r in V2:
        assert r["name"] not in names_v1, r["name"]
        for k in ("primary_label", "second_scorer", "third_scorer"):
            c = r[k]
            if c.startswith("passt."):
                assert c.split(".", 1)[1] in AS, c
            if c.split(".", 1)[0] in DSP:
                pass                                         # resolved by check_catalogue.py
        dense = r["primary_label"].split(".")[0] in DSP
        r["label_strength"] = "descriptor" if dense else ("classifier" if r["primary_label"].startswith("passt.") else "text_only")
        r["label_type"] = "dense" if dense else "sparse"
        if r.pop("rule") or dense:
            r["class_rule"] = "quartile (top 25% vs bottom 25% of population)"
        elif r["population"] == "solo":
            r["class_rule"] = "top 25% vs bottom 25% of population"
        else:
            r["class_rule"] = "top 250 by score vs bottom 25% of population"
        r["population_n"] = POP_N[r["population"]]
        kws = [w for w in re.findall(r"[a-z][a-z-]{3,}", r["pos_anchor"].lower())][:1] if r["pos_anchor"] else []
        r["expected_tagged_clips"] = sum(any(re.search(r"\b" + re.escape(k) + r"\b", t) for k in kws) for t in texts)
        r["thin"] = ""
        r.pop("kw", None)
    # A redefinition that only changes the second scorer shares its v1 twin's label, sign and population, so
    # build_directions gives both the same direction and the same effect. dedupe sorts by effect with a stable
    # sort, so on the tie the row listed first becomes the keeper: twins go FIRST so the _r2 row (with the
    # fixed second scorer) is the one screened and the v1 twin is absorbed into its cluster.
    key = lambda r: (r["primary_label"], r["sign"], r["population"])
    v1keys = {key(r): r["name"] for r in rows}
    twins = [r for r in V2 if r["name"] in RD and key(r) in v1keys]
    for r in twins:
        r["blurb"] += f" Same direction as {v1keys[key(r)]}: listed before the v1 rows so it wins the dedupe tie."
    allrows = twins + rows + [r for r in V2 if r not in twins]
    names = [r["name"] for r in allrows]
    assert len(names) == len(set(names)), [n for n, c in collections.Counter(names).items() if c > 1]
    seen = {}
    for r in allrows:                                      # one direction per (label, sign, population)
        k = key(r)
        assert k not in seen or (seen[k] in RD and r["name"] == seen[k][:-3]), (k, seen.get(k), r["name"])
        seen.setdefault(k, r["name"])
    with open(out / "concept_catalogue_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(allrows)
    new = [r for r in V2 if r["name"] not in RD]
    summ = {"rows": len(allrows), "v1_rows": len(rows), "new_rows": len(new), "redefinitions": len(RD),
            "new_by_category": dict(collections.Counter(r["category"] for r in new)),
            "new_by_population": dict(collections.Counter(r["population"] for r in new)),
            "new_by_strength": dict(collections.Counter(r["label_strength"] for r in new)),
            "redefined": RD, "twins_listed_first": [r["name"] for r in twins]}
    json.dump(summ, open(out / "catalogue_v2_summary.json", "w"), indent=1)
    print(json.dumps(summ))
