"""Generate concept_catalogue.csv, render_prompts.json, holdout_sfx_prompts.json for the many-knobs plan.
Deterministic (random.Random(20261005)). Planning artefact only.

--v2: rebuild the v1 rows and prompts in memory (v1 files are NOT rewritten), append the v2 rows from
catalogue_v2_rows.py (next to this file) and write concept_catalogue_v2.csv + catalogue_v2_summary.json.
--out DIR: write into DIR instead of this notes folder (regeneration checks)."""
import argparse, ast, csv, json, random, re, collections, sys
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--v2", action="store_true", help="write concept_catalogue_v2.csv (v1 outputs untouched)")
_ap.add_argument("--out", default=r"C:\_dev\projects\DEMON\notes\steering_pr")
ARGS = _ap.parse_args()
OUT = Path(ARGS.out)
AS_CSV = Path(r"C:\Users\ryanf\panns_data\class_labels_indices.csv")
MC_CSV = Path(r"E:\Projects\tada-replication\data\musiccaps-public.csv")
TADA = Path(r"C:\_dev\projects\DEMON-tada-sa3\acestep\tada\data\benchmark_prompts.json")
AS = {r["display_name"] for r in csv.DictReader(open(AS_CSV, encoding="utf-8"))}

rows = []  # catalogue rows
VOCAB = collections.defaultdict(list)  # slot -> list of (concept, phrase)

def add(name, cat, primary, sign, second, strength, diff, blurb, pos="", neg="", third="",
        phrases=None, slot=None, pop="music", kw=None):
    scorer = primary.split(".", 1)[0]
    if scorer == "passt":
        assert primary.split(".", 1)[1] in AS, primary
    if second.startswith("passt."):
        assert second.split(".", 1)[1] in AS, second
    rows.append(dict(name=name, category=cat, primary_label=primary, sign=sign, second_scorer=second,
                     third_scorer=third, label_strength=strength, difficulty=diff, blurb=blurb,
                     pos_anchor=pos, neg_anchor=neg, population=pop,
                     screen_set="sfx" if pop == "sfx" else "music", kw=kw or []))
    if slot:
        for p in (phrases or [name.replace("_", " ")]):
            VOCAB[slot].append((name, p))

def txt(name):
    return f"clap_music.{name}"

# ---------------------------------------------------------------- timbre (DSP-primary first)
T = "timbre"
add("bright", T, "proxies.centroid", "+", "muq.bright", "descriptor", "easy",
    "Moves energy up the spectrum: bright and airy at +, dark and dull at -.", "bright, crisp, sparkling sound",
    "dark, dull, muffled sound", "timbral.brightness", ["bright", "sparkling", "crisp"], "timbre", kw=["bright"])
add("warm", T, "proxies.lowhigh_db", "+", "clap_music.warm", "descriptor", "medium",
    "Low-end weight against top end: warm and round at +, thin at -. (MuQ cannot see warm.)",
    "warm, round, full-bodied sound", "thin, cold, tinny sound", "timbral.warmth", ["warm", "round"], "timbre", kw=["warm"])
add("noisy", T, "proxies.flatness", "+", "muq.noisy", "descriptor", "medium",
    "Noise-like spectrum versus clean tones.", "noisy, hissy, textured sound", "clean, pure tones", "spectral.hf_flatness",
    ["noisy", "hissy"], "timbre", kw=["noisy"])
add("percussive", T, "proxies.perc_ratio", "+", "muq.percussive", "descriptor", "medium",
    "Share of transient (percussive) energy versus sustained tone.", "percussive, drum-driven music",
    "sustained, smooth, legato music", "demucs.drums_share", ["percussive", "drum-heavy"], "timbre", kw=["percussive", "percussion"])
for col, nm, pos, neg, blurb, d in [
    ("hardness", "hard_attack", "hard, sharp attacks", "soft, gentle attacks", "Attack hardness (AudioCommons): hard hits at +, soft onsets at -.", "medium"),
    ("depth", "deep", "deep, low, resonant sound", "shallow, thin sound", "Perceived depth (AudioCommons): deep and resonant at +.", "medium"),
    ("brightness", "timbral_brightness", "bright sound", "dark sound", "AudioCommons brightness model; check against proxies.centroid in dedupe.", "easy"),
    ("roughness", "rough", "rough, gritty, buzzing sound", "smooth, pure sound", "Sensory roughness (AudioCommons), the redefined rough knob.", "hard"),
    ("warmth", "timbral_warmth", "warm sound", "cold sound", "AudioCommons warmth model; second opinion on warm.", "medium"),
    ("sharpness", "sharp", "sharp, piercing sound", "blunt, rounded sound", "Sharpness (AudioCommons, Zwicker-like).", "medium"),
    ("boominess", "boomy", "boomy, booming bass", "tight, lean bass", "Boominess (AudioCommons): low-mid boom.", "medium"),
]:
    add(nm, T, f"timbral.{col}", "+", f"muq.{nm}", "descriptor", d, blurb, pos, neg, "", [pos.split(",")[0]], "timbre")
for col, nm, pos, neg, blurb, d, ph in [
    ("tilt_db_oct", "spectral_tilt", "bright, top-heavy mix", "bass-heavy, dark mix", "Spectral slope in dB/octave; the single axis behind bright and warm.", "easy", None),
    ("sub_db", "sub_bass", "deep sub-bass", "no sub-bass", "Energy share below 60 Hz.", "easy", ["deep sub-bass"]),
    ("bass_db", "bass_heavy", "heavy bass", "light bass", "Energy share 60-250 Hz.", "easy", ["heavy bass", "bass-heavy"]),
    ("lowmid_db", "muddy", "muddy, boxy low mids", "clear low mids", "Energy share 200-500 Hz (mud).", "medium", ["muddy"]),
    ("nasal_db", "nasal", "nasal, honky midrange", "scooped midrange", "Energy share 800-1500 Hz.", "hard", ["nasal", "honky"]),
    ("presence_db", "present", "forward, present midrange", "recessed, distant midrange", "Energy share 2-5 kHz (presence).", "medium", ["upfront"]),
    ("air_db", "airy", "airy, shimmering highs", "no high-frequency air", "Energy share above 12 kHz.", "medium", ["airy", "shimmering"]),
    ("hf_db", "hf_energy", "sizzling high end", "rolled-off high end", "Energy share 8-16 kHz (sa3judge tone.hf).", "easy", ["sizzling"]),
    ("bandwidth", "full_range", "full-range, wide-band sound", "narrow-band, band-limited sound", "Spectral bandwidth.", "medium", ["full-range"]),
    ("harmonic_ratio", "tonal_harmonic", "harmonic, tonal sound", "noisy, atonal sound", "HPSS harmonic share (1 - percussive), with flatness as control.", "medium", ["harmonic", "tonal"]),
]:
    add(nm, T, f"spectral.{col}", "+", f"muq.{nm}", "descriptor", d, blurb, pos, neg, "", ph, "timbre" if ph else None)
add("high_register", T, "tonal.pitch_centroid", "+", "muq.high_register", "descriptor", "medium",
    "Pitch register of the tonal content: high at +, low at -.", "high-pitched melody", "low-pitched, bass register", "",
    ["high-pitched"], "timbre")
add("dissonant", T, "tonal.dissonance", "+", "muq.dissonant", "descriptor", "hard",
    "Sensory dissonance between partials.", "dissonant, clashing harmony", "consonant, sweet harmony", "timbral.roughness",
    ["dissonant"], "timbre")
for nm, pos, neg, third, d in [
    ("glassy", "glassy, crystalline tones", "", "spectral.air_db", "hard"), ("metallic", "metallic, clanging timbre", "", "spectral.nasal_db", "medium"),
    ("wooden", "wooden, organic timbre", "", "", "hard"), ("velvety", "velvety, smooth timbre", "", "timbral.warmth", "hard"),
    ("grainy", "grainy, granular texture", "", "proxies.flatness", "hard"), ("fuzzy", "fuzzy, fuzzed-out tone", "", "passt.Distortion", "medium"),
    ("silky", "silky, soft timbre", "", "", "hard"), ("hollow", "hollow, thin timbre", "", "", "hard"),
    ("breathy", "breathy, airy tone", "", "spectral.air_db", "hard"), ("muted", "muted, damped tone", "", "proxies.centroid", "medium"),
    ("round", "round, rounded tone", "", "timbral.warmth", "hard"), ("crisp", "crisp, articulate tone", "", "timbral.hardness", "medium"),
    ("mellow", "mellow, gentle tone", "harsh, aggressive tone", "proxies.centroid", "medium"), ("harsh", "harsh, abrasive tone", "", "timbral.sharpness", "medium"),
    ("soft_timbre", "soft, delicate sound", "", "level.lufs", "medium"), ("punchy", "punchy, tight sound", "", "level.transient_db", "medium"),
    ("shimmering", "shimmering, sparkling texture", "", "spectral.air_db", "medium"), ("lush", "lush, rich, layered sound", "", "spectral.bandwidth", "medium"),
    ("thin", "thin, small sound", "big, full sound", "proxies.lowhigh_db", "medium"), ("gritty", "gritty, dirty sound", "", "timbral.roughness", "medium"),
]:
    add(nm, T, txt(nm), "+", f"muq.{nm}", "text_only", d, f"Texture word '{nm.replace('_', ' ')}' scored by CLAP; DSP column as a sanity third.",
        pos, neg or "plain, neutral sound", third, [pos.split(",")[0]], "timbre", kw=[nm.split("_")[0]])

# ---------------------------------------------------------------- dynamics
D = "dynamics"
for col, nm, sg, pos, neg, sec, blurb, d, ph in [
    ("lufs", "loud", "+", "loud, intense music", "quiet, soft music", "muq.loud", "Integrated loudness (LUFS).", "easy", ["loud"]),
    ("dr_db", "dynamic", "+", "dynamic music with quiet and loud passages", "flat, constant-level music", "muq.dynamic", "Spread of momentary loudness (p95 - p10).", "medium", ["dynamic"]),
    ("crest_db", "uncompressed", "+", "open, uncompressed, dynamic mix", "heavily compressed, squashed mix", "audiobox.PQ", "Crest factor: open at +, squashed at -.", "medium", ["heavily compressed"]),
    ("loudness_slope", "crescendo", "+", "music building up in a crescendo", "music fading out, decrescendo", "muq.crescendo", "Loudness trend over the clip: building at +, fading at -.", "hard", ["building crescendo", "slow build"]),
    ("momentary_std", "accented", "+", "strongly accented, punchy dynamics", "even, smooth dynamics", "muq.accented", "Momentary loudness variance.", "medium", ["accented"]),
    ("silence_frac", "sparse_silences", "+", "music with pauses and silences", "continuous wall of sound", "muq.sparse_silences", "Share of near-silent frames.", "hard", ["with pauses"]),
    ("transient_db", "transient_punch", "+", "punchy transients", "smeared, soft transients", "muq.transient_punch", "Onset peak over local floor.", "medium", ["punchy transients"]),
    ("am_tremolo", "tremolo", "+", "tremolo, pulsing volume", "steady volume", "muq.tremolo", "Amplitude modulation depth at 4-8 Hz.", "hard", ["tremolo"]),
    ("peak_to_lufs", "limited", "-", "brickwall limited, loud master", "quiet master with headroom", "audiobox.PQ", "Peak-to-loudness ratio (low = limited).", "medium", ["loud master"]),
    ("clip_frac", "clipping", "+", "clipping, overloaded audio", "clean audio", "passt.Distortion", "Fraction of near-full-scale samples.", "medium", None),
]:
    add(nm, D, f"level.{col}", sg, sec, "descriptor", d, blurb, pos, neg, "", ph, "dyn" if ph else None)
for nm, pos, neg, d in [("intense", "intense, powerful music", "gentle, restrained music", "medium"),
                       ("quiet_soft", "quiet, soft, delicate music", "loud music", "medium"),
                       ("sustained", "long sustained notes", "short detached notes", "medium"),
                       ("staccato", "staccato, short detached notes", "legato, connected notes", "hard")]:
    add(nm, D, txt(nm), "+", f"muq.{nm}", "text_only", d, f"'{pos}' by CLAP text similarity.", pos, neg,
        "basicpitch.mean_note_dur" if nm in ("sustained", "staccato") else "level.lufs", [pos.split(",")[0]], "dyn", kw=[nm.split("_")[0]])

# ---------------------------------------------------------------- rhythm
R = "rhythm"
for col, nm, sg, pos, neg, sec, blurb, d, ph in [
    ("bpm", "fast_tempo", "+", "fast tempo, uptempo music", "slow tempo music", "muq.fast_tempo", "Tempo (beat_this / librosa).", "medium", ["fast tempo", "uptempo"]),
    ("beat_strength", "strong_beat", "+", "strong, driving beat", "weak, floating beat", "muq.strong_beat", "Onset energy at beats over mean.", "medium", ["driving beat"]),
    ("pulse_clarity", "clear_pulse", "+", "steady, clear pulse", "free-time, rubato, no pulse", "muq.clear_pulse", "Tempogram peak sharpness.", "medium", ["steady pulse"]),
    ("tempo_stability", "steady_tempo", "+", "metronomic, steady tempo", "rubato, free tempo", "muq.steady_tempo", "1 - CV of inter-beat intervals.", "hard", ["rubato"]),
    ("offbeat_share", "syncopated", "+", "syncopated, offbeat rhythm", "on-the-beat, square rhythm", "muq.syncopated", "Onset energy on off-beat subdivisions.", "hard", ["syncopated"]),
    ("swing_ratio", "swing", "+", "swung, shuffled rhythm", "straight eighth notes", "muq.swing", "Long-short eighth ratio from onsets.", "hard", ["swung", "shuffle rhythm"]),
    ("kick_quarter", "four_on_floor", "+", "four-on-the-floor kick drum", "broken, irregular kick pattern", "muq.four_on_floor", "Low-band onset periodicity at the quarter note.", "medium", ["four-on-the-floor"]),
    ("hf_onset_rate", "busy_hats", "+", "busy, rapid hi-hats", "sparse cymbals", "passt.Hi-hat", "Onset rate above 6 kHz.", "medium", ["rapid hi-hats"]),
    ("superflux_rate", "event_density", "+", "busy, dense rhythm", "sparse, minimal rhythm", "muq.event_density", "SuperFlux onset rate (the density axis; production density is the minus sign).", "medium", ["busy", "sparse"]),
    ("tempo_slope", "accelerando", "+", "accelerating tempo", "slowing down, ritardando", "muq.accelerando", "Tempo trend within the clip.", "hard", ["accelerating"]),
    ("repetition", "repetitive", "+", "repetitive, looping pattern", "through-composed, evolving music", "muq.repetitive", "Chroma self-similarity at bar lags.", "medium", ["looping", "repetitive"]),
]:
    add(nm, R, f"rhythm.{col}", sg, sec, "descriptor", d, blurb, pos, neg, "", ph, "rhythm")
add("onset_rate", R, "proxies.onset_rate", "-", "muq.onset_rate", "descriptor", "medium",
    "The production density knob (+ = sparse), kept for like-for-like comparison.", "sparse, few notes", "busy, many notes",
    "rhythm.superflux_rate")
add("dense_arrangement", R, "basicpitch.polyphony", "+", "muq.dense_arrangement", "descriptor", "medium",
    "Simultaneous notes (polyphony).", "dense, many-layered arrangement", "single line, monophonic", "", ["dense arrangement"], "rhythm")
for nm, pos, neg, d, third in [
    ("halftime", "half-time groove", "double-time groove", "hard", "rhythm.bpm"), ("triplet_feel", "triplet feel, 12/8 rhythm", "straight 4/4 rhythm", "hard", "rhythm.swing_ratio"),
    ("groovy", "groovy, funky rhythm", "stiff, mechanical rhythm", "medium", ""),
    ("polyrhythmic", "polyrhythmic, interlocking rhythms", "simple rhythm", "hard", ""), ("drum_fill", "drum fills and rolls", "steady beat without fills", "hard", "passt.Drum roll"),
    ("arpeggiated", "arpeggiated pattern", "sustained chords", "medium", "basicpitch.note_rate"), ("ostinato", "ostinato riff", "free melody", "hard", "rhythm.repetition"),
    ("danceable", "danceable, club-ready groove", "undanceable, ambient", "medium", "essentia.danceability"), ("no_percussion", "no drums, no percussion", "with drums and percussion", "easy", "demucs.drums_share"),
    ("marching", "marching rhythm", "flowing rhythm", "medium", ""), ("waltz", "waltz in three-four time", "four-four time", "hard", ""),
    ("off_beat_skank", "off-beat guitar skank", "on-beat strumming", "hard", "rhythm.offbeat_share"), ("rolling_rhythm", "rolling, galloping rhythm", "static rhythm", "hard", ""),
]:
    add(nm, R, txt(nm), "+", f"muq.{nm}", "text_only", d, f"'{pos}' by CLAP text similarity.", pos, neg, third, [pos.split(",")[0]], "rhythm",
        kw=[nm.split("_")[0]])

# ---------------------------------------------------------------- space
S = "space"
add("reverb", S, "timbral.reverb", "+", "passt.Reverberation", "descriptor", "easy", "Reverb amount: wet at +, dry at -.",
    "reverberant, wet, spacious sound", "dry, close, dead room", "", ["reverberant", "drenched in reverb", "dry"], "space", kw=["reverb", "reverberant"])
add("wide_stereo", S, "stereo.side_mid_db", "+", "clap_music.wide_stereo", "descriptor", "medium",
    "Stereo width (side over mid). CLAP/MuQ are mono, so the second scorer is weak: provisional grade.",
    "wide stereo image", "narrow mono image", "stereo.lr_corr", ["wide stereo", "mono"], "space", kw=["mono", "stereo"])
add("pan_motion", S, "stereo.pan_motion", "+", "stereo.hf_side_mid_db", "descriptor", "hard",
    "Movement of the stereo balance over time (auto-pan). Same-family second: provisional.", "sounds panning left and right", "static center image",
    "", ["auto-panned"], "space")
for cls, nm, pos, d, ph in [
    ("Echo", "echo_delay", "echo and delay effects", "medium", ["dub delay", "echoing"]),
    ("Inside, small room", "small_room", "recorded in a small room", "medium", ["in a small room"]),
    ("Inside, large room or hall", "large_hall", "recorded in a large hall", "medium", ["in a concert hall", "in a large hall"]),
    ("Inside, public space", "public_space", "recorded in a public space", "hard", ["in a train station"]),
    ("Outside, rural or natural", "outdoors_nature", "recorded outdoors in nature", "medium", ["outdoors in nature"]),
    ("Outside, urban or manmade", "outdoors_urban", "recorded outdoors in the city", "medium", ["on a city street"]),
    ("Field recording", "field_recording", "field recording", "medium", ["field recording"]),
]:
    add(nm, S, f"passt.{cls}", "+", f"muq.{nm}", "classifier", d, f"AudioSet '{cls}' probability.", pos, "studio recording", "timbral.reverb",
        ph, "space", pop="all")
for nm, pos, neg, third, d in [("close_mic", "close-miked, intimate sound", "distant, roomy sound", "timbral.reverb", "hard"),
                               ("distant", "distant, far away sound", "close, upfront sound", "spectral.presence_db", "hard"),
                               ("cavernous", "cavernous, cathedral-like space", "small dry room", "timbral.reverb", "medium"),
                               ("underwater_space", "underwater, submerged sound", "clear open air sound", "proxies.centroid", "medium")]:
    add(nm, S, txt(nm), "+", f"muq.{nm}", "text_only", d, f"'{pos}' by CLAP text similarity.", pos, neg, third, [pos.split(",")[0]], "space", pop="all")

# ---------------------------------------------------------------- production
P = "production"
for col, nm, sg, pos, neg, d, ph in [
    ("PQ", "production_quality", "+", "polished, professionally produced", "low quality amateur recording", "medium", ["polished studio production", "low quality recording"]),
    ("PC", "production_complexity", "+", "complex, intricate production", "simple, bare production", "medium", ["intricate production"]),
    ("CE", "enjoyment", "+", "enjoyable, engaging music", "boring, unpleasant music", "hard", None),
    ("CU", "usefulness", "+", "well-crafted, usable music", "unusable, broken audio", "hard", None),
]:
    add(nm, P, f"audiobox.{col}", sg, f"muq.{nm}", "classifier", d, f"Audiobox Aesthetics {col}.", pos, neg, "", ph, "prod" if ph else None, pop="all",
        kw=["low quality", "poor audio quality", "amateur recording"] if col == "PQ" else None)
for cls, nm, pos, d, ph in [
    ("Distortion", "distortion", "heavily distorted, overdriven sound", "easy", ["distorted", "overdriven"]),
    ("Crackle", "vinyl_crackle", "vinyl crackle", "medium", ["vinyl crackle"]),
    ("Static", "static_noise", "radio static, noise", "medium", ["static noise"]),
    ("Mains hum", "mains_hum", "electrical hum", "hard", ["with electrical hum"]),
    ("Radio", "radio_filtered", "sounds like an old AM radio", "medium", ["through an old radio"]),
    ("Chorus effect", "chorus_fx", "chorus effect, detuned shimmer", "medium", ["chorus-drenched"]),
    ("Effects unit", "effects_pedal", "guitar effects pedals", "hard", ["effects-laden"]),
    ("Sound effect", "sfx_laden", "music with sound effects", "medium", ["with sound effects"]),
    ("Noise", "noise_bed", "a bed of noise", "medium", ["noisy"]),
    ("Television", "tv_audio", "sounds like television audio", "hard", ["from a TV"]),
]:
    add(nm, P, f"passt.{cls}", "+", f"muq.{nm}", "classifier", d, f"AudioSet '{cls}' probability.", pos, "clean studio mix", "", ph, "prod", pop="all",
        kw=[nm.split("_")[0]])
add("muffled", P, "spectral.rolloff85", "-", "muq.muffled", "descriptor", "easy", "85% spectral rolloff: muffled at +.",
    "muffled, low-passed sound", "open, clear sound", "", ["muffled", "low-passed"], "prod", kw=["muffled"])
add("noise_floor", P, "spectral.floor_db", "+", "passt.Static", "descriptor", "medium", "Background noise floor (tape hiss).",
    "tape hiss, noisy background", "silent background", "", ["tape hiss"], "prod")
add("tonal_artifacts", P, "spectral.tone_persist_db", "+", "audiobox.PQ", "descriptor", "hard",
    "Persistent high-pitched tone (render artifact detector); expect it NOT to be a musical knob.", "high-pitched whine", "clean", "")
add("sidechain", P, "level.am_beat", "+", "muq.sidechain", "descriptor", "medium", "Loudness pumping at the beat rate.",
    "sidechain-compressed pumping", "steady sustained mix", "", ["sidechain pumping"], "prod")
for nm, pos, neg, third, d in [
    ("lofi", "lo-fi, degraded, warm tape sound", "hi-fi, pristine sound", "audiobox.PQ", "easy"), ("tape_saturation", "tape saturation, analog warmth", "clean digital sound", "timbral.warmth", "medium"),
    ("bitcrushed", "bitcrushed, aliased 8-bit sound", "clean high-resolution sound", "spectral.hf_flatness", "medium"), ("phaser", "phaser effect, swirling sound", "dry sound", "", "hard"),
    ("flanger", "flanger, jet-plane sweep", "dry sound", "", "hard"), ("filter_sweep", "filter sweep, opening low-pass filter", "static filter", "spectral.rolloff85", "hard"),
    ("glitchy", "glitchy, stuttering edits", "smooth continuous playback", "rhythm.superflux_rate", "medium"), ("reversed", "reversed sounds", "forward sounds", "", "hard"),
    ("granular", "granular, smeared texture", "clean, natural playback", "proxies.flatness", "hard"), ("live_recording", "live concert recording with audience", "studio recording", "passt.Crowd", "medium"),
    ("amateur", "amateur home recording", "professional studio recording", "audiobox.PQ", "medium"), ("home_video", "home video audio", "studio recording", "audiobox.PQ", "hard"),
    ("minimal", "minimal, sparse arrangement", "maximal, dense arrangement", "basicpitch.polyphony", "medium"), ("wall_of_sound", "wall of sound, huge layered mix", "sparse mix", "spectral.bandwidth", "medium"),
    ("analog", "analog synthesizers and tape", "digital, in-the-box production", "", "hard"), ("digital_cold", "cold, digital, clinical production", "warm, analog production", "", "hard"),
    ("cassette", "cassette tape recording", "pristine digital recording", "spectral.air_db", "medium"), ("dj_mix", "DJ mix, beatmatched transition", "single song", "", "hard"),
    ("solo_instrument", "a single instrument alone", "full band", "basicpitch.polyphony", "medium"), ("demo_rough", "rough demo recording", "finished master", "audiobox.PQ", "hard"),
    ("overcompressed", "overcompressed, pumping, loud", "natural dynamics", "level.crest_db", "medium"), ("stereo_chorus_pad", "wide shimmering pad", "narrow dry sound", "stereo.side_mid_db", "medium"),
    ("dubby", "dub mixing with spring reverb and delay", "dry mix", "passt.Echo", "medium"), ("pristine", "pristine audiophile recording", "noisy recording", "audiobox.PQ", "medium"),
]:
    add(nm, P, txt(nm), "+", f"muq.{nm}", "text_only", d, f"'{pos}' by CLAP text similarity.", pos, neg, third, [pos.split(",")[0]], "prod", pop="all",
        kw=[nm.split("_")[0]])

# ---------------------------------------------------------------- instruments
I = "instrument"
AS_INST = [  # (name, class, phrase, diff, third)
    ("piano", "Piano", "piano", "easy", "musetimbre.piano"), ("electric_piano", "Electric piano", "Rhodes electric piano", "medium", "musetimbre.electric_piano"),
    ("organ", "Organ", "pipe organ", "easy", ""), ("hammond_organ", "Hammond organ", "Hammond organ", "medium", ""),
    ("synthesizer", "Synthesizer", "analog synthesizer", "easy", ""), ("harpsichord", "Harpsichord", "harpsichord", "medium", "musetimbre.harpsichord"),
    ("acoustic_guitar", "Acoustic guitar", "acoustic guitar", "easy", "musetimbre.acoustic_guitar"), ("electric_guitar", "Electric guitar", "electric guitar", "easy", "musetimbre.electric_guitar"),
    ("bass_guitar", "Bass guitar", "electric bass guitar", "medium", "demucs.bass_share"), ("slide_guitar", "Steel guitar, slide guitar", "slide guitar", "medium", ""),
    ("banjo", "Banjo", "banjo", "easy", ""), ("sitar", "Sitar", "sitar", "medium", ""), ("mandolin", "Mandolin", "mandolin", "medium", ""),
    ("zither", "Zither", "zither", "hard", ""), ("ukulele", "Ukulele", "ukulele", "medium", ""), ("strummed_guitar", "Strum", "strummed guitar chords", "medium", ""),
    ("drum_kit", "Drum kit", "acoustic drum kit", "easy", "demucs.drums_share"), ("drum_machine", "Drum machine", "drum machine", "easy", "demucs.drums_share"),
    ("snare", "Snare drum", "crisp snare drum", "medium", ""), ("rimshot", "Rimshot", "rimshots", "hard", ""), ("drum_roll", "Drum roll", "drum rolls", "hard", ""),
    ("kick_drum", "Bass drum", "booming kick drum", "medium", "spectral.sub_db"), ("timpani", "Timpani", "timpani", "medium", ""),
    ("tabla", "Tabla", "tabla", "medium", ""), ("cymbal", "Cymbal", "crash cymbals", "medium", "spectral.hf_db"), ("hi_hat", "Hi-hat", "hi-hats", "medium", "rhythm.hf_onset_rate"),
    ("wood_block", "Wood block", "wood block", "hard", ""), ("tambourine", "Tambourine", "tambourine", "medium", ""), ("maracas", "Maraca", "maracas", "hard", ""),
    ("gong", "Gong", "gong", "medium", ""), ("tubular_bells", "Tubular bells", "tubular bells", "medium", ""), ("marimba", "Marimba, xylophone", "marimba", "medium", ""),
    ("glockenspiel", "Glockenspiel", "glockenspiel", "medium", ""), ("vibraphone", "Vibraphone", "vibraphone", "medium", ""), ("steelpan", "Steelpan", "steel drums", "medium", ""),
    ("orchestra", "Orchestra", "full orchestra", "easy", ""), ("french_horn", "French horn", "French horn", "medium", ""), ("trumpet", "Trumpet", "trumpet", "easy", ""),
    ("trombone", "Trombone", "trombone", "medium", ""), ("brass_section", "Brass instrument", "brass section", "easy", ""), ("string_section", "String section", "string section", "easy", ""),
    ("violin", "Violin, fiddle", "violin", "easy", "musetimbre.violin"), ("pizzicato", "Pizzicato", "pizzicato strings", "medium", ""), ("cello", "Cello", "cello", "medium", "musetimbre.cello"),
    ("double_bass", "Double bass", "upright bass", "medium", "demucs.bass_share"), ("flute", "Flute", "flute", "easy", "musetimbre.flute"), ("saxophone", "Saxophone", "saxophone", "easy", "musetimbre.saxophone"),
    ("clarinet", "Clarinet", "clarinet", "medium", "musetimbre.clarinet"), ("harp", "Harp", "harp", "medium", ""), ("church_bells", "Church bell", "church bells", "medium", ""),
    ("jingle_bells", "Jingle bell", "sleigh bells", "medium", ""), ("chimes", "Chime", "chimes", "medium", ""), ("wind_chimes", "Wind chime", "wind chimes", "medium", ""),
    ("harmonica", "Harmonica", "harmonica", "medium", ""), ("accordion", "Accordion", "accordion", "easy", ""), ("bagpipes", "Bagpipes", "bagpipes", "easy", ""),
    ("didgeridoo", "Didgeridoo", "didgeridoo", "medium", ""), ("theremin", "Theremin", "theremin", "hard", ""), ("singing_bowl", "Singing bowl", "singing bowls", "medium", ""),
    ("turntable_scratch", "Scratching (performance technique)", "turntable scratching", "medium", ""), ("cowbell", "Cowbell", "cowbell", "medium", ""),
    ("handclaps", "Clapping", "handclaps", "easy", ""), ("finger_snaps", "Finger snapping", "finger snaps", "medium", ""), ("guitar_tapping", "Tapping (guitar technique)", "tapped guitar", "hard", ""),
    ("electronic_organ", "Electronic organ", "electronic organ", "hard", ""), ("sampler", "Sampler", "chopped samples", "hard", ""),
]
for nm, cls, ph, d, third in AS_INST:
    add(nm, I, f"passt.{cls}", "+", f"muq.{nm}", "classifier", d, f"Adds {ph}; AudioSet '{cls}' probability.", f"music featuring {ph}", "music",
        third, [ph], "inst", kw=[ph, nm.replace("_", " ")])
TXT_INST = [
    ("wurlitzer", "Wurlitzer electric piano", "medium"), ("clavinet", "funky clavinet", "medium"), ("mellotron", "Mellotron flutes", "hard"),
    ("celesta", "celesta", "hard"), ("music_box", "music box", "medium"), ("toy_piano", "toy piano", "hard"), ("kalimba", "kalimba", "medium"),
    ("koto", "koto", "medium"), ("shamisen", "shamisen", "hard"), ("erhu", "erhu", "medium"), ("guzheng", "guzheng", "medium"), ("oud", "oud", "medium"),
    ("bouzouki", "bouzouki", "hard"), ("balalaika", "balalaika", "hard"), ("bansuri", "bansuri flute", "hard"), ("shakuhachi", "shakuhachi", "medium"),
    ("duduk", "duduk", "hard"), ("kora", "kora", "hard"), ("djembe", "djembe", "medium"), ("congas", "congas", "medium"), ("bongos", "bongos", "medium"),
    ("cajon", "cajon", "hard"), ("taiko", "taiko drums", "medium"), ("handpan", "handpan", "medium"), ("pan_flute", "pan flute", "medium"),
    ("ocarina", "ocarina", "hard"), ("tin_whistle", "tin whistle", "medium"), ("oboe", "oboe", "medium"), ("bassoon", "bassoon", "medium"),
    ("tuba", "tuba", "medium"), ("piccolo", "piccolo", "hard"), ("recorder", "recorder", "hard"), ("triangle", "triangle", "hard"),
    ("castanets", "castanets", "hard"), ("shaker", "shaker", "medium"), ("ride_cymbal", "ride cymbal", "hard"), ("brushed_drums", "brushed drums", "medium"),
    ("string_quartet", "string quartet", "medium"), ("synth_pad", "lush synth pad", "easy"), ("synth_lead", "screaming synth lead", "easy"),
    ("synth_bass", "synth bass", "easy"), ("bass_808", "808 bass", "easy"), ("supersaw", "supersaw chords", "medium"), ("wobble_bass", "wobble bass", "medium"),
    ("acid_303", "acid 303 bassline", "medium"), ("arp_synth", "synth arpeggiator", "easy"), ("fm_bells", "FM synth bells", "medium"),
    ("modular_synth", "modular synthesizer bleeps", "hard"), ("nylon_guitar", "nylon-string classical guitar", "medium"), ("twelve_string", "12-string guitar", "hard"),
    ("slap_bass", "slap bass", "medium"), ("fretless_bass", "fretless bass", "hard"), ("muted_trumpet", "muted trumpet", "medium"), ("flugelhorn", "flugelhorn", "hard"),
    ("hammered_dulcimer", "hammered dulcimer", "hard"), ("hurdy_gurdy", "hurdy-gurdy", "hard"), ("santoor", "santoor", "hard"), ("tanpura_drone", "tanpura drone", "medium"),
    ("harmonium", "harmonium", "medium"), ("lute", "lute", "hard"), ("darbuka", "darbuka", "hard"), ("frame_drum", "frame drum", "hard"), ("bodhran", "bodhran", "hard"),
    ("gamelan", "gamelan ensemble", "medium"), ("talking_drum", "talking drum", "hard"), ("timbales", "timbales", "hard"), ("glass_harmonica", "glass harmonica", "hard"),
]
for nm, ph, d in TXT_INST:
    third = "demucs.bass_share" if "bass" in nm else ("demucs.drums_share" if nm in ("djembe", "congas", "bongos", "cajon", "taiko", "brushed_drums") else "")
    add(nm, I, txt(nm), "+", f"muq.{nm}", "text_only", d, f"Adds {ph}; no AudioSet class, so CLAP/MuQ text only (plus MuseTimbre prototype).",
        f"music featuring {ph}", "music", third or f"musetimbre.{nm}", [ph], "inst", kw=[ph, nm.replace("_", " ")])

# ---------------------------------------------------------------- genres
G = "genre"
AS_GEN = [("pop", "Pop music", "pop", "easy"), ("hip_hop", "Hip hop music", "hip hop", "easy"), ("rock", "Rock music", "rock", "easy"),
          ("heavy_metal", "Heavy metal", "heavy metal", "easy"), ("punk", "Punk rock", "punk rock", "medium"), ("grunge", "Grunge", "grunge", "medium"),
          ("prog_rock", "Progressive rock", "progressive rock", "medium"), ("rock_and_roll", "Rock and roll", "rock and roll", "medium"),
          ("psych_rock", "Psychedelic rock", "psychedelic rock", "medium"), ("rnb", "Rhythm and blues", "R&B", "medium"), ("soul", "Soul music", "soul", "medium"),
          ("reggae", "Reggae", "reggae", "easy"), ("country", "Country", "country", "easy"), ("swing_music", "Swing music", "swing", "medium"),
          ("bluegrass", "Bluegrass", "bluegrass", "medium"), ("funk", "Funk", "funk", "easy"), ("folk", "Folk music", "folk", "easy"),
          ("middle_eastern", "Middle Eastern music", "Middle Eastern", "medium"), ("jazz", "Jazz", "jazz", "easy"), ("disco", "Disco", "disco", "easy"),
          ("classical", "Classical music", "classical", "easy"), ("electronic", "Electronic music", "electronic", "easy"), ("house", "House music", "house", "easy"),
          ("techno", "Techno", "techno", "easy"), ("dubstep", "Dubstep", "dubstep", "easy"), ("drum_and_bass", "Drum and bass", "drum and bass", "easy"),
          ("electronica", "Electronica", "electronica", "medium"), ("edm", "Electronic dance music", "EDM", "easy"), ("ambient", "Ambient music", "ambient", "easy"),
          ("trance", "Trance music", "trance", "easy"), ("latin", "Music of Latin America", "Latin", "medium"), ("salsa", "Salsa music", "salsa", "medium"),
          ("flamenco", "Flamenco", "flamenco", "medium"), ("blues", "Blues", "blues", "easy"), ("childrens", "Music for children", "children's", "hard"),
          ("new_age", "New-age music", "new age", "medium"), ("african", "Music of Africa", "African", "medium"), ("afrobeat", "Afrobeat", "afrobeat", "medium"),
          ("asian", "Music of Asia", "Asian traditional", "medium"), ("carnatic", "Carnatic music", "Carnatic", "hard"), ("bollywood", "Music of Bollywood", "Bollywood", "hard"),
          ("ska", "Ska", "ska", "medium"), ("traditional", "Traditional music", "traditional folk", "medium"), ("indie", "Independent music", "indie", "medium"),
          ("background", "Background music", "background", "hard"), ("theme", "Theme music", "TV theme", "hard"), ("jingle", "Jingle (music)", "advertising jingle", "hard"),
          ("soundtrack", "Soundtrack music", "film soundtrack", "medium"), ("lullaby", "Lullaby", "lullaby", "medium"), ("video_game", "Video game music", "video game", "medium"),
          ("christmas", "Christmas music", "Christmas", "medium"), ("dance", "Dance music", "dance", "easy"), ("wedding", "Wedding music", "wedding", "hard")]
for nm, cls, ph, d in AS_GEN:
    add(nm, G, f"passt.{cls}", "+", f"muq.{nm}", "classifier", d, f"Pushes toward {ph}; AudioSet '{cls}'.", f"{ph} music", "music",
        "essentia.discogs", [ph], "genre", kw=[ph.lower()])
TXT_GEN = ["lofi hip hop", "trap", "drill", "boom bap", "synthwave", "vaporwave", "chillwave", "shoegaze", "dream pop", "post-rock", "math rock",
           "indie folk", "IDM", "breakbeat", "UK garage", "jungle", "footwork", "hardstyle", "dub", "dub techno", "minimal techno", "acid house",
           "deep house", "future bass", "downtempo", "trip hop", "glitch", "industrial", "noise", "drone", "bossa nova", "samba", "tango", "cumbia",
           "reggaeton", "dancehall", "amapiano", "highlife", "city pop", "chiptune", "baroque", "romantic orchestral", "minimalist classical",
           "marching band", "big band", "bebop", "free jazz", "jazz fusion", "smooth jazz", "celtic", "klezmer", "polka", "balkan brass", "surf rock",
           "rockabilly", "doom metal", "black metal", "post-punk", "new wave", "krautrock", "space rock", "stoner rock", "phonk", "spaghetti western",
           "film noir jazz", "epic trailer", "horror score", "gospel organ", "motown", "neo-soul"]
for ph in TXT_GEN:
    nm = re.sub(r"[^a-z0-9]+", "_", ph.lower()).strip("_")
    d = "medium" if ph in ("trap", "synthwave", "lofi hip hop", "bossa nova", "chiptune", "baroque", "big band", "dub", "deep house", "surf rock") else "hard"
    add(nm, G, txt(nm), "+", f"muq.{nm}", "text_only", d, f"Pushes toward {ph}; text only unless Essentia Discogs-EffNet is installed.",
        f"{ph} music", "music", "essentia.discogs", [ph], "genre", kw=[ph.lower()])

# ---------------------------------------------------------------- moods
M = "mood"
for nm, cls, d in [("happy", "Happy music", "easy"), ("funny", "Funny music", "hard"), ("sad", "Sad music", "easy"), ("tender", "Tender music", "medium"),
                   ("exciting", "Exciting music", "medium"), ("angry", "Angry music", "medium"), ("scary", "Scary music", "medium")]:
    add(nm, M, f"passt.{cls}", "+", f"muq.{nm}", "classifier", d, f"Mood '{nm}' (AudioSet '{cls}').", f"{nm} music", "music",
        "essentia.mood", [nm], "mood", kw=[nm])
add("major_mode", M, "tonal.mode_score", "+", "muq.major_mode", "descriptor", "hard", "Major versus minor (Krumhansl profile difference).",
    "music in a major key", "music in a minor key", "passt.Happy music", ["in a major key", "in a minor key"], "mood")
add("key_clarity", M, "tonal.key_clarity", "+", "muq.key_clarity", "descriptor", "hard", "Clear tonal centre versus atonal/ambiguous.",
    "clearly tonal, diatonic music", "atonal, chromatic music", "tonal.chroma_entropy", ["atonal"], "mood")
for nm in ["energetic", "calm", "melancholic", "romantic", "aggressive", "dreamy", "epic", "uplifting", "dark", "mysterious", "playful", "nostalgic",
           "suspenseful", "hopeful", "triumphant", "peaceful", "ominous", "joyful", "bittersweet", "eerie", "hypnotic", "soothing", "chaotic", "serene",
           "heroic", "sensual", "lonely", "anxious", "majestic", "whimsical", "somber", "euphoric", "brooding", "meditative", "cheerful", "relaxed",
           "haunting", "sentimental", "spiritual", "menacing", "carefree", "determined", "gloomy", "passionate", "groovy_mood"]:
    ph = nm.replace("_mood", "")
    add(nm, M, txt(nm), "+", f"muq.{nm}", "text_only", "medium" if ph in ("energetic", "calm", "dark", "epic", "dreamy", "aggressive", "peaceful") else "hard",
        f"Mood '{ph}' by CLAP text similarity; needs a human spot check.", f"{ph} music", "music", "essentia.mood", [ph], "mood", kw=[ph])

# ---------------------------------------------------------------- sound effects (AudioSet, pop = sfx)
X = "sound_effect"
SFX = [("wind", "Wind", "howling wind"), ("rustling_leaves", "Rustling leaves", "rustling leaves"), ("thunderstorm", "Thunderstorm", "a thunderstorm"),
       ("thunder", "Thunder", "rolling thunder"), ("rain", "Rain", "heavy rain"), ("rain_on_surface", "Rain on surface", "rain on a tin roof"),
       ("stream", "Stream", "a babbling brook"), ("waterfall", "Waterfall", "a waterfall"), ("ocean_waves", "Waves, surf", "ocean waves"),
       ("fire", "Fire", "a crackling campfire"), ("birdsong", "Bird vocalization, bird call, bird song", "birdsong"), ("crickets", "Cricket", "crickets at night"),
       ("frogs", "Frog", "frogs croaking"), ("owl", "Owl", "an owl hooting"), ("crow", "Crow", "crows cawing"), ("wolf_howl", "Howl", "a wolf howling"),
       ("dog_bark", "Bark", "a dog barking"), ("cat_meow", "Meow", "a cat meowing"), ("horse", "Horse", "horse hooves"), ("cow", "Moo", "cows mooing"),
       ("rooster", "Crowing, cock-a-doodle-doo", "a rooster crowing"), ("bees", "Bee, wasp, etc.", "buzzing bees"), ("insects", "Insect", "buzzing insects"),
       ("whale", "Whale vocalization", "whale song"), ("lion_roar", "Roar", "a lion roaring"), ("car_passing", "Car passing by", "a car passing by"),
       ("race_car", "Race car, auto racing", "race cars"), ("truck", "Truck", "a heavy truck"), ("motorcycle", "Motorcycle", "a motorcycle revving"),
       ("train", "Train", "a passing train"), ("train_horn", "Train horn", "a train horn"), ("subway", "Subway, metro, underground", "a subway train"),
       ("jet", "Jet engine", "a jet engine"), ("helicopter", "Helicopter", "a helicopter"), ("airplane", "Fixed-wing aircraft, airplane", "a propeller airplane"),
       ("boat", "Motorboat, speedboat", "a motorboat"), ("police_siren", "Police car (siren)", "a police siren"), ("ambulance", "Ambulance (siren)", "an ambulance siren"),
       ("car_horn", "Vehicle horn, car horn, honking", "car horns honking"), ("skidding", "Tire squeal", "tires screeching"), ("engine_idle", "Idling", "an idling engine"),
       ("revving", "Accelerating, revving, vroom", "an engine revving"), ("traffic", "Traffic noise, roadway noise", "city traffic"),
       ("door_slam", "Slam", "a door slamming"), ("knock", "Knock", "knocking on a door"), ("footsteps", "Walk, footsteps", "footsteps"),
       ("clock_tick", "Tick-tock", "a ticking clock"), ("typing", "Typing", "typing on a keyboard"), ("typewriter", "Typewriter", "a typewriter"),
       ("keys_jangling", "Keys jangling", "jangling keys"), ("dishes", "Dishes, pots, and pans", "clattering dishes"), ("glass_clink", "Chink, clink", "glasses clinking"),
       ("glass_shatter", "Shatter", "glass shattering"), ("coins", "Coin (dropping)", "coins dropping"), ("phone_ring", "Telephone bell ringing", "an old telephone ringing"),
       ("alarm_clock", "Alarm clock", "an alarm clock"), ("doorbell", "Doorbell", "a doorbell"), ("vacuum", "Vacuum cleaner", "a vacuum cleaner"),
       ("water_tap", "Water tap, faucet", "a running faucet"), ("pouring", "Pour", "water pouring"), ("dripping", "Drip", "water dripping"),
       ("boiling", "Boiling", "boiling water"), ("frying", "Frying (food)", "food sizzling in a pan"), ("explosion", "Explosion", "an explosion"),
       ("gunshot", "Gunshot, gunfire", "gunshots"), ("fireworks", "Fireworks", "fireworks"), ("thud", "Thump, thud", "heavy thuds"),
       ("smash", "Smash, crash", "a crash"), ("whoosh", "Whoosh, swoosh, swish", "whooshes"), ("hammer", "Hammer", "hammering"),
       ("wood_crack", "Crack", "wood cracking"), ("whip", "Whip", "a whip crack"), ("gears", "Gears", "grinding gears"), ("fan", "Mechanical fan", "a whirring fan"),
       ("sewing_machine", "Sewing machine", "a sewing machine"), ("chainsaw", "Chainsaw", "a chainsaw"), ("power_drill", "Drill", "a power drill"),
       ("jackhammer", "Jackhammer", "a jackhammer"), ("cash_register", "Cash register", "a cash register"), ("applause", "Applause", "applause"),
       ("cheering", "Cheering", "a cheering crowd"), ("crowd", "Crowd", "a crowd murmuring"), ("children_playing", "Children playing", "children playing"),
       ("laughter", "Laughter", "people laughing"), ("beep", "Beep, bleep", "electronic beeps"), ("ping", "Ping", "pings"), ("boing", "Boing", "cartoon boings"),
       ("sine_tone", "Sine wave", "a pure sine tone"), ("white_noise", "White noise", "white noise"), ("sonar", "Sonar", "sonar pings"),
       ("heartbeat", "Heart sounds, heartbeat", "a heartbeat"), ("breathing", "Breathing", "heavy breathing"), ("creak", "Creak", "a creaking door"),
       ("rumble", "Rumble", "a low rumble"), ("buzzer", "Buzzer", "a buzzer"), ("church_bell_toll", "Change ringing (campanology)", "bell ringing peals")]
for nm, cls, ph in SFX:
    d = "medium" if cls in ("Rain", "Wind", "Waves, surf", "Thunder", "Bird vocalization, bird call, bird song", "Applause", "Fire", "Explosion", "Train", "Crowd") else "hard"
    add(nm, X, f"passt.{cls}", "+", f"clap_general.{nm}", "classifier", d, f"Sound effect: {ph} (AudioSet '{cls}').", f"the sound of {ph}", "silence",
        "muq." + nm, [ph], "sfx", pop="sfx", kw=[re.sub(r"^(a|an|the) ", "", ph)])
for nm, ph in [("laser_zap", "sci-fi laser zaps"), ("riser", "a cinematic riser"), ("impact_boom", "a cinematic boom impact"), ("glitch_sfx", "digital glitch noises"),
               ("ui_click", "interface clicks"), ("magic_sparkle", "magical sparkle chimes")]:
    add(nm, X, f"clap_general.{nm}", "+", f"muq.{nm}", "text_only", "hard", f"Design sound '{ph}'; no AudioSet class.", f"the sound of {ph}", "silence",
        "passt.Sound effect", [ph], "sfx", pop="sfx")

# ---------------------------------------------------------------- abstract
A = "abstract"
ABS = ["spring", "summer", "autumn", "winter", "dawn", "afternoon", "sunset", "midnight", "rainy day", "snowfall", "storm", "fog", "sunny",
       "underwater", "outer space", "desert", "forest", "city at night", "cathedral", "cave", "beach", "mountains", "jungle", "arctic",
       "cyberpunk city", "haunted house", "carnival", "tavern", "temple", "laboratory", "train journey", "nightclub", "coffee shop",
       "countryside", "wild west town", "ancient ruins", "dreamland", "1920s", "1950s", "1960s", "1970s", "1980s", "1990s", "futuristic",
       "retro", "medieval", "ancient", "red", "blue", "golden", "neon", "pastel", "cosmic", "industrial", "pastoral", "sacred", "tropical",
       "childlike", "mechanical", "organic", "alien", "cinematic", "heroic quest", "romantic evening", "road trip"]
for ph in ABS:
    nm = "abs_" + re.sub(r"[^a-z0-9]+", "_", ph.lower()).strip("_")
    add(nm, A, txt(nm), "+", f"muq.{nm}", "text_only", "hard", f"Evokes '{ph}'. Two text scorers only: needs a human spot check before shipping.",
        f"music that evokes {ph}", "music", "clap_general." + nm, [ph], "abs", kw=[ph.lower()])

DROP = set("""drill boom_bap chillwave math_rock indie_folk footwork dub_techno glitch noise amapiano highlife city_pop romantic_orchestral
minimalist_classical marching_band jazz_fusion smooth_jazz klezmer balkan_brass rockabilly doom_metal post_punk new_wave krautrock space_rock
stoner_rock phonk spaghetti_western film_noir_jazz gospel_organ motown neo_soul dancehall background wedding jingle theme
toy_piano shamisen bouzouki balalaika duduk kora cajon ocarina piccolo recorder triangle castanets ride_cymbal twelve_string fretless_bass
flugelhorn hurdy_gurdy santoor lute darbuka frame_drum bodhran talking_drum timbales glass_harmonica electronic_organ sampler guitar_tapping rimshot zither
bittersweet soothing lonely anxious brooding cheerful relaxed sentimental menacing carefree determined gloomy passionate groovy_mood
whale lion_roar cow rooster subway airplane boat ambulance skidding engine_idle keys_jangling coins doorbell vacuum water_tap boiling
sewing_machine cash_register ping sonar buzzer church_bell_toll breathing horse
abs_afternoon abs_fog abs_sunny abs_mountains abs_arctic abs_tavern abs_temple abs_laboratory abs_coffee_shop abs_wild_west_town abs_dreamland
abs_1920s abs_1960s abs_pastel abs_red abs_sacred abs_childlike abs_organic abs_heroic_quest abs_romantic_evening
home_video demo_rough stereo_chorus_pad digital_cold dj_mix reversed hollow silky velvety round""".split())
rows[:] = [r for r in rows if r["name"] not in DROP]

# ---------------------------------------------------------------- label names -> columns label_corpus.py writes
# (2026-10-05) The planned scorer prefixes above are mapped onto the columns that
# DEMON-steer-bench scripts/steering_bench/label_corpus.py actually writes:
# desc.* / dyn.* / timbral.* / passt.* / clap_music|clap_general|muq.<concept name>
# (= cos(pos_anchor) - cos(neg_anchor), anchors.json "concepts" synced from this CSV by
# check_catalogue.py --sync-anchors). Descriptors with no column (stereo: shards are mono;
# audiobox, basicpitch, demucs, essentia, musetimbre: not run; swing, kick periodicity,
# tempo slope, repetition, dissonance, transient level, sidechain AM, tone persistence)
# become text rows: primary clap_music.<name>; an unresolved second becomes muq.<name>;
# an unresolved third becomes empty. scripts/steering_bench/check_catalogue.py must print 0.
COL_MAP = {
    **{f"proxies.{k}": f"desc.{k}" for k in ("centroid", "lowhigh_db", "flatness", "onset_rate", "perc_ratio")},
    **{f"spectral.{k}": f"desc.{k}" for k in ("sub_db", "bass_db", "lowmid_db", "nasal_db", "presence_db", "hf_db",
                                              "air_db", "tilt_db_oct", "bandwidth", "rolloff85", "floor_db")},
    "spectral.harmonic_ratio": "dyn.hp_ratio_db",
    "tonal.pitch_centroid": "dyn.pitch_centroid", "tonal.key_clarity": "dyn.key_strength",
    "tonal.mode_score": "dyn.major_minus_minor", "tonal.chroma_entropy": "dyn.chroma_entropy",
    "level.lufs": "dyn.lufs", "level.crest_db": "dyn.crest_db", "level.dr_db": "dyn.lra",
    **{f"level.{k}": f"dyn.{k}" for k in ("loudness_slope", "momentary_std", "silence_frac", "peak_to_lufs",
                                          "clip_frac", "am_tremolo")},
    "rhythm.bpm": "dyn.tempo", "rhythm.beat_strength": "dyn.beat_strength", "rhythm.superflux_rate": "dyn.flux_mean",
    **{f"rhythm.{k}": f"dyn.{k}" for k in ("pulse_clarity", "tempo_stability", "offbeat_share", "hf_onset_rate")},
}
KEEP_PREFIX = ("timbral.", "passt.", "clap_music.", "clap_general.", "muq.", "desc.", "dyn.")


def _ok(col):
    return col.startswith(KEEP_PREFIX)


for r in rows:
    p, s2, t3 = (COL_MAP.get(r[k], r[k]) for k in ("primary_label", "second_scorer", "third_scorer"))
    if not _ok(p):
        p, r["label_strength"] = f"clap_music.{r['name']}", "text_only"
    if not _ok(s2) or s2 == p:
        s2 = f"muq.{r['name']}"
    if t3 and (not _ok(t3) or t3 in (p, s2)):
        t3 = ""
    r["primary_label"], r["second_scorer"], r["third_scorer"] = p, s2, t3
for k in list(VOCAB):
    VOCAB[k] = [it for it in VOCAB[k] if it[0] not in DROP]

# uniqueness of (primary, sign)
seen = {}
for r in rows:
    k = (r["primary_label"], r["sign"])
    assert k not in seen, (k, seen.get(k), r["name"])
    seen[k] = r["name"]
names = [r["name"] for r in rows]
assert len(names) == len(set(names)), [n for n, c in collections.Counter(names).items() if c > 1]

# ================================================================ prompts
rng = random.Random(20261005)
use = collections.Counter()

def pick(slot, k=1, exclude=()):
    """balanced: choose among the least-used items of the slot"""
    out = []
    for _ in range(k):
        items = [it for it in VOCAB[slot] if it[0] not in exclude and it[0] not in [o[0] for o in out]]
        m = min(use[it[0]] for it in items)
        cand = [it for it in items if use[it[0]] <= m + 1]
        it = rng.choice(cand)
        use[it[0]] += 1
        out.append(it)
    return out

prompts = []
def emit(text, category, tags):
    prompts.append({"prompt": text, "category": category, "tags": sorted(set(tags))})

MUSIC_T = ["{g} with {i1} and {i2}, {t}, {m}", "{m} {g} track featuring {i1}, {i2}, {t}", "{g} piece: {i1} over {i2}, {t}, {m} mood",
           "{t} {g} with {i1} and {i2}, {m}", "instrumental {g}, {i1}, {i2}, {t}, {m} atmosphere"]
for n in range(2500):
    g = pick("genre"); i = pick("inst", 3); m = pick("mood")
    mod_slot = rng.choice(["timbre", "prod", "space", "dyn", "rhythm"])
    t = pick(mod_slot)
    tags = [g[0][0], i[0][0], i[1][0], m[0][0], t[0][0]]
    s = rng.choice(MUSIC_T).format(g=g[0][1], i1=i[0][1], i2=i[1][1], t=t[0][1], m=m[0][1])
    if rng.random() < 0.5:
        s += f", plus {i[2][1]}"; tags.append(i[2][0])
    else:
        use[i[2][0]] -= 1
    if rng.random() < 0.4:
        g2 = pick("genre", exclude=[g[0][0]]); s += f", with {g2[0][1]} influences"; tags.append(g2[0][0])
    if rng.random() < 0.7:
        a = pick("abs"); s += f", evoking {a[0][1]}"; tags.append(a[0][0])
    if rng.random() < 0.8:
        x = pick(rng.choice(["rhythm", "timbre", "rhythm", "timbre", "dyn", "prod"]), exclude=[t[0][0]]); s += f", {x[0][1]}"; tags.append(x[0][0])
    emit(s, "music", tags)
for n in range(300):
    i = pick("inst"); t = pick(rng.choice(["timbre", "dyn", "space"])); m = pick("mood"); g = pick("genre")
    emit(f"solo {i[0][1]}, no other instruments, {t[0][1]}, {m[0][1]}, {g[0][1]} style", "solo",
         [i[0][0], t[0][0], m[0][0], g[0][0], "solo_instrument"])
SFX_T = ["{a} and {b}, {p}", "the sound of {a} with {b} in the background, {p}", "{a}, {b}, {p}, {q}", "foley: {a} followed by {b}, {p}"]
for n in range(1300):
    a, b = pick("sfx", 2); p = pick("space"); tmpl = rng.choice(SFX_T)
    tags = [a[0], b[0], p[0][0]]
    q = pick("prod") if "{q}" in tmpl else [("", "")]
    if q[0][0]:
        tags.append(q[0][0])
    s = tmpl.format(a=a[1], b=b[1], p=p[0][1], q=q[0][1])
    if rng.random() < 0.6:
        c = pick("sfx", exclude=[a[0], b[0]]); s += f", then {c[0][1]}"; tags.append(c[0][0])
    emit(s, "sfx", tags)
for n in range(300):
    g = pick("genre"); x = pick("sfx"); m = pick("mood"); i = pick("inst"); q = pick("prod")
    emit(f"{m[0][1]} {g[0][1]} beat with {i[0][1]} and {x[0][1]} in the background, {q[0][1]}", "hybrid", [g[0][0], x[0][0], m[0][0], i[0][0], q[0][0]])
for n in range(200):
    a = pick("abs", 2); g = pick("genre"); i = pick("inst"); m = pick("mood"); q = pick(rng.choice(["prod", "timbre"]))
    emit(f"{a[0][1]} {g[0][1]} soundscape, {i[0][1]}, {m[0][1]}, {q[0][1]}, like {a[1][1]}", "abstract", [a[0][0], a[1][0], g[0][0], i[0][0], m[0][0], q[0][0]])

KW = [(r["name"], [k for k in r["kw"] if len(k) >= 3]) for r in rows]
def kw_tags(text):
    t = text.lower()
    return [n for n, ks in KW if any(re.search(r"\b" + re.escape(k) + r"\b", t) for k in ks)]

tada = json.load(open(TADA, encoding="utf-8"))
for p in tada["test_prompts"]:
    emit(p, "tada_test", kw_tags(p))
holdout = set(tada["holdout_prompts"])
mc = list(csv.DictReader(open(MC_CSV, encoding="utf-8")))
vocal = re.compile(r"\b(vocal|vocals|sing|singer|singing|sings|voice|voices|rap|rapping|lyrics|choir|chant)\b", re.I)
mc_ok = [r for r in mc if not vocal.search(r["caption"])]
rng.shuffle(mc_ok)
for r in mc_ok[:300]:
    asp = " ".join(ast.literal_eval(r["aspect_list"]))
    emit(r["caption"], "musiccaps", kw_tags(r["caption"] + " " + asp))
assert len(prompts) == 5000, len(prompts)
assert not any(p["prompt"] in holdout for p in prompts)
for k, p in enumerate(prompts):
    p_id = k
    prompts[k] = {"id": p_id, "prompt": p["prompt"], "seed": 30000 + p_id, "category": p["category"], "tags": p["tags"]}

# sfx holdout (excluded from corpus)
hrng = random.Random(777)
SFXV = VOCAB["sfx"]
hold = []
for k in range(20):
    a, b = hrng.sample(SFXV, 2)
    s = f"{a[1]} and {b[1]}, recorded outdoors" if k % 2 else f"the sound of {a[1]} in a large room"
    hold.append({"id": 900000 + k, "prompt": s, "seed": 2115, "category": "sfx_holdout"})
corpus_text = {p["prompt"] for p in prompts}
assert not any(h["prompt"] in corpus_text for h in hold)

# counts
cnt = collections.Counter(t for p in prompts for t in p["tags"])
dense_scorers = {"desc", "dyn", "timbral"}
pop_n = {"music": sum(p["category"] != "sfx" for p in prompts), "sfx": sum(p["category"] == "sfx" for p in prompts), "all": len(prompts)}
for r in rows:
    r["expected_tagged_clips"] = cnt.get(r["name"], 0)
    dense = r["primary_label"].split(".")[0] in dense_scorers
    r["label_type"] = "dense" if dense else "sparse"
    r["class_rule"] = "quartile (top 25% vs bottom 25% of population)" if dense else "top 250 by score vs bottom 25% of population"
    r["population_n"] = pop_n[r["population"]]
    r["thin"] = "yes" if (not dense and r["expected_tagged_clips"] < 40) else ""

cols = ["name", "category", "primary_label", "sign", "second_scorer", "third_scorer", "label_strength", "label_type", "class_rule",
        "population", "population_n", "screen_set", "expected_tagged_clips", "thin", "difficulty", "pos_anchor", "neg_anchor", "blurb"]
if ARGS.v2:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import catalogue_v2_rows
    catalogue_v2_rows.write_v2(rows, prompts, cols, AS, OUT)
    raise SystemExit(0)
with open(OUT / "concept_catalogue.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
json.dump(prompts, open(OUT / "render_prompts.json", "w", encoding="utf-8"), indent=0, ensure_ascii=False)
json.dump(hold, open(OUT / "holdout_sfx_prompts.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# report
bycat = collections.Counter(r["category"] for r in rows)
bystr = collections.Counter(r["label_strength"] for r in rows)
print("rows", len(rows), dict(bycat), dict(bystr))
print("prompt categories", collections.Counter(p["category"] for p in prompts))
sp = [r for r in rows if r["label_type"] == "sparse"]
vals = sorted(r["expected_tagged_clips"] for r in sp)
print("sparse n", len(sp), "tagged min/median/max", vals[0], vals[len(vals) // 2], vals[-1])
print("thin", [r["name"] for r in rows if r["thin"]])
dn = [r for r in rows if r["label_type"] == "dense"]
print("dense n", len(dn), "tagged median", sorted(r["expected_tagged_clips"] for r in dn)[len(dn) // 2])
summary = {"rows": len(rows), "by_category": dict(bycat), "by_strength": dict(bystr),
           "prompt_categories": dict(collections.Counter(p["category"] for p in prompts)),
           "sparse_tagged_min_median_max": [vals[0], vals[len(vals) // 2], vals[-1]],
           "by_category_strength": {c: dict(collections.Counter(r["label_strength"] for r in rows if r["category"] == c)) for c in bycat}}
json.dump(summary, open(OUT / "catalogue_summary.json", "w"), indent=1)
print(json.dumps(summary["by_category_strength"]))
print(prompts[0]); print(prompts[2600]); print(prompts[3000]); print(prompts[4200]); print(prompts[4600])

# MuseTimbre prototype renders (runbook phase 1b): 8 solo clips per instrument, audio only, never labelled
proto = []
for nm, ph in VOCAB["inst"]:
    for s in range(8):
        proto.append({"id": 100000 + len(proto), "prompt": f"solo {ph}, no other instruments", "seed": 40000 + s,
                      "category": "musetimbre_proto", "tags": [nm]})
json.dump(proto, open(OUT / "proto_prompts.json", "w", encoding="utf-8"), indent=0, ensure_ascii=False)
print("proto", len(proto))
