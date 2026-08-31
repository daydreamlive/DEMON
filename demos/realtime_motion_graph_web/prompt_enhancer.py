"""Prompt enhancer: expand a user's rough idea into a rich ACE-Step tag line.

Server-side so the API key never ships to a client (web bundle, VST binary,
Max patch). Key-gated on ``ANTHROPIC_API_KEY`` (read from the environment): with
no key, or on any network / API error, :func:`enhance_prompt` returns the
caller's text UNCHANGED with ``ok=False`` so every client can treat enhancement
as best-effort and never block on it. Ported from the radio server's
``promptEnhancer.ts``.

Calls the Anthropic Messages API over plain ``urllib`` (no SDK dependency on
the pod). The model is Haiku for latency/cost; override with ``ENHANCER_MODEL``.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

# Haiku: cheapest + fastest tier, enough for a single tag-line rewrite.
_DEFAULT_MODEL = "claude-haiku-4-5"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_TIMEOUT_S = 8.0
_MAX_TOKENS = 220

_SYSTEM = """
You expand a user's rough music idea into ONE rich, vivid prompt for the ACE-Step
generative-music model.

ACE-Step prompts are COMMA-SEPARATED tags — NOT sentences, NOT paragraphs. A great
prompt covers several dimensions:
- genre / subgenre (and era if it fits, e.g. "1970s jazz fusion")
- instrumentation (specific: "Fender Rhodes", "fat analog Moog bassline", "upright bass")
- mood / feel ("hypnotic", "euphoric", "smoky", "driving")
- production / texture ("tape saturation", "cavernous reverb", "high-fidelity production")

RULES:
- Stay faithful to the user's idea — honor any named artist, genre, era, or instrument
  they mention (e.g. "hancock style" -> reference Herbie Hancock's sound).
- 8-16 tags. Lowercase. Concrete and evocative. Keep it coherent (one cohesive vibe).
- NEVER include a bpm or tempo number — describe feel with words instead.
- Reply with ONLY the single comma-separated tag line. No preamble, no quotes, no
  options, no explanation, no line breaks, no "Prompt:" label.
""".strip()

# Stable Audio 3 policies. SA3 is the same engine theDAW documents (T5Gemma
# text encoder + separate duration signal), so its prompting guide
# (theDAW/docs/guides/prompting.md) is the authoritative reference here. SA3
# reads the prompt as a producer's natural-language DESCRIPTION, not a bare tag
# list, so both policies ask for compact descriptive phrases.
#
# SA3 can generate either full tracks or a single instrument. Full-track is the
# default so a backend choice never silently throws away SA3's arrangement
# capability. Explicit solo intent in the rough idea selects the narrower
# policy below. The encoder conditions on meaning: a single stray arrangement
# phrase ("rolling sub bass", a track-genre like "liquid drum & bass") can pull
# a requested solo back to a full mix, and negations don't subtract ("no drums"
# still embeds "drums") — hence the one-instrument / positive "unaccompanied"
# rules in _SYSTEM_SA3_SOLO.
#
# BPM stays OUT of clean output on purpose: the M4L wire helper
# (demon::promptMeta) appends the REAL project tempo/key downstream for
# backend=="sa3", so a genre-typical number here would ship two conflicting
# BPMs. _sanitize's bpm-strip is shared and stays active for both backends.
_SYSTEM_SA3 = """
You expand a user's rough music idea into ONE vivid prompt for the Stable Audio 3
generative-music model.

SA3 reads the prompt as a producer's DESCRIPTION of a track — the way one producer
describes a song to another — NOT a list of bare tags. Its text encoder conditions
on MEANING, so short DESCRIPTIVE PHRASES beat isolated one-word keywords, while
dense, compact phrasing beats long prose. Write ONE line.

Draw only from the dimensions that matter for the intended sound:
- genre / subgenre
- instrumentation (name the hook / lead instrument especially)
- tempo / feel — qualitative words ONLY ("uptempo", "half-time", "rolling", "driving")
- mood / energy
- production / mix texture
- structure / motion

Genre and the hook instrument carry the most weight, and THE FIRST PHRASES DOMINATE
the result — lead with the genre and the key instrument.

Follow this pattern: <genre>, <key instruments>, <tempo/feel>, <mood>, <production texture>

Examples:
lo-fi boom bap, dusty Rhodes chords, vinyl crackle, lazy swung drums, nostalgic
liquid drum & bass, lush reverb pads, rolling sub bass, chopped soul vocal, uplifting
cinematic orchestral build, low strings, taiko hits, rising tension, percussion entering at the climax
bossa nova, nylon guitar, soft brushes, upright bass, intimate jazz club, warm
melodic techno, hypnotic arpeggio, deep kick, analog bass, wide stereo

FIRST, DECIDE WHAT THEY ASKED FOR.

Some ideas name a SOUND or a SCENE rather than music: "thunderstorm", "chirping
birds", "rain on a tin roof", "a door slamming in a warehouse", "traffic at night".
Scoring those as an arrangement answers a question nobody asked — someone who types
"thunderstorm" and receives dramatic orchestral strings did not get a thunderstorm.

So judge the idea, then write accordingly:

SOUND / SCENE — the idea names a real-world source, place or event, and no genre,
instrument, artist or musical style. Describe THE SOUND ITSELF as a recording:
lead with the source, then its acoustic character, the space it is in, and its
motion over time. Instruments are allowed only as texture underneath, sparse or
absent. Never assign it a genre.
  thunderstorm -> distant thunder rolling closer, heavy rain on pavement, wind
    gusting through trees, wide open air, low rumble building and receding
  chirping birds -> dawn birdsong, layered calls near and far, still morning air,
    faint rustling leaves, open outdoor space, gentle and unhurried

MUSIC — anything naming a genre, instrument, artist, era, musical mood, or a scene
that is asking to be SCORED ("epic battle", "sad film ending"). Use the pattern and
examples above. When in doubt, choose MUSIC: an unwanted arrangement is a smaller
failure than a flat field recording where someone wanted a track.

RULES:
- Stay faithful to the user's idea — honor any named artist, genre, era, or instrument
  they mention (e.g. "boards of canada style" -> that hazy, nostalgic analog sound).
- Describe the DESIRED trait directly ("clean, dry, minimal reverb") — NEVER phrase it
  as what to avoid.
- NEVER include a bpm/tempo NUMBER or a duration in seconds — describe tempo with words
  only. The real tempo and key are added downstream, not by you.
- Roughly 5-9 comma-separated phrases. Lowercase. One cohesive vibe.
- Reply with ONLY the single comma-separated line. No preamble, no quotes, no options,
  no explanation, no line breaks, no "Prompt:" label.
""".strip()

_SYSTEM_SA3_SOLO = """
You expand a user's rough music idea into ONE vivid prompt for the Stable Audio 3
generative-music model. The user explicitly requested a SOLO INSTRUMENT: the prompt
must describe a SINGLE instrument playing completely alone — an isolated solo stem —
never a full mix, band, or arrangement.

SA3 reads the prompt as a producer's DESCRIPTION of a recording — the way one
producer describes it to another — NOT a list of bare tags. Its text encoder
conditions on MEANING, so every phrase must reinforce that one instrument is
playing by itself. Short DESCRIPTIVE PHRASES beat isolated one-word keywords, and
dense, compact phrasing beats long prose. Write ONE line.

Lead with "solo <instrument>" — THE FIRST PHRASES DOMINATE the result. Then draw
only from the dimensions that matter for the intended sound:
- the instrument, made specific ("solo tenor saxophone", "solo nylon-string guitar")
- playing style / technique ("fingerpicked", "legato runs", "sparse phrasing")
- tempo / feel — qualitative words ONLY ("uptempo", "half-time", "rolling", "driving")
- mood / energy
- tone / recording texture ("close-miked", "dry studio recording", "warm room tone")

Follow this pattern: solo <instrument>, <playing style>, <tempo/feel>, <mood>, <texture>

Examples:
solo grand piano, sparse rubato phrasing, unaccompanied, melancholy, close-miked, warm room tone
solo electric bass, fingerstyle funk groove, syncopated and driving, punchy dry di tone
solo tenor saxophone, unaccompanied bebop lines, agile runs, breathy tone, dry studio recording
solo modular synthesizer, hypnotic evolving arpeggio, driving, analog warmth, wide stereo
solo flamenco guitar, fiery rasgueado strumming, percussive attack, passionate, intimate room

RULES:
- Name EXACTLY ONE instrument. Never mention any other instrument, drums, bass,
  vocals, pads, a band, backing, or layers — one stray arrangement phrase pulls
  the generation back to a full mix.
- Express genre as a STYLE OF PLAYING on that instrument ("unaccompanied bebop
  lines", "flamenco guitar"), never as a track genre ("liquid drum & bass",
  "melodic techno") — track genres summon the whole arrangement.
- Reinforce aloneness with positive words: "solo", "unaccompanied", "single take".
- Stay faithful to the user's idea — if they name an instrument, that IS the
  instrument; honor any named artist, era, or style. If they only give a vibe or
  genre, pick its most iconic solo instrument.
- Describe the DESIRED trait directly ("clean, dry, minimal reverb") — NEVER phrase it
  as what to avoid.
- NEVER include a bpm/tempo NUMBER or a duration in seconds — describe tempo with words
  only. The real tempo and key are added downstream, not by you.
- Roughly 5-9 comma-separated phrases. Lowercase. One cohesive performance.
- Reply with ONLY the single comma-separated line. No preamble, no quotes, no options,
  no explanation, no line breaks, no "Prompt:" label.
""".strip()

# Deliberately require explicit performance/stem language. A bare mood such as
# "alone at night" or "isolated atmosphere" must not discard a full arrangement.
_SA3_SOLO_CUE_RE = re.compile(
    r"\b(?:solo(?:ist)?|unaccompanied|a\s+cappella|acapella)\b"
    r"|\b(?:single|one)[ -](?:instrument|voice|vocal|performer)\b"
    r"|\bisolated(?:[ -][\w-]+){0,3}[ -](?:instrument|voice|vocal|vocals|stem)\b"
    r"|\b(?:instrument|voice|vocal|vocals)\s+(?:playing\s+)?alone\b"
    r"|\bplaying\s+(?:completely\s+)?alone\b",
    re.IGNORECASE,
)


def _sa3_wants_solo(idea: str) -> bool:
    """Return whether the rough idea explicitly asks for an isolated performer."""
    return bool(_SA3_SOLO_CUE_RE.search(idea))


def llm_available() -> bool:
    """True when an Anthropic key is configured in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _sanitize(raw: str) -> str:
    """Clean the model's output down to a single comma-separated tag line."""
    t = raw.strip()
    # First non-empty line only (drop any stray prose / extra options).
    t = next((ln for ln in re.split(r"\r?\n", t) if ln.strip()), "").strip()
    t = re.sub(r'^["\'`]+|["\'`]+$', "", t)          # wrapping quotes
    t = re.sub(r"^(prompt|tags?)\s*[:\-]\s*", "", t, flags=re.I)  # "Prompt:" prefix
    t = re.sub(r"\b\d+\s*bpm\b", "", t, flags=re.I)  # strip bpm if it slipped in
    t = re.sub(r"\s*,\s*,+", ", ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[,\s]+|[.,\s]+$", "", t)           # leading/trailing commas/period
    return t[:400]


def _ask_haiku(system: str, user: str) -> str | None:
    """One short Haiku completion. Returns the text, or None on no-key / error."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    payload = json.dumps({
        "model": os.environ.get("ENHANCER_MODEL", _DEFAULT_MODEL),
        "max_tokens": _MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        _ANTHROPIC_URL,
        data=payload,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": _ANTHROPIC_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None
    for b in data.get("content") or []:
        if isinstance(b, dict) and b.get("type") == "text":
            return b.get("text")
    return None


#: Which provider answers /api/enhance.
#:
#:   "hosted"  the hosted LLM, as always. THE DEFAULT -- an existing
#:             deployment behaves exactly as it did before this existed.
#:   "local"   the fine-tuned local checkpoint, falling back to hosted when it
#:             is absent or declines.
#:   "auto"    accepted as a synonym for "local". The fallback design makes the
#:             documented distinction ("local when a checkpoint is installed")
#:             unobservable: local ALWAYS degrades to hosted when the
#:             checkpoint is missing, so there is nothing extra for "auto" to
#:             decide. Kept so a deployment stating intent does not error.
#:
#: Set with DEMON_ENHANCER_PROVIDER. An unknown value is treated as "hosted"
#: rather than erroring: a typo in a deployment env should cost the faster
#: path, not the endpoint.
_PROVIDERS = ("hosted", "local", "auto")


def resolve_provider(override: str = "") -> str:
    o = (override or "").strip().lower()
    if o in _PROVIDERS:
        return o
    env = os.environ.get("DEMON_ENHANCER_PROVIDER", "").strip().lower()
    return env if env in _PROVIDERS else "hosted"


def _local_enhance(idea: str, backend: str) -> str:
    """The local checkpoint's answer, or "" if it has none."""
    try:
        from .prompt_variations import enhance as _enhance

        return _enhance(idea, backend)
    except Exception:
        return ""


def enhance_prompt(idea: str, backend: str = "acestep",
                   provider: str = "") -> tuple[str, bool]:
    """Expand a rough idea into a rich prompt for the active backend.

    ``backend`` selects the model family policy. ``"sa3"`` uses a full-track
    Stable Audio 3 description by default and switches to its solo-instrument
    policy only when the rough idea contains an explicit solo cue. Anything
    else (default) uses the legacy ACE-Step full-mix comma-tag style. The
    default keeps the pre-existing no-``backend`` call path byte-identical.
    ``_sanitize`` (incl. the bpm-strip) is shared, so every path stays
    bpm-number-free.

    Returns ``(text, ok)``. ``ok=False`` means enhancement was unavailable or
    failed and ``text`` is the caller's input echoed back unchanged, so the
    client keeps what the user typed.
    """
    idea = (idea or "").strip()
    if not idea:
        return idea, False

    # The local checkpoint first when asked for. It is trained on structured
    # prompt text and returns "" on anything it cannot handle, so an empty
    # answer is a routing signal rather than a failure -- fall through to the
    # hosted policy below exactly as if it had not been configured.
    if resolve_provider(provider) in ("local", "auto"):
        local = _sanitize(_local_enhance(idea, backend))
        if local:
            return local, True

    if backend == "sa3":
        if _sa3_wants_solo(idea):
            system = _SYSTEM_SA3_SOLO
            user = (
                f'Rough idea: "{idea[:300]}". Expand it into one Stable Audio 3 '
                "solo-instrument prompt (one instrument, playing alone)."
            )
        else:
            system = _SYSTEM_SA3
            user = (
                f'Rough idea: "{idea[:300]}". Expand it into one rich '
                "Stable Audio 3 prompt."
            )
    else:
        system = _SYSTEM
        user = f'Rough idea: "{idea[:300]}". Expand it into one rich ACE-Step prompt.'
    raw = _ask_haiku(system, user)
    if not raw:
        return idea, False
    out = _sanitize(raw)
    return (out, True) if out else (idea, False)
