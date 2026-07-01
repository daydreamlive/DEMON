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

# Stable Audio 3 policy. SA3 is the same engine theDAW documents (T5Gemma text
# encoder + separate duration signal), so its prompting guide
# (theDAW/docs/guides/prompting.md) is the authoritative reference here. SA3
# reads the prompt as a producer's natural-language DESCRIPTION of a track, not a
# bare tag list, so this system prompt asks for compact descriptive phrases,
# genre + hook instrument first (they carry the most weight and the first phrases
# dominate the result). BPM stays OUT of the clean output on purpose: the M4L
# wire helper (demon::promptMeta) appends the REAL project tempo/key downstream
# for backend=="sa3", so a genre-typical number here would ship two conflicting
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


def enhance_prompt(idea: str, backend: str = "acestep") -> tuple[str, bool]:
    """Expand a rough idea into a rich prompt for the active backend.

    ``backend`` selects the LLM policy: ``"sa3"`` uses the Stable Audio 3
    natural-language description style; anything else (default) uses the legacy
    ACE-Step comma-tag style. The default keeps the pre-existing no-``backend``
    call path byte-identical. ``_sanitize`` (incl. the bpm-strip) is shared, so
    both backends stay bpm-number-free.

    Returns ``(text, ok)``. ``ok=False`` means enhancement was unavailable or
    failed and ``text`` is the caller's input echoed back unchanged, so the
    client keeps what the user typed.
    """
    idea = (idea or "").strip()
    if not idea:
        return idea, False
    if backend == "sa3":
        system = _SYSTEM_SA3
        user = f'Rough idea: "{idea[:300]}". Expand it into one rich Stable Audio 3 prompt.'
    else:
        system = _SYSTEM
        user = f'Rough idea: "{idea[:300]}". Expand it into one rich ACE-Step prompt.'
    raw = _ask_haiku(system, user)
    if not raw:
        return idea, False
    out = _sanitize(raw)
    return (out, True) if out else (idea, False)
