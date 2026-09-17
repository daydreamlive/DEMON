"""Conservative text constraints for local prompt rewrites.

These checks reject known contradictions, not acoustic bleed. Unknown solo
subjects are kept verbatim rather than guessed. A valid anchor is preferable
to a distinct variation that changes the requested performers.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

GUARD_VERSION = "lineup-3"

# Common instrument names, synonyms and component sounds. Longest matches win,
# so "electric piano" is not reduced to "piano", or "drum machine" to "drum".
_ALIASES = {
    "saxophone": "saxophone|sax|tenor sax|alto sax",
    "acoustic guitar": "acoustic guitar|steel string guitar",
    "electric guitar": "electric guitar",
    "nylon guitar": "nylon string guitar|nylon guitar|classical guitar",
    "slide guitar": "slide guitar|lap steel",
    "pedal steel": "pedal steel|pedal steel guitar",
    "guitar": "guitar",
    "piano": "piano|grand piano|acoustic piano",
    "electric piano": "electric piano|rhodes|wurlitzer|electric keys",
    "keys": "keys|keyboard",
    "synth": "analog synthesizer|analog synth|synthesizer|synth",
    "modular synth": "modular synthesizer|modular synth",
    "bass synth": "bass synthesizer|bass synth|synth bass",
    "bass guitar": "bass guitar|electric bass|fretless bass",
    "double bass": "double bass|upright bass|contrabass",
    "bass": "bass|bassline",
    "drums": "drum kit|drums|drum|kick|snare|cymbal|hi hat|hi hats|toms|rimshot",
    "percussion": "percussion|shaker|tambourine|clap|claps",
    "drum machine": "drum machine|drum machines",
    "hand drums": "hand drums|hand drum|conga|bongo|djembe",
    "tabla": "tabla",
    "timpani": "timpani|kettledrum",
    "steel drum": "steel drum|steel pan|steelpan",
    "violin": "violin|fiddle",
    "viola": "viola",
    "cello": "cello|violoncello",
    "strings": "strings|string ensemble|string section",
    "flute": "flute",
    "clarinet": "clarinet",
    "oboe": "oboe",
    "bassoon": "bassoon",
    "woodwinds": "woodwinds|woodwind",
    "trumpet": "trumpet",
    "trombone": "trombone",
    "french horn": "french horn",
    "tuba": "tuba",
    "brass": "brass|horn|horns|horn section",
    "organ": "organ|hammond|tonewheel organ",
    "pipe organ": "pipe organ",
    "accordion": "accordion",
    "harmonica": "harmonica|mouth organ",
    "harp": "harp",
    "harpsichord": "harpsichord",
    "celesta": "celesta|celeste",
    "glockenspiel": "glockenspiel",
    "vibraphone": "vibraphone|vibes",
    "marimba": "marimba",
    "xylophone": "xylophone",
    "bells": "bells|bell|chime|chimes",
    "ukulele": "ukulele|uke",
    "banjo": "banjo",
    "mandolin": "mandolin",
    "sitar": "sitar",
    "oud": "oud",
    "koto": "koto",
    "dulcimer": "dulcimer",
    "kalimba": "kalimba|thumb piano",
    "theremin": "theremin",
    "erhu": "erhu",
    "bagpipes": "bagpipes|bagpipe",
    "tin whistle": "tin whistle|penny whistle",
    "music box": "music box",
    "texture": "ambient texture|soundscape|ambient soundscape",
    "voice": "voice|vocal|vocals|singer|singing",
    "choir": "choir|choral|chorus of voices",
}
_LOOKUP = {alias: subject for subject, aliases in _ALIASES.items() for alias in aliases.split("|")}
_NAMES = re.compile(r"(?<!\w)(?:" + "|".join(
    re.escape(alias) for alias in sorted(_LOOKUP, key=len, reverse=True)
) + r")(?:s|es)?(?!\w)")

# A specific instrument may be described by its family, but family membership
# never permits an unrelated sibling (a violin cannot become a cello).
_FAMILIES = {
    **{x: {"guitar"} for x in ("acoustic guitar", "electric guitar", "nylon guitar", "slide guitar", "pedal steel")},
    "piano": {"keys"}, "electric piano": {"piano", "keys"},
    "modular synth": {"synth"}, "bass synth": {"synth", "bass"},
    "bass guitar": {"bass"}, "double bass": {"bass", "strings"},
    **{x: {"strings"} for x in ("violin", "viola", "cello")},
    **{x: {"woodwinds"} for x in ("flute", "clarinet", "oboe", "bassoon", "saxophone")},
    **{x: {"brass"} for x in ("trumpet", "trombone", "french horn", "tuba")},
    "pipe organ": {"organ", "keys"}, "organ": {"keys"},
    "drum machine": {"drums", "percussion"}, "drums": {"percussion"},
    "hand drums": {"drums", "percussion"}, "tabla": {"hand drums", "drums", "percussion"},
    "timpani": {"drums", "percussion"}, "steel drum": {"drums", "percussion"},
    "choir": {"voice"},
    "glockenspiel": {"bells"}, "music box": {"bells"},
}
_PERCUSSION = {"drums", "percussion", "drum machine", "hand drums", "tabla", "timpani", "steel drum"}
_SOLO = re.compile(r"\b(?:solo|unaccompanied|single (?:unaccompanied )?instrument|single (?:evolving )?layer|playing alone)\b")
_LINEUPS = {
    "band": r"\b(?:full (?:band|arrangement|production|mix)|backing band|rhythm section|with (?:a )?band)\b",
    "duo": r"\b(?:duo|duet)\b", "trio": r"\btrio\b", "quartet": r"\bquartet\b",
    "quintet": r"\bquintet\b", "sextet": r"\bsextet\b",
    "septet": r"\bseptet\b", "octet": r"\boctet\b",
    "big band": r"\b(?:big|large) band\b",
    "chamber": r"\bchamber (?:group|ensemble)\b",
    "overdubbed": r"\b(?:multi tracked|multitracked|overdubbed|layered takes)\b",
    "accompanied": r"\b(?:with (?:light )?accompaniment|accompanied by)\b",
    "orchestra": r"\b(?:orchestra|orchestral)\b",
    "ensemble": r"\b(?:ensemble|group)\b",
}
_ACCOMPANIMENT = re.compile(r"\b(?:accompanied by|accompaniment|backing|backbeat|arrangement|band|orchestra|ensemble|rhythm section)\b")
_RHYTHM = re.compile(r"\b(?:grooves?|beats?|rhythms?)\b")
# Radio-frequency bands describe filtering, not additional performers.
_RADIO_BAND = re.compile(r"\b(?:(?:am|fm) radio|radio frequency) band\b")
_NEGATED = re.compile(r"\b(?:no|without) (?:any )?(?:vocals?|singing|accompaniment|backing band|drums?)\b")
_COMPONENTS = re.compile(r"\b(?:singing (?:vibrato|tone|sustain)|(?:single )?bowed voice|(?:struck|plucked|bowed|nylon|steel) strings|struck bells)\b")


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("-", " ")).strip()


def subjects(text: str) -> tuple[str, ...]:
    found = []
    for match in _NAMES.finditer(_COMPONENTS.sub("", _NEGATED.sub("", _normal(text)))):
        word = match.group()
        subject = _LOOKUP.get(word) or _LOOKUP.get(word[:-1]) or _LOOKUP.get(word[:-2])
        if subject and subject not in found:
            found.append(subject)
    return tuple(found)


def _lineups(text: str) -> frozenset[str]:
    # "String ensemble" can name the selected source, not its accompaniment.
    text = text.replace("string ensemble", "strings").replace("vocal ensemble", "choir")
    return frozenset(key for key, pattern in _LINEUPS.items() if re.search(pattern, text))


@dataclass(frozen=True)
class PromptConstraint:
    source: str
    instruments: tuple[str, ...]
    solo: bool
    lineups: frozenset[str]
    active: bool

    @classmethod
    def infer(cls, source: str, deck: str) -> "PromptConstraint":
        normal = _normal(source)
        instruments = subjects(source)
        # Drop generic family mentions already explained by a specific source.
        families = set().union(*(_FAMILIES.get(x, set()) for x in instruments)) if instruments else set()
        instruments = tuple(x for x in instruments if x not in families)
        return cls(source.strip(), instruments, bool(_SOLO.search(normal)),
                   _lineups(normal), deck == "sa3")

    def violations(self, candidate: str) -> tuple[str, ...]:
        if not self.active:
            return ()
        normal = _normal(candidate)
        mentioned = set(subjects(candidate))
        reasons = []
        if not candidate.strip():
            return ("empty",)
        if re.search(r"\b(\w+)\s+\1\b", normal) and not re.search(r"\b(\w+)\s+\1\b", _normal(self.source)):
            reasons.append("repetition")
        if self.solo and not self.instruments:
            return () if candidate.strip() == self.source else ("unknown_solo_subject",)
        if any(x not in mentioned for x in self.instruments):
            reasons.append("missing_subject")
        if self.solo:
            if not _SOLO.search(normal):
                reasons.append("missing_solo")
            if self.lineups or _lineups(normal) or _ACCOMPANIMENT.search(_RADIO_BAND.sub("", _NEGATED.sub("", normal))):
                reasons.append("accompaniment")
            allowed = set(self.instruments)
            for subject in self.instruments:
                allowed.update(_FAMILIES.get(subject, set()))
            if mentioned - allowed or len(self.instruments) > 1:
                reasons.append("other_source")
            # A separate rhythm clause can imply backing even without naming
            # another instrument (e.g. "driving syncopated rhythm"). Preserve
            # explicit rhythm requests and percussion, but don't invent them
            # from a genre or mood. "Syncopated guitar phrasing" stays valid.
            if not set(self.instruments) & _PERCUSSION and not _RHYTHM.search(_normal(self.source)):
                if _RHYTHM.search(normal):
                    reasons.append("added_rhythm")
        elif self.lineups:
            if _SOLO.search(normal) or not self.lineups <= _lineups(normal):
                reasons.append("changed_lineup")
        # Reject known technique contradictions instead of merely checking
        # that the right instrument name appears somewhere in the sentence.
        if set(self.instruments) & {"piano", "electric piano", "violin", "viola", "cello", "guitar", "acoustic guitar", "electric guitar", "ukulele"}:
            if re.search(r"\b(?:reedy|breath driven|tonguing|embouchure)\b", normal) and not re.search(r"\b(?:reedy|breath driven|tonguing|embouchure)\b", _normal(self.source)):
                reasons.append("incompatible_technique")
        return tuple(reasons)

    def accept(self, candidate: str, fallback: str = "") -> str:
        """Choose a valid result, or a validated fallback. Never splice text."""
        if not self.violations(candidate):
            return candidate.strip()
        if fallback and not self.violations(fallback):
            return fallback.strip()
        return ""
