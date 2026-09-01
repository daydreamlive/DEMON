"""Token-window matcher for the artist-name filter — SPEC.md.

Window-hash lookup, not an automaton: windows are token-aligned by
construction (no `queen`-in-`frequency` substring hits), the loop ports 1:1
to the TypeScript twin in demon-public-demo, and the whole thing is a few
hundred dict probes per prompt — microseconds, safe upstream of the audio
hot path.
"""

from __future__ import annotations

from dataclasses import dataclass

from .normalize import Normalizer, Token

MULTI = "multi"
SINGLE_RARE = "single_rare"
SINGLE_COMMON = "single_common"

_APOSTROPHES = ("'", "’", "ʼ")


@dataclass(frozen=True)
class ArtistMatch:
    display: str          # canonical display name, e.g. "Taylor Swift"
    cls: str              # multi | single_rare | single_common
    evidence: str         # exact | fragmented | obfuscated | cue
    span: tuple[int, int]  # original-text char span [start, end)


class Matcher:
    def __init__(self, normalizer: Normalizer,
                 entries: dict[str, tuple[str, str, int]], config: dict):
        """``entries``: squash key -> (display, class, canonical token count)."""
        self._norm = normalizer
        self._entries = entries
        # Cues are DIRECTIONAL: "queen style" reads as a style reference,
        # "style, queen bassline" does not — an undirected radius let a
        # trailing cue poison the words after it ("french house style, four
        # to the floor" must never fire on "four").
        self._cues_before = frozenset(config["cues_before"])
        self._cues_after = frozenset(config["cues_after"])
        self._genres = frozenset(config.get("genre_terms", ()))
        self._cue_radius = int(config["cue_radius"])
        self._max_chars = int(config["max_window_chars"])
        self._max_tokens = int(config["max_window_tokens"])
        self._affix_min = int(config["affix_min_len"])

    # ------------------------------------------------------------------
    def scan(self, text: str) -> ArtistMatch | None:
        tokens = self._norm.tokenize(text)
        for i in range(len(tokens)):
            best: ArtistMatch | None = None
            squash = ""
            for j in range(i, min(len(tokens), i + self._max_tokens)):
                squash += tokens[j].norm
                if len(squash) > self._max_chars:
                    break
                m = self._probe(text, tokens, i, j, squash)
                if m is not None:
                    best = m  # longest window at this start wins
            if best is None and len(tokens[i].norm) >= self._affix_min:
                best = self._probe_affixes(tokens[i])
            if best is not None:
                return best
        return None

    # ------------------------------------------------------------------
    def _probe(self, text: str, tokens: list[Token], i: int, j: int,
               squash: str) -> ArtistMatch | None:
        window = tokens[i:j + 1]
        table_folded = any(t.changed for t in window)
        entry = self._entries.get(squash)
        leet_folded = False
        if entry is None and any(c.isalpha() for c in squash):
            folded = self._norm.fold_digits(squash)
            if folded != squash:
                entry = self._entries.get(folded)
                leet_folded = entry is not None
        if entry is None:
            return None
        display, cls, ntokens = entry
        obfuscated = table_folded or leet_folded
        # Fragmentation = MORE pieces than the name canonically has. A
        # two-token name matched by a two-token window ("the band") is just
        # the name written normally, not evasion.
        fragmented = (j - i + 1) > ntokens
        cue = self._has_cue(text, tokens, i, j)
        if cls == SINGLE_COMMON and not (fragmented or obfuscated):
            # A common-word name spoken plainly. Only a style cue makes it an
            # artist reference — and a genre word RIGHT NEXT to it makes it a
            # genre phrase again ("chicago house style", "detroit techno
            # vibes"): the place/word reading wins over the band reading.
            if not cue or self._genre_adjacent(tokens, i, j):
                return None
        if obfuscated:
            evidence = "obfuscated"
        elif fragmented and cls != MULTI:
            evidence = "fragmented"
        elif cue and cls == SINGLE_COMMON:
            evidence = "cue"
        else:
            evidence = "exact"
        return ArtistMatch(display, cls, evidence,
                           (window[0].start, window[-1].end))

    def _genre_adjacent(self, tokens: list[Token], i: int, j: int) -> bool:
        if i > 0 and tokens[i - 1].norm in self._genres:
            return True
        return j + 1 < len(tokens) and tokens[j + 1].norm in self._genres

    def _probe_affixes(self, token: Token) -> ArtistMatch | None:
        """`taylorswift4ever`: long single tokens, multi-class keys only."""
        t = token.norm
        for length in range(self._affix_min, len(t)):
            for cand in (t[:length], t[len(t) - length:]):
                entry = self._entries.get(cand)
                if entry is not None and entry[1] == MULTI:
                    return ArtistMatch(entry[0], MULTI, "exact",
                                       (token.start, token.end))
        return None

    def _has_cue(self, text: str, tokens: list[Token], i: int, j: int) -> bool:
        for k in range(max(0, i - self._cue_radius), i):
            if tokens[k].norm in self._cues_before:
                return True
        for k in range(j + 1, min(len(tokens), j + 1 + self._cue_radius)):
            if tokens[k].norm in self._cues_after:
                return True
        # Possessive: the matched window immediately followed by 's
        # (tokenized as a bare `s` right after an apostrophe).
        if j + 1 < len(tokens):
            nxt = tokens[j + 1]
            between = text[tokens[j].end:nxt.start]
            if nxt.norm == "s" and between and all(c in _APOSTROPHES for c in between):
                return True
        return False
