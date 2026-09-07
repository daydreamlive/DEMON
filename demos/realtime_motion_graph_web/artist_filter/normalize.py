"""Normalization for the artist-name filter — SPEC.md steps 1-5.

Everything here is per-character with an index map back into the ORIGINAL
text, so a match can report the exact span it fired on even after NFKD
expansion (ß -> ss) or invisible-character deletion. The tables (invisibles,
homoglyphs, leet) live in data/filter_config.v1.json, shared verbatim with
the TypeScript port in demon-public-demo — change the data, not the code,
when tuning.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    """One normalized token with its original-text span.

    ``changed`` marks that homoglyph/symbol-leet folding altered the token —
    obfuscation evidence for the single_common decision rule (SPEC.md rule c).
    Digit-leet is NOT applied here; the matcher folds digits per window so a
    pure-digit token ("808") can never be folded into letters on its own.
    """

    norm: str
    start: int
    end: int
    changed: bool


def _codepoints(values: list[str]) -> frozenset[str]:
    return frozenset(chr(int(v, 16)) for v in values)


class Normalizer:
    """Tokenizer configured by filter_config tables."""

    def __init__(self, config: dict):
        self._invisibles = _codepoints(config["invisibles"])
        self._homoglyphs = dict(config["homoglyphs"])
        self._symbol_leet = dict(config["symbol_leet"])
        self._digit_leet = dict(config["digit_leet"])

    # ------------------------------------------------------------------
    def key(self, text: str) -> str:
        """Squash a NAME into its lookup key (build-time and query-time)."""
        return "".join(t.norm for t in self.tokenize(text))

    def fold_digits(self, squash: str) -> str:
        """Digit-leet fold of a window squash (matcher applies the alpha gate)."""
        return "".join(self._digit_leet.get(c, c) for c in squash)

    # ------------------------------------------------------------------
    def tokenize(self, text: str) -> list[Token]:
        """SPEC.md steps 1-5: normalize ``text`` into span-tracked tokens."""
        # Steps 1-4 in one pass: per original char, emit zero or more
        # normalized chars, each remembering its original index and whether
        # a confusables/leet table (not plain Unicode normalization) touched it.
        chars: list[str] = []
        origin: list[int] = []
        changed: list[bool] = []
        for idx, ch in enumerate(text):
            if ch in self._invisibles:
                continue
            folded = self._homoglyphs.get(ch)
            if folded is None:
                folded = self._symbol_leet.get(ch)
            was_table_fold = folded is not None
            base = folded if folded is not None else ch
            for d in unicodedata.normalize("NFKD", base):
                if unicodedata.category(d) == "Mn":
                    continue
                for c in d.casefold():
                    chars.append(c)
                    origin.append(idx)
                    changed.append(was_table_fold)

        # Step 5: split on anything outside [a-z0-9].
        tokens: list[Token] = []
        i, n = 0, len(chars)
        while i < n:
            c = chars[i]
            if not (c.isascii() and (c.isalpha() or c.isdigit())):
                i += 1
                continue
            j = i
            while j < n and chars[j].isascii() and (chars[j].isalpha() or chars[j].isdigit()):
                j += 1
            tokens.append(Token(
                norm="".join(chars[i:j]),
                start=origin[i],
                end=origin[j - 1] + 1,
                changed=any(changed[i:j]),
            ))
            i = j
        return tokens
