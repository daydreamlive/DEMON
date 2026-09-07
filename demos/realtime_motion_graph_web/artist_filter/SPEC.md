# Artist-name filter — normative algorithm (v1)

This spec is the contract between the Python implementation here (the pod,
which enforces) and the TypeScript implementation in demon-public-demo (the
publish path). Both are driven by the SAME versioned data artifacts and must
pass the SAME golden vectors (`data/golden_vectors.v1.json`). Change the
algorithm here first, regenerate vectors, then port.

Motivation: name-based artist invocation must be rejected even under evasion
(Isbell v. Suno, Aug 2026 — "t a y l o r s w i f t" defeated Suno's filter).
Deterministic, microsecond-scale, hot-path safe: no LLM in the decision.

## Inputs

- `data/artists.v1.json.gz` — `{version, source, generated, entries}` where
  each entry is `[squash_key, display_name, class, ntokens]`,
  class ∈ `multi | single_rare | single_common`; `ntokens` is the name's
  canonical token count (fragmentation evidence compares against it).
- `data/filter_config.v1.json` — normalization tables (invisibles, homoglyphs,
  symbol/digit leet), cue tokens, window bounds.

## Normalization (ordered; produces tokens with original-text spans)

1. **Strip invisibles** (delete, do not split): ZWSP/ZWNJ/ZWJ, soft hyphen,
   BOM/word-joiner, variation selectors — the `invisibles` codepoint list.
   Deleting merges `tay​lor` → `taylor`.
2. **Fold homoglyphs**: explicit confusables table (Cyrillic/Greek lookalikes,
   Latin extensions NFKD won't split: ø→o, đ→d, ł→l, æ→ae, œ→oe). Data-driven.
3. **Fold symbol leet**: `$→s @→a !→i` (must happen BEFORE tokenization or the
   symbols act as separators: `T@ylor $wift` would shatter).
4. **NFKD → drop combining marks (category Mn) → casefold.** `Beyoncé`→
   `beyonce`, fullwidth→ASCII, ß→ss. TS uses NFKD + strip `\p{M}` +
   `toLowerCase()`; golden vectors pin divergences (İ, ß, ligatures).
5. **Tokenize** on any char outside `[a-z0-9]`, keeping each token's original
   char span and a `changed` flag (steps 2–3 altered it = obfuscation
   evidence).

Steps 1–4 are per-character with an index map so spans survive expansion.

## Matching: token-window hash lookup

Build (offline): every name/alias → `squash` = its normalized tokens joined
with nothing (`daft punk`→`daftpunk`); map `squash → (display, class)`.
Keys shorter than `min_key_len` (3) are dropped at build.

Query: for tokens `t_i..t_j` contiguous, bounded by `max_window_chars` (34)
summed length and `max_window_tokens` (24):

- look up `concat(t_i..t_j)`;
- if the window contains ≥1 alphabetic char, also look up the **digit-leet
  fold** of the concat (`0→o 1→i 3→e 4→a 5→s 7→t 8→b`). Pure-digit windows
  (`808`, `140`) never fold and never match.

Additionally, for a single token of length ≥ `affix_min_len` (8): every
prefix and suffix of length ≥ 8 is looked up, **multi-class keys only**
(`taylorswift4ever`). Interior embedding (`xxtaylorswiftxx`) is a documented
v1 gap.

Token alignment is what kills substring false positives: `queen` never
matches inside the single token `frequency`.

## Decision rules per matched class

- `multi` — fires on any hit (evasion forms included; `tail or swift`
  squashes to `tailorswift` ≠ `taylorswift`).
- `single_rare` — fires on any hit, including fragmented (`e m i n e m`).
- `single_common` (name is a common English word: prince, queen, air, low…)
  — fires only if at least one of:
  (a) a **directional cue** within `cue_radius` (2) tokens: `cues_after`
      tokens after the window (`queen style`, `drake type beat`,
      `madonna-esque`) or `cues_before` tokens before it (`like prince`,
      `inspired by queen`) — direction matters, or a trailing "…style," would
      poison the words after it — or the token is **possessive**
      (immediately followed by an apostrophe + `s` token) — UNLESS a
      `genre_terms` token sits immediately adjacent to the window
      (`chicago house style`, `detroit techno vibes`: the genre-phrase
      reading wins over the band reading);
  (b) the window has **more tokens than the name canonically has**
      (`q u e e n` — fragmentation is evasion; a two-token window matching
      a two-token name is just the name written normally);
  (c) normalization **changed** the matched text (leet/homoglyph — `Qu33n`,
      Cyrillic `Аir` — obfuscation is evasion).

Build-time class assignment mirrors this. A token is "common" when it is
frequent (Zipf ≥ 3.5) AND either dictionary English (with inflections) or
very frequent (Zipf ≥ 4.3 — chicago, berlin). Single common tokens →
`single_common`; other singles → `single_rare`. Non-Person names whose
every token is common gate as `single_common` when the artist has a
Wikipedia article (The Band, Air Supply) and are dropped when it does not
(Instrumental Music, S.O.L.O.); Person-type names are never demoted
(michael jackson, james brown stay unconditional). Names that squash to a
pure-genre term or pure digits are dropped entirely.

First match wins (leftmost window start, longest window at that start).

## Result

`ArtistMatch {display, cls, evidence: exact|fragmented|obfuscated|cue,
span: [start, end)}` or no match. `evidence` is diagnostic (logs/analytics),
not part of the accept/reject decision beyond the single_common gates.

## Modes

`DEMON_ARTIST_FILTER` (pod) / `ARTIST_FILTER` (webapp): `on` (default) |
`log` (scan + log `artist_filter_hit`, never the prompt text, and allow) |
`off`. Unknown values mean `on` — fail closed.

## Deferred (v1.1)

Edit-distance-1 fuzzy via SymSpell delete index over multi-class keys
(squash ≥ 10 chars); artifact format reserves a `fuzzy` section. Tune only
after log-mode data exists.
