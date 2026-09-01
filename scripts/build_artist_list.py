# /// script
# requires-python = ">=3.11"
# dependencies = ["wordfreq>=3.0", "english-words>=2.0"]
# ///
"""Build data/artists.v1.json.gz for the artist-name filter from a
MusicBrainz JSON dump.

Offline tooling only — pods never run this; they load the committed
artifact. Streams the NDJSON `artist` entity file on stdin so the 10+ GB
decompressed dump never touches disk:

    curl -O https://data.metabrainz.org/pub/musicbrainz/data/json-dumps/<DATE>/artist.tar.xz
    tar -xJOf artist.tar.xz mbdump/artist | uv run scripts/build_artist_list.py \
        --dump-date <DATE> --out demos/realtime_motion_graph_web/artist_filter/data/artists.v1.json.gz

Inclusion (notability) — an artist makes the list when any of:
  * a direct Wikipedia URL relation;
  * a community rating (votes >= 1);
  * genre votes summing >= 2;
  * a Wikidata relation AND at least one genre or tag.
MusicBrainz has no listener counts; these are the strongest signals the
artist entity itself carries. Tune with --min-genre-votes etc. after a
log-mode review, not by editing code.

Classification per SPEC.md: multi (>=2 tokens) fires unconditionally.
A token is "common" when it is frequent (Zipf >= --common-zipf) AND either
dictionary English (web2, with cheap inflections) or very frequent
(Zipf >= --high-zipf — "chicago", "berlin"). Single common tokens are
single_common (cue-gated), other singles single_rare. Non-Person names
whose EVERY token is common gate as single_common when the artist has a
Wikipedia article ("The Band", "Air Supply") and are DROPPED when it does
not ("Instrumental Music") — Person names (michael jackson) never demote.
Entries that squash to a pure genre term, to pure digits, or onto the
manual stoplist are dropped.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from demos.realtime_motion_graph_web.artist_filter.normalize import Normalizer  # noqa: E402

DATA_DIR = REPO / "demos/realtime_motion_graph_web/artist_filter/data"

MULTI, RARE, COMMON = "multi", "single_rare", "single_common"

import re  # noqa: E402

_NUMERIC_UNIT_RE = re.compile(r"\d+(bpm|hz|khz|ms|db|st)")

# MusicBrainz special-purpose artists and similar non-artists.
_SPECIAL = {"variousartists", "unknown", "anonymous", "traditional",
            "noartist", "data", "dialogue", "silence", "christmasmusic"}


_zipf_cache: dict[str, float] = {}


def _zipf(word: str) -> float:
    z = _zipf_cache.get(word)
    if z is None:
        from wordfreq import zipf_frequency
        z = _zipf_cache[word] = zipf_frequency(word, "en")
    return z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "artists.v1.json.gz"))
    ap.add_argument("--dump-date", required=True,
                    help="MusicBrainz dump date, recorded in the artifact")
    ap.add_argument("--common-zipf", type=float, default=3.5)
    ap.add_argument("--high-zipf", type=float, default=4.3,
                    help="a token this frequent gates as common even when "
                         "no dictionary lists it (place names)")
    ap.add_argument("--min-genre-votes", type=int, default=2)
    ap.add_argument("--strong-rating-votes", type=int, default=3,
                    help="rating votes that make a mononym/common-phrase "
                         "name protectable")
    ap.add_argument("--strong-genre-votes", type=int, default=10,
                    help="summed genre votes that do the same")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    from english_words import get_english_words_set
    english = get_english_words_set(["web2"], lower=True)

    def in_dict(t: str) -> bool:
        """Dictionary membership including cheap inflections — web2 has
        "pad" but not "pads", and an artist named PADS must not classify as
        a rare token just because the wordlist skips plurals."""
        if t in english:
            return True
        for stem in (t[:-1] if t.endswith("s") else None,
                     t[:-2] if t.endswith(("es", "ed")) else None,
                     t[:-3] + "y" if t.endswith("ies") else None,
                     t[:-3] if t.endswith("ing") else None):
            if stem and len(stem) >= 2 and stem in english:
                return True
        return False

    def is_common(t: str) -> bool:
        """A token a prompt could plausibly use as a plain word: frequent
        AND (dictionary English OR very frequent — 'chicago', 'berlin' are
        no dictionary words but must still gate like common words)."""
        z = _zipf(t)
        return z >= args.common_zipf and (in_dict(t) or z >= args.high_zipf)

    config = json.loads((DATA_DIR / "filter_config.v1.json").read_text())
    norm = Normalizer(config)
    min_key_len = int(config["min_key_len"])
    genre_terms = set(config["genre_terms"])
    stoplist = {
        norm.key(line.strip())
        for line in (DATA_DIR / "stoplist.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    } | genre_terms | _SPECIAL

    def classify(tokens: list[str], artist_type: str,
                 strong: bool) -> str | None:
        """None = drop this name."""
        squash = "".join(tokens)
        if len(squash) < min_key_len or squash in stoplist:
            return None
        if not any(c.isalpha() for c in squash):
            return None  # "311", "808" — pure digits collide with gear/bpm
        if _NUMERIC_UNIT_RE.fullmatch(squash):
            return None  # "140 b.p.m.", "808 Hz" — gear/tempo phrases
        if len(tokens) >= 2 and all(len(t) == 1 or t.isdigit() for t in tokens):
            # Acronym names ("S.O.L.O.", "A.N.D.") squash to an ordinary
            # word and would match it as an unconditional multi. Treat the
            # squash as the mononym it effectively is.
            tokens = [squash]
        if len(tokens) >= 2:
            if artist_type == "Person":
                # First+last name phrases identify a person even when every
                # token is common (michael jackson, james brown) — never
                # demote or drop these.
                return MULTI
            if all(is_common(t) for t in tokens):
                # A generic English phrase as a group name. With a strong
                # notability signal the name has real currency — gate it like
                # a common word ("The Band", "Air Supply"). Without one it's
                # an obscure act whose name would poison ordinary prompts
                # ("Instrumental Music") — drop it.
                return COMMON if strong else None
            return MULTI
        # Single-token names are the false-positive tail: MusicBrainz is
        # full of obscure acts named after ordinary music words ("Reverb",
        # "Arpeggio", "Angelic"). A mononym earns list protection only with
        # a strong notability signal — which every famous mononym (Beyoncé,
        # Eminem, Prince, Madonna) clears easily.
        if not strong:
            return None
        t = tokens[0]
        if _zipf(t) >= 5.5:
            return None  # function words ("and", "the") — never protectable
        return COMMON if is_common(t) else RARE

    def alias_ok(tokens: list[str], alias_type: str | None) -> bool:
        if len(tokens) == 1:
            # "West" for Kanye West, "Prince" for anyone — a single common
            # word is too ambiguous to inherit an alias's intent.
            return not is_common(tokens[0])
        if (alias_type or "") == "Search hint" and all(is_common(t) for t in tokens):
            return False  # "The Weekend" — a common phrase as a typo hint
        return True

    stats = {"records": 0, "notable": 0, "wikipedia": 0, "rated": 0,
             "genre_voted": 0, "wikidata_tagged": 0, "names": 0,
             "aliases_kept": 0, "aliases_dropped": 0, "dropped_names": 0}
    entries: dict[str, tuple[str, str, int]] = {}
    rank = {MULTI: 2, RARE: 1, COMMON: 0}

    def add(name: str, artist_type: str, strong: bool,
            alias_type: str | None = None, is_alias: bool = False) -> None:
        tokens = [t.norm for t in norm.tokenize(name)]
        if not tokens:
            return
        if is_alias and not alias_ok(tokens, alias_type):
            stats["aliases_dropped"] += 1
            return
        cls = classify(tokens, artist_type, strong)
        if cls is None:
            stats["dropped_names"] += 1
            return
        key = "".join(tokens)
        prev = entries.get(key)
        if prev is not None and rank[prev[1]] >= rank[cls]:
            return
        entries[key] = (name, cls, len(tokens))
        stats["aliases_kept" if is_alias else "names"] += 1

    for line in sys.stdin:
        stats["records"] += 1
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rel_types = {r.get("type") for r in d.get("relations") or ()}
        wikipedia = "wikipedia" in rel_types
        rated = (d.get("rating") or {}).get("votes-count") or 0
        genre_votes = sum(g.get("count") or 0 for g in d.get("genres") or ())
        wikidata_tagged = "wikidata" in rel_types and bool(
            d.get("genres") or d.get("tags"))
        if wikipedia:
            stats["wikipedia"] += 1
        if rated >= 1:
            stats["rated"] += 1
        if genre_votes >= args.min_genre_votes:
            stats["genre_voted"] += 1
        if wikidata_tagged:
            stats["wikidata_tagged"] += 1
        if not (wikipedia or rated >= 1
                or genre_votes >= args.min_genre_votes or wikidata_tagged):
            continue
        stats["notable"] += 1
        if args.stats_only:
            continue
        artist_type = d.get("type") or ""
        # "Strong" notability gates the FP-prone name shapes (mononyms,
        # all-common-word phrases): a direct Wikipedia rel is rare in modern
        # MusicBrainz (links migrated to Wikidata), so community engagement
        # carries the signal instead.
        strong = bool(
            wikipedia
            or rated >= args.strong_rating_votes
            or genre_votes >= args.strong_genre_votes
        )
        add(d["name"], artist_type, strong)
        for alias in d.get("aliases") or ():
            add(alias.get("name") or "", artist_type, strong,
                alias_type=alias.get("type"), is_alias=True)
        if stats["records"] % 500_000 == 0:
            print(f"  …{stats['records']} records, {len(entries)} keys",
                  file=sys.stderr)

    print(json.dumps(stats, indent=2), file=sys.stderr)
    if args.stats_only:
        return 0

    artifact = {
        "version": "artists.v1",
        "source": f"MusicBrainz JSON dump {args.dump_date} (CC0)",
        "generated_from": {
            "dump_date": args.dump_date,
            "common_zipf": args.common_zipf,
            "high_zipf": args.high_zipf,
            "min_genre_votes": args.min_genre_votes,
        },
        "entries": sorted(
            [k, disp, cls, n] for k, (disp, cls, n) in entries.items()
        ),
    }
    out = Path(args.out)
    # mtime=0 keeps the artifact byte-stable across rebuilds of identical data.
    with gzip.GzipFile(out.name, "wb", fileobj=open(out, "wb"), mtime=0) as fh:
        fh.write(json.dumps(artifact, ensure_ascii=False,
                            separators=(",", ":")).encode())
    print(f"wrote {out} ({len(entries)} entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
