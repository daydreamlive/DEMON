"""Artist-name filter: golden-vector conformance, decision rules, latency.

The golden vectors (`artist_filter/data/golden_vectors.v1.json`) are the
cross-repo conformance suite — the TypeScript twin in demon-public-demo runs
the SAME file against the SAME shipped artifact, so a behavior change here
that isn't reflected there fails one side visibly. Rule tests use a small
inline fixture so they stay meaningful even when the shipped list evolves.

Pure CPU/source tests: no torch, no GPU, no network.
"""

import json
import time
from pathlib import Path

import pytest

from demos.realtime_motion_graph_web import artist_filter
from demos.realtime_motion_graph_web.artist_filter.matcher import Matcher
from demos.realtime_motion_graph_web.artist_filter.normalize import Normalizer

DATA = Path(artist_filter.__file__).parent / "data"
CONFIG = json.loads((DATA / "filter_config.v1.json").read_text())
VECTORS = json.loads((DATA / "golden_vectors.v1.json").read_text())


# ---------------------------------------------------------------------------
# Golden vectors against the SHIPPED artifact.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", VECTORS["cases"], ids=[c["text"][:40] or "<empty>" for c in VECTORS["cases"]]
)
def test_golden_vector(case):
    match = artist_filter.scan(case["text"])
    if case["expect"] == "reject":
        assert match is not None, f"expected reject: {case['note']}"
        if "display" in case:
            assert match.display == case["display"], case["note"]
    else:
        assert match is None, (
            f"expected pass but matched {match and match.display!r} "
            f"({match and match.evidence}): {case['note']}"
        )


def test_version_is_loaded():
    assert artist_filter.filter_version() == "artists.v1"


# ---------------------------------------------------------------------------
# Decision rules on a fixed fixture (independent of the shipped list).
# ---------------------------------------------------------------------------

FIXTURE = {
    "taylorswift": ("Taylor Swift", "multi", 2),
    "daftpunk": ("Daft Punk", "multi", 2),
    "theband": ("The Band", "single_common", 2),
    "beyonce": ("Beyoncé", "single_rare", 1),
    "queen": ("Queen", "single_common", 1),
    "prince": ("Prince", "single_common", 1),
    "chicago": ("Chicago", "single_common", 1),
}


@pytest.fixture(scope="module")
def matcher():
    return Matcher(Normalizer(CONFIG), FIXTURE, CONFIG)


def test_multi_fires_plain(matcher):
    m = matcher.scan("taylor swift breakup pop")
    assert m and m.display == "Taylor Swift" and m.evidence == "exact"


def test_spaced_letters_fire(matcher):
    m = matcher.scan("t a y l o r s w i f t")
    assert m and m.display == "Taylor Swift"


def test_symbol_and_digit_leet_fire_as_obfuscated(matcher):
    m = matcher.scan("T@Yl0r $wift")
    assert m and m.evidence == "obfuscated"


def test_single_common_needs_a_cue(matcher):
    assert matcher.scan("queen of the night vocals") is None
    m = matcher.scan("queen style vocals")
    assert m and m.display == "Queen" and m.evidence == "cue"


def test_single_common_fragmentation_is_evidence(matcher):
    m = matcher.scan("q u e e n")
    assert m and m.evidence == "fragmented"


def test_canonical_token_count_is_not_fragmentation(matcher):
    # "the band" written normally is a 2-token window on a 2-token name.
    assert matcher.scan("the band swells into the chorus") is None
    m = matcher.scan("t h e b a n d")
    assert m and m.display == "The Band" and m.evidence == "fragmented"
    m = matcher.scan("the band style breakdown")
    assert m and m.evidence == "cue"


def test_cues_are_directional(matcher):
    # A trailing style-cue must not poison the words after it.
    assert matcher.scan("french house style, queen bassline") is None
    m = matcher.scan("queen style bassline")
    assert m and m.display == "Queen"
    m = matcher.scan("inspired by queen")
    assert m and m.display == "Queen"
    m = matcher.scan("queen inspired chords")
    assert m and m.display == "Queen"


def test_single_common_obfuscation_is_evidence(matcher):
    m = matcher.scan("qu33n harmonies")
    assert m and m.evidence == "obfuscated"


def test_possessive_is_a_cue(matcher):
    m = matcher.scan("prince's falsetto")
    assert m and m.display == "Prince"
    m = matcher.scan("prince’s falsetto")
    assert m and m.display == "Prince"


def test_genre_adjacency_suppresses_cue(matcher):
    assert matcher.scan("chicago house style groove") is None
    m = matcher.scan("chicago style horns")
    assert m and m.display == "Chicago"


def test_token_alignment_kills_substrings(matcher):
    assert matcher.scan("frequency modulated bass") is None
    assert matcher.scan("princely fanfare") is None


def test_affix_catches_glued_multi(matcher):
    m = matcher.scan("taylorswift4ever type beat")
    assert m and m.display == "Taylor Swift"


def test_affix_never_fires_for_single_class(matcher):
    # beyonce is single_rare: affix rule is multi-only by spec.
    assert matcher.scan("beyoncefan playlist") is None


def test_pure_digit_windows_never_fold(matcher):
    assert matcher.scan("808 bass and 303 acid") is None


def test_zero_width_space_is_deleted(matcher):
    m = matcher.scan("tay​lor swift")
    assert m and m.display == "Taylor Swift"


def test_cyrillic_homoglyphs_fold(matcher):
    m = matcher.scan("tаylor swift")  # Cyrillic а
    assert m and m.evidence == "obfuscated"


def test_span_points_into_original_text(matcher):
    text = "warm pads, taylor swift, tape hiss"
    m = matcher.scan(text)
    assert m and text[m.span[0]:m.span[1]] == "taylor swift"


# ---------------------------------------------------------------------------
# Mode flag.
# ---------------------------------------------------------------------------

def test_filter_mode_defaults_on_and_fails_closed(monkeypatch):
    monkeypatch.delenv("DEMON_ARTIST_FILTER", raising=False)
    assert artist_filter.filter_mode() == "on"
    monkeypatch.setenv("DEMON_ARTIST_FILTER", "log")
    assert artist_filter.filter_mode() == "log"
    monkeypatch.setenv("DEMON_ARTIST_FILTER", "off")
    assert artist_filter.filter_mode() == "off"
    monkeypatch.setenv("DEMON_ARTIST_FILTER", "banana")
    assert artist_filter.filter_mode() == "on"


# ---------------------------------------------------------------------------
# Latency budget: upstream of the audio hot path.
# ---------------------------------------------------------------------------

def _best_of(fn, n=5):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def test_latency_realistic_prompt():
    text = ("hypnotic deep groove, fat analog bassline, dusty rhodes chords, "
            "tape saturation, four to the floor, cavernous reverb, "
            "late night warehouse energy, crisp hats") * 2  # ~300 chars
    artist_filter.scan(text)  # warm the map
    assert _best_of(lambda: artist_filter.scan(text)) < 0.002


def test_latency_worst_case_fragmented():
    text = " ".join("x" * 1 for _ in range(2048))  # 4095 chars of 1-char tokens
    artist_filter.scan(text)
    assert _best_of(lambda: artist_filter.scan(text)) < 0.005
