"""Semantic regression examples use synthetic prompts and no model weights."""
import pytest
from demos.realtime_motion_graph_web.prompt_constraints import PromptConstraint, subjects

@pytest.mark.parametrize("source,bad", [
    ("solo piano", "solo piano with kick and snare"),
    ("solo clarinet", "solo clarinet with a rhythm section"),
    ("solo acoustic guitar", "solo acoustic guitar and clarinet"),
    ("solo violin", "solo violin with reedy breath-driven notes"),
    ("solo flute", "solo flute, hip hop groove"),
    ("solo bass guitar", "solo bass guitar with vocals"),
    ("solo piano", "solo electric piano"),
    ("solo electric guitar", "solo acoustic guitar"),
    ("solo ukulele", "solo acoustic guitar and ukulele"),
    ("solo marimba", "marimba in a quartet"),
    ("solo piano", "piano, intimate close miking"),
    ("solo piano", "solo piano, controlled controlled tone"),
    ("single evolving layer, ambient texture", "single evolving layer, ambient texture with hi-hat rolls"),
    ("flute leading a full band", "unaccompanied solo flute"),
    ("piano in a trio", "piano in a quartet"),
    ("solo waterphone", "solo waterphone with piano"),
])
def test_contradictory_rewrites_use_valid_source(source, bad):
    contract = PromptConstraint.infer(source, "sa3")
    assert contract.violations(bad)
    assert contract.accept(bad, source) == source

@pytest.mark.parametrize("source,good", [
    ("solo drum kit", "unaccompanied solo drums, kick and snare, cymbal groove"),
    ("solo violin", "unaccompanied solo violin, singing vibrato"),
    ("solo violin, single bowed voice", "solo violin, balanced resonance"),
    ("solo piano", "unaccompanied solo piano, no vocals, expressive dynamics"),
    ("solo hammered dulcimer, shimmering struck strings", "solo dulcimer, ringing harmonics"),
    ("solo glockenspiel, tiny struck bells", "solo glockenspiel, bright articulation"),
    ("flute leading a full band", "flute leading a full band with guitar and drums"),
    ("piano in a trio", "piano in a trio, expressive phrasing"),
    ("solo waterphone", "solo waterphone"),
])
def test_valid_rewrites_survive(source, good):
    assert PromptConstraint.infer(source, "sa3").accept(good, source) == good


def test_conflicting_solo_input_is_not_a_valid_fallback():
    c = PromptConstraint.infer("solo piano and drums", "sa3")
    assert c.accept("solo guitar", c.source) == ""


def test_compound_instrument_names_are_not_split():
    assert subjects("electric piano with nylon-string guitar and drum machine") == (
        "electric piano", "nylon guitar", "drum machine")


def test_acestep_keeps_its_existing_unconstrained_arrangement_path():
    assert PromptConstraint.infer("techno", "acestep").accept("bass and drums") == "bass and drums"

@pytest.mark.parametrize("source,good", [
    ("solo tabla, hand drums", "solo tabla, resonant drums, balanced articulation"),
    ("solo glockenspiel", "solo glockenspiel, bright bell tones"),
    ("solo music box", "solo music box, delicate chimes"),
])
def test_instrument_components_do_not_invalidate_the_anchor(source, good):
    contract = PromptConstraint.infer(source, "sa3")
    assert not contract.violations(source)
    assert contract.accept(good, source) == good


@pytest.mark.parametrize("band", ["AM radio band", "FM radio band", "radio-frequency band"])
def test_radio_filtering_is_not_accompaniment(band):
    source = f"solo piano, filtered through a narrow {band}"
    contract = PromptConstraint.infer(source, "sa3")
    assert not contract.violations(source)
    assert contract.accept(source + ", soft notes", source) == source + ", soft notes"
    assert contract.accept(source + ", with a rock band", source) == source


@pytest.mark.parametrize("rhythm", [
    "driving syncopated rhythm", "driving rhythms", "a shuffling groove",
    "syncopated grooves", "a steady beat", "driving beats", "a rhythm track",
])
def test_solo_guitar_does_not_gain_an_unrequested_rhythm_clause(rhythm):
    source = (
        "solo electric guitar, expressive lead lines, single instrument, stratvema, "
        "classic 70s rock, uplifting, euphoric, soaring, clean, polished studio recording"
    )
    candidate = (
        f"unaccompanied solo electric guitar, expressive lead lines, {rhythm}, "
        "uplifting and euphoric, soaring melodic lines, polished studio recording"
    )
    contract = PromptConstraint.infer(source, "sa3")
    assert "added_rhythm" in contract.violations(candidate)
    assert contract.accept(candidate, source) == source


@pytest.mark.parametrize("source,candidate", [
    ("solo electric guitar", "solo electric guitar, syncopated guitar phrasing"),
    ("solo electric guitar", "solo electric guitar, rhythmic picking and expressive bends"),
    ("solo electric guitar, syncopated rhythm", "solo electric guitar, driving syncopated rhythm"),
    ("solo electric guitar, steady beat", "solo electric guitar, steady beat, clean articulation"),
    ("solo electric guitar, shuffling groove", "solo electric guitar, shuffling grooves"),
    ("solo drum kit", "solo drums, driving syncopated rhythm"),
    ("electric guitar leading a full band", "electric guitar leading a full band, driving syncopated rhythm"),
])
def test_instrument_phrasing_and_requested_rhythm_remain_valid(source, candidate):
    assert not PromptConstraint.infer(source, "sa3").violations(candidate)
