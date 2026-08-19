import pytest

from acestep.engine.sa3_denoise_mapping import (
    denoise_mapping_mode,
    dial_to_entry_sigma,
    map_denoise_to_entry_sigma,
)


def test_mapping_is_monotonic_and_preserves_endpoints():
    values = [dial_to_entry_sigma(i / 100) for i in range(101)]

    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_mapping_matches_measured_knots():
    assert dial_to_entry_sigma(0.1) == pytest.approx(0.3786)
    assert dial_to_entry_sigma(0.5) == pytest.approx(0.7683)
    assert dial_to_entry_sigma(0.8) == pytest.approx(0.8843)


def test_mapping_interpolates_between_knots():
    midpoint = (0.7683 + 0.8376) / 2
    assert dial_to_entry_sigma(0.575) == pytest.approx(midpoint)


def test_mapping_clamps_inputs():
    assert dial_to_entry_sigma(-1) == 0.0
    assert dial_to_entry_sigma(2) == 1.0


def test_default_mapping_lifts_the_measured_dead_region(monkeypatch):
    monkeypatch.delenv("DEMON_SA3_DENOISE_MAPPING", raising=False)

    assert denoise_mapping_mode() == "calibrated"
    assert map_denoise_to_entry_sigma(0.2) > 0.5
    assert map_denoise_to_entry_sigma(0.5) > 0.7


def test_identity_mode_is_a_bugfix_preserving_rollback(monkeypatch):
    monkeypatch.setenv("DEMON_SA3_DENOISE_MAPPING", "identity")

    assert map_denoise_to_entry_sigma(0.2) == 0.2
    assert map_denoise_to_entry_sigma(0.5) == 0.5


def test_invalid_mode_fails_loudly(monkeypatch):
    monkeypatch.setenv("DEMON_SA3_DENOISE_MAPPING", "mystery")

    with pytest.raises(ValueError, match=r"calibrated\|identity"):
        map_denoise_to_entry_sigma(0.5)
