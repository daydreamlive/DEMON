"""Cross-backend knob homonym rule (backend-seam plan §3.3).

A knob name means ONE thing everywhere: if two backend families expose
the same knob name, the specs must be semantically identical (type,
default, bounds, options, group, bank). If the semantics differ even
slightly, the knob must be renamed (family prefix or ``group``) — the
live example is ACE's ``denoise`` (the k1 strength knob) vs SA3's
"denoise" (``init_noise_level``), which ships as ``sa3_denoise``.

Runs over the registered families' knob UNIVERSES
(``acestep.streaming.families.FAMILY_KNOB_UNIVERSES``) — the full spec
set a family can ever expose, obtainable without constructing the
GPU-heavy backend — never over source greps. With a single registered
family today, the cross-family comparison is vacuous; the scaffolding
(and the keys-in-sync guard) is the point: the first added family is
automatically held to the rule.
"""

from acestep.streaming.families import FAMILIES, FAMILY_KNOB_UNIVERSES


def _semantics(spec) -> dict:
    """The fields that define what a knob MEANS on the wire. The
    free-text description is deliberately excluded (wording may vary);
    everything a client validates or renders from is included."""
    return {
        "type": spec.type,
        "default": spec.default,
        "min_val": spec.min_val,
        "max_val": spec.max_val,
        "options": tuple(spec.options),
        "group": spec.group,
        "bank": spec.bank,
    }


def _family_manifest(name: str) -> dict:
    """``knob name -> semantics`` for one family, asserting the family's
    own universe doesn't fork a name across its internal variants (for
    ACE: the SDE / non-SDE mode manifests)."""
    out: dict = {}
    for spec in FAMILY_KNOB_UNIVERSES[name]():
        sem = _semantics(spec)
        prior = out.get(spec.name)
        assert prior is None or prior == sem, (
            f"family {name!r} forks knob {spec.name!r} across its own "
            f"variants: {prior} != {sem}"
        )
        out[spec.name] = sem
    return out


def test_every_family_declares_a_knob_universe():
    # Adding a backend family without registering its knob universe
    # would silently exempt it from the homonym rule. Keep the two
    # registries keyed identically.
    assert set(FAMILY_KNOB_UNIVERSES) == set(FAMILIES)


def test_family_universes_are_nonempty_and_internally_consistent():
    for name in FAMILY_KNOB_UNIVERSES:
        manifest = _family_manifest(name)
        assert manifest, f"family {name!r} declares an empty knob universe"


def test_shared_knob_names_have_identical_semantics():
    # The homonym rule, enforced across every registered family pair.
    # name -> (first family seen, semantics); any later family using the
    # name must match exactly or rename (prefix / group).
    seen: dict = {}
    violations: list = []
    for name in sorted(FAMILY_KNOB_UNIVERSES):
        for knob, sem in _family_manifest(name).items():
            prior = seen.get(knob)
            if prior is None:
                seen[knob] = (name, sem)
            elif prior[1] != sem:
                violations.append(
                    f"{knob!r}: {prior[0]} declares {prior[1]}, "
                    f"{name} declares {sem}"
                )
    assert not violations, (
        "knob homonyms with diverging semantics (rename with a family "
        "prefix or align the specs):\n  " + "\n  ".join(violations)
    )
