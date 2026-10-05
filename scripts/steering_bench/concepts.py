"""The five production SA3 steering knobs, as the bench sees them.

Per concept: the descriptor from :mod:`proxies` and its sign (``+1``: a
higher descriptor value is the knob's POSITIVE pole; density is ``-1``
because positive = sparse = fewer onsets), the production pack's block
(knob_provenance.md section 1), the label/blurb the production packs carry,
and the MusicCaps keyword classes (tada_bench_reuse.md section 5) matched on
``caption`` + ``aspect_list``, lower-cased.

Keyword kinds: ``two_sided`` keeps captions matching exactly one pole;
``presence`` takes as negatives captions matching none of ``exclude``.
"""

from __future__ import annotations

CONCEPTS = ("bright", "warm", "rough", "density", "percussive")

SPEC = {
    "bright": {
        "descriptor": "centroid", "sign": 1, "prod_block": 23,
        "label": "Bright", "blurb": "positive brightens (spectral centroid up)",
        "kind": "two_sided",
        "pos": r"\bbright\w*",
        "neg": r"\b(dark\w*|muffled|dull\w*|muddy)\b",
    },
    "warm": {
        "descriptor": "lowhigh_db", "sign": 1, "prod_block": 15,
        "label": "Warm", "blurb": "positive tilts the spectrum toward bass (warmer)",
        "kind": "two_sided",
        "pos": r"\bwarm\w*",
        # no \w* after thin/cold: it would match "thing", "coldplay"-like words
        "neg": r"\b(cold|colder|harsh\w*|thin|thinner|bright\w*)\b",
    },
    "rough": {
        "descriptor": "flatness", "sign": 1, "prod_block": 2,
        "label": "Rough", "blurb": "positive adds grit and noise (spectral flatness up)",
        "kind": "two_sided",
        "pos": r"\b(distort\w*|gritty|grit|rough\w*|fuzz\w*|overdriv\w*)\b",
        "neg": r"\b(smooth\w*|clean\w*|mellow\w*)\b",
    },
    "density": {
        # POSITIVE = sparse, matching the production knob (discover.py:94-95)
        "descriptor": "onset_rate", "sign": -1, "prod_block": 1,
        "label": "Density",
        "blurb": "positive thins the texture toward sparse/minimal "
                 "(same direction as the ACE steer_density axis)",
        "kind": "two_sided",
        "pos": r"\b(sparse\w*|minimal\w*|simple|simplistic)\b",
        "neg": r"\b(dense|density|busy|full|complex)\b",
    },
    "percussive": {
        "descriptor": "perc_ratio", "sign": 1, "prod_block": 15,
        "label": "Percussive", "blurb": "positive pushes toward drums and percussion",
        "kind": "presence",
        "pos": r"\b(percussive|percussion\w*|drum\w*)\b",
        "exclude": r"\b(percuss\w*|drum\w*|beat\w*|groove\w*)\b",
    },
}

DESCRIPTORS = ("centroid", "lowhigh_db", "flatness", "onset_rate", "perc_ratio")
