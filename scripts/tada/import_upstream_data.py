"""Regenerate ``acestep/tada/data/*.json`` from a steer-audio checkout.

The TADA reference code (github.com/luk-st/steer-audio, MIT, Copyright
(c) 2026 Lukasz Staniszewski) keeps its concept lists, prompt templates,
evaluation queries and calibrated steering ranges as Python modules and
YAML configs. This script loads them unchanged and writes them out as
JSON, so the data in ``acestep/tada/data`` is the upstream data verbatim.

Usage::

    python scripts/tada/import_upstream_data.py --repo <steer-audio checkout>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parents[2] / "acestep" / "tada" / "data"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"wrote {OUT / name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    args = ap.parse_args()
    repo: Path = args.repo
    rev = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    source = {"repo": "https://github.com/luk-st/steer-audio", "commit": rev,
              "license": "MIT, Copyright (c) 2026 Lukasz Staniszewski"}

    # Steering concepts: contrastive prompt pairs (paper Table 8, App. G.1).
    sp = _load_module(
        repo / "src/steering/methods/sae/lib/configs/steer_prompts.py", "tada_steer_prompts",
    )
    concepts = {}
    for key, fn in sp.CONCEPT_TO_PROMPTS.items():
        neg, pos, lyrics = fn()
        concepts[key] = {"positive": list(pos), "negative": list(neg), "lyrics": lyrics}
    ev = _load_module(repo / "src/steering/methods/sae/lib/configs/eval.py", "tada_eval")

    # Calibrated CAA strength ranges and steering settings (released configs).
    caa = {}
    for cfg_path in sorted((repo / "configs/steering/ace/caa").glob("eval_*.yaml")):
        cfg = _yaml(cfg_path)
        concept = cfg.get("concept")
        if not concept or not cfg_path.stem.endswith("_" + concept):
            continue
        # eval_<variant>_<concept>: loc, all, ablated, ablated_calib
        variant = cfg_path.stem[len("eval_"):-len("_" + concept)]
        caa.setdefault(concept, {})[variant] = {
            k: cfg.get(k) for k in (
                "layers", "steer-mode", "min-range", "max-range", "steps-per-side",
                "duration", "steps", "guidance-scale", "seed", "method-kwargs", "alphas",
            )
        }
    compute = {}
    for cfg_path in sorted((repo / "configs/steering/ace/caa").glob("compute_*.yaml")):
        cfg = _yaml(cfg_path)
        compute[cfg["concept"]] = cfg.get("scorer-kwargs", {})
    sao = {}
    for cfg_path in sorted((repo / "configs/steering/stable_audio/stable_audio_caa").glob("*.yaml")):
        sao[cfg_path.stem] = _yaml(cfg_path)

    _write("steering_concepts.json", {
        "source": source,
        "note": "CONCEPT_TO_PROMPTS (steer_prompts.py) and CONCEPT_TO_EVAL_PROMPTS (configs/eval.py).",
        "concepts": concepts,
        "eval_prompts": ev.CONCEPT_TO_EVAL_PROMPTS,
        "caa_eval": caa,
        "caa_compute": compute,
        "stable_audio_caa": sao,
    })

    tp = _load_module(repo / "src/steering/eval/test_prompts.py", "tada_test_prompts")
    _write("benchmark_prompts.json", {
        "source": source,
        "note": "TEST_PROMPTS (100, paper App. G.2), HOLDOUT_PROMPTS (20, hyperparameter selection) and paired lyrics; vocal concepts sing the paired LYRICS, others use [inst].",
        "test_prompts": list(tp.TEST_PROMPTS),
        "test_lyrics": list(tp.LYRICS),
        "holdout_prompts": list(tp.HOLDOUT_PROMPTS),
        "holdout_lyrics": list(tp.LYRICS_HOLDOUT),
        "vocal_concepts": ["vocal_gender", "vocal_style"],
    })

    feats = _load_module(repo / "src/preprocess/features.py", "tada_features")
    patch_data = {}
    for cfg_path in sorted((repo / "configs/patch_data/musiccaps").glob("*.yaml")):
        patch_data[cfg_path.stem] = _yaml(cfg_path)
    patch_cfg = {
        "ace": _yaml(repo / "configs/patch_config/ace.yaml"),
        "stableaudio": _yaml(repo / "configs/patch_config/stableaudio.yaml"),
    }
    experiments = {}
    for sub in ("patch_ace", "patch_stableaudio"):
        for cfg_path in sorted((repo / "configs/experiment" / sub).glob("*.yaml")):
            experiments[f"{sub}/{cfg_path.stem}"] = _yaml(cfg_path)
    evcfg = _yaml(repo / "configs/eval_audio.yaml")
    _write("localization.json", {
        "source": source,
        "note": "Counterfactual-prompt construction (src/preprocess/features.py, paper Table 5), per-concept patching data and eval prompts (configs/patch_data/musiccaps), patching settings (configs/patch_config), experiment overrides, and the similarity prompt templates (configs/eval_audio.yaml). Prompt pairs themselves: HF dataset lukasz-staniszewski/patching-music-musiccaps-prompts.",
        "original_features": feats.MUSICCAPS_ORIGINAL_FEATURES,
        "counterfactual_features": feats.MUSICCAPS_COUNTERFACTUAL_FEATURES,
        "swap_features": feats.MUSICCAPS_SWAPS_FEATURES,
        "patch_data": patch_data,
        "patch_config": patch_cfg,
        "experiments": experiments,
        "clap_prompt_template": evcfg.get("clap_prompt_template"),
        "muqt_prompt_template": evcfg.get("muqt_prompt_template"),
        "concepts_by_model": {
            "ace": ["female", "male", "fast", "slow", "happy", "sad", "drums", "violin", "reggae", "jazz"],
            "stableaudio": ["female", "male", "fast", "slow", "happy", "sad", "violin", "flute",
                            "maracas", "guitar", "techno", "reggae"],
        },
        "tau": 0.10,
    })


if __name__ == "__main__":
    main()
