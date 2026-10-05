"""TADA on Stable Audio 3: scoring with the reference evaluation code.

Runs in the evaluation environment (laion_clap, muq, audiobox_aesthetics)
against a clone of github.com/luk-st/steer-audio (MIT), using its own
metric functions so the numbers mean what the paper's numbers mean:

* ``patch``: for every patch set rendered by ``sa3_tada_run.py patch``,
  the reference CLAP model (``enable_fusion=True``, default checkpoint, as
  ``src/metrics/metrics.py: calculate_clap``) scores every clip against
  the concept's two ``eval_clap_prompts`` with the template
  ``"This is a music of {p}"``. Writes ``patch/<concept>/scores.json``:
  ``{set: [mean sim prompt 1, mean sim prompt 2]}``. Impacts and the
  selected blocks are computed from it by ``sa3_tada_run.py localize``
  (core :mod:`acestep.tada.patching`).
* ``protocol``: ``eval_steering_protocol.main`` (CLAP, MuQ-T, LPAPS,
  aesthetics) on every ``eval/<method>_<concept>`` directory.
* ``auc``: alignment AUC and CSM per direction with the paper cutoff
  ``min(PCI-all, PCI-loc)`` max LPAPS (``auc.py: pci_cutoff``); writes
  ``eval/auc.json`` and prints a table.

    <eval env python> scripts/tada/sa3_tada_score.py patch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: Working root for the replication data (audio, vectors, packs, the
#: steer-audio checkout); set TADA_ROOT to relocate it.
TADA_ROOT = Path(os.environ.get("TADA_ROOT", "tada-replication"))


REF = Path(os.environ.get("TADA_REF", str(TADA_ROOT / "steer-audio")))
DEFAULT_OUT = TADA_ROOT / "sa3"
CLAP_TEMPLATE = "This is a music of {p}"
SA3_SR = 44100

sys.path.insert(0, str(REF / "editing" / "AudioEditingCode"))
sys.path.insert(0, str(REF))
os.chdir(REF)  # the reference resolves res/ checkpoints relative to its root


def _bare_packages() -> None:
    """Register the reference's package directories without running their
    ``__init__`` files: ``src/steering/__init__.py`` imports every steering
    method (and through them the ACE-Step pipeline), none of which the
    evaluation code uses."""
    import types

    for name in ("src.steering", "src.steering.methods", "src.steering.methods.sae",
                 "src.steering.methods.sae.lib", "src.steering.methods.sae.lib.configs"):
        if name in sys.modules:
            continue
        m = types.ModuleType(name)
        m.__path__ = [str(REF / Path(*name.split(".")))]
        sys.modules[name] = m


_bare_packages()


def _clap_model():
    import laion_clap
    import torch

    orig = torch.load
    torch.load = lambda *a, **kw: orig(*a, **{**kw, "weights_only": False})
    try:
        m = laion_clap.CLAP_Module(enable_fusion=True)
        m.load_ckpt(verbose=False)
    finally:
        torch.load = orig
    return m.to("cuda").eval()


def _patch_muq(args) -> None:
    """MuQ-T eq. 2 scores (SA3 primary metric; the audit found CLAP barely
    separates the real positive and negative prompts on SA3): reference
    ``calculate_muqt`` (MuQ-MuLan large, 24 kHz, template ``{p}``) of every
    patch set against the concept's ``eval_muqt_prompts``. Writes
    ``scores_muq.json`` beside the CLAP ``scores.json``."""
    from src.metrics.metrics import calculate_muqt

    _cache_models()
    for cdir in sorted((args.out / args.patch_dir).iterdir()):
        meta = cdir / "done.json"
        if not meta.exists() or (args.concepts and cdir.name not in args.concepts):
            continue
        dst = cdir / "scores_muq.json"
        if dst.exists() and not args.force:
            continue
        prompts = json.loads(meta.read_text())["eval_muqt_prompts"]
        scores = {}
        for sdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            _, per = calculate_muqt(audio_dir=str(sdir), prompts=prompts, sr=SA3_SR,
                                    resample_to_24k=True)
            # full precision (the summary is formatted to 3 decimals)
            scores[sdir.name] = [sum(r[f"muqt_sim_p{i}"] for r in per.values()) / len(per)
                                 for i in range(len(prompts))]
        dst.write_text(json.dumps({"prompts": prompts, "metric": "muq", "template": "{p}",
                                   "scores": scores}, indent=1))
        print(f"patch muq {cdir.name}: {len(scores)} sets", flush=True)


def cmd_patch(args) -> None:
    """Reference ``calculate_clap`` with the model loaded once."""
    import torch
    from src.steering.eval.audio_io import as_wav_dir

    if args.metric == "muq":
        return _patch_muq(args)
    model = _clap_model()
    for cdir in sorted((args.out / args.patch_dir).iterdir()):
        meta = cdir / "done.json"
        if not meta.exists() or (args.concepts and cdir.name not in args.concepts):
            continue
        dst = cdir / "scores.json"
        if dst.exists() and not args.force:
            continue
        prompts = json.loads(meta.read_text())["eval_clap_prompts"]
        with torch.no_grad():
            text = torch.tensor(model.get_text_embedding(
                [CLAP_TEMPLATE.format(p=p) for p in prompts])).cpu()
            text = torch.nn.functional.normalize(text, dim=1)
        scores = {}
        for sdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            with as_wav_dir(str(sdir)) as wd:
                files = sorted(str(p) for p in Path(wd).glob("*.wav"))
                embs = []
                with torch.no_grad():
                    for i in range(0, len(files), 64):
                        embs.append(torch.tensor(
                            model.get_audio_embedding_from_filelist(x=files[i:i + 64])).cpu())
            a = torch.nn.functional.normalize(torch.cat(embs), dim=1)
            sims = a @ text.t()
            scores[sdir.name] = [float(x) for x in sims.mean(dim=0)]
        dst.write_text(json.dumps({"prompts": prompts, "metric": "clap",
                                   "template": CLAP_TEMPLATE, "scores": scores}, indent=1))
        print(f"patch {cdir.name}: {len(scores)} sets", flush=True)


def _eval_dirs(args):
    for d in sorted((args.out / args.sub).iterdir(), reverse=bool(getattr(args, "reverse", False))):
        if not d.is_dir():
            continue
        method, site, concept = d.name.split("_", 2)
        if not args.sub.startswith("calib") and method != "pci" and not (d / "sweep.json").exists():
            continue  # sweep still rendering (sweep.json is written last)
        if args.methods and method not in args.methods:
            continue
        if args.labels and f"{method}_{site}" not in args.labels:
            continue
        if args.concepts and concept not in args.concepts:
            continue
        yield d, f"{method}_{site}", concept


def _cache_models() -> None:
    """The reference protocol builds its CLAP, LPAPS and MuQ-MuLan models
    once per strength (``editing/eval.py``, ``calculate_muqt``). Memoize
    the constructors so each is built once per process: same models,
    same weights, same scores; only the reload time goes away."""
    import functools

    import editing.eval as ev
    from muq import MuQMuLan

    def memo(factory):
        cache = {}

        @functools.wraps(factory)
        def wrapped(*a, **kw):
            key = repr((a, sorted(kw.items())))
            if key not in cache:
                cache[key] = factory(*a, **kw)
            return cache[key]
        return wrapped

    ev.LPAPS = memo(ev.LPAPS)
    ev.CLAPTextConsistencyMetric = memo(ev.CLAPTextConsistencyMetric)
    MuQMuLan.from_pretrained = memo(MuQMuLan.from_pretrained)
    try:
        import audiobox_aesthetics.infer as aes
        aes.AesPredictor = memo(aes.AesPredictor)
    except ImportError:
        pass


#: Alignment anchors for the production SA3 steering-pack concepts (the
#: reference ``CONCEPT_TO_EVAL_PROMPTS`` has only the 9 TADA concepts).
#: Same form as the reference table: CLAP short prompt, MuQ "This is a
#: music of ...". One anchor per concept, both directions (the negative
#: direction scores as a decrease). density: positive = sparse.
PACK_ANCHORS = {
    "bright": "a bright track",
    "warm": "a warm track",
    "percussive": "a percussive track",
    "rough": "a rough, gritty track",
    "density": "a sparse, minimal track",
}


def load_catalogue_anchors(path) -> dict:
    """``{name: pos_anchor}`` from a many-knobs catalogue CSV (name,
    pos_anchor columns) or a JSON ``{name: [pos, neg]}`` / ``{name: {"pos":
    ...}}`` / ``{name: "text"}`` (make_packs.py pci_descriptors.json)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        import csv

        with open(path, newline="", encoding="utf-8") as f:
            return {r["name"].strip(): r["pos_anchor"].strip() for r in csv.DictReader(f)
                    if (r.get("name") or "").strip() and (r.get("pos_anchor") or "").strip()}
    d = json.loads(path.read_text(encoding="utf-8"))
    d = d.get("concepts", d) if isinstance(d, dict) else {}
    out = {}
    for k, v in d.items():
        t = v if isinstance(v, str) else (v.get("pos") or v.get("pos_anchor") if isinstance(v, dict) else
                                          (v[0] if isinstance(v, list) and v else None))
        if isinstance(t, str) and t.strip() and not str(k).startswith("_"):
            out[str(k)] = t.strip()
    return out


def _register_anchors(extra: dict | None = None) -> dict:
    """Add :data:`PACK_ANCHORS` (then ``extra``, the ``--catalogue`` pos
    anchors) to the reference table in memory. The reference ``main`` looks
    the concept up BEFORE applying its ``eval_prompt=`` override
    (eval_steering_protocol.py:810), so the override alone would still
    KeyError on a new concept. Returns ``{concept: anchor}`` for every
    concept that gets the MuQ-form override: PACK_ANCHORS first, then
    catalogue names the reference table does not already have (the 9 TADA
    concepts keep their reference prompts)."""
    from src.steering.methods.sae.lib.configs.eval import CONCEPT_TO_EVAL_PROMPTS as table

    own = dict(PACK_ANCHORS)
    for c, a in (extra or {}).items():
        if c not in own and c not in table:
            own[c] = a
    for c, a in own.items():
        table.setdefault(c, {"clap": a, "muqt": f"This is a music of {a}"})
    return own


def cmd_protocol(args) -> None:
    from src.steering.eval.eval_steering_protocol import main as protocol

    _cache_models()
    own = _register_anchors(load_catalogue_anchors(args.catalogue) if args.catalogue else None)

    for d, label, concept in _eval_dirs(args):
        # Re-checked per directory: another scorer may have finished it.
        if (d / "protocol_results" / "lpaps.csv").exists() and not args.force:
            continue
        print(f"protocol {d.name}", flush=True)
        if args.lpaps_only:
            # calibration probes need only the preservation curve
            from src.steering.eval.eval_steering_protocol import compute_lpaps_preservation
            alphas = sorted(float(p.name[len("alpha_"):]) for p in d.glob("alpha_*"))
            df = compute_lpaps_preservation(str(d), alphas)
            (d / "protocol_results").mkdir(parents=True, exist_ok=True)
            df.to_csv(d / "protocol_results" / "lpaps.csv", index=False)
            continue
        kw = {}
        if concept in own:
            # MuQ form for both metrics (CLAP is secondary on SA3)
            kw["eval_prompt"] = f"This is a music of {own[concept]}"
        protocol(str(d), concept, skip_aesthetics=args.skip_aesthetics, **kw)


def cmd_cutoff(args) -> None:
    """Reference LPAPS (``compute_lpaps_preservation``) of every PCI
    directory at its strongest switch lengths only, ``protocol_results/
    lpaps_endpoints.csv``: the PCI cutoff is the max LPAPS, reached at
    the full prompt swap (reference ``PCI_CUTOFFS.md``), so strength
    calibration can start before the full PCI protocol finishes."""
    from src.steering.eval.eval_steering_protocol import compute_lpaps_preservation

    _cache_models()
    for d, label, concept in _eval_dirs(args):
        if not label.startswith("pci_"):
            continue
        out = d / "protocol_results" / "lpaps_endpoints.csv"
        if out.exists() and not args.force:
            continue
        alphas = sorted(float(p.name[len("alpha_"):]) for p in d.glob("alpha_*"))
        ends = [0.0, min(alphas), max(alphas)]
        df = compute_lpaps_preservation(str(d), ends)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"cutoff {d.name}: {df.to_dict('records')}", flush=True)


def _pci_cutoff(root: Path, concept: str, direction: str, reference) -> float:
    """``auc.pci_cutoff`` when the PCI directories carry the full
    protocol; otherwise the same min-of-max from ``lpaps_endpoints.csv``
    (the max is the full-swap endpoint)."""
    import pandas as pd

    full = [root / f"pci_{s}_{concept}" / "protocol_results" / "lpaps.csv" for s in ("all", "loc")]
    if all(f.exists() for f in full):
        return reference(root, concept, direction)
    vals = []
    for s in ("all", "loc"):
        f = root / f"pci_{s}_{concept}" / "protocol_results" / "lpaps_endpoints.csv"
        if f.exists():
            df = pd.read_csv(f)
            df = df[df["alpha"] > 0] if direction == "pos" else df[df["alpha"] < 0]
            vals.append(float(df["mean"].max()))
    return min(vals) if vals else float("nan")


def cmd_auc(args) -> None:
    from src.steering.eval.auc import (
        N_QUALITY_POINTS, compute_alignment_auc_direction, compute_csm_direction,
        compute_quality_metrics, pci_cutoff,
    )

    root = args.out / args.sub
    res = {}
    for d, label, concept in _eval_dirs(args):
        pr = d / "protocol_results"
        if not (pr / "lpaps.csv").exists():
            continue
        row = res.setdefault(concept, {}).setdefault(label, {})
        for direction in ("pos", "neg"):
            cut = _pci_cutoff(root, concept, direction, pci_cutoff)
            auc = compute_alignment_auc_direction(pr, direction, cut, ["muqt", "clap"])
            csm = compute_csm_direction(pr, direction, cut, ["muqt", "clap"])
            row[direction] = {"cutoff": cut, **auc, **csm}
        if (pr / "aesthetics.csv").exists():
            q = compute_quality_metrics([pr], [label], row["pos"]["cutoff"], row["neg"]["cutoff"],
                                        N_QUALITY_POINTS, True)
            row["quality"] = q[label]
    (root / "auc.json").write_text(json.dumps(res, indent=1))
    for concept, rows in res.items():
        for label, r in rows.items():
            line = " ".join(
                f"{dn}: muq {r[dn].get('muqt', float('nan')):.3f} clap {r[dn].get('clap', float('nan')):.3f}"
                for dn in ("pos", "neg") if dn in r)
            print(f"{concept:18s} {label:12s} {line}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("patch", "protocol", "auc", "cutoff"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--concepts", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-aesthetics", action="store_true")
    ap.add_argument("--lpaps-only", action="store_true",
                    help="protocol: LPAPS preservation curve only (calibration probes)")
    ap.add_argument("--sub", default="eval", help="eval or calib")
    ap.add_argument("--patch-dir", default="patch", help="patch, patch_xattn_out or patch_resid")
    ap.add_argument("--metric", choices=("clap", "muq"), default="clap", help="patch scoring metric")
    ap.add_argument("--reverse", action="store_true", help="walk directories in reverse order")
    ap.add_argument("--methods", nargs="*", default=None, help="caa, austeer, pci")
    ap.add_argument("--labels", nargs="*", default=None, help="method_site filter, e.g. pci_all caa_loc")
    ap.add_argument("--catalogue", default=None,
                    help="protocol: alignment anchor (pos_anchor) for any concept name, from the many-knobs "
                         "concept_catalogue.csv or a JSON {name: [pos, neg]} (pci_descriptors.json); "
                         "PACK_ANCHORS and the reference TADA concepts keep their own anchors")
    args = ap.parse_args()
    args.out = args.out.resolve()
    {"patch": cmd_patch, "protocol": cmd_protocol, "auc": cmd_auc,
     "cutoff": cmd_cutoff}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
