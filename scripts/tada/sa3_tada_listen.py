"""Ear package for TADA on SA3: for each chosen concept, one benchmark
prompt at strength 0, at the strongest positive strength and at the
strongest negative strength whose mean LPAPS stays within the PCI cutoff
(the localised CAA sweep). The prompt is the one whose MuQ alignment
moves most between the two signed strengths. Writes six wavs and a
README to ``--dst``.

    <eval env python> scripts/tada/sa3_tada_listen.py piano vocal_gender
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

#: Working root for the replication data (audio, vectors, packs, the
#: steer-audio checkout); set TADA_ROOT to relocate it.
TADA_ROOT = Path(os.environ.get("TADA_ROOT", "tada-replication"))


import numpy as np
import pandas as pd
import soundfile as sf

#: Positive / negative pole of each benchmark concept (reference PCI prompts).
POLES = {
    "piano": ("piano", "no piano"), "mood": ("happy", "sad"), "tempo": ("fast", "slow"),
    "vocal_gender": ("female vocal", "male vocal"), "vocal_style": ("rap vocal", "sung vocal"),
    "guitar_electronic": ("acoustic guitar", "electric guitar"), "violin": ("violin", "no violin"),
    "rock_genre": ("jazz", "rock"), "electronic_music": ("classical", "electronic"),
}

ROOT = TADA_ROOT / "sa3/eval50"


def _with_pci(args) -> int:
    """Four files per concept from one benchmark prompt, all from the
    same seed and batch layout: zero, the real positive prompt (PCI at
    full switch length, ``sa3_tada_run.py swap``), CAA at the strongest
    positive strength within the PCI cutoff, CAA at the largest positive
    grid strength. The prompt is the one whose MuQ alignment gains most
    from zero to the admitted strength (stated in the README)."""
    root = TADA_ROOT / "sa3" / args.sub
    auc = json.loads((root / "auc.json").read_text())
    bench = json.loads((Path(__file__).resolve().parents[2] / "acestep" / "tada" / "data"
                        / "benchmark_prompts.json").read_text(encoding="utf-8"))["test_prompts"]
    lines = [f"# TADA on Stable Audio 3: ear package ({args.sub}, CAA {args.site})", "",
             "One benchmark prompt per concept: the one whose MuQ alignment gains most from strength 0 to",
             "the admitted strength (a best case chosen by the metric, not a typical one).", ""]
    for c in args.concepts:
        d = root / f"{args.method}_{args.site}_{c}"
        pr = d / "protocol_results"
        lp = pd.read_csv(pr / "lpaps.csv")
        mq = pd.read_csv(pr / "muqt.csv")
        cut = auc[c][f"{args.method}_{args.site}"]["pos"]["cutoff"]
        a_adm = float(lp[(lp.alpha > 0) & (lp["mean"] <= cut)].alpha.max())
        a_max = float(lp.alpha.max())
        sc = {float(r.alpha): np.array(ast.literal_eval(r.scores)) for r in mq.itertuples()}
        k = int(np.argmax(sc[a_adm] - sc[0.0]))
        sweep = json.loads((d / "sweep.json").read_text())
        pos_pole, neg_pole = POLES.get(c, ("positive pole", "negative pole"))
        lines.append(f"## {c} (toward {pos_pole})")
        lines.append(f"Prompt {k}: \"{bench[k]}\". Blocks {sweep['blocks']}, MuQ query \"{mq.prompt_used.iloc[0]}\", "
                     f"PCI cutoff (mean LPAPS) {cut:.2f}.")
        files = [("1_zero", d / "alpha_0.0", f"strength 0 (the prompt as written), MuQ {sc[0.0][k]:.3f}"),
                 ("2_pci_real_prompt", root / f"swap_full_{c}" / "pos",
                  "REFERENCE: the real positive prompt of the PCI triple rendered from step 0 "
                  "(its own wording, so it is a different piece of text, same seed and layout)"),
                 (f"3_caa_admitted_alpha{a_adm:+g}", d / f"alpha_{a_adm}",
                  f"CAA at the strongest strength within the PCI cutoff (mean LPAPS "
                  f"{float(lp[lp.alpha == a_adm]['mean'].iloc[0]):.2f}), MuQ {sc[a_adm][k]:.3f}"),
                 (f"4_caa_max_alpha{a_max:+g}", d / f"alpha_{a_max}",
                  f"CAA at the largest strength on the grid (mean LPAPS "
                  f"{float(lp[lp.alpha == a_max]['mean'].iloc[0]):.2f}, beyond the cutoff when above it), "
                  f"MuQ {sc[a_max][k]:.3f}")]
        for tag, src, what in files:
            z = np.load(src / "audios.npz")
            x = z["audio"][k].astype(np.float32) / 32768.0
            name = f"{c}_{tag}.wav"
            sf.write(args.dst / name, x.T, int(z["sr"]))
            lines.append(f"- `{name}`: {what}")
        lines.append(f"- Listen for: files 3 and 4 moving toward {pos_pole} relative to file 1, and how far that is "
                     f"from what the real prompt (file 2) does. The piece should stay recognisable in file 3.")
        lines.append("")
    lines += ["All files: SA3 medium, 8 steps, cfg 1, seed 2115, 10 s mono, same batch layout (batch 25), so",
              "zero and steered files share their initial noise. Files in `superseded/` are the earlier packages",
              "(old vectors; the quick piano render there drew different noise for its zero file)."]
    (args.dst / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("concepts", nargs="+")
    ap.add_argument("--method", default="caa")
    ap.add_argument("--site", default="loc")
    ap.add_argument("--dst", type=Path, default=TADA_ROOT / "listen_sa3")
    ap.add_argument("--sub", default=None, help="eval directory under sa3/ (default eval50)")
    ap.add_argument("--with-pci", action="store_true",
                    help="E3 layout: zero, real prompt (PCI full swap), CAA at the strongest admitted "
                         "and at the largest grid strength, positive direction")
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    if args.with_pci:
        return _with_pci(args)
    auc = json.loads((ROOT / "auc.json").read_text())
    lines = ["# TADA on Stable Audio 3: ear package", "",
             f"Method {args.method}, site {args.site} (blocks 3, 6, 7), 8 steps, cfg 1, seed 2115, 10 s mono.",
             "Each concept: the same prompt and seed at strength 0, at the strongest positive strength and at the",
             "strongest negative strength whose mean LPAPS stays within the PCI cutoff (prompt-swap distortion).", ""]
    for c in args.concepts:
        d = ROOT / f"{args.method}_{args.site}_{c}"
        pr = d / "protocol_results"
        lp = pd.read_csv(pr / "lpaps.csv")
        mq = pd.read_csv(pr / "muqt.csv")
        cut = auc[c][f"{args.method}_{args.site}"]
        a_pos = float(lp[(lp.alpha > 0) & (lp["mean"] <= cut["pos"]["cutoff"])].alpha.max())
        a_neg = float(lp[(lp.alpha < 0) & (lp["mean"] <= cut["neg"]["cutoff"])].alpha.min())
        sc = {float(r.alpha): np.array(ast.literal_eval(r.scores)) for r in mq.itertuples()}
        k = int(np.argmax(sc[a_pos] - sc[a_neg]))
        sweep = json.loads((d / "sweep.json").read_text())
        lines.append(f"## {c}")
        bench = json.loads((Path(__file__).resolve().parents[2] / "acestep" / "tada" / "data"
                            / "benchmark_prompts.json").read_text(encoding="utf-8"))["test_prompts"]
        lines.append(f"Prompt {k}: \"{bench[k]}\". MuQ query \"{mq.prompt_used.iloc[0]}\".")
        for tag, a in (("zero", 0.0), ("pos", a_pos), ("neg", a_neg)):
            z = np.load(d / f"alpha_{a}" / "audios.npz")
            x = z["audio"][k].astype(np.float32) / 32768.0
            name = f"{c}_{tag}_alpha{a:+g}.wav"
            sf.write(args.dst / name, x.T, int(z["sr"]))
            m = float(lp[lp.alpha == a]["mean"].iloc[0])
            lines.append(f"- `{name}`: alpha {a:+g}, MuQ {sc[a][k]:.3f} (mean over prompts "
                         f"{float(np.mean(sc[a])):.3f}), mean LPAPS {m:.2f}")
        pos_pole, neg_pole = POLES.get(c, ("positive pole", "negative pole"))
        lines.append(f"- Listen for: pos toward {pos_pole}, neg toward {neg_pole}, against zero; the piece "
                     f"should stay recognisable. Sweep range {sweep['alphas'][0]} to {sweep['alphas'][-1]}.")
        lines.append("")
    (args.dst / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
