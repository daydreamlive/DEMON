#!/usr/bin/env python3
"""Aggregate int8 variant bench JSONs into a final markdown report.

Picks up everything matching ``bench_outputs/vae_int8/json/*.json`` and
writes ``bench_outputs/vae_int8/REPORT.md``.

Usage:
    uv run python scripts/write_int8_report.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "bench_outputs" / "vae_int8"
JSON_DIR = RESULTS_DIR / "json"


# Ordered list of (variant_id, json_filename_pattern, description)
ORDERED = [
    ("A",  "A_vae_decode_int8_60s.json",                        "TRT entropy, 32 latents, no pinning"),
    ("B",  "B_vae_decode_int8_60s_minmax.json",                 "TRT minmax, 32 latents, no pinning"),
    ("C",  "C_vae_decode_int8_60s_pin.json",                    "TRT entropy, 32 latents, pin first/last conv to fp16"),
    ("D",  "D_vae_decode_int8_60s_combined.json",               "TRT entropy + pin (orchestrator pick, dup of C)"),
    ("E",  "E_vae_decode_int8_60s_minmax_pin.json",             "TRT minmax, 32 latents, pin first/last conv"),
    ("F",  "F_vae_decode_int8_60s_data128.json",                "TRT entropy, 128 transient-biased latents"),
    ("G1", "G1_vae_decode_int8_60s_modelopt_pct32.json",        "modelopt percentile-99.999, 32 latents"),
    ("G2", "G2_vae_decode_int8_60s_modelopt_ent32.json",        "modelopt entropy, 32 latents"),
    ("G3", "G3_vae_decode_int8_60s_modelopt_pct128.json",       "modelopt percentile-99.999, 128 latents"),
]


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _row(variant_id: str, desc: str, j: dict) -> dict:
    syn = j.get("synthetic", {})
    wav = j.get("wav_roundtrip", {})
    return {
        "id": variant_id,
        "desc": desc,
        "engine": Path(j.get("int8_engine", "")).parent.name,
        "engine_mb": j.get("engine_size_mb", 0),
        "syn_int8_ms": syn.get("int8", {}).get("median_ms"),
        "syn_int8_mse_vs_pt": syn.get("int8", {}).get("mse_vs_pt"),
        "syn_fp16_ms": syn.get("fp16", {}).get("median_ms"),
        "syn_fp16_mse_vs_pt": syn.get("fp16", {}).get("mse_vs_pt"),
        "syn_pt_ms": syn.get("pt", {}).get("median_ms"),
        "syn_fp16_int8_psnr": syn.get("fp16_vs_int8", {}).get("psnr_db"),
        "wav_int8_ms": wav.get("decode", {}).get("int8", {}).get("median_ms"),
        "wav_int8_mse_vs_pt": wav.get("decode", {}).get("int8", {}).get("mse_vs_pt"),
        "wav_fp16_ms": wav.get("decode", {}).get("fp16", {}).get("median_ms"),
        "wav_pt_ms": wav.get("decode", {}).get("pt", {}).get("median_ms"),
        "wav_int8_psnr_vs_orig": wav.get("vs_original_audio", {}).get("int8", {}).get("psnr_db"),
        "wav_fp16_psnr_vs_orig": wav.get("vs_original_audio", {}).get("fp16", {}).get("psnr_db"),
        "wav_pt_psnr_vs_orig": wav.get("vs_original_audio", {}).get("pt", {}).get("psnr_db"),
        "wav_fp16_int8_psnr": wav.get("fp16_vs_int8", {}).get("psnr_db"),
    }


def _fmt(v, fmt_str):
    if v is None:
        return "-"
    try:
        return fmt_str.format(v)
    except (TypeError, ValueError):
        return "-"


def _pick_winner(rows: list[dict]) -> dict:
    """Pick on synthetic MSE vs PT (clean decoder-only signal).

    The wav-round-trip PSNR-vs-original is encoder-dominated -- decoder
    differences sit in the noise floor of that metric (~0.2 dB across all
    variants). The synthetic MSE isolates decoder quality and is what
    actually distinguishes the variants.
    """
    if not rows:
        return {}

    by_mse = sorted(rows, key=lambda r: r["syn_int8_mse_vs_pt"] or 1e9)
    by_speed = sorted(rows, key=lambda r: r["syn_int8_ms"] or 1e9)

    return {
        "best_quality": by_mse[0],
        "fastest":      by_speed[0],
    }


def write_report(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    winners = _pick_winner(rows)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# INT8 VAE decode -- variant comparison\n\n")
        f.write(f"_{time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n")

        f.write("## TL;DR\n\n")
        f.write(
            "**The purpose of an int8 engine is to be both faster AND "
            "close-enough quality vs fp16.** A variant that is slower "
            "than fp16 is disqualified, no matter how good its quality.\n\n"
            "**Practical winner: Variant A** (`vae_decode_int8_60s`) -- "
            "TRT entropy, 32 latents, no pinning.\n\n"
            "- **1.64x faster than fp16** (32.9 ms vs 54 ms / 60 s of audio)\n"
            "- **32.52 dB PSNR vs fp16** -- audible loss but the closest of "
            "any variant that is actually faster than fp16\n"
            "- 245 MB engine\n\n"
            "**Why not the higher-quality modelopt variants (G1/G2)?** "
            "They reach 35.51 dB PSNR vs fp16 (+3 dB closer), but at "
            "0.69x fp16 speed (79 ms vs 54 ms). At that point you should "
            "just ship fp16: it has *infinite* PSNR vs itself and is "
            "faster. modelopt int8 is dominated by fp16 in the "
            "speed/quality plane for this graph.\n\n"
            "**There is no variant in this matrix that beats fp16 on "
            "BOTH axes.** The remaining experiment that could deliver "
            "one is described in 'Unfinished work' below.\n\n"
        )
        f.write(
            "## Key findings\n\n"
            "1. **Profile-matching the calibration data** to the deployed "
            "engine profile (1500-frame opt instead of 250-frame opt) was "
            "the first big win -- moved fp16-vs-int8 PSNR from 28.3 dB to "
            "32.5 dB on the same recipe (4.2 dB free).\n"
            "2. **modelopt closes the quality gap further** (+3.0 dB to "
            "35.5 dB) but at a 2.4x latency penalty, ending up *slower* "
            "than fp16 -- defeating the purpose. modelopt-entropy and "
            "modelopt-percentile produce essentially identical scales; "
            "the quality win is the modelopt pipeline (offline ORT scale "
            "computation), not the calibration algorithm.\n"
            "3. **Layer-pinning, MinMax, and more calibration data with "
            "the TRT-native pipeline all hurt or did nothing.**\n"
            "4. **The Pareto frontier vs fp16** has only one int8 point "
            "that is strictly better than fp16 on the speed axis: "
            "Variant A. All modelopt variants are dominated by fp16.\n\n"
        )

        f.write("## Setup\n\n")
        f.write("- All INT8 variants share the same 60s dynamic profile "
                "(min=125, opt=1500, max=1500 latent frames).\n")
        f.write("- Calibration set: 32 generated latents at 1500 frames each "
                "(prompt-diverse: dance / jazz / ambient / metal / hip hop / "
                "classical / folk / synthwave; cfg=7.5; 8 steps).\n")
        f.write("- fp16 baseline is `vae_decode_fp16_60s` (existing build, "
                "same 60s profile).\n")
        f.write("- PT eager reference is the bf16 ACE-Step Oobleck VAE "
                "(`session.handler.vae`).\n")
        f.write("- All measurements on RTX 5090, TRT 10.13.3, "
                "torch 2.9.1+cu128.\n\n")

        f.write("## Variant matrix\n\n")
        f.write("| Variant | Description | Engine | MB |\n")
        f.write("|---|---|---|---:|\n")
        for r in rows:
            f.write(f"| {r['id']} | {r['desc']} | `{r['engine']}` | "
                    f"{_fmt(r['engine_mb'], '{:.0f}')} |\n")
        f.write("\n")

        # ---------------- Synthetic ----------------
        f.write("## Synthetic latent (Session.generate, 60s, seed=42)\n\n")
        f.write("Decode-only quality, measured on a freshly-generated 1500-frame "
                "latent. **Speedup vs fp16 = fp16_ms / int8_ms** (>1.0x = "
                "int8 is faster, <1.0x = int8 is slower than fp16 and "
                "should be discarded).\n\n")
        f.write("| Variant | int8 ms | fp16 ms | **Speedup vs fp16** | "
                "int8 MSE vs PT | fp16 MSE vs PT | "
                "fp16-vs-int8 PSNR (dB) | Verdict |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            ms_int8 = r['syn_int8_ms']
            ms_fp16 = r['syn_fp16_ms']
            speedup = ms_fp16 / ms_int8 if ms_int8 and ms_fp16 else None
            if speedup is None:
                verdict = "-"
            elif speedup >= 1.0:
                verdict = f"**{speedup:.2f}x faster**"
            else:
                verdict = f"_slower than fp16 ({speedup:.2f}x) -- DISCARD_"
            f.write(
                f"| {r['id']} | "
                f"{_fmt(r['syn_int8_ms'], '{:.1f}')} | "
                f"{_fmt(r['syn_fp16_ms'], '{:.1f}')} | "
                f"{_fmt(speedup, '{:.2f}x')} | "
                f"{_fmt(r['syn_int8_mse_vs_pt'], '{:.2e}')} | "
                f"{_fmt(r['syn_fp16_mse_vs_pt'], '{:.2e}')} | "
                f"{_fmt(r['syn_fp16_int8_psnr'], '{:.2f}')} | "
                f"{verdict} |\n")
        f.write("\n")

        # ---------------- Wav round-trip ----------------
        f.write("## Wav round-trip (encode->decode of `tests/fixtures/inside_confusion.wav`, first 60s)\n\n")
        f.write("The wav is PT-encoded once and the same latents are decoded "
                "by all three paths. PSNR vs fp16 is the metric that matters "
                "for the int8-vs-fp16 question; PSNR vs original is mostly "
                "encoder-dominated noise (~0.2 dB spread across all "
                "decoders).\n\n")
        f.write("| Variant | int8 ms | fp16 ms | **Speedup vs fp16** | "
                "fp16-vs-int8 PSNR (dB) | int8 PSNR vs orig | "
                "fp16 PSNR vs orig | Verdict |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            ms_int8 = r['wav_int8_ms']
            ms_fp16 = r['wav_fp16_ms']
            speedup = ms_fp16 / ms_int8 if ms_int8 and ms_fp16 else None
            if speedup is None:
                verdict = "-"
            elif speedup >= 1.0:
                verdict = f"**{speedup:.2f}x faster**"
            else:
                verdict = f"_slower than fp16 ({speedup:.2f}x) -- DISCARD_"
            f.write(
                f"| {r['id']} | "
                f"{_fmt(r['wav_int8_ms'], '{:.1f}')} | "
                f"{_fmt(r['wav_fp16_ms'], '{:.1f}')} | "
                f"{_fmt(speedup, '{:.2f}x')} | "
                f"{_fmt(r['wav_fp16_int8_psnr'], '{:.2f}')} | "
                f"{_fmt(r['wav_int8_psnr_vs_orig'], '{:.2f}')} | "
                f"{_fmt(r['wav_fp16_psnr_vs_orig'], '{:.2f}')} | "
                f"{verdict} |\n")
        f.write("\n")

        # ---------------- Decision guide ----------------
        f.write("## Decision guide\n\n")
        f.write("- The **synthetic int8-MSE-vs-PT** is the cleanest signal of "
                "decoder fidelity. Use that to pick the variant. modelopt "
                "variants (G1, G2, G3) win by 2-3x on this metric.\n")
        f.write("- The **wav round-trip PSNR vs the original audio is "
                "dominated by the encoder**: PT, fp16, and int8 all sit "
                "within ~0.2 dB of each other. Decoder differences are not "
                "audible at this granularity for full encode/decode cycles, "
                "but they do matter when the latent comes from generation "
                "(not encoding), since there is no encoder error to mask.\n")
        f.write("- TRT-native int8 variants run at ~33 ms/60s (1800x "
                "realtime). modelopt variants run at ~79 ms/60s (760x "
                "realtime). For non-realtime decoding, latency doesn't matter. "
                "For streaming/realtime, both are fine.\n\n")

        f.write("Engines and wavs:\n")
        f.write("- Engines: `~/.daydream-scope/models/demon/trt_engines/<engine_name>/`\n")
        f.write("- A/B wavs: `bench_outputs/vae_int8/<engine_name>/`\n")
        f.write("  - `wav_original.wav` -- the source clip, trimmed to 60s\n")
        f.write("  - `wav_pt_eager.wav` / `wav_trt_fp16.wav` / `wav_trt_int8.wav` -- decoded outputs\n")
        f.write("  - `synth_*.wav` -- decoded synthetic latents\n\n")

        f.write("## What didn't work\n\n")
        f.write("**MinMax calibrator (B, E):** MinMax sets per-tensor int8 "
                "scales from observed (min, max), with no clipping. "
                "Snake-activated audio decoders have heavy-tailed "
                "activations -- a single transient sample sets the scale "
                "and most of the int8 range is wasted. KL-entropy "
                "calibration clips outliers for better resolution on the "
                "bulk distribution. MSE doubled going entropy->minmax in "
                "TRT-native (A 6.87e-4 -> B 1.53e-3).\n\n")
        f.write("**Pinning first/last Conv to fp16 (C, D, E):** in theory "
                "the input and output convs are the most sensitive "
                "layers. In practice TRT inserts reformat kernels at "
                "int8/fp16 boundaries and `OBEY_PRECISION_CONSTRAINTS` "
                "removes tactics from the search. Pinned layers run fp16 "
                "but eat quantization error from input reformats, and "
                "neighbouring layers can't opportunistically stay fp16. "
                "A's MSE 6.87e-4 vs C's 9.98e-4 is a 45% degradation.\n\n")
        f.write("**More calibration data with TRT-native (F):** going "
                "from 32 to 128 prompt-diverse, transient-biased latents "
                "with the same TRT-entropy recipe slightly *hurt* "
                "synthetic MSE (8.03e-4 vs 6.87e-4). TRT's calibrator "
                "appears to plateau quickly; the bottleneck is the "
                "calibrator, not the data volume. Same data through "
                "modelopt (G3) does help.\n\n")
        f.write("**modelopt percentile vs entropy (G1 vs G2):** "
                "**identical** results to four significant figures (MSE "
                "3.46e-04 each). The win is the modelopt pipeline "
                "(offline ORT-based scale computation, more aggressive "
                "fp16 fallback for non-quantized ops) -- not the choice "
                "of calibration algorithm.\n\n")
        f.write("**modelopt percentile + transient-biased 128 latents "
                "(G3) was *worse* than 32 mixed-prompt (G1/G2):** MSE "
                "8.54e-04 vs 3.46e-04, a 2.5x degradation. Reason: the "
                "extra prompts I added were percussion- and quiet-"
                "ambient-heavy (drums, claps, vinyl crackle, drone). "
                "Their activation tails are *wider* than the original "
                "diverse mix, so the 99.999th percentile picks a wider "
                "per-tensor scale, sacrificing precision on the bulk "
                "distribution to accommodate transients that the "
                "production stream rarely sees. **Lesson**: for "
                "percentile calibration, the calibration set must "
                "match the inference distribution, not artificially "
                "stretch it. The original 32-latent mixed-prompt set "
                "was already the right distribution. \"More data\" was "
                "the wrong axis to optimize.\n\n")

        f.write("## Unfinished work -- the experiment that could deliver "
                "an int8 engine that beats fp16 on BOTH axes\n\n")
        f.write("The matrix above tested two clean recipes (TRT-native "
                "calibration and modelopt-QDQ quantization) at full strength. "
                "Neither produced a Pareto win over fp16 on speed AND "
                "quality. The hybrid approach -- never tested -- is:\n\n"
                "**Hybrid: extract modelopt's per-tensor scales, feed "
                "them to a TRT-native build.**\n\n"
                "1. modelopt computes high-quality scales offline via "
                "ORT (the win we observed in G1/G2).\n"
                "2. Walk the modelopt QDQ ONNX, extract `y_scale` from "
                "each `QuantizeLinear` node.\n"
                "3. Write those scales into a TRT calibration cache file "
                "(text format, one tensor per line).\n"
                "4. Build a TRT-native engine with INT8+FP16 flags and a "
                "custom `IInt8Calibrator` whose `read_calibration_cache()` "
                "returns the modelopt-derived cache.\n"
                "5. TRT then has modelopt's superior scales AND TRT-native's "
                "aggressive int8/fp16 fusion -- expected: ~35 dB PSNR at "
                "~33 ms (1.6x fp16 speedup).\n\n"
                "Risk: ONNX tensor names may not map cleanly to the names "
                "in TRT's calibration cache after parser fusions. Mitigation: "
                "do a 1-batch dummy TRT calibration first to learn what "
                "names TRT asks for, then write modelopt scales for "
                "matching tensors and fall back to TRT's own calibration "
                "for the rest.\n\n")
        f.write("Other options if hybrid fails:\n\n")
        f.write("- **modelopt `autotune=True`**: searches per-region "
                "quantization schemes via trtexec. May find a config "
                "that's faster than the default modelopt build. Adds "
                "30-60 min to a build but no custom code.\n")
        f.write("- **SmoothQuant via modelopt**: rebalances activation "
                "outliers into weights pre-quant. Primarily designed for "
                "matmul-heavy LLM topologies; uncertain win on 1D-conv "
                "VAE.\n")
        f.write("- **QAT**: highest ceiling, requires training infra.\n\n")
        f.write("Skip layer-pinning and TRT MinMax -- both confirmed to "
                "regress on this graph topology.\n")

    print(f"Report: {path}", flush=True)


def main():
    rows: list[dict] = []
    for variant_id, fname, desc in ORDERED:
        j = _load(JSON_DIR / fname)
        if j is None:
            print(f"  [skip] {variant_id}: {fname} not found")
            continue
        rows.append(_row(variant_id, desc, j))
    if not rows:
        print("No bench JSONs found.")
        sys.exit(1)
    write_report(rows, RESULTS_DIR / "REPORT.md")


if __name__ == "__main__":
    main()
