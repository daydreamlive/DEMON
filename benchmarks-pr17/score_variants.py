"""Compute additional quality metrics for the archived FP8 variants.

For each variant under ``benchmarks-pr17/variants/<tag>/audio/`` we load
the 6 prompt renders, then compute:

- ``text_clap``  : LAION CLAP cosine(audio_emb, text_emb) where text is
                  the original generation prompt. Absolute prompt
                  adherence; bf16 reference is included for sanity.
- ``mel_l1``     : L1 distance between log-mel spectrograms of this
                  variant's audio and bf16's matching audio at the same
                  prompt and seed. Lower = closer to bf16 at the
                  waveform level. ``bf16`` is by definition 0.
- ``audio_clap_vs_bf16`` : cosine(CLAP_audio_emb(variant), CLAP_audio_emb(bf16))
                  for the same prompt+seed pair. Perceptual-embedding
                  fidelity to bf16. Higher = closer.
- ``mrstft``     : Multi-resolution STFT loss vs bf16 (sum of spectral
                  convergence + log-magnitude L1 at three window sizes).
                  Captures phase coherence + transient fidelity in a way
                  power-mel L1 does not. Lower = closer.

Both metrics use the prompt list from ``fp8_listen_test.py`` so the
order/labels are consistent with the listen-test layout.

CLAP backend is ``laion/larger_clap_music_and_speech`` loaded via
``transformers.ClapModel`` (no separate ``laion_clap`` install — that
package conflicts with newer transformers).

Run from repo root::

    python benchmarks-pr17/score_variants.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import ClapModel, ClapProcessor


# Keep prompt list in sync with benchmarks-pr17/fp8_listen_test.py.
PROMPTS: list[tuple[str, str, int, str]] = [
    ("dance", "dance music, four on the floor, kick drum, electronic, club, energetic synth bass, bright leads", 128, "F minor"),
    ("jazz",  "jazz piano trio, brushed drums, walking bass, late-night club", 140, "Bb major"),
    ("ambient", "ambient electronic, slow evolving pads, glassy textures, no drums", 80, "C minor"),
    ("metal", "metal, aggressive guitar riffs, fast double kick, growling vocals", 180, "E minor"),
    ("orch", "classical orchestral, sweeping strings, brass, timpani", 90, "D major"),
    ("folk", "acoustic folk, fingerpicked guitar, soft harmonica, brushes", 100, "G major"),
]

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_SR = 48_000

# Mel-L1 uses a single-scale mel spectrogram at 48 kHz. Single scale is
# enough for ranking; multi-scale buys precision we don't need here.
MEL_SR = 48_000
MEL_N_FFT = 2048
MEL_HOP = 512
MEL_N_MELS = 128


def _load_wav_mono(path: Path, target_sr: int) -> torch.Tensor:
    """Load WAV as 1D float32 tensor at ``target_sr``."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # wav shape: [T, C]
    x = torch.from_numpy(wav.mean(axis=1))  # to mono
    if sr != target_sr:
        x = torchaudio.functional.resample(x, sr, target_sr)
    return x


def _log_mel(x: torch.Tensor, mel: torchaudio.transforms.MelSpectrogram) -> torch.Tensor:
    s = mel(x)
    return torch.log(s.clamp_min(1e-5))


def _mel_l1(a: torch.Tensor, b: torch.Tensor, mel: torchaudio.transforms.MelSpectrogram) -> float:
    n = min(a.numel(), b.numel())
    a = a[:n]
    b = b[:n]
    la = _log_mel(a, mel)
    lb = _log_mel(b, mel)
    return float((la - lb).abs().mean().item())


def _build_mel(device: torch.device) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=MEL_SR,
        n_fft=MEL_N_FFT,
        hop_length=MEL_HOP,
        n_mels=MEL_N_MELS,
        power=2.0,
    ).to(device)


# Multi-resolution STFT: standard MelGAN/Parallel-WaveGAN window set.
MRSTFT_FFT_SIZES = (512, 1024, 2048)
MRSTFT_HOP_SIZES = (50, 120, 240)
MRSTFT_WIN_SIZES = (240, 600, 1200)


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    window = torch.hann_window(win, device=x.device)
    s = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=True,
    )
    return s.abs()


def _mrstft(a: torch.Tensor, b: torch.Tensor) -> float:
    """Sum of spectral-convergence + log-magnitude L1 across 3 window sizes."""
    n = min(a.numel(), b.numel())
    a = a[:n]
    b = b[:n]
    total = 0.0
    for n_fft, hop, win in zip(MRSTFT_FFT_SIZES, MRSTFT_HOP_SIZES, MRSTFT_WIN_SIZES):
        ma = _stft_mag(a, n_fft, hop, win)
        mb = _stft_mag(b, n_fft, hop, win)
        sc = (mb - ma).norm(p="fro") / mb.norm(p="fro").clamp_min(1e-8)
        lm = (torch.log(ma.clamp_min(1e-7)) - torch.log(mb.clamp_min(1e-7))).abs().mean()
        total += float(sc.item()) + float(lm.item())
    return total / len(MRSTFT_FFT_SIZES)


def _list_variant_dirs(variants_root: Path) -> list[str]:
    out = []
    for p in sorted(variants_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "audio").is_dir():
            out.append(p.name)
    return out


def _load_metrics(variants_root: Path, tag: str) -> dict:
    p = variants_root / tag / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@torch.no_grad()
def score_all(variants_root: Path, device: torch.device) -> dict:
    print(f"[setup] device={device}")
    print(f"[setup] loading CLAP model: {CLAP_MODEL_ID}")
    t0 = time.perf_counter()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device).eval()
    print(f"[setup] CLAP ready in {time.perf_counter() - t0:.1f}s")

    mel = _build_mel(device)

    # bf16 reference audio for mel-L1 baseline.
    bf16_dir = variants_root / "bf16" / "audio"
    if not bf16_dir.is_dir():
        raise FileNotFoundError(f"bf16 reference missing: {bf16_dir}")

    print("[setup] loading bf16 reference audio (mel + CLAP)")
    bf16_mel_wavs: dict[str, torch.Tensor] = {}
    bf16_clap_embs: dict[str, torch.Tensor] = {}
    for i, (tag, _prompt, _bpm, _key) in enumerate(PROMPTS):
        fname = f"{i+1:02d}_{tag}.wav"
        wav_mel = _load_wav_mono(bf16_dir / fname, MEL_SR)
        bf16_mel_wavs[fname] = wav_mel.to(device)
        wav_clap_np = (
            wav_mel.cpu().numpy() if CLAP_SR == MEL_SR
            else _load_wav_mono(bf16_dir / fname, CLAP_SR).cpu().numpy()
        )
        a_in = processor(audios=[wav_clap_np], sampling_rate=CLAP_SR, return_tensors="pt").to(device)
        emb = model.get_audio_features(**a_in)
        bf16_clap_embs[fname] = torch.nn.functional.normalize(emb, dim=-1)[0]

    # Precompute text embeddings once.
    prompt_texts = [p[1] for p in PROMPTS]
    text_inputs = processor(text=prompt_texts, return_tensors="pt", padding=True).to(device)
    text_emb = model.get_text_features(**text_inputs)
    text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
    # text_emb: [6, D]

    variants = _list_variant_dirs(variants_root)
    print(f"[setup] variants: {variants}")

    results: dict = {"variants": {}}
    for vtag in variants:
        v_audio_dir = variants_root / vtag / "audio"
        per_prompt = []
        for i, (ptag, prompt_text, _bpm, _key) in enumerate(PROMPTS):
            fname = f"{i+1:02d}_{ptag}.wav"
            wav_mel = _load_wav_mono(v_audio_dir / fname, MEL_SR).to(device)
            mel_l1 = _mel_l1(wav_mel, bf16_mel_wavs[fname], mel)
            mrstft = _mrstft(wav_mel, bf16_mel_wavs[fname])

            wav_clap_np = wav_mel.cpu().numpy() if CLAP_SR == MEL_SR else _load_wav_mono(v_audio_dir / fname, CLAP_SR).cpu().numpy()
            audio_inputs = processor(
                audios=[wav_clap_np],
                sampling_rate=CLAP_SR,
                return_tensors="pt",
            ).to(device)
            audio_emb = model.get_audio_features(**audio_inputs)
            audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
            cos_text = float((audio_emb[0] @ text_emb[i]).item())
            cos_audio_bf16 = float((audio_emb[0] @ bf16_clap_embs[fname]).item())

            per_prompt.append({
                "prompt_tag": ptag,
                "file": fname,
                "text_clap": cos_text,
                "audio_clap_vs_bf16": cos_audio_bf16,
                "mel_l1_vs_bf16": mel_l1,
                "mrstft_vs_bf16": mrstft,
            })

        m = _load_metrics(variants_root, vtag)
        text_clap_vals = [r["text_clap"] for r in per_prompt]
        audio_clap_vals = [r["audio_clap_vs_bf16"] for r in per_prompt]
        mel_l1_vals = [r["mel_l1_vs_bf16"] for r in per_prompt]
        mrstft_vals = [r["mrstft_vs_bf16"] for r in per_prompt]
        agg = {
            "tag": vtag,
            "cosine_sim": m.get("cosine_sim_avg"),
            "speedup": m.get("speedup_vs_bf16"),
            "engine_mb": m.get("engine_size_mb"),
            "text_clap_mean": float(np.mean(text_clap_vals)),
            "text_clap_std": float(np.std(text_clap_vals)),
            "audio_clap_vs_bf16_mean": float(np.mean(audio_clap_vals)),
            "audio_clap_vs_bf16_std": float(np.std(audio_clap_vals)),
            "mel_l1_mean": float(np.mean(mel_l1_vals)),
            "mel_l1_std": float(np.std(mel_l1_vals)),
            "mrstft_mean": float(np.mean(mrstft_vals)),
            "mrstft_std": float(np.std(mrstft_vals)),
            "per_prompt": per_prompt,
        }
        results["variants"][vtag] = agg
        print(
            f"  {vtag:>14s}  text_clap={agg['text_clap_mean']:.4f}  "
            f"a_clap_bf16={agg['audio_clap_vs_bf16_mean']:.4f}  "
            f"mel_l1={agg['mel_l1_mean']:.4f}  "
            f"mrstft={agg['mrstft_mean']:.4f}  "
            f"cos={agg['cosine_sim']}  speedup={agg['speedup']}"
        )

    return results


def _format_table(results: dict) -> str:
    rows = list(results["variants"].values())
    # Sort fp8 variants by speedup descending; bf16 last for reference.
    bf16 = [r for r in rows if r["tag"] == "bf16"]
    fp8 = [r for r in rows if r["tag"] != "bf16"]
    fp8.sort(key=lambda r: -(r["speedup"] or 0.0))
    ordered = fp8 + bf16

    header = "| tag | cos | text-CLAP | audio-CLAP vs bf16 | mel-L1 vs bf16 | MR-STFT vs bf16 | speedup | engine MB |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in ordered:
        cos = r["cosine_sim"]
        cos_s = f"{cos:.4f}" if cos is not None else "—"
        sp = r["speedup"]
        sp_s = f"{sp:.2f}x" if sp is not None else "—"
        eng = r["engine_mb"]
        eng_s = f"{eng:.0f}" if eng is not None else "—"
        lines.append(
            f"| {r['tag']} | {cos_s} | "
            f"{r['text_clap_mean']:.4f} ± {r['text_clap_std']:.4f} | "
            f"{r['audio_clap_vs_bf16_mean']:.4f} ± {r['audio_clap_vs_bf16_std']:.4f} | "
            f"{r['mel_l1_mean']:.4f} ± {r['mel_l1_std']:.4f} | "
            f"{r['mrstft_mean']:.4f} ± {r['mrstft_std']:.4f} | "
            f"{sp_s} | {eng_s} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variants-root",
        type=str,
        default=str(Path(__file__).parent / "variants"),
    )
    ap.add_argument("--out", type=str, default=None,
                    help="Output JSON (default: <variants-root>/scores.json)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    variants_root = Path(args.variants_root).resolve()
    out_path = Path(args.out) if args.out else variants_root / "scores.json"
    device = torch.device(args.device)

    results = score_all(variants_root, device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[save] {out_path}")
    print()
    print(_format_table(results))


if __name__ == "__main__":
    main()
