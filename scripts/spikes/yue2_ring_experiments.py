"""M2/M3/M4 in-process experiments on the production YuE2 ring.

Drives :class:`YuE2Backend` (built exactly as the create path builds it)
on one GPU, no server, and writes one JSON plus wavs per command.

``quality``  E2, E3, E5 with objective proxies (latent relative L2 and
             spectral style descriptors: long-term mel spectrum distance,
             centroid, sub-150 Hz share, onset strength). Ear verdicts
             are the user's. (CLAP was tried first: laion/
             larger_clap_music under transformers 4.57 returned collapsed
             embeddings, text similarity ~0 for every clip, so it was
             dropped.)
  E2  re-noise the anchor with its OWN noise (song seed) to d in
      {0.25, 0.5, 0.75} and integrate: should land near the anchor.
  E3  same score + semantics, style tags A vs B, full solve each, seed
      fixed; reference points: seed change under A (no style change) and
      a fresh composition under B (new semantics).
  E5  after a restyle to B, re-noise the anchor with seed 1 to d in
      {0.25, 0.5, 0.75, 1.0} and integrate under B.

``latency``  ring tick p50/p95 and knob-to-ear with the spike method:
             first changed PCM (> 1e-5 vs the pre-change render of the
             same window) and first fully updated PCM (rendered from a
             latent generated entirely under the new knobs), from the
             control change to CPU PCM. Two phases per knob: from a
             settled ring (idle) and with the change landing half way
             through another solve (busy). Runner pacing, transport and
             the client buffer are excluded.

    python scripts/spikes/yue2_ring_experiments.py quality --seconds 60 --out DIR
    python scripts/spikes/yue2_ring_experiments.py latency --seconds 30 --depth 1 --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from acestep.streaming.generator_backend import TickContext
from acestep.streaming.knobs import KnobState
from acestep.streaming.yue2_backend import YuE2Backend, playable_seconds, yue2_knob_specs
from acestep.streaming.yue2_session import compose_song, nar_backend_for

STYLE_PAIRS = [
    ("city pop, female vocal, bright synths, groovy bass",
     "dark industrial techno, distorted kick, male vocal"),
    ("acoustic folk, fingerpicked guitar, warm, gentle",
     "heavy metal, distorted guitars, aggressive drums, fast"),
]
SR = 48000
PCM_EPS = 1e-5


# ---- shared ------------------------------------------------------------------


def save_wav(path: Path, audio_2n: torch.Tensor):
    pcm = (audio_2n.clamp(-1, 1).float().cpu().numpy().T * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def rel_l2(a, b) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def make_backend(ctx, song, prompt, depth=1):
    return YuE2Backend.from_context(
        ctx, composition=song["composition"], bundle=song["bundle"],
        bundle_b=song["bundle_b"], anchor_latent=song["anchor"],
        knob_state=KnobState(yue2_knob_specs()), prompt_tags=prompt, depth=depth,
    )


def knobs(**over) -> dict:
    values = {"yue2_denoise": 1.0, "seed": 0, "x0_target": 0.0, "feedback": 0.0, "feedback_depth": 1}
    values.update(over)
    return values


def settle(backend, values, ctx, limit=400) -> torch.Tensor:
    """Produce until the ring settles under ``values``; the latent it
    settled on."""
    for _ in range(limit):
        backend.produce(values, ctx, "generate")
        if backend.is_settled(backend._last_prep):
            return backend._last_result_latent
    raise RuntimeError("ring did not settle")


def compose(ctx, prompt, seconds):
    return compose_song(ctx, prompt=prompt, prompt_b=prompt, lyrics="", seed=0,
                        max_frames=int(round(seconds * 25)))


# ---- quality (E2, E3, E5) ------------------------------------------------------


def spectral(audio_2n: torch.Tensor) -> dict:
    """Style proxies a tag change should move: the long-term average mel
    spectrum (dB), spectral centroid, sub-150 Hz energy share and mean
    onset strength (mono, 48 kHz)."""
    import librosa

    mono = audio_2n.float().mean(0).cpu().numpy()
    mel = librosa.feature.melspectrogram(y=mono, sr=SR, n_fft=4096, hop_length=2048, n_mels=64)
    power = np.abs(librosa.stft(mono, n_fft=4096, hop_length=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=4096)
    return {
        "ltas_db": librosa.power_to_db(mel.mean(axis=1) + 1e-10),
        "centroid_hz": float(librosa.feature.spectral_centroid(S=np.sqrt(power), sr=SR).mean()),
        "low_share": float(power[freqs < 150].sum() / power.sum()),
        "onset_mean": float(librosa.onset.onset_strength(y=mono, sr=SR).mean()),
    }


def describe(audio, anchor_features) -> dict:
    f = spectral(audio)
    return {
        "ltas_db_dist_vs_anchor": float(np.abs(f["ltas_db"] - anchor_features["ltas_db"]).mean()),
        "centroid_hz": f["centroid_hz"], "low_share": f["low_share"], "onset_mean": f["onset_mean"],
    }


def run_quality(ctx, seconds, out: Path) -> dict:
    report = {}
    pctx = TickContext(playhead_s=0.0, buffer_duration_s=seconds)
    for pair_index, (style_a, style_b) in enumerate(STYLE_PAIRS):
        tag = f"pair{pair_index}"
        song = compose(ctx, style_a, seconds)
        bundle_a = song["bundle"]
        anchor = song["anchor"]
        anchor_audio = ctx.decode_full(anchor)
        anchor_features = spectral(anchor_audio)
        save_wav(out / f"{tag}_anchor_A.wav", anchor_audio)
        rows = {"anchor_A": describe(anchor_audio, anchor_features)}
        rows["anchor_A"].update(frames=bundle_a.frames, nar=nar_backend_for(bundle_a, ctx.has_trt_nar))

        # E3 reference: seed change, same style (no style change at all).
        seed1 = ctx.solve(bundle_a, seed=1)
        rows["A_seed1"] = describe(ctx.decode_full(seed1), anchor_features)
        rows["A_seed1"]["latent_rel_l2_vs_anchor"] = rel_l2(seed1, anchor)

        # E3: same semantics, tags B, same seed (fast restyle, full solve).
        bundle_b = ctx.bundle(song["composition"], style=style_b, epoch=1)
        restyled = ctx.solve(bundle_b, seed=0)
        audio = ctx.decode_full(restyled)
        save_wav(out / f"{tag}_E3_restyle_B_seed0.wav", audio)
        rows["E3_restyle_B"] = describe(audio, anchor_features)
        rows["E3_restyle_B"]["latent_rel_l2_vs_anchor"] = rel_l2(restyled, anchor)

        # E3 reference: new semantics under B (what a re-compose gives).
        fresh = compose(ctx, style_b, seconds)
        audio = ctx.decode_full(fresh["anchor"])
        save_wav(out / f"{tag}_E3_recompose_B.wav", audio)
        rows["E3_recompose_B"] = describe(audio, anchor_features)
        rows["E3_recompose_B"]["frames"] = fresh["bundle"].frames
        ctx.release_bundle(fresh["bundle"])

        # E2: own noise, partial re-noise, through the production ring.
        backend = make_backend(ctx, song, style_a)
        for d in (0.25, 0.5, 0.75):
            lat = settle(backend, knobs(yue2_denoise=d, seed=0), pctx)
            rows[f"E2_d{d}"] = {"latent_rel_l2_vs_anchor": rel_l2(lat, anchor)}
        # E5: restyle to B in the ring, then audio-to-audio at seed 1.
        backend.handle_set_prompt(style_b)
        for d in (0.25, 0.5, 0.75, 1.0):
            lat = settle(backend, knobs(yue2_denoise=d, seed=1), pctx)
            audio = ctx.decode_full(lat)
            save_wav(out / f"{tag}_E5_B_seed1_d{d}.wav", audio)
            rows[f"E5_d{d}"] = describe(audio, anchor_features)
            rows[f"E5_d{d}"]["latent_rel_l2_vs_anchor"] = rel_l2(lat, anchor)
        backend.close()
        ctx.release_bundle(bundle_b)
        report[tag] = {"style_A": style_a, "style_B": style_b, "rows": rows}
        print(tag, json.dumps(rows, indent=1), flush=True)
    return report


# ---- latency (ticks + knob-to-ear) ---------------------------------------------


class RealtimeLoop:
    """The runner's produce-then-render cadence at a fixed probe window:
    every iteration produces once and renders the window at ``probe_s``."""

    def __init__(self, backend, seconds, probe_s):
        self.backend = backend
        self.ctx = TickContext(playhead_s=probe_s, buffer_duration_s=seconds)
        self.probe_s = probe_s
        self.ticks: list[float] = []

    def step(self, values):
        fresh = self.backend.produce(values, self.ctx, "generate")
        if self.backend.last_tick_ms > 1.0:
            self.ticks.append(self.backend.last_tick_ms)
        chunk = self.backend.render_window(self.probe_s)
        return fresh, chunk.pcm.copy()


def measure_change(loop, before, after, *, busy=None, limit=600) -> dict:
    """Settle under ``before``; with ``busy=(n, values)`` start a solve
    under ``values`` and let it run ``n`` ticks; then switch to ``after``
    and time first changed / fully updated PCM."""
    backend = loop.backend
    settle(backend, before, loop.ctx)
    current = before
    if busy is not None:
        n, current = busy
        for _ in range(n):
            loop.step(current)
    _, reference = loop.step(current)
    target_sig = None
    first_changed = fully_updated = None
    t0 = time.perf_counter()
    for _ in range(limit):
        fresh, pcm = loop.step(after)
        if target_sig is None:
            target_sig = backend._signature(backend._last_prep, backend._active_bundle)
        now = time.perf_counter() - t0
        if first_changed is None and np.abs(pcm - reference).max() > PCM_EPS:
            first_changed = now
        if fresh and backend._emerged_signature == target_sig:
            fully_updated = now
            break
    return {"first_changed_s": first_changed, "fully_updated_s": fully_updated}


def run_latency(ctx, seconds, depth, out: Path) -> dict:
    prompt = STYLE_PAIRS[0][0]
    song = compose(ctx, prompt, seconds)
    backend = make_backend(ctx, song, prompt, depth=depth)
    frames = song["bundle"].frames
    duration = playable_seconds(frames)
    loop = RealtimeLoop(backend, duration, probe_s=duration / 2)
    base = knobs(seed=1)
    changes = {
        "seed": knobs(seed=2),
        "yue2_denoise": knobs(seed=1, yue2_denoise=0.5),
        "x0_target": knobs(seed=1, x0_target=0.5),
    }
    report = {"frames": frames, "depth": depth, "cond_tokens": song["bundle"].cond_tokens,
              "nar": nar_backend_for(song["bundle"], ctx.has_trt_nar),
              "create_ms": song["timings_ms"], "changes": {}}
    torch.cuda.reset_peak_memory_stats()
    for name, after in changes.items():
        idle = measure_change(loop, base, after)
        busy = measure_change(loop, base, after, busy=(16, knobs(seed=3)))
        report["changes"][name] = {"idle": idle, "busy": busy}
        print(name, json.dumps(report["changes"][name]), flush=True)
    report["tick_ms_p50"] = float(np.percentile(loop.ticks, 50))
    report["tick_ms_p95"] = float(np.percentile(loop.ticks, 95))
    report["ticks_measured"] = len(loop.ticks)
    report["peak_torch_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
    backend.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["quality", "latency"])
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from acestep.engine.yue2_context import YuE2Context
    from acestep.engine.yue2_runtime import trt_dir, weights_root

    ctx = YuE2Context(weights_root(), trt_dir=trt_dir())
    if args.command == "quality":
        report = run_quality(ctx, args.seconds, args.out)
        name = f"quality_{int(args.seconds)}s.json"
    else:
        report = run_latency(ctx, args.seconds, args.depth, args.out)
        name = f"latency_{int(args.seconds)}s_d{args.depth}.json"
    (args.out / name).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "changes"}, indent=2, default=str))
    ctx.close()


if __name__ == "__main__":
    main()
