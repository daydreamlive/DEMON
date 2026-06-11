"""ACE bit-identical drain parity rail (GPU, eager).

The safety rail for the ModelAdapter seam (backend-seam plan round 3,
Phase 3a step 1; canonical SA3 plan §4): drain a fixed battery of
seeded ticks through the production ``Session.stream()`` /
``StreamPipeline`` path with REAL ACE weights and compare the finished
latents bit-for-bit against a capture taken on the pre-seam tree.

Usage::

    # On the pre-seam tree: write the blessed capture
    .venv/Scripts/python.exe scripts/ace_drain_parity.py --capture

    # After the seam refactor: require bit-identical latents
    .venv/Scripts/python.exe scripts/ace_drain_parity.py --check

The capture lives under ``<MODELS_DIR>/parity/`` (machine-local GPU
artifact, never committed). Eager decoder + eager VAE so the drain
exercises the PyTorch forward path the seam relocates; the TRT path is
covered end-to-end by the golden harness.

The battery sweeps denoise (schedule cache + partial-denoise init),
seed, and shift across enough ticks to fill and refill the ring buffer,
in both ODE and SDE solver modes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Force THIS repo to the front of sys.path (sibling ACE-Step editable
# installs otherwise shadow our acestep; see scripts/gen_wire_types.py).
_REPO_ROOT = Path(__file__).resolve().parents[1]
while str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from acestep.paths import checkpoints_dir, models_dir  # noqa: E402

CHECKPOINT = "acestep-v15-turbo"
FIXTURE = "low_fi_Gm_loop_60s_gnm.wav"
PROMPT = "driving cinematic synthwave, analog arpeggios, 152 bpm, G minor"
SEED = 1528
STEPS = 8
DEPTH = 4

# (denoise, seed, shift) per tick — enough ticks to fill the ring
# (DEPTH) and then cycle finished latents with varying schedules.
TICK_PLAN = [
    (1.00, SEED, 3.0),
    (1.00, SEED, 3.0),
    (0.80, SEED, 3.0),
    (0.80, SEED + 1, 3.0),
    (0.55, SEED + 1, 3.5),
    (0.55, SEED + 1, 3.5),
    (0.40, SEED + 2, 3.5),
    (1.00, SEED + 2, 3.0),
    (0.75, SEED + 3, 4.0),
    (0.75, SEED + 3, 4.0),
    (0.30, SEED + 4, 3.0),
    (0.30, SEED + 4, 3.0),
]


def out_path() -> Path:
    d = models_dir() / "parity"
    d.mkdir(parents=True, exist_ok=True)
    return d / "ace_drain_eager.pt"


def _load_audio(path: Path, duration: float = 60.0):
    """Fixture loader, mirroring demos/test_stream_cover_graph.py."""
    import soundfile as sf

    from acestep.nodes.types import Audio

    sample_rate = 48000
    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != sample_rate:
        import torchaudio

        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    waveform = waveform[:2, : int(duration * sample_rate)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, : waveform.shape[-1] - rem]
    return Audio(waveform=waveform, sample_rate=sample_rate)


def run_drain() -> dict:
    from acestep.engine.session import Session
    from acestep.fixtures import audio_fixture

    session = Session(
        project_root=str(checkpoints_dir()),
        config_path=CHECKPOINT,
        decoder_backend="eager",
        vae_backend="eager",
    )

    audio = _load_audio(audio_fixture(FIXTURE))
    # The eager VAE encode SAMPLES the latent posterior from the global
    # RNG; without seeding here the source latent differs per process
    # and every drained latent diverges at noise magnitude.
    torch.manual_seed(SEED)
    source = session.prepare_source(audio)
    duration_s = audio.waveform.shape[-1] / audio.sample_rate
    conditioning = session.encode_text(
        tags=PROMPT, duration=duration_s,
    )

    results: dict = {}
    for mode in ("ode", "sde"):
        # Pin the global RNG at drain start: per-request manual_seed
        # covers initial noise, but SDE re-noise draws consume the
        # global CUDA stream between those reseeds.
        torch.manual_seed(SEED + 99)
        handle = session.stream(
            source=source,
            conditioning=conditioning,
            steps=STEPS,
            shift=3.0,
            method=mode,
            pipeline_depth=DEPTH,
        )
        latents = []
        for i, (denoise, seed, shift) in enumerate(TICK_PLAN):
            t0 = time.perf_counter()
            lat = handle.tick(
                denoise=denoise,
                seed=seed,
                shift=shift,
                source_latent=source.latent,
                steps=STEPS,
            )
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            got = lat is not None
            print(f"  [{mode}] tick {i:2d} denoise={denoise:.2f} "
                  f"seed={seed} {'EMIT' if got else '....'} ({ms:.0f} ms)")
            if got:
                latents.append(lat.tensor.detach().float().cpu())
        assert latents, f"no latents emitted in {mode} drain"
        results[mode] = latents
        handle.close()

    session.close()
    return results


def main() -> int:
    args = set(sys.argv[1:])
    path = out_path()

    if "--capture" in args:
        results = run_drain()
        torch.save(results, path)
        n = {k: len(v) for k, v in results.items()}
        print(f"[capture] wrote {path} latents={n}")
        return 0

    if "--check" in args:
        if not path.exists():
            print(f"[check] no blessed capture at {path}; run --capture "
                  f"on the pre-seam tree first")
            return 2
        blessed = torch.load(path, weights_only=True)
        results = run_drain()
        failures = []
        for mode, lats in results.items():
            ref = blessed.get(mode, [])
            if len(ref) != len(lats):
                failures.append(
                    f"{mode}: emitted {len(lats)} latents, blessed has {len(ref)}"
                )
                continue
            for i, (a, b) in enumerate(zip(lats, ref)):
                if a.shape != b.shape:
                    failures.append(f"{mode}[{i}]: shape {a.shape} != {b.shape}")
                elif not torch.equal(a, b):
                    md = (a - b).abs().max().item()
                    failures.append(f"{mode}[{i}]: NOT bit-identical (max_diff={md:.3e})")
        if failures:
            print("[check] PARITY FAILED:")
            for f in failures:
                print(f"  {f}")
            return 1
        total = sum(len(v) for v in results.values())
        print(f"[check] PARITY OK: {total} latents bit-identical to {path}")
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
