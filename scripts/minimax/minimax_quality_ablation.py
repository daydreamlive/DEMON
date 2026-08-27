"""Quality ablation ladder for the MiniMax-Music3 backend.

Answers one question: *where* between the reference trajectory and the
streamed ring buffer does audio quality get lost? Parity gates already
say the DiT and the DAV are right to ~1.0 cosine, so any audible delta
has to come from the sampler configuration or the streaming machinery
wrapped around them -- and those are exactly the two things a cosine
against a stored ``final_latent`` cannot see.

The ladder walks one variable at a time from ground truth outward:

    L0  decode the reference run's own final latent      (decode path)
    L1  our sampler, reference settings (30 steps, CFG)  (sampler path)
    L2  ..drop to the streaming step count               (step count)
    L3  ..drop guidance                                  (CFG)
    L4  one generation through the real StreamPipeline   (solver path)
    L5  the full streaming ring buffer                   (streaming path)

Each rung writes a WAV and a row of metrics against L0. Latent-domain
numbers catch trajectory error; audio-domain numbers catch everything
downstream of it, including phase damage that a cosine on a latent is
blind to.

    .venv/Scripts/python.exe scripts/minimax/minimax_quality_ablation.py \
        --fixture <models>/minimax/fixtures/minimax_music3_8s_seed7.safetensors \
        --out-dir out/ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# A sibling ACE-Step checkout shadows `acestep` otherwise.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from acestep.engine.minimax_adapter import MiniMaxAdapter  # noqa: E402
from acestep.engine.minimax_context import get_minimax_context  # noqa: E402


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 96) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(20.0), hz2mel(sr / 2.0), n_mels + 2)
    freqs = mel2hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        hi = min(hi, fb.shape[1] - 1)
        mid = min(max(mid, lo + 1), hi - 1)
        if mid <= lo:
            continue
        fb[i, lo:mid] = np.linspace(0.0, 1.0, mid - lo, endpoint=False)
        fb[i, mid:hi] = np.linspace(1.0, 0.0, hi - mid, endpoint=False)
    return fb


def _stft_mag(x: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    win = np.hanning(n_fft).astype(np.float64)
    n = 1 + max(0, (len(x) - n_fft)) // hop
    frames = np.stack([x[i * hop:i * hop + n_fft] * win for i in range(n)])
    return np.abs(np.fft.rfft(frames, axis=1))


def audio_metrics(x: np.ndarray, sr: int, ref: np.ndarray | None = None,
                  ref_sr: int | None = None) -> dict:
    """``x`` is ``[N, C]`` float32.

    ``lr_corr`` is the load-bearing one for this family: incoherent
    overlap-add and channel-order damage both show up as a negative
    left/right correlation while every latent-domain number stays clean.
    """
    if x.ndim == 1:
        x = x[:, None]
    mono = x.mean(axis=1).astype(np.float64)
    out = {
        "peak": float(np.abs(x).max()),
        "rms_db": float(20 * np.log10(max(np.sqrt((mono ** 2).mean()), 1e-12))),
    }
    if x.shape[1] > 1:
        lo, hi = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
        out["lr_corr"] = float(np.corrcoef(lo, hi)[0, 1])
        mid, side = (lo + hi) / 2, (lo - hi) / 2
        out["side_mid_db"] = float(
            10 * np.log10(max((side ** 2).mean(), 1e-20)
                          / max((mid ** 2).mean(), 1e-20))
        )
    S = _stft_mag(mono)
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    p = S ** 2 + 1e-20
    out["centroid_hz"] = float((p * freqs).sum() / p.sum())
    out["hf_ratio"] = float(p[:, freqs > 8000].sum() / p.sum())
    out["flatness"] = float(
        np.exp(np.log(S + 1e-12).mean()) / (S.mean() + 1e-12)
    )
    if ref is not None:
        rm = (ref if ref.ndim == 1 else ref.mean(axis=1)).astype(np.float64)
        if ref_sr is not None and ref_sr != sr:
            # Frame-aligned metrics against a reference at another rate
            # are meaningless -- 48 kHz vs 44.1 kHz drifts 8.8% across
            # the file, so identical audio scores as badly damaged. Put
            # both on the reference's clock first.
            import torchaudio
            mono = torchaudio.functional.resample(
                torch.from_numpy(mono).float(), sr, ref_sr,
            ).numpy().astype(np.float64)
            sr = ref_sr
        n = min(len(mono), len(rm))
        fb = _mel_filterbank(sr, 2048)
        a = np.log10(_stft_mag(mono[:n]) @ fb.T + 1e-8)
        b = np.log10(_stft_mag(rm[:n]) @ fb.T + 1e-8)
        m = min(a.shape[0], b.shape[0])
        out["logmel_l1"] = float(np.abs(a[:m] - b[:m]).mean())
    return out


def latent_metrics(x: torch.Tensor, ref: torch.Tensor) -> dict:
    a, b = x.float().flatten(), ref.float().flatten()
    return {
        "cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "rel_rms": float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()),
        "std": float(x.float().std()),
    }


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def warp(schedule: torch.Tensor, alpha: float) -> torch.Tensor:
    """The adapter's Flux/SD3 shift, applied the same way it applies it."""
    if abs(alpha - 1.0) < 1e-6:
        return schedule
    s_max = schedule[0].clone()
    u = schedule / s_max.clamp_min(1e-9)
    u = alpha * u / (1.0 + (alpha - 1.0) * u)
    out = u * s_max
    out[0] = s_max
    return out


@torch.no_grad()
def sample(
    adapter, noise_btc, cond, *, steps, cfg, denoise=1.0, source_btc=None,
    shift=1.0, guidance="vanilla", rcfg=None,
):
    """Plain Euler in DEMON convention, the reference trajectory shape.

    ``adapter.batched_forward`` already returns the velocity in DEMON's
    descending-``s`` sign, so the step is ``x += (s_next - s_cur) * v``
    with a negative ``ds`` -- which reproduces MiniMax's ``+dt`` step.

    ``guidance`` picks the combine operator: ``vanilla`` is the
    reference pipeline's ``v_u + w*(v_c - v_u)``, ``apg`` is what
    :class:`StreamPipeline` actually applies. ``rcfg`` picks how the
    negative velocity is obtained, mirroring ``SlotRequest.rcfg_mode``:
    ``None`` runs an uncond forward every step, ``initialize`` runs one
    at step 0 and reuses it, ``self`` never runs one and substitutes the
    slot's initial noise (in flow matching ``v = noise - x0``, so with
    the prior ``x0_uncond ~ 0`` the noise IS the uncond velocity).
    """
    from acestep.engine import ode_steps

    sched = warp(torch.linspace(denoise, 0.0, steps + 1), shift)
    zero = torch.zeros_like(cond)
    s0 = float(sched[0])
    if source_btc is not None and denoise < 1.0:
        x = s0 * noise_btc + (1.0 - s0) * source_btc
    else:
        x = noise_btc.clone()
    x0_noise = x.clone()

    mb = ode_steps.MomentumBuffer()
    v_u_cached = None
    for i in range(steps):
        s = float(sched[i])
        row = ([s], [None], [None], [None])
        v = adapter.batched_forward(x, *row, [{"encoder_hidden_states": cond}])
        if cfg != 1.0:
            if rcfg == "self":
                v_u = x0_noise
            elif rcfg == "initialize" and v_u_cached is not None:
                v_u = v_u_cached
            else:
                v_u = adapter.batched_forward(
                    x, *row, [{"encoder_hidden_states": zero}],
                )
                if rcfg == "initialize":
                    v_u_cached = v_u
            if guidance == "vanilla":
                v = v_u + cfg * (v - v_u)
            elif guidance == "apg":
                v = ode_steps.apg_forward(v, v_u, cfg, mb, -0.75)
            else:
                raise ValueError(guidance)
        x = x + (float(sched[i + 1]) - s) * v
    return x


def write_wav(path: Path, audio_cn: torch.Tensor, sr: int) -> np.ndarray:
    """``[C, N]`` torch -> ``[N, C]`` numpy, written to disk."""
    pcm = audio_cn.detach().float().cpu().transpose(0, 1).contiguous().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(str(path), pcm, sr)
    return pcm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out-dir", default="out/ablation")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float32"))
    ap.add_argument("--cfg", type=float, default=1.7)
    ap.add_argument("--stream-steps", type=int, default=16)
    ap.add_argument("--stream-depth", type=int, default=4)
    ap.add_argument("--stream-denoise", type=float, default=0.6)
    ap.add_argument("--stream-shift", type=float, default=2.0,
                    help="schedule warp used by the streaming rungs; "
                         "matched to --stream-steps")
    ap.add_argument("--skip-stream", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="grid steps x guidance x schedule shift instead of "
                         "the ladder; finds the cheapest usable setting")
    ap.add_argument("--sweep-steps", default="6,8,10,12,16,20,24,30,40")
    ap.add_argument("--sweep-cfg", default="1.0,1.4,1.7,2.5")
    ap.add_argument("--sweep-shift", default="1.0,1.5,2.0,3.0")
    ap.add_argument("--sweep-guidance", default="vanilla:full",
                    help="comma-separated <guidance>:<rcfg>, e.g. "
                         "'vanilla:full,apg:full,vanilla:initialize,"
                         "vanilla:self'")
    ap.add_argument("--sweep-wav", default=None,
                    help="also write a WAV for each grid point")
    ap.add_argument("--cover-sweep", action="store_true",
                    help="grid the partial-denoise cover path, which is "
                         "what actually streams, against its own anchor")
    ap.add_argument("--cover-denoise", default="0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    ctx = get_minimax_context(dtype=dtype, ar_policy="absent")
    dev = ctx.device

    data = load_file(args.fixture)
    cond = data["encoder_hidden_states"].unsqueeze(0).to(dev, ctx.dtype)
    noise = data["initial_noise"].unsqueeze(0).to(dev, ctx.dtype).movedim(1, 2)
    ref_latent = data["final_latent"].unsqueeze(0).to(dev, torch.float32)
    ref_btc = ref_latent.movedim(1, 2)
    frames = cond.shape[1]
    print(f"[fixture] frames={frames} dtype={args.dtype}")

    adapter = MiniMaxAdapter(
        ctx.make_dit(latent_frames=frames), schedule_builder=None,
        device=dev, dtype=ctx.dtype,
    )
    codec = ctx.make_codec()
    sr = ctx.sample_rate

    rows: list = []

    def record(name, latent_btc, audio_cn, ref_pcm=None):
        pcm = write_wav(out_dir / f"{name}.wav", audio_cn, sr)
        row = {"name": name}
        if latent_btc is not None:
            row.update(latent_metrics(latent_btc, ref_btc))
        row.update(audio_metrics(pcm, sr, ref_pcm))
        rows.append(row)
        return pcm

    # -- L0: ground truth latent through our decoder -----------------------
    ref_pcm = record("L0_ref_final_latent",
                     ref_btc, codec.decode_full(ref_latent))
    rows[-1]["logmel_l1"] = 0.0
    print(f"  L0_ref_final_latent: lr={rows[-1].get('lr_corr', 0):.3f} "
          f"rms={rows[-1]['rms_db']:.1f}dB")

    if args.sweep:
        return sweep(args, adapter, codec, cond, noise, ref_btc, ref_pcm,
                     sr, out_dir)
    if args.cover_sweep:
        return cover_sweep(args, adapter, codec, cond, noise, sr, out_dir)

    # -- L1..L3: our sampler, one variable at a time -----------------------
    ladder = [
        ("L1_s30_cfg%.2g" % args.cfg, 30, args.cfg),
        ("L2_s%d_cfg%.2g" % (args.stream_steps, args.cfg),
         args.stream_steps, args.cfg),
        ("L3_s%d_nocfg" % args.stream_steps, args.stream_steps, 1.0),
        ("L3b_s30_nocfg", 30, 1.0),
    ]
    for name, steps, cfg in ladder:
        lat = sample(adapter, noise, cond, steps=steps, cfg=cfg)
        record(name, lat, codec.decode_full(lat.movedim(1, 2).float()), ref_pcm)
        print(f"  {name}: cos={rows[-1]['cos']:.5f} "
              f"logmel={rows[-1]['logmel_l1']:.4f} "
              f"lr={rows[-1].get('lr_corr', float('nan')):.3f} "
              f"rms={rows[-1]['rms_db']:.1f}dB")

    # -- controls the streaming rungs have to be read against ----------
    # L4/L5 draw their own noise, so their distance from L0 is partly
    # just "a different take of the same composition". Measure that
    # floor explicitly at the reference settings, or every streaming
    # number reads as damage.
    alt = torch.randn(
        noise.shape, generator=torch.Generator().manual_seed(1528),
        dtype=torch.float32,
    ).to(noise.device, noise.dtype)
    lat = sample(adapter, alt, cond, steps=30, cfg=args.cfg)
    record("C1_alt_noise_s30_cfg%.2g" % args.cfg, lat,
           codec.decode_full(lat.movedim(1, 2).float()), ref_pcm)
    print(f"  C1_alt_noise (take-to-take floor): "
          f"logmel={rows[-1]['logmel_l1']:.4f} "
          f"lr={rows[-1].get('lr_corr', float('nan')):.3f} "
          f"rms={rows[-1]['rms_db']:.1f}dB")

    # The shipped sampler settings on that same alternate noise: the
    # honest like-for-like target for what streaming should achieve.
    lat = sample(adapter, alt, cond, steps=args.stream_steps, cfg=args.cfg,
                 shift=args.stream_shift)
    record("C2_alt_noise_shipped", lat,
           codec.decode_full(lat.movedim(1, 2).float()), ref_pcm)
    print(f"  C2_alt_noise_shipped (s{args.stream_steps} shift"
          f"{args.stream_shift:g}): logmel={rows[-1]['logmel_l1']:.4f} "
          f"lr={rows[-1].get('lr_corr', float('nan')):.3f} "
          f"rms={rows[-1]['rms_db']:.1f}dB")

    if not args.skip_stream:
        run_stream(args, ctx, codec, cond, ref_pcm, out_dir, rows, record)

    print()
    hdr = ["name", "cos", "rel_rms", "logmel_l1", "lr_corr", "side_mid_db",
           "rms_db", "centroid_hz", "hf_ratio"]
    print("".join(f"{h:>13s}" if h != "name" else f"{h:24s}" for h in hdr))
    for r in rows:
        cells = [f"{r['name']:24s}"]
        for h in hdr[1:]:
            v = r.get(h)
            cells.append("            -" if v is None else f"{v:13.4f}")
        print("".join(cells))
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {out_dir}/metrics.json")
    return 0


def sweep(args, adapter, codec, cond, noise, ref_btc, ref_pcm, sr, out_dir):
    """Grid ``steps x guidance x shift`` against the reference decode.

    The ladder says *what* broke; this says what to set instead. Two
    numbers per point: latent cosine against the reference trajectory,
    and log-mel distance against its decoded audio. They disagree in a
    useful way -- cosine rewards landing on the same sample, log-mel
    rewards sounding equally clean -- so a point has to win both before
    it earns a default.
    """
    import time

    steps_list = [int(v) for v in args.sweep_steps.split(",") if v]
    cfg_list = [float(v) for v in args.sweep_cfg.split(",") if v]
    shift_list = [float(v) for v in args.sweep_shift.split(",") if v]
    # "<guidance>:<rcfg>"; rcfg "full" means an uncond forward per step.
    modes = [m for m in args.sweep_guidance.split(",") if m]

    grid = []
    n = len(steps_list) * len(cfg_list) * len(shift_list) * len(modes)
    print(f"\n[sweep] {len(steps_list)}x{len(cfg_list)}x{len(shift_list)}"
          f"x{len(modes)} = {n} points")
    print(f"{'steps':>6s}{'cfg':>6s}{'shift':>7s}{'mode':>18s}{'fwd':>6s}"
          f"{'cos':>10s}{'logmel':>9s}{'lr':>8s}{'rms_db':>9s}"
          f"{'hf_ratio':>10s}{'ms':>8s}")
    for steps in steps_list:
        for cfg in cfg_list:
            for shift in shift_list:
                for mode in modes:
                    guidance, _, rcfg = mode.partition(":")
                    rcfg = rcfg or "full"
                    if cfg == 1.0 and mode != modes[0]:
                        continue  # unguided is the same run in every mode
                    t0 = time.perf_counter()
                    lat = sample(
                        adapter, noise, cond, steps=steps, cfg=cfg,
                        shift=shift, guidance=guidance,
                        rcfg=None if rcfg == "full" else rcfg,
                    )
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) * 1000.0
                    audio = codec.decode_full(lat.movedim(1, 2).float())
                    pcm = audio.transpose(0, 1).cpu().numpy()
                    m = latent_metrics(lat, ref_btc)
                    m.update(audio_metrics(pcm, sr, ref_pcm))
                    if cfg == 1.0:
                        fwd = steps
                    elif rcfg == "self":
                        fwd = steps
                    elif rcfg == "initialize":
                        fwd = steps + 1
                    else:
                        fwd = steps * 2
                    m.update(steps=steps, cfg=cfg, shift=shift,
                             guidance=guidance, rcfg=rcfg,
                             forwards=fwd, sample_ms=ms)
                    grid.append(m)
                    if args.sweep_wav:
                        write_wav(
                            out_dir / f"sweep_s{steps}_c{cfg:g}_x{shift:g}"
                                      f"_{guidance}_{rcfg}.wav", audio, sr,
                        )
                    print(f"{steps:6d}{cfg:6.2f}{shift:7.2f}"
                          f"{guidance + ':' + rcfg:>18s}{fwd:6d}"
                          f"{m['cos']:10.5f}{m['logmel_l1']:9.4f}"
                          f"{m.get('lr_corr', float('nan')):8.3f}"
                          f"{m['rms_db']:9.2f}{m['hf_ratio']:10.4f}{ms:8.0f}")

    (out_dir / "sweep.json").write_text(json.dumps(grid, indent=2))
    print("\n  best by log-mel distance:")
    for r in sorted(grid, key=lambda r: r["logmel_l1"])[:8]:
        print(f"    steps={r['steps']:3d} cfg={r['cfg']:.2f} "
              f"shift={r['shift']:.2f} {r['guidance']}:{r['rcfg']:10s} -> "
              f"logmel {r['logmel_l1']:.4f} cos {r['cos']:.5f} "
              f"({r['forwards']} forwards)")
    print("\n  best per forward-count budget:")
    by_budget: dict = {}
    for r in grid:
        cur = by_budget.get(r["forwards"])
        if cur is None or r["logmel_l1"] < cur["logmel_l1"]:
            by_budget[r["forwards"]] = r
    for fwd in sorted(by_budget):
        r = by_budget[fwd]
        print(f"    {fwd:3d} forwards: logmel {r['logmel_l1']:.4f} "
              f"cos {r['cos']:.5f}  (steps={r['steps']} cfg={r['cfg']:.2f} "
              f"shift={r['shift']:.2f} {r['guidance']}:{r['rcfg']})")
    print(f"\n  wrote {out_dir}/sweep.json")
    return 0


def cover_sweep(args, adapter, codec, cond, noise, sr, out_dir):
    """Grid the partial-denoise cover, measured against its own anchor.

    Streaming does not sample from noise; it re-covers a fixed anchor at
    ``minimax_denoise`` forever. That path has its own failure mode and
    the from-noise grid cannot see it: a cover can be perfectly stable,
    perfectly reproducible, and still land systematically duller than
    the anchor it came from, because the schedule spends its steps
    differently over a truncated range.

    So the reference here is the ANCHOR, not the fixture. What matters
    is whether the cover keeps the anchor's tonal balance -- ``hf_ratio``
    and ``rms_db`` relative to it -- not whether it reproduces it, since
    a cover that reproduced it exactly would not be a cover.
    """
    anchor = sample(adapter, noise, cond, steps=args.stream_steps,
                    cfg=args.cfg, shift=args.stream_shift)
    a_audio = codec.decode_full(anchor.movedim(1, 2).float())
    a_pcm = write_wav(out_dir / "cover_anchor.wav", a_audio, sr)
    am = audio_metrics(a_pcm, sr)
    print(f"\n[cover] anchor: rms {am['rms_db']:.2f} dB  "
          f"hf {am['hf_ratio']:.4f}  centroid {am['centroid_hz']:.0f} Hz  "
          f"lr {am['lr_corr']:.3f}")
    print(f"{'denoise':>8s}{'shift':>7s}{'logmel':>9s}{'d_rms':>8s}"
          f"{'hf':>9s}{'d_hf':>8s}{'centroid':>10s}{'lr':>7s}")

    grid = []
    # Re-covering uses a different noise draw than the anchor did, which
    # is the point: the cover is a variation, not a repeat.
    g = torch.Generator().manual_seed(4242)
    cover_noise = torch.randn(
        1, adapter.latent_channels, cond.shape[1], generator=g,
    ).movedim(-1, -2).to(noise.device, noise.dtype)

    for shift in [float(v) for v in args.sweep_shift.split(",") if v]:
        for d in [float(v) for v in args.cover_denoise.split(",") if v]:
            lat = sample(adapter, cover_noise, cond, steps=args.stream_steps,
                         cfg=args.cfg, denoise=d, source_btc=anchor,
                         shift=shift)
            audio = codec.decode_full(lat.movedim(1, 2).float())
            pcm = audio.transpose(0, 1).cpu().numpy()
            m = audio_metrics(pcm, sr, a_pcm)
            m.update(denoise=d, shift=shift)
            grid.append(m)
            if args.sweep_wav:
                write_wav(out_dir / f"cover_d{d:g}_x{shift:g}.wav", audio, sr)
            print(f"{d:8.2f}{shift:7.2f}{m['logmel_l1']:9.4f}"
                  f"{m['rms_db'] - am['rms_db']:8.2f}{m['hf_ratio']:9.4f}"
                  f"{m['hf_ratio'] / am['hf_ratio']:8.2f}"
                  f"{m['centroid_hz']:10.0f}{m['lr_corr']:7.3f}")

    (out_dir / "cover_sweep.json").write_text(json.dumps(grid, indent=2))
    print(f"\n  wrote {out_dir}/cover_sweep.json")
    return 0


def run_stream(args, ctx, codec, cond, ref_pcm, out_dir, rows, record):
    """L4/L5: the real backend and the real ring buffer."""
    from acestep.streaming.knobs import KnobState
    from acestep.streaming.generator_backend import TickContext
    from acestep.streaming.minimax_backend import (
        DELIVERY_SAMPLE_RATE, MiniMaxBackend, minimax_knob_specs,
    )
    from acestep.streaming.minimax_session import (
        MINIMAX_DURATION_S, MINIMAX_VAE_WINDOW_S,
    )

    # -- L4: ONE generation through the real StreamPipeline ---------------
    # depth 1 and denoise 1.0 so the only difference from L3 is the
    # solver plumbing itself, not the cover path or the ring.
    backend = MiniMaxBackend.from_context(
        ctx, cond={"encoder_hidden_states": cond},
        knob_state=KnobState(minimax_knob_specs()),
        duration_s=MINIMAX_DURATION_S, steps=args.stream_steps, depth=1,
        vae_window_s=MINIMAX_VAE_WINDOW_S,
    )
    backend.knob_state.update({
        "minimax_denoise": 1.0,
        "minimax_shift": args.stream_shift,
    })
    # Capture the noise the pipeline draws, so L4b can re-run the same
    # trajectory by hand. Comparing L4 to a DIFFERENT noise draw only
    # ever measures take-to-take variation; comparing it to the same one
    # measures the solver.
    drawn: list = []
    orig_make_noise = backend.pipeline._make_noise

    def _spy(request):
        out = orig_make_noise(request)
        drawn.append(out.detach().clone())
        return out

    backend.pipeline._make_noise = _spy
    lat = None
    for _ in range(args.stream_steps + 2):
        backend.produce(backend.read_knobs(),
                        TickContext(playhead_s=0.0,
                                    buffer_duration_s=MINIMAX_DURATION_S),
                        "generate")
        if backend._last_result_latent is not None:
            lat = backend._last_result_latent
            break
    backend.pipeline._make_noise = orig_make_noise
    if lat is not None:
        record("L4_pipeline_1gen", lat,
               codec.decode_full(lat.movedim(1, 2).float()), ref_pcm)
        print(f"  L4_pipeline_1gen: cos={rows[-1]['cos']:.5f} "
              f"logmel={rows[-1]['logmel_l1']:.4f} "
              f"lr={rows[-1].get('lr_corr', float('nan')):.3f} "
              f"rms={rows[-1]['rms_db']:.1f}dB")

    # -- L4b: the same noise, sampled by hand -----------------------------
    # The equivalence gate. StreamPipeline reaches the velocity through
    # SlotConditions, an APG combine and a compiled Euler kernel; this
    # script reaches it with four lines of Python. On identical noise the
    # two must land on the same latent, or something in the streaming
    # solver is not the sampler anyone measured.
    if lat is not None and drawn:
        by_hand = sample(
            backend.adapter, drawn[0].to(lat.device, lat.dtype), cond,
            steps=args.stream_steps, cfg=args.cfg, shift=args.stream_shift,
        )
        record("L4b_same_noise_by_hand", by_hand,
               codec.decode_full(by_hand.movedim(1, 2).float()), ref_pcm)
        agree = float(torch.nn.functional.cosine_similarity(
            by_hand.float().flatten(), lat.float().flatten(), dim=0))
        rel = float((by_hand.float() - lat.float()).pow(2).mean().sqrt()
                    / lat.float().pow(2).mean().sqrt())
        rows[-1]["pipeline_agreement_cos"] = agree
        print(f"  L4b_same_noise_by_hand: pipeline agreement cos={agree:.6f} "
              f"rel_rms={rel:.2e}  "
              f"lr={rows[-1].get('lr_corr', float('nan')):.3f}")
    backend.close()

    # -- L5: the full ring, exactly as the smoke script drives it ---------
    backend = MiniMaxBackend.from_context(
        ctx, cond={"encoder_hidden_states": cond},
        knob_state=KnobState(minimax_knob_specs()),
        duration_s=MINIMAX_DURATION_S, steps=args.stream_steps,
        depth=args.stream_depth, vae_window_s=MINIMAX_VAE_WINDOW_S,
    )
    backend.knob_state.update({
        "minimax_denoise": args.stream_denoise,
        "minimax_shift": args.stream_shift,
    })
    total = int(round(MINIMAX_DURATION_S * DELIVERY_SAMPLE_RATE))
    buf = np.zeros((total, 2), dtype=np.float32)
    gens: list = []
    playhead, lead = 0.0, 0.35
    while playhead < 20.0:
        fresh = backend.produce(
            backend.read_knobs(),
            TickContext(playhead_s=playhead % MINIMAX_DURATION_S,
                        buffer_duration_s=MINIMAX_DURATION_S),
            "generate",
        )
        if fresh and backend._last_result_latent is not None:
            gens.append(backend._last_result_latent.detach().float().cpu())
        chunk = backend.render_window((playhead + lead) % MINIMAX_DURATION_S)
        if chunk is not None:
            _crossfade(buf, chunk.pcm, chunk.start_sample)
        playhead += 0.105

    out_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(str(out_dir / "L5_ring_buffer.wav"), buf, DELIVERY_SAMPLE_RATE)
    row = {"name": "L5_ring_buffer"}
    row.update(audio_metrics(buf, DELIVERY_SAMPLE_RATE, ref_pcm,
                             ref_sr=ctx.sample_rate))
    rows.append(row)
    print(f"  L5_ring_buffer: logmel={row['logmel_l1']:.4f} "
          f"lr={row.get('lr_corr', float('nan')):.3f} "
          f"rms={row['rms_db']:.1f}dB  ({len(gens)} generations)")

    # The ring against what it should have become: the same backend's
    # own whole-song render. Anything here is assembly damage -- window
    # placement, crossfade, coverage -- and nothing to do with the
    # sampler. Reported per second so a localized hole is visible as a
    # hole rather than averaged into a mediocre global score.
    ideal = backend.render_full()
    if ideal is not None:
        sf.write(str(out_dir / "L5c_ideal_whole_render.wav"),
                 ideal.pcm, DELIVERY_SAMPLE_RATE)
        n = min(len(buf), len(ideal.pcm))
        a, b = buf[:n], ideal.pcm[:n]
        err = float(np.sqrt(((a - b) ** 2).mean())
                    / max(np.sqrt((b ** 2).mean()), 1e-12))
        vs_ideal = audio_metrics(a, DELIVERY_SAMPLE_RATE, b)
        print(f"  L5 vs its own whole render: rel_rms={err:.4f} "
              f"logmel={vs_ideal['logmel_l1']:.4f}")
        sec = DELIVERY_SAMPLE_RATE
        prof = [
            float(np.sqrt(((a[i:i + sec] - b[i:i + sec]) ** 2).mean())
                  / max(np.sqrt((b[i:i + sec] ** 2).mean()), 1e-12))
            for i in range(0, n - sec + 1, sec)
        ]
        print("  per-second rel_rms vs ideal: "
              + " ".join(f"{v:.3f}" for v in prof))

    # A generation the ring never crossfades: the same latent decoded
    # whole. Separates "the cover latents are bad" from "overlap-add
    # destroyed them".
    if gens:
        last = gens[-1].to(ctx.device)
        record("L5b_last_gen_whole", last,
               codec.decode_full(last.movedim(1, 2).float()), ref_pcm)
        print(f"  L5b_last_gen_whole: cos={rows[-1]['cos']:.5f} "
              f"logmel={rows[-1]['logmel_l1']:.4f} "
              f"lr={rows[-1].get('lr_corr', float('nan')):.3f} "
              f"rms={rows[-1]['rms_db']:.1f}dB")
        # Do consecutive covers agree? If they do not, the ring is
        # overlap-adding decorrelated audio and the level drop is
        # explained by that alone.
        pair = [
            float(torch.nn.functional.cosine_similarity(
                a.flatten(), b.flatten(), dim=0))
            for a, b in zip(gens[:-1], gens[1:])
        ]
        if pair:
            print(f"  consecutive-generation cosine: n={len(pair)} "
                  f"min={min(pair):.4f} median={float(np.median(pair)):.4f} "
                  f"max={max(pair):.4f}")
    backend.close()


def _crossfade(buf, chunk, start):
    n, total = chunk.shape[0], buf.shape[0]
    if n <= 0:
        return
    xf = min(1200, n // 4)
    patch = chunk.copy()
    if xf > 0:
        ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
        head = np.arange(start, start + xf) % total
        tail = np.arange(start + n - xf, start + n) % total
        patch[:xf] = buf[head] * (1.0 - ramp) + patch[:xf] * ramp
        patch[n - xf:] = buf[tail] * ramp[::-1] + patch[n - xf:] * (1.0 - ramp[::-1])
    buf[np.arange(start, start + n) % total] = patch


if __name__ == "__main__":
    raise SystemExit(main())
