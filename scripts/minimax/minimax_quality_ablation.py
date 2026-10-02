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
    ap.add_argument("--frames-capture", default=None,
                    help="capture carrying raw frame_hiddens, for the "
                         "L5 chunked-vs-single-pass assembly gate")
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
        run_stream(args, ctx, codec, cond, noise, ref_pcm, out_dir, rows, record)

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


def run_stream(args, ctx, codec, cond, noise, ref_pcm, out_dir, rows, record):
    """L4/L5: the shipping streaming renderer, not a sampler nobody runs.

    These two rungs used to measure the ring buffer and the cover path.
    Neither exists any more: MiniMax is autoregressive and the backend
    drives its own chunked render loop (docs/MINIMAX.md section 2), so
    what has to be gated is different.

    **L4 is the equivalence gate.** Every rung above is measured with
    this file's own four-line Euler loop. L4 drives
    ``MiniMaxChunkRenderer.render_cond`` -- the code a session actually
    runs -- from the SAME noise and requires the two to land on the same
    latent. Without it every number above describes a sampler that does
    not ship.

    **L5 is the assembly gate.** It renders one span two ways: as the
    stream does, in overlapping chunks with a locked carry, and in a
    single pass. Any difference is assembly (carry, commit placement,
    decode guard) rather than sampling. It needs a capture carrying at
    least chunk + hop AR frames, so it is skipped with a note when
    ``--frames-capture`` is absent.
    """
    from acestep.engine.minimax_render import (
        CARRY_LATENT_FRAMES,
        CHUNK_AR_FRAMES,
        HOP_AR_FRAMES,
        MINIMAX_UPSAMPLE,
        MiniMaxChunkRenderer,
        MiniMaxLatentStream,
        RenderControls,
        latent_origin,
    )

    renderer = MiniMaxChunkRenderer(
        ctx.make_dit(latent_frames=cond.shape[1], backend="eager"),
        ctx.condition_encoder,
        device=ctx.device, dtype=ctx.dtype,
        chunk_ar_frames=CHUNK_AR_FRAMES,
        carry_latent_frames=CARRY_LATENT_FRAMES,
        latent_channels=ctx.latent_channels,
    )
    controls = RenderControls(
        steps=args.stream_steps, shift=args.stream_shift,
        guidance=args.cfg, cond_strength=1.0, seed=0,
    )

    # -- L4: the shipping sampler, on this file's own noise ---------------
    # The reference loop works in DEMON convention over [B, T, C]; the
    # renderer works in MiniMax's own over [B, C, T]. Same trajectory, so
    # the noise is handed over transposed and the result transposed back,
    # and nothing else may differ.
    # The reference run's OWN initial noise, the same draw L1-L3 use, so
    # this rung is directly comparable to them rather than a fresh take.
    noise_btc = noise
    shipped = renderer.render_cond(
        cond, carry=None, controls=controls, chunk_index=0,
        noise=noise_btc.movedim(1, 2),
    )
    shipped_btc = shipped.movedim(1, 2)
    record("L4_shipping_sampler", shipped_btc,
           codec.decode_full(shipped), ref_pcm)
    print(f"  L4_shipping_sampler: cos={rows[-1]['cos']:.5f} "
          f"logmel={rows[-1]['logmel_l1']:.4f} "
          f"rms={rows[-1]['rms_db']:.1f}dB "
          f"({renderer.last_forwards} forwards)")

    by_hand = sample(
        _demon_convention_adapter(renderer, ctx), noise_btc, cond,
        steps=args.stream_steps, cfg=args.cfg, shift=args.stream_shift,
    )
    agree = float(torch.nn.functional.cosine_similarity(
        by_hand.float().flatten(), shipped_btc.float().flatten(), dim=0))
    rel = float((by_hand.float() - shipped_btc.float()).pow(2).mean().sqrt()
                / shipped_btc.float().pow(2).mean().sqrt())
    rows[-1]["sampler_agreement_cos"] = agree
    rows[-1]["sampler_agreement_rel_rms"] = rel
    print(f"  L4b_vs_reference_loop: cos={agree:.6f} rel_rms={rel:.2e}")

    # -- L5: chunked assembly vs a single pass over the same span ---------
    if not args.frames_capture:
        print("  L5_chunked_stream: skipped (pass --frames-capture to a "
              f"capture with at least {CHUNK_AR_FRAMES + HOP_AR_FRAMES} "
              "AR frames)")
        return

    from safetensors.torch import load_file

    frames = load_file(str(args.frames_capture)).get("frame_hiddens")
    if frames is None:
        print("  L5_chunked_stream: skipped (capture has no frame_hiddens)")
        return
    have = int(frames.shape[1])
    if have < CHUNK_AR_FRAMES + HOP_AR_FRAMES:
        print(f"  L5_chunked_stream: skipped ({have} AR frames, need "
              f"{CHUNK_AR_FRAMES + HOP_AR_FRAMES})")
        return

    stream = MiniMaxLatentStream(renderer, hop_ar_frames=HOP_AR_FRAMES)
    stream.push_frames(frames.to(ctx.device))
    while stream.render_next(controls) is not None:
        pass
    committed = stream.latent_slice(0, stream.committed_frames)

    # The same span in one pass. The renderer is length-agnostic (RoPE is
    # built for whatever sequence arrives), so this is a real comparison
    # rather than a differently-shaped one.
    whole_cond = renderer.cond_encoder(
        frames.to(ctx.device, ctx.dtype)
    )[:, :stream.committed_frames]
    single = renderer.render_cond(
        whole_cond, carry=None, controls=controls, chunk_index=0,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    a = write_wav(out_dir / "L5_chunked_stream.wav",
                  codec.decode_full(committed), ctx.sample_rate)
    b = write_wav(out_dir / "L5b_single_pass.wav",
                  codec.decode_full(single), ctx.sample_rate)
    # NO ref_pcm here: the capture is a different composition from the
    # parity fixture, so a distance to the fixture's reference would be
    # a distance between two different songs. What L5 measures is the
    # chunked assembly against ITS OWN single-pass render, below.
    for name, pcm in (("L5_chunked_stream", a), ("L5b_single_pass", b)):
        row = {"name": name}
        row.update(audio_metrics(pcm, ctx.sample_rate))
        rows.append(row)
        print(f"  {name}: lr={row.get('lr_corr', float('nan')):.3f} "
              f"rms={row['rms_db']:.1f}dB "
              f"centroid={row['centroid_hz']:.0f}Hz")

    # The two are DIFFERENT takes -- different noise, different windows --
    # so they cannot be compared sample for sample. What must match is
    # the character, and what must not appear is a seam. Reported per
    # second so a localized defect shows as a defect rather than being
    # averaged into a mediocre global score.
    sec = ctx.sample_rate
    n = min(len(a), len(b))
    prof = [
        float(np.sqrt((a[i:i + sec] ** 2).mean())
              / max(np.sqrt((b[i:i + sec] ** 2).mean()), 1e-12))
        for i in range(0, n - sec + 1, sec)
    ]
    print("  per-second RMS ratio, chunked / single pass: "
          + " ".join(f"{v:.2f}" for v in prof))

    # Seam check: the sample-to-sample delta at each commit boundary
    # against the same statistic over the whole signal. A carry that is
    # not locked shows up here and nowhere else.
    d = np.abs(np.diff(a, axis=0)).max(axis=1)
    p999 = float(np.quantile(d, 0.999))
    seams = []
    for k in range(1, stream.chunks_rendered):
        idx = (
            latent_origin(k * HOP_AR_FRAMES) + CARRY_LATENT_FRAMES
        ) * MINIMAX_UPSAMPLE
        if 1 <= idx < len(d) - 1:
            seams.append(float(d[idx - 1:idx + 2].max()))
    if seams:
        verdict = "OK" if max(seams) <= 4 * p999 else "DISCONTINUITY"
        print(f"  seam |diff| max={max(seams):.5f} vs signal "
              f"p99.9={p999:.5f} -> {verdict}")


def _demon_convention_adapter(renderer, ctx):
    """A DEMON-convention shim over the streaming renderer's DiT.

    ``sample()`` above speaks ``batched_forward`` in DEMON's descending
    ``s`` over [B, T, C]; the renderer's DiT speaks MiniMax's ascending
    ``t`` over [B, C, T], with the opposite velocity sign. Bridging here
    rather than inside the renderer is deliberate: the shipping path
    works in the model's own convention and needs no conversion at all,
    which is one of the things the streaming rewrite bought.
    """
    from acestep.engine.minimax_adapter import MiniMaxAdapter

    return MiniMaxAdapter(
        renderer.dit, schedule_builder=None,
        device=ctx.device, dtype=ctx.dtype,
    )


if __name__ == "__main__":
    raise SystemExit(main())
