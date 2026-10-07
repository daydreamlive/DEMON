"""Seam probe: does an SA3 loop audibly "restart" at the lap boundary? (#365)

An 8-bar loop is cut from a server fixture at a detected downbeat and
uploaded as the audio-to-audio source of a headless PRIMARY session (one
per fixture x sa3_denoise x seed). After a full lap of fresh audio the
client mirror IS the loop buffer. The seam is then measured on
x = [loop|loop] against every other bar line of x:

  novelty   self-similarity novelty (log-mel and chroma, checkerboard
            kernel ~2 s); z-score of the seam vs the other bar lines
  beat      DP beat tracking over x; the largest deviation of a beat
            interval around the seam from the median interval (ms)
  spectral  RMS log-mel distance between the 1 s before and the 1 s after
            each bar line; z-score of the seam vs the other bar lines
  level     RMS of the loop's last / first 1 s vs the body median (dB)

Controls through the same metrics:
  cut       the source cut looped ([cut|cut]): a true loop
  misalign  the cut truncated by half a bar, looped: a bar-phase jump at
            the seam (the beat grid itself stays continuous in 4/4)
  offbeat   the cut truncated by 1.5 beats, looped: a beat-phase jump
  contig    16 contiguous source bars, "seam" at bar 8: no splice at all
            (the [cut|cut] control is only a true loop when the source
            repeats every 8 bars at exactly the sidecar bpm)
  rolled    each generated loop rolled by 4 bars before looping, so the
            measured "seam" is an interior bar line of the generation

  python scripts/sa3/seam_probe.py --label main \
      --fixtures low_fi_Gm_loop_60s_gnm.wav prog_rock_loop_60s_enm.wav \
      --denoise 0.9 1.0 --seeds 1 2
  # recompute metrics from the saved loop.wav files, no server:
  python scripts/sa3/seam_probe.py --label main --reuse ...
  # controls only (no server):
  python scripts/sa3/seam_probe.py --label main --controls-only ...
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Repo root FIRST on sys.path (AGENTS.md: a sibling ACE-Step checkout can
# shadow ``acestep`` / ``demos`` otherwise).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SR = 48000
BARS = 8
HOP = 512
N_MELS = 128
KERNEL_S = 1.0          # checkerboard half-width: ~2 s kernel
SNAP_WIN_S = 0.06       # novelty peak search around a bar line
PROMPT = ("upbeat electronic dance track, driving four-on-the-floor drums, "
          "punchy synth bass, bright arpeggios, 124 bpm")
FIXTURE_DIR = os.path.expanduser("~/.daydream-scope/models/demon/fixtures")


# ---------------------------------------------------------------- audio io

def read_wav(path: str) -> np.ndarray:
    """(channels, samples) float32 at SR."""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.T
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    if x.shape[0] == 1:
        x = np.concatenate([x, x])
    return np.ascontiguousarray(x[:2])


def write_wav(path: str, x: np.ndarray) -> None:
    """x: (channels, samples)."""
    import soundfile as sf
    sf.write(path, np.clip(x.T, -1, 1), SR, subtype="PCM_16")


def fixture_bpm(name: str) -> float:
    with open(os.path.join(FIXTURE_DIR, name + ".sidecar.json")) as f:
        return float(json.load(f)["bpm"])


# ------------------------------------------------------------ the 8-bar cut

def find_downbeat(mono: np.ndarray, bpm: float) -> float:
    """First downbeat (s): beat_track at the sidecar bpm, bar phase = the
    beat phase (mod 4) with the strongest mean onset, refined to the
    nearest onset peak."""
    import librosa
    env = librosa.onset.onset_strength(y=mono, sr=SR, hop_length=HOP)
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP,
                                       start_bpm=bpm, tightness=400, units="frames")
    beats = np.asarray(beats)
    if len(beats) < 8:
        raise RuntimeError("beat tracker found too few beats")
    phase = int(np.argmax([env[beats[p::4]].mean() for p in range(4)]))
    b0 = beats[phase]
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP,
                                        units="frames")
    near = onsets[np.abs(onsets - b0) <= 3]
    if len(near):
        b0 = near[np.argmin(np.abs(near - b0))]
    return float(librosa.frames_to_time(b0, sr=SR, hop_length=HOP))


def cut_loop(wav: np.ndarray, bpm: float) -> tuple[np.ndarray, float, float, int]:
    mono = wav.mean(axis=0)
    t0 = find_downbeat(mono, bpm)
    bar_s = 240.0 / bpm
    n = int(round(BARS * bar_s * SR))
    s0 = int(round(t0 * SR))
    if s0 + n > wav.shape[1]:
        s0 = 0
    return np.ascontiguousarray(wav[:, s0:s0 + n]), t0, bar_s, s0


# ------------------------------------------------------------------ metrics

def _logmel(y: np.ndarray) -> np.ndarray:
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=HOP,
                                       n_mels=N_MELS)
    return librosa.power_to_db(m, ref=np.max(m) + 1e-12, top_db=80.0)


def _novelty(feat: np.ndarray) -> np.ndarray:
    """Foote novelty: Gaussian-tapered checkerboard over a cosine SSM."""
    f = feat - feat.mean(axis=1, keepdims=True)
    f = f / (np.linalg.norm(f, axis=0, keepdims=True) + 1e-9)
    ssm = f.T @ f
    w = int(round(KERNEL_S * SR / HOP))
    ax = np.arange(-w, w) + 0.5
    g = np.exp(-0.5 * (ax / (0.5 * w)) ** 2)
    k = np.outer(g, g) * np.outer(np.sign(ax), np.sign(ax))
    n = ssm.shape[0]
    nov = np.zeros(n)
    for i in range(w, n - w):
        nov[i] = float((ssm[i - w:i + w, i - w:i + w] * k).sum())
    return nov / (np.abs(k).sum())


def _lines(n: int, bar: float) -> tuple[list[float], float]:
    """Bar lines (s) of [a|a] where a has n samples and bar length `bar` s.
    Returns (lines, seam)."""
    lap = n / SR
    lines = []
    for base in (0.0, lap):
        k = 1
        while k * bar < lap - 1e-3:
            lines.append(base + k * bar)
            k += 1
    lines.append(lap)
    return sorted(lines), lap


def _z(seam: float, others: list[float]) -> float:
    o = np.asarray(others, dtype=float)
    sd = float(o.std())
    return float((seam - o.mean()) / sd) if sd > 1e-12 else float("nan")


def seam_metrics(a: np.ndarray, bar: float, x: np.ndarray | None = None) -> dict:
    """a: one lap, (channels, samples); bar: bar length in seconds. The
    seam is measured on x = [a|a] unless `x` (mono) is given, in which case
    x is measured with its "seam" at len(a) (used for the contiguous
    control, where x is real source audio running straight through)."""
    import librosa
    mono = a.mean(axis=0)
    n = len(mono)
    if x is None:
        x = np.concatenate([mono, mono])
    dur = len(x) / SR
    lines, seam = _lines(n, bar)
    edge = max(KERNEL_S, 1.0) + 0.05
    lines = [t for t in lines if edge <= t <= dur - edge]
    others = [t for t in lines if abs(t - seam) > 1e-6]
    fr = lambda t: int(round(t * SR / HOP))  # noqa: E731
    snap = int(round(SNAP_WIN_S * SR / HOP))

    mel = _logmel(x)
    chroma = librosa.feature.chroma_cqt(y=x, sr=SR, hop_length=HOP)
    out = {}
    for name, feat in (("novelty_mel", mel), ("novelty_chroma", chroma)):
        nov = _novelty(feat)
        val = lambda t: float(nov[max(0, fr(t) - snap):fr(t) + snap + 1].max())  # noqa: E731
        s, o = val(seam), [val(t) for t in others]
        out[name] = {"seam": round(s, 4), "others_mean": round(float(np.mean(o)), 4),
                     "others_max": round(float(np.max(o)), 4), "z": round(_z(s, o), 2)}

    # spectral: RMS dB distance of mean log-mel, 1 s before vs 1 s after
    one = fr(1.0)
    def sdist(t):
        i = fr(t)
        return float(np.sqrt(((mel[:, i - one:i].mean(1) - mel[:, i:i + one].mean(1)) ** 2).mean()))
    s, o = sdist(seam), [sdist(t) for t in others]
    out["spectral"] = {"seam_db": round(s, 2), "others_mean_db": round(float(np.mean(o)), 2),
                       "z": round(_z(s, o), 2)}

    # beat: DP beat tracking over x; a timing jump at the seam shows as an
    # irregular beat interval spanning it
    env = librosa.onset.onset_strength(y=x, sr=SR, hop_length=HOP)
    _, bts = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP,
                                     start_bpm=240.0 / bar, tightness=100, units="time")
    bts = np.asarray(bts)
    ioi = np.diff(bts)
    if len(ioi) > 4:
        med = float(np.median(ioi))
        j = int(np.searchsorted(bts, seam))  # bts[j-1] < seam <= bts[j]
        if 0 < j < len(bts):
            span = [k for k in (j - 2, j - 1, j) if 0 <= k < len(ioi)]
            seam_dev = max((ioi[k] - med for k in span), key=abs) * 1000.0
            interior = np.array([abs(ioi[k] - med) for k in range(len(ioi))
                                 if abs(bts[k] - seam) > 2 * bar and abs(bts[k + 1] - seam) > 2 * bar])
            interior = interior * 1000.0 if len(interior) else np.array([np.nan])
            out["beat"] = {"median_ms": round(med * 1000, 1),
                           "seam_dev_ms": round(float(seam_dev), 1),
                           "interior_absdev_median_ms": round(float(np.nanmedian(interior)), 1),
                           "interior_absdev_p90_ms": round(float(np.nanpercentile(interior, 90)), 1)}

    # level: last / first 1 s of the lap vs body median of 1 s bins
    nb = n // SR
    rms = lambda seg: 20 * np.log10(max(float(np.sqrt((seg ** 2).mean())), 1e-6))  # noqa: E731
    bins = [rms(mono[i * SR:(i + 1) * SR]) for i in range(nb)]
    body = float(np.median(bins[1:-1])) if nb > 2 else float(np.median(bins))
    out["level"] = {"body_db": round(body, 1),
                    "last1_vs_body_db": round(rms(mono[-SR:]) - body, 1),
                    "first1_vs_body_db": round(rms(mono[:SR]) - body, 1)}
    return out


def seam_png(a: np.ndarray, bar: float, path: str, title: str,
             x: np.ndarray | None = None) -> None:
    """Log-mel of +-4 s around the seam of [a|a] and around an interior bar
    line (bar 4 of the first lap), side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display
    mono = a.mean(axis=0)
    n = len(mono)
    if x is None:
        x = np.concatenate([mono, mono])
    seam = n / SR
    interior = 4 * bar
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (t, lab) in zip(axes, ((seam, "seam (lap boundary)"),
                                   (interior, "interior bar line (bar 4)"))):
        lo = max(0, int((t - 4.0) * SR))
        hi = min(len(x), int((t + 4.0) * SR))
        m = _logmel(x[lo:hi])
        librosa.display.specshow(m, sr=SR, hop_length=HOP, x_axis="time",
                                 y_axis="mel", ax=ax, cmap="magma")
        ax.axvline(t - lo / SR, color="cyan", lw=1.5, ls="--")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("s (window start = line - 4 s)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ session

def generate(args, cut: np.ndarray, denoise: float, seed: int) -> np.ndarray | None:
    from demos.realtime_motion_graph_web.headless_client import HeadlessClient
    cfg = {"prompt": args.prompt, "sde": False, "lora": False, "depth": 4, "steps": 8,
           "client_id": f"seam-probe-{args.label}"}
    client = HeadlessClient(args.url, cfg, cut)
    client._raw = {"sa3_denoise": float(denoise), "seed": int(seed)}
    t0 = time.time()
    ready = client.start(timeout_s=900)
    dur = float(ready["duration"])
    settle = args.settle if args.settle > 0 else dur + 20.0
    while time.time() - t0 < settle and client.running:
        time.sleep(0.5)
    if not client.running:
        print(f"FAIL client died: {client.closed_reason}", flush=True)
        client.stop()
        return None
    m = client.mirror.copy()
    slices = client.slice_count
    client.stop()
    print(f"  session dur={dur:.3f}s mirror={len(m)} samples slices={slices}", flush=True)
    return np.ascontiguousarray(m.T)


def save_run(dirpath: str, a: np.ndarray, bar: float, title: str,
             write_audio: bool = True, x: np.ndarray | None = None) -> dict:
    os.makedirs(dirpath, exist_ok=True)
    if write_audio:
        write_wav(os.path.join(dirpath, "loop.wav"), a)
        write_wav(os.path.join(dirpath, "passes3.wav"), np.concatenate([a, a, a], axis=1))
    met = seam_metrics(a, bar, x)
    seam_png(a, bar, os.path.join(dirpath, "seam.png"), title, x)
    with open(os.path.join(dirpath, "metrics.json"), "w") as f:
        json.dump(met, f, indent=1)
    return met


def row(run: str, kind: str, m: dict) -> dict:
    return {"run": run, "kind": kind,
            "nov_mel_z": m["novelty_mel"]["z"], "nov_chroma_z": m["novelty_chroma"]["z"],
            "spectral_z": m["spectral"]["z"], "spectral_seam_db": m["spectral"]["seam_db"],
            "beat_seam_dev_ms": m.get("beat", {}).get("seam_dev_ms"),
            "beat_interior_p90_ms": m.get("beat", {}).get("interior_absdev_p90_ms"),
            "last1_db": m["level"]["last1_vs_body_db"],
            "first1_db": m["level"]["first1_vs_body_db"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", default="ws://127.0.0.1:1318/")
    ap.add_argument("--fixtures", nargs="+",
                    default=["low_fi_Gm_loop_60s_gnm.wav", "prog_rock_loop_60s_enm.wav",
                             "thrash_metal_loop_60s_enm.wav"])
    ap.add_argument("--denoise", type=float, nargs="+", default=[0.9, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--settle", type=float, default=0.0,
                    help="seconds per session (default: duration + 20)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "notes", "sa3_seam_365", "out"))
    ap.add_argument("--reuse", action="store_true",
                    help="recompute from saved loop.wav files; no server")
    ap.add_argument("--controls-only", action="store_true")
    args = ap.parse_args()

    root = os.path.join(args.out, args.label)
    os.makedirs(root, exist_ok=True)
    rows, cuts = [], {}
    for fx in args.fixtures:
        stem = os.path.splitext(fx)[0]
        bpm = fixture_bpm(fx)
        src = read_wav(os.path.join(FIXTURE_DIR, fx))
        cut, t0, bar, s0 = cut_loop(src, bpm)
        cuts[stem] = {"bpm": bpm, "downbeat_s": round(t0, 4), "bar_s": round(bar, 4),
                      "cut_s": round(cut.shape[1] / SR, 4)}
        print(f"[{stem}] bpm={bpm} downbeat={t0:.3f}s bar={bar:.4f}s "
              f"cut={cut.shape[1] / SR:.3f}s", flush=True)
        m = save_run(os.path.join(root, f"{stem}_control_cut"), cut, bar,
                     f"{stem} control: source cut looped")
        rows.append(row(f"{stem}", "control_cut", m))
        half = int(round(bar / 2 * SR))
        mis = np.ascontiguousarray(cut[:, : cut.shape[1] - half])
        m = save_run(os.path.join(root, f"{stem}_control_misalign"), mis, bar,
                     f"{stem} control: cut minus half a bar, looped")
        rows.append(row(f"{stem}", "control_misalign", m))
        off = int(round(1.5 * bar / 4 * SR))
        ob = np.ascontiguousarray(cut[:, : cut.shape[1] - off])
        m = save_run(os.path.join(root, f"{stem}_control_offbeat"), ob, bar,
                     f"{stem} control: cut minus 1.5 beats, looped")
        rows.append(row(f"{stem}", "control_offbeat", m))
        n = cut.shape[1]
        if s0 + 2 * n <= src.shape[1]:
            contig = src[:, s0:s0 + 2 * n].mean(axis=0)
            m = save_run(os.path.join(root, f"{stem}_control_contig"), cut, bar,
                         f"{stem} control: 16 contiguous source bars, line at bar 8",
                         write_audio=False, x=contig)
            rows.append(row(f"{stem}", "control_contig", m))
        if args.controls_only:
            continue
        for d in args.denoise:
            for s in args.seeds:
                run = f"{stem}_d{d:.2f}_s{s}"
                rd = os.path.join(root, run)
                print(f"[{run}]", flush=True)
                if args.reuse:
                    a = read_wav(os.path.join(rd, "loop.wav"))
                else:
                    a = generate(args, cut, d, s)
                    if a is None:
                        rows.append({"run": run, "kind": "generated", "error": "client died"})
                        continue
                gbar = a.shape[1] / SR / BARS
                m = save_run(rd, a, gbar, f"{run} generated", write_audio=not args.reuse)
                rows.append(row(run, "generated", m))
                rolled = np.roll(a, -int(round(4 * gbar * SR)), axis=1)
                m = save_run(os.path.join(rd, "rolled"), rolled, gbar,
                             f"{run} control: generated rolled 4 bars (seam = interior)",
                             write_audio=False)
                rows.append(row(run, "control_rolled", m))
                print(f"  gen  {rows[-2]}\n  roll {rows[-1]}", flush=True)
                if not args.reuse:
                    time.sleep(1.0)

    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump({"label": args.label, "prompt": args.prompt, "cuts": cuts, "rows": rows}, f,
                  indent=1)
    cols = ["run", "kind", "nov_mel_z", "nov_chroma_z", "spectral_z", "spectral_seam_db",
            "beat_seam_dev_ms", "beat_interior_p90_ms", "last1_db", "first1_db"]
    lines = [f"# seam probe: {args.label}", "", f"prompt: {args.prompt}", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, r.get("error", ""))) for c in cols) + " |")
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
