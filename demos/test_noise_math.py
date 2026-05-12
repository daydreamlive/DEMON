"""Synthetic validation of the noise-sharing math.

No model required. Validates:

1. The current EMA scheme decays as alpha^N between gen 1 and gen N.
2. An "anchor" scheme (blend with frozen anchor noise, no decay) holds a
   stable alpha across all gens.
3. The elementwise blend produces no temporal structure (autocorrelation
   along T stays a delta).
4. Position-aligned "scrolling" noise produces frame-by-frame
   correlation that tracks the playhead shift.

Usage:
    uv run python demos/test_noise_math.py
"""
import math
import torch


T = 1500            # frames (60s @ 25fps), matches realtime_motion demo
D = 64              # latent channels
N_GENS = 8          # how many generations to roll out per scheme
ALPHAS = (0.3, 0.5, 0.7, 0.9, 0.95, 0.99)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().double()
    bf = b.flatten().double()
    return float((af @ bf) / (af.norm() * bf.norm()).clamp_min(1e-12))


def gen_ema(alpha: float, n: int, gen: torch.Generator) -> list[torch.Tensor]:
    """Replicates acestep/engine/stream.py:391-396 exactly."""
    out, last = [], None
    for _ in range(n):
        noise = torch.randn(1, T, D, generator=gen)
        if alpha > 0.0 and last is not None:
            noise = alpha * last + math.sqrt(1.0 - alpha**2) * noise
        last = noise.clone()
        out.append(noise)
    return out


def gen_anchor(alpha: float, n: int, gen: torch.Generator) -> list[torch.Tensor]:
    """Anchor mode: anchor = first fresh noise; every subsequent gen is
    alpha*anchor + sqrt(1-a^2)*fresh. Stable alpha-weight, no decay."""
    out = []
    anchor = None
    for _ in range(n):
        fresh = torch.randn(1, T, D, generator=gen)
        if anchor is None:
            anchor = fresh.clone()
            out.append(fresh)
        else:
            out.append(alpha * anchor + math.sqrt(1.0 - alpha**2) * fresh)
    return out


def gen_scrolling(stride_frames: int, n: int, gen: torch.Generator) -> list[torch.Tensor]:
    """Position-aligned scrolling noise. The same audio frame across
    consecutive gens reads from the same position in a long noise buffer.

    Models a streaming use case where each tick advances by
    ``stride_frames`` along T. Frame N of gen K+1 should equal frame
    (N + stride) of gen K's underlying buffer, so noise is *aligned to
    the playhead*, not to the tensor index.
    """
    buf_len = T + stride_frames * (n - 1)
    buf = torch.randn(1, buf_len, D, generator=gen)
    return [buf[:, k * stride_frames : k * stride_frames + T, :].clone() for k in range(n)]


def variance_table(label: str, gens: list[torch.Tensor]) -> None:
    var = torch.stack([g.var(unbiased=False) for g in gens]).mean().item()
    print(f"  {label:>18s}  mean per-element variance: {var:.4f}  (target 1.0)")


def corr_matrix(gens: list[torch.Tensor]) -> torch.Tensor:
    n = len(gens)
    M = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            M[i, j] = cos(gens[i], gens[j])
    return M


def print_corr_to_gen1(label: str, gens: list[torch.Tensor]) -> None:
    n = len(gens)
    cells = " ".join(f"{cos(gens[0], gens[k]):+.3f}" for k in range(n))
    print(f"  {label:>18s}  cos(gen1, genK) k=1..{n}: {cells}")


def autocorr_along_T(g: torch.Tensor, lags: tuple[int, ...] = (0, 1, 2, 5, 10, 25)) -> list[float]:
    """Cosine sim between g[:, :T-lag, :] and g[:, lag:, :].
    A delta at lag=0 with ~0 elsewhere means i.i.d. across T."""
    out = []
    for lag in lags:
        if lag == 0:
            out.append(1.0)
            continue
        a = g[:, : g.shape[1] - lag, :]
        b = g[:, lag:, :]
        out.append(cos(a, b))
    return out


def main():
    print("=" * 78)
    print("Noise-sharing math validation")
    print(f"  shape per gen: [1, {T}, {D}],  N_GENS={N_GENS}")
    print("=" * 78)

    print("\n[1] CURRENT EMA SCHEME (acestep/engine/stream.py:391-396)")
    print("    noise = alpha*_last_noise + sqrt(1-a^2)*fresh; _last_noise=noise.clone()")
    print("    Predicted cos(gen1, gen_{1+k}) = alpha**k.\n")
    print(f"  {'predicted (alpha^k)':>30s}  " +
          " ".join(f"k={k}" for k in range(N_GENS)))
    for a in ALPHAS:
        gen = torch.Generator().manual_seed(0)
        gens = gen_ema(a, N_GENS, gen)
        actual = [cos(gens[0], gens[k]) for k in range(N_GENS)]
        pred = [a**k for k in range(N_GENS)]
        print(f"  alpha={a:.2f} predicted: " + " ".join(f"{p:+.3f}" for p in pred))
        print(f"             actual  : " + " ".join(f"{x:+.3f}" for x in actual))
    print()
    print("    -> Half-life in ticks (cos drops below 0.5):")
    for a in ALPHAS:
        if a >= 1.0:
            print(f"       alpha={a:.2f}: infinite (degenerate)")
        else:
            hl = math.log(0.5) / math.log(a)
            print(f"       alpha={a:.2f}: ~{hl:.1f} ticks")

    print("\n" + "=" * 78)
    print("[2] ANCHOR SCHEME  (proposed fix #1)")
    print("    anchor=first_noise; noise_k = alpha*anchor + sqrt(1-a^2)*fresh_k")
    print("    Predicted cos(gen1, gen_k) = alpha  for all k>=2.\n")
    for a in ALPHAS:
        gen = torch.Generator().manual_seed(0)
        gens = gen_anchor(a, N_GENS, gen)
        actual = [cos(gens[0], gens[k]) for k in range(N_GENS)]
        print(f"  alpha={a:.2f} cos(gen1, gen_k) k=0..{N_GENS-1}: " +
              " ".join(f"{x:+.3f}" for x in actual))

    print("    Pairwise (non-gen-1) cos at alpha=0.7:")
    gen = torch.Generator().manual_seed(0)
    gens = gen_anchor(0.7, N_GENS, gen)
    M = corr_matrix(gens)
    for i in range(N_GENS):
        print("       " + " ".join(f"{M[i,j].item():+.3f}" for j in range(N_GENS)))
    print("    -> off-diagonal cells should all be ~alpha^2 = 0.49 (shared anchor weight)")

    print("\n" + "=" * 78)
    print("[3] VARIANCE PRESERVATION CHECK")
    gen = torch.Generator().manual_seed(0)
    variance_table("ema a=0.5", gen_ema(0.5, N_GENS, gen))
    gen = torch.Generator().manual_seed(0)
    variance_table("ema a=0.9", gen_ema(0.9, N_GENS, gen))
    gen = torch.Generator().manual_seed(0)
    variance_table("anchor a=0.5", gen_anchor(0.5, N_GENS, gen))
    gen = torch.Generator().manual_seed(0)
    variance_table("anchor a=0.9", gen_anchor(0.9, N_GENS, gen))
    gen = torch.Generator().manual_seed(0)
    variance_table("fresh randn", [torch.randn(1, T, D, generator=gen) for _ in range(N_GENS)])

    print("\n" + "=" * 78)
    print("[4] AUTOCORRELATION ALONG T (frame-to-frame structure)")
    print("    All noise should be ~delta(lag=0) along T — no temporal smoothing.")
    lags = (0, 1, 2, 5, 10, 25)
    print(f"  {'scheme':>22s}  " + " ".join(f"lag={l:>2d}" for l in lags))
    gen = torch.Generator().manual_seed(0)
    g = gen_ema(0.9, 1, gen)[0]
    print(f"  {'ema (single gen)':>22s}  " + " ".join(f"{x:+.3f}" for x in autocorr_along_T(g, lags)))
    gen = torch.Generator().manual_seed(0)
    g = gen_anchor(0.9, 1, gen)[0]
    print(f"  {'anchor (single gen)':>22s}  " + " ".join(f"{x:+.3f}" for x in autocorr_along_T(g, lags)))

    print("\n" + "=" * 78)
    print("[5] SCROLLING NOISE  (proposed fix #2 — position-aligned for streaming)")
    print("    Long noise buffer; each gen reads an offset window. Frame N at")
    print("    tick K+1 == frame (N-stride) at tick K. Models walk-mode.")
    stride = 25  # 1s @ 25fps
    gen = torch.Generator().manual_seed(0)
    gens = gen_scrolling(stride, N_GENS, gen)
    print(f"    stride={stride} frames, N={N_GENS}")
    print(f"    overlap frac per consecutive pair: {(T - stride) / T:.3f}")
    print("    Pairwise cos (whole-tensor, unaligned):")
    M = corr_matrix(gens)
    for i in range(N_GENS):
        print("       " + " ".join(f"{M[i,j].item():+.3f}" for j in range(N_GENS)))
    print("    -> unaligned cos is small. The point is per-frame alignment.")
    print("    Now compare gen_k frame N vs gen_{k+1} frame N-stride:")
    g0, g1 = gens[0], gens[1]
    aligned = cos(g0[:, stride:, :], g1[:, :T - stride, :])
    misaligned = cos(g0, g1)
    print(f"       aligned cos (shifted by stride): {aligned:+.3f}")
    print(f"       misaligned cos (raw):            {misaligned:+.3f}")

    print("\n" + "=" * 78)
    print("Summary:")
    print(" - EMA: cos(gen1, gen_k) = alpha^k. At alpha=0.5 you lose half of")
    print("   gen 1 every tick. After 4 ticks at alpha=0.5, only 6% remains.")
    print(" - Anchor: stable alpha across all gens. Pairwise cos ~= alpha^2")
    print("   between any two non-anchor gens. No decay.")
    print(" - Noise is i.i.d. along T regardless of blend — no musical")
    print("   beat/bar structure is induced by either scheme.")
    print(" - Scrolling noise gives per-frame alignment at the cost of needing")
    print("   to know the streaming stride.")


if __name__ == "__main__":
    main()
