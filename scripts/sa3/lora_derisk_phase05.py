"""SA3 LoRA Phase 0.5 de-risk prototype (throwaway harness).

Gates the design decisions D4 (manager-owned slots, transactional
enable) and D6a (eager fallback viability) of notes/SA3_LORA_PLAN.md
BEFORE any production wiring. Five checks against the real vendored
``stable_audio_3`` package + the medium checkpoint:

1. churn    — enable A,B,C; disable B (middle slot); enable D; verify
              every adapter's contribution and per-id strength
              targeting survive the physical ParametrizationList shift
              that ``remove_lora_by_index`` causes.
2. rollback — simulate a mid-enable failure (partial weight install),
              roll back via ``remove_lora_by_index``, verify the model
              is bit-identical to its pre-enable state.
3. svd      — measure the real CPU SVD wall time an ``-xs`` adapter
              would pay at apply time on medium (sizes the disk-cache
              work item, or confirms the Phase-1 rejection).
4. hygiene  — enable adapters (DiT + conditioner), tear down the way
              ``SA3Backend.close()`` will, verify the process-cached
              model is pristine: bitwise params/buffers, no leftover
              parametrizations, and a bit-identical DiT forward.
5. bench    — eager medium DiT per-tick latency with 0/1/3 rank-16
              LoRAs at batch (pipeline depth) 1 and 4, on real
              conditioning. Prices D1's parametrize overhead and D6a's
              eager fallback.

Synthetic rank-16 adapters (deterministic randn fills) stand in for
trained files: every check here is about mechanics and cost, not
audio quality, and synthetic weights exercise the identical code path
(``add_lora`` + direct tensor install + ``set_lora_strength``).

Run (idle GPU recommended for stable bench numbers):
    .venv/Scripts/python.exe scripts/sa3/lora_derisk_phase05.py
    .venv/Scripts/python.exe scripts/sa3/lora_derisk_phase05.py --checks churn,rollback
    .venv/Scripts/python.exe scripts/sa3/lora_derisk_phase05.py --checks svd --svd-limit 40
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

# --- sys.path: repo root FIRST (so `acestep` is ours, not a sibling shadow).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists())
for _p in (str(_REPO_ROOT),):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import torch  # noqa: E402


# ---------------------------------------------------------------------------
# Adapter plumbing (the exact mechanics SA3LoRAManager will use)
# ---------------------------------------------------------------------------


def _lora_modules():
    """Vendored package imports, resolved after ensure_sa3_paths."""
    from stable_audio_3.models.lora.model import (
        LoRAParametrization,
        add_lora,
        remove_lora_by_index,
        set_lora_strength,
    )
    from stable_audio_3.models.lora.utils import get_lora_layers, has_lora

    return (
        LoRAParametrization, add_lora, remove_lora_by_index,
        set_lora_strength, get_lora_layers, has_lora,
    )


def _roots(sam):
    """The exact application roots the vendored loader uses for
    diffusion_cond models: model.model (DiT) + model.conditioner."""
    return [sam.model.model, sam.model.conditioner]


def _iter_params_of_index(sam, lora_index: int):
    """All LoRAParametrization objects belonging to one adapter,
    selected by their ``lora_index`` attribute (NOT physical position —
    this is the D4 install mechanism that is immune to prior
    removals)."""
    LoRAParametrization = _lora_modules()[0]
    for root in _roots(sam):
        for _name, mod in root.named_modules():
            plist = getattr(getattr(mod, "parametrizations", None), "weight", None)
            if plist is None:
                continue
            for p in plist:
                if isinstance(p, LoRAParametrization) and p.lora_index == lora_index:
                    yield p


def add_adapter(sam, lora_index: int, *, rank: int = 16, seed: int, strength: float = 1.0):
    """Register a plain-LoRA adapter on both roots and install synthetic
    weights by direct copy into the parametrization objects.

    Mirrors load_and_apply_loras' registration exactly (from_linear /
    from_conv1d partials, per-index), then installs weights the D4 way:
    write into the objects selected by lora_index, bypassing the
    index-remapped load_state_dict path entirely.
    """
    import torch.nn as nn
    (LoRAParametrization, add_lora, _rm, set_lora_strength, _gl, _hl) = _lora_modules()

    cfg = {
        nn.Linear: {"weight": partial(
            LoRAParametrization.from_linear, rank=rank, lora_alpha=rank,
            adapter_type="lora", lora_index=lora_index,
        )},
        nn.Conv1d: {"weight": partial(
            LoRAParametrization.from_conv1d, rank=rank, lora_alpha=rank,
            adapter_type="lora", lora_index=lora_index,
        )},
    }
    for root in _roots(sam):
        add_lora(root, cfg)

    g = torch.Generator(device="cpu").manual_seed(seed)
    n = 0
    with torch.no_grad():
        for p in _iter_params_of_index(sam, lora_index):
            # Deterministic non-zero A and B so the delta is non-trivial.
            p.lora_A.copy_(torch.randn(
                p.lora_A.shape, generator=g, dtype=torch.float32,
            ).mul_(0.02).to(p.lora_A.device))
            p.lora_B.copy_(torch.randn(
                p.lora_B.shape, generator=g, dtype=torch.float32,
            ).mul_(0.02).to(p.lora_B.device))
            n += 1
    for root in _roots(sam):
        set_lora_strength(root, strength, lora_index=lora_index)
    return n


def remove_adapter(sam, lora_index: int):
    (_L, _a, remove_lora_by_index, _s, _gl, _hl) = _lora_modules()
    for root in _roots(sam):
        remove_lora_by_index(root, lora_index)


def set_strength(sam, lora_index: int, strength: float):
    (_L, _a, _r, set_lora_strength, _gl, _hl) = _lora_modules()
    for root in _roots(sam):
        set_lora_strength(root, strength, lora_index=lora_index)


# ---------------------------------------------------------------------------
# Probes and snapshots
# ---------------------------------------------------------------------------


def probe_module(sam, fqn: str):
    """Resolve a module by FQN; a ``cond:`` prefix roots the lookup at
    the conditioner instead of the DiT."""
    if fqn.startswith("cond:"):
        mod, fqn = sam.model.conditioner, fqn[len("cond:"):]
    else:
        mod = sam.model.model
    for part in fqn.split("."):
        mod = getattr(mod, part)
    return mod


PROBE_FQNS = [
    "model.transformer.layers.0.self_attn.to_qkv",
    "model.transformer.layers.11.ff.ff.0.proj",
    "model.transformer.layers.23.self_attn.to_out",
    # The single conditioner-side target on medium (the seconds_total
    # embedder Linear) — proves conditioner application + teardown.
    "cond:conditioners.seconds_total.embedder.embedding.1",
]


def expected_weight(mod) -> torch.Tensor:
    """Recompute the effective weight by replaying the parametrization
    chain in physical order with the same op order/dtype casts as the
    vendored lora_forward, so a correct install is BIT-identical."""
    plist = mod.parametrizations.weight
    w = plist.original.detach().clone()
    for p in plist:
        delta = (p.lora_B.detach() @ p.lora_A.detach()).view(w.shape)
        delta = p.scaling * p.lora_strength * delta
        w = w + delta.to(w.dtype)
    return w


def check_probes(sam, label: str) -> bool:
    ok = True
    for fqn in PROBE_FQNS:
        mod = probe_module(sam, fqn)
        if not hasattr(mod, "parametrizations"):
            continue
        got = mod.weight.detach()
        want = expected_weight(mod)
        if not torch.equal(got, want):
            max_err = (got.float() - want.float()).abs().max().item()
            print(f"    PROBE MISMATCH [{label}] {fqn}: max_err={max_err:.3e}")
            ok = False
    return ok


def snapshot_state(sam) -> dict[str, torch.Tensor]:
    """CPU clones of every param + buffer under both application roots,
    keyed by root-qualified name. The pristine reference for rollback
    and hygiene bit-identity checks."""
    snap: dict[str, torch.Tensor] = {}
    for ri, root in enumerate(_roots(sam)):
        for name, p in root.named_parameters():
            snap[f"{ri}.{name}"] = p.detach().cpu().clone()
        for name, b in root.named_buffers():
            snap[f"{ri}.{name}"] = b.detach().cpu().clone()
    return snap


def compare_state(sam, snap: dict[str, torch.Tensor], label: str) -> bool:
    now = snapshot_state(sam)
    ok = True
    missing = set(snap) - set(now)
    extra = set(now) - set(snap)
    if missing:
        print(f"    STATE [{label}]: {len(missing)} tensors MISSING, e.g. {sorted(missing)[:3]}")
        ok = False
    if extra:
        print(f"    STATE [{label}]: {len(extra)} EXTRA tensors, e.g. {sorted(extra)[:3]}")
        ok = False
    n_diff = 0
    for k in snap.keys() & now.keys():
        if not torch.equal(snap[k], now[k]):
            if n_diff < 3:
                print(f"    STATE [{label}]: tensor differs: {k}")
            n_diff += 1
    if n_diff:
        print(f"    STATE [{label}]: {n_diff} tensors differ bitwise")
        ok = False
    return ok


def no_parametrizations_left(sam) -> bool:
    (_L, _a, _r, _s, _gl, has_lora) = _lora_modules()
    for root in _roots(sam):
        if has_lora(root):
            return False
        for _name, mod in root.named_modules():
            if getattr(mod, "parametrizations", None):
                return False
    return True


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_churn(sam) -> bool:
    print("[churn] enable A(0), B(1), C(2)...")
    for idx, seed in ((0, 100), (1, 200), (2, 300)):
        n = add_adapter(sam, idx, seed=seed, strength=1.0)
        print(f"    adapter idx={idx}: {n} parametrizations installed")
    ok = check_probes(sam, "A+B+C")

    print("[churn] disable B (middle slot, physical positions shift)...")
    remove_adapter(sam, 1)
    ok &= check_probes(sam, "A+C after removing B")

    print("[churn] enable D(3) after the shift (direct-copy install)...")
    add_adapter(sam, 3, seed=400, strength=1.0)
    ok &= check_probes(sam, "A+C+D")

    print("[churn] strength targeting by id after churn...")
    set_strength(sam, 2, 0.5)   # C
    ok &= check_probes(sam, "C@0.5")
    set_strength(sam, 3, 0.25)  # D
    ok &= check_probes(sam, "D@0.25")
    set_strength(sam, 0, 0.0)   # A muted
    ok &= check_probes(sam, "A@0")

    # Verify the STRENGTH landed on the right adapters (attribute-level).
    for idx, want in ((0, 0.0), (2, 0.5), (3, 0.25)):
        for p in _iter_params_of_index(sam, idx):
            got = float(p.lora_strength)
            if got != want:
                print(f"    STRENGTH MISMATCH idx={idx}: got {got}, want {want}")
                ok = False
            break

    # Cleanup for the next check.
    for idx in (0, 2, 3):
        remove_adapter(sam, idx)
    if not no_parametrizations_left(sam):
        print("    CLEANUP FAILED: parametrizations left after removing all")
        ok = False
    print(f"[churn] {'PASS' if ok else 'FAIL'}")
    return ok


def check_rollback(sam) -> bool:
    print("[rollback] pre-state: adapter A(0) enabled...")
    add_adapter(sam, 0, seed=100, strength=1.0)
    pre = snapshot_state(sam)

    print("[rollback] attempt enable of X(1), fail mid-install...")
    try:
        import torch.nn as nn
        (LoRAParametrization, add_lora, _r, _s, _gl, _hl) = _lora_modules()
        cfg = {
            nn.Linear: {"weight": partial(
                LoRAParametrization.from_linear, rank=16, lora_alpha=16,
                adapter_type="lora", lora_index=1,
            )},
            nn.Conv1d: {"weight": partial(
                LoRAParametrization.from_conv1d, rank=16, lora_alpha=16,
                adapter_type="lora", lora_index=1,
            )},
        }
        for root in _roots(sam):
            add_lora(root, cfg)
        # Partial install: fill only the first few parametrizations,
        # then hit the simulated truncated-state-dict error.
        g = torch.Generator(device="cpu").manual_seed(999)
        with torch.no_grad():
            for i, p in enumerate(_iter_params_of_index(sam, 1)):
                if i >= 5:
                    raise RuntimeError("simulated truncated state dict")
                p.lora_B.copy_(torch.randn(
                    p.lora_B.shape, generator=g, dtype=torch.float32,
                ).to(p.lora_B.device))
    except RuntimeError as exc:
        print(f"    enable failed as intended: {exc}")
        remove_adapter(sam, 1)  # the transactional rollback

    ok = compare_state(sam, pre, "post-rollback vs pre-enable")
    ok &= check_probes(sam, "A after rollback")

    remove_adapter(sam, 0)
    if not no_parametrizations_left(sam):
        print("    CLEANUP FAILED after rollback check")
        ok = False
    print(f"[rollback] {'PASS' if ok else 'FAIL'}")
    return ok


def check_svd(sam, limit: int | None) -> bool:
    """Time the CPU SVDs an -xs enable would run (upstream model.py:
    W0.view(out,-1).cpu().float() -> torch.linalg.svd(full_matrices=False))."""
    import torch.nn as nn
    mods = []
    for root_name, root in (("model", sam.model.model), ("conditioner", sam.model.conditioner)):
        for name, mod in root.named_modules():
            if isinstance(mod, (nn.Linear, nn.Conv1d)):
                mods.append((root_name, name, mod))
    total = len(mods)
    if limit:
        mods = mods[:limit]
    print(f"[svd] timing CPU SVD over {len(mods)}/{total} Linear/Conv1d weights...")
    t_total = 0.0
    worst = (0.0, "")
    cache_bytes_fp16 = 0
    for i, (root_name, name, mod) in enumerate(mods):
        W = mod.weight.detach()
        W2 = W.view(W.shape[0], -1).cpu().float()
        t0 = time.perf_counter()
        torch.linalg.svd(W2, full_matrices=False)
        dt = time.perf_counter() - t0
        t_total += dt
        if dt > worst[0]:
            worst = (dt, f"{root_name}.{name} {tuple(W2.shape)}")
        r = min(W2.shape)
        cache_bytes_fp16 += (W2.shape[0] * r + W2.shape[1] * r) * 2
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(mods)}: cumulative {t_total:.1f}s")
    scale = total / len(mods) if mods else 1.0
    print(
        f"[svd] measured {len(mods)} layers: {t_total:.1f}s total "
        f"(worst {worst[0] * 1000:.0f}ms @ {worst[1]}); "
        f"extrapolated full enable: ~{t_total * scale:.1f}s; "
        f"full-basis cache ~{cache_bytes_fp16 * scale / 1e9:.2f} GB fp16"
    )
    print("[svd] PASS (measurement only)")
    return True


def _bench_cond(ctx, duration: float, steps: int):
    cond = ctx.prepare_cond(prompt="warm analog house groove, 124 bpm",
                            duration=duration, steps=steps)
    return cond


def _dit_call_ms(ctx, cond, batch: int, iters: int = 50, warmup: int = 10) -> float:
    from acestep.engine.sa3_stream_helpers import stack_sa3_cond_bundles
    stacked = stack_sa3_cond_bundles([cond.cond_bundle] * batch)
    x = torch.randn(
        batch, ctx.latent_channels, cond.latent_frames,
        device=ctx.device, dtype=ctx.dtype,
    )
    t = torch.full((batch,), 0.5, device=ctx.device, dtype=ctx.dtype)
    dit = ctx.dit
    with torch.no_grad():
        for _ in range(warmup):
            dit(x, t, **stacked)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            dit(x, t, **stacked)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def check_bench(sam, ctx) -> bool:
    print("[bench] preparing real conditioning (duration=30s, steps=8)...")
    cond = _bench_cond(ctx, duration=30.0, steps=8)
    print(f"    latent_frames={cond.latent_frames}")

    results: dict[str, dict[int, float]] = {}
    configs = [("0 LoRA", []), ("1 LoRA", [0]), ("3 LoRA", [0, 1, 2])]
    installed: list[int] = []
    for label, idxs in configs:
        for idx in idxs:
            if idx not in installed:
                add_adapter(sam, idx, seed=100 + idx, strength=1.0)
                installed.append(idx)
        results[label] = {}
        for batch in (1, 4):
            ms = _dit_call_ms(ctx, cond, batch)
            results[label][batch] = ms
            print(f"    {label:7s} B={batch}: {ms:7.2f} ms/step (median)")

    # parametrize.cached() experiment (D1 candidate mitigation).
    import torch.nn.utils.parametrize as parametrize
    with parametrize.cached():
        ms_cached = _dit_call_ms(ctx, cond, 1, iters=30, warmup=5)
    print(f"    3 LoRA B=1 under parametrize.cached(): {ms_cached:7.2f} ms/step")

    for idx in installed:
        remove_adapter(sam, idx)
    torch.cuda.empty_cache()

    base = results["0 LoRA"]
    print("[bench] overhead vs 0 LoRA:")
    for label in ("1 LoRA", "3 LoRA"):
        for batch in (1, 4):
            ovh = (results[label][batch] / base[batch] - 1.0) * 100
            print(f"    {label} B={batch}: +{ovh:.1f}%")
    print("[bench] PASS (measurement only)")
    return True


def check_hygiene(sam, ctx) -> bool:
    print("[hygiene] baseline DiT forward on the pristine model...")
    cond = _bench_cond(ctx, duration=10.0, steps=8)
    from acestep.engine.sa3_stream_helpers import stack_sa3_cond_bundles
    stacked = stack_sa3_cond_bundles([cond.cond_bundle])
    x = torch.randn(
        1, ctx.latent_channels, cond.latent_frames,
        device=ctx.device, dtype=ctx.dtype,
        generator=torch.Generator(device=ctx.device.type).manual_seed(1528),
    )
    t = torch.full((1,), 0.5, device=ctx.device, dtype=ctx.dtype)
    with torch.no_grad():
        ref_out = ctx.dit(x, t, **stacked).clone()
    pre = snapshot_state(sam)
    pre_use_lora = getattr(sam.model, "use_lora", None)
    pre_lora_names = getattr(sam.model, "lora_names", None)

    print("[hygiene] session lifetime: enable 2 adapters + loader flags...")
    add_adapter(sam, 0, seed=100, strength=1.0)
    add_adapter(sam, 1, seed=200, strength=0.7)
    sam.model.use_lora = True
    sam.model.lora_names = ["synthetic-a", "synthetic-b"]
    with torch.no_grad():
        lora_out = ctx.dit(x, t, **stacked).clone()
    if torch.equal(lora_out, ref_out):
        print("    WARNING: adapters had NO effect on the forward (unexpected)")

    print("[hygiene] teardown (the SA3Backend.close() contract)...")
    remove_adapter(sam, 0)
    remove_adapter(sam, 1)
    if pre_use_lora is None:
        if hasattr(sam.model, "use_lora"):
            del sam.model.use_lora
    else:
        sam.model.use_lora = pre_use_lora
    if pre_lora_names is None:
        if hasattr(sam.model, "lora_names"):
            del sam.model.lora_names
    else:
        sam.model.lora_names = pre_lora_names
    torch.cuda.empty_cache()

    ok = no_parametrizations_left(sam)
    if not ok:
        print("    FAIL: parametrizations remain after teardown")
    ok &= compare_state(sam, pre, "post-teardown vs pristine")

    print("[hygiene] next-session forward on the same cached context...")
    with torch.no_grad():
        post_out = ctx.dit(x, t, **stacked)
    if not torch.equal(post_out, ref_out):
        max_err = (post_out.float() - ref_out.float()).abs().max().item()
        print(f"    FAIL: post-teardown forward differs (max_err={max_err:.3e})")
        ok = False
    print(f"[hygiene] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="medium")
    ap.add_argument(
        "--checks", default="churn,rollback,svd,hygiene,bench",
        help="comma list from: churn,rollback,svd,hygiene,bench",
    )
    ap.add_argument(
        "--svd-limit", type=int, default=None,
        help="only time the first N SVDs and extrapolate",
    )
    args = ap.parse_args()
    wanted = [c.strip() for c in args.checks.split(",") if c.strip()]

    from acestep.engine.sa3_context import SA3Context
    print(f"loading SA3Context(model_id={args.model!r})...")
    t0 = time.perf_counter()
    ctx = SA3Context(model_id=args.model)
    sam = ctx.sam
    print(f"loaded in {time.perf_counter() - t0:.1f}s; dtype={ctx.dtype}")

    results: dict[str, bool] = {}
    for name in wanted:
        if name == "churn":
            results[name] = check_churn(sam)
        elif name == "rollback":
            results[name] = check_rollback(sam)
        elif name == "svd":
            results[name] = check_svd(sam, args.svd_limit)
        elif name == "hygiene":
            results[name] = check_hygiene(sam, ctx)
        elif name == "bench":
            results[name] = check_bench(sam, ctx)
        else:
            print(f"unknown check: {name}")
            results[name] = False

    print("\n=== Phase 0.5 summary ===")
    for name, ok in results.items():
        print(f"  {name:9s} {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
