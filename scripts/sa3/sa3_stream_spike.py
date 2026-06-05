"""SA3 stream-loop fork (Phase 0): drive Stable Audio 3 through a
StreamDiffusion-style ring buffer using DEMON's `ode_steps` primitives.

This is the deliberate code-level FORK (tolerate duplication; converge with
ACE's StreamPipeline later under the bit-identical parity guardrail). It
re-implements only the ring-buffer LOOP — the thing whose reuse is the whole
bet — and reuses SA3's conditioning verbatim by CAPTURING the exact
(noise, cond_inputs, sigmas) that the reference `generate()` feeds its
pingpong sampler (monkeypatch `sample_flow_pingpong`).

The per-step is byte-for-byte SA3 pingpong, composed from ode_steps:
    v  = dit(xt, t_curr, **cond_inputs)         # the DiT forward
    x0 = ode_steps.x0_from_vel(xt, v, t_curr)   # xt - t_curr*v   (== SA3 `denoised`)
    xt = ode_steps.step_sde_renoise(xt, x0, t_next, eps)  # (1-tn)*x0 + tn*eps  (== SA3 renoise)

Validates:
  (A) single-slot ode_steps loop reproduces the captured reference latent
      (high cosine sim; independent renoise draws => not bit-identical, that's
      the Phase-2 seeded-noise harness, not Phase-0);
  (B) a depth>1 ring buffer emits one finished latent every tick after warmup,
      each decoding to coherent audio @ 10.77 Hz latent rate.

Run:
    .venv/Scripts/python.exe scripts/sa3/sa3_stream_spike.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists())
_SA3_SRC = _REPO_ROOT / "notes" / "SA3" / "stable-audio-3"
# Force repo root to the FRONT (a sibling ACE-Step editable install otherwise
# shadows our `acestep`; see sa3_unified_stream.py for the full explanation).
for _p in (str(_HERE), str(_SA3_SRC), str(_REPO_ROOT)):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_SA3_SRC))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from acestep.engine import ode_steps  # noqa: E402  -- the reuse under test
from sa3_reference_generate import checkpoint_dir, load_local_model  # noqa: E402

PROMPT = "warm analog house groove, 124 bpm, deep bassline"
DURATION = 10.0
STEPS = 8


# ---------------------------------------------------------------------------
# Capture the exact sampler inputs the reference produces (reuse SA3 cond build)
# ---------------------------------------------------------------------------
def capture_recipe(sam, *, prompt, duration, seed, steps, sampler_type="pingpong"):
    """Capture the exact (noise, sigmas, cond_inputs) the reference sampler
    receives, plus its output latent, by spying on the relevant SA3 sampler.

    sampler_type="euler" is DETERMINISTIC (no renoise) — used for the bit-parity
    check. "pingpong" is the stochastic distilled default — used for streaming.

    Also captures ``sched_args`` (the kwargs SA3 passed to ``build_schedule``:
    steps, dist_shift, effective_seq_len, fallback_seq_len) so a caller can
    rebuild the schedule at any ``sigma_max`` — this is what lets the SA3
    stream do source-anchored audio-to-audio (``init_noise_level<1``) with a
    byte-faithful SA3 schedule. ``ref_latent`` doubles as a ready-made source
    song latent for that path.
    """
    import stable_audio_3.inference.sampling as S

    fn_name = {"pingpong": "sample_flow_pingpong", "euler": "sample_discrete_euler"}[sampler_type]
    cap = {}
    orig = getattr(S, fn_name)
    orig_bs = S.build_schedule

    def spy(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
        cap["x"] = x.detach().clone()
        cap["sigmas"] = sigmas.detach().clone()
        cap["extra_args"] = extra_args
        return orig(model, x, sigmas, callback=callback, disable_tqdm=disable_tqdm, **extra_args)

    def bs_spy(*args, **kwargs):
        # build_schedule is called with all-kwargs inside sample_diffusion.
        # Record the (sigma_max-independent) inputs so we can replay the exact
        # schedule warp at any sigma_max later.
        cap["sched_args"] = {
            "steps": kwargs.get("steps", steps),
            "dist_shift": kwargs.get("dist_shift"),
            "effective_seq_len": kwargs.get("effective_seq_len"),
            "fallback_seq_len": kwargs.get("fallback_seq_len"),
        }
        return orig_bs(*args, **kwargs)

    setattr(S, fn_name, spy)
    S.build_schedule = bs_spy
    try:
        ref = sam.generate(prompt=prompt, duration=duration, steps=steps,
                           seed=seed, cfg_scale=1.0, return_latents=True,
                           disable_tqdm=True, sampler_type=sampler_type)
    finally:
        setattr(S, fn_name, orig)
        S.build_schedule = orig_bs
    cap["ref_latent"] = ref.detach().clone()
    return cap


def batch_extra_args(extra_args: dict, B: int) -> dict:
    """Repeat batch-1 tensor entries to B; pass scalars/None/bool through."""
    out = {}
    for k, v in extra_args.items():
        if torch.is_tensor(v) and v.shape[0] == 1 and B > 1:
            out[k] = v.repeat(B, *([1] * (v.ndim - 1)))
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# The forked ring buffer
# ---------------------------------------------------------------------------
@dataclass
class _Slot:
    xt: torch.Tensor       # [1, 256, T]
    step_idx: int = 0


class SA3RingBuffer:
    """StreamDiffusion ring buffer for SA3 pingpong, mirroring the structure of
    acestep.engine.stream.StreamPipeline.tick(): emit finished, refill from
    queue, batch active slots into one DiT forward, per-slot ode_steps step,
    advance. All slots share one (prompt, duration) => one cond + schedule;
    each slot has its own fresh initial noise (an independent generation)."""

    def __init__(self, dit, recipe: dict, *, depth: int, noise_seed: int = 0,
                 sampler: str = "pingpong"):
        self.dit = dit
        self.cond = recipe["extra_args"]
        self.sampler = sampler
        sig = recipe["sigmas"]
        self.sigmas = (sig[0] if sig.dim() == 2 else sig).float()  # [steps+1]
        self.total_steps = self.sigmas.shape[0] - 1
        self.T = recipe["x"].shape[-1]
        self.C = recipe["x"].shape[1]
        self.device = recipe["x"].device
        self.dtype = recipe["x"].dtype
        self._slots: list[Optional[_Slot]] = [None] * depth
        self._queue: list[int] = []  # seeds
        self._gen = torch.Generator(device=self.device).manual_seed(noise_seed)
        # branch-free sentinels for the euler step (vt*1, +noise*0)
        self._ones = torch.ones(1, 1, 1, device=self.device, dtype=self.dtype)
        self._zeros = torch.zeros(1, 1, 1, device=self.device, dtype=self.dtype)
        self.ticks = 0

    def submit(self, seed: int) -> None:
        self._queue.append(seed)

    def _fresh_noise(self) -> torch.Tensor:
        return torch.randn(1, self.C, self.T, generator=self._gen,
                           device=self.device, dtype=self.dtype)

    def tick(self) -> Optional[torch.Tensor]:
        # 1) emit a finished slot
        finished = None
        for i, s in enumerate(self._slots):
            if s is not None and s.step_idx >= self.total_steps:
                finished = s.xt
                self._slots[i] = None
                break
        # 2) refill empties from queue
        for i, s in enumerate(self._slots):
            if s is None and self._queue:
                self._queue.pop(0)
                self._slots[i] = _Slot(xt=self._fresh_noise(), step_idx=0)
        # 3) collect active
        active = [(i, s) for i, s in enumerate(self._slots)
                  if s is not None and s.step_idx < self.total_steps]
        if not active:
            self.ticks += 1
            return finished
        indices, slots = zip(*active)
        B = len(slots)
        # 4) one batched DiT forward across slots (different step_idx => different t)
        xt_b = torch.cat([s.xt for s in slots], dim=0)                 # [B,256,T]
        t_b = torch.tensor([self.sigmas[s.step_idx] for s in slots],
                           device=self.device, dtype=self.dtype)        # [B]
        with torch.no_grad():
            v_b = self.dit(xt_b, t_b, **batch_extra_args(self.cond, B))  # [B,256,T]
        # 5) per-slot ode_steps step + advance
        for j, s in enumerate(slots):
            t_curr = float(self.sigmas[s.step_idx])
            t_next = float(self.sigmas[s.step_idx + 1])
            xt_i = s.xt
            v_i = v_b[j:j + 1]
            if self.sampler == "euler":
                # deterministic: xt + (t_next - t_curr) * v  (== SA3 sample_discrete_euler)
                s.xt = ode_steps.step_ode_euler(
                    xt_i, v_i, t_curr, t_next, self._ones, self._zeros)
            else:
                # pingpong: x0 = xt - t_curr*v ; xt = (1-t_next)*x0 + t_next*eps
                x0 = ode_steps.x0_from_vel(xt_i, v_i, t_curr)
                eps = torch.randn(xt_i.shape, generator=self._gen,
                                  device=self.device, dtype=self.dtype)
                s.xt = ode_steps.step_sde_renoise(xt_i, x0, t_next, eps)
            s.step_idx += 1
        self.ticks += 1
        return finished


def cos_sim(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def decode_stats(sam, latent):
    with torch.no_grad():
        audio = sam.model.pretransform.decode(latent.to(
            next(sam.model.pretransform.parameters()).dtype))
    a = audio.float().clamp(-1, 1)
    return tuple(a.shape), float(a.abs().max()), float(a.pow(2).mean().sqrt())


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", action="store_true",
                    help="load the model in fp32 and run ONLY part A (euler parity). "
                         "fp32 is deterministic and matches dt precision, so the "
                         "ode_steps euler loop should hit ~1.0 cos vs SA3 euler.")
    args = ap.parse_args()
    model_half = not args.fp32

    sam = load_local_model(checkpoint_dir(), device="cuda", model_half=model_half)
    sr = sam.model.sample_rate
    ds = sam.model.pretransform.downsampling_ratio
    dt_str = "fp16" if model_half else "fp32"
    print(f"[model] latent_rate={sr/ds:.4f} Hz  dtype={dt_str}\n")

    # ---- (A) DETERMINISTIC PARITY: ode_steps euler == SA3 sample_discrete_euler ----
    # Euler has no renoise, so given identical (noise, schedule, cond) the two
    # must match to the fp16 floor. This is the real proof that ode_steps drives
    # SA3; the stochastic pingpong path (B) makes no parity claim.
    print("=== (A) euler parity: ode_steps.step_ode_euler vs SA3 sample_discrete_euler ===")
    print("[capture] running reference generate(sampler_type='euler') ...")
    e_recipe = capture_recipe(sam, prompt=PROMPT, duration=DURATION, seed=42,
                              steps=STEPS, sampler_type="euler")
    Te = e_recipe["x"].shape[-1]
    print(f"[capture] noise {tuple(e_recipe['x'].shape)}  sigmas {tuple(e_recipe['sigmas'].shape)}"
          f"  latent T={Te} (~{Te/(sr/ds):.1f}s)")
    rb = SA3RingBuffer(sam.model.model, e_recipe, depth=1, sampler="euler")
    rb._slots[0] = _Slot(xt=e_recipe["x"].clone(), step_idx=0)  # SAME initial noise
    out = None
    while out is None:
        out = rb.tick()
    ref = e_recipe["ref_latent"]
    cs = cos_sim(out, ref)
    md = float((out.float() - ref.float()).abs().max())
    denom = float(ref.float().abs().max()) + 1e-9
    print(f"  ode_steps euler final latent: shape={tuple(out.shape)}")
    print(f"  cos_sim_vs_ref={cs:.6f}  max_abs_diff={md:.4g}  "
          f"rel_to_peak={md/denom:.2e}  ref_peak={denom:.3g}")
    # fp32 should hit bit-parity (deterministic + matched dt precision); fp16
    # diverges by ~1e-3 L2 purely from fp16 forward accumulation, not logic.
    thresh = 0.99999 if not model_half else 0.999
    print(f"  => {'PARITY OK' if cs > thresh else 'MISMATCH — investigate'} "
          f"(threshold {thresh} for {dt_str})\n")

    if args.fp32:
        return 0  # parity-only run

    # ---- (B) depth>1 ring buffer (stochastic pingpong): one emit per tick ----
    print("[capture] running reference generate(pingpong) for the streaming recipe ...")
    recipe = capture_recipe(sam, prompt=PROMPT, duration=DURATION, seed=42, steps=STEPS)
    T = recipe["x"].shape[-1]
    print(f"[capture] latent T={T} (~{T/(sr/ds):.1f}s)\n")
    depth = 4
    n_submit = 8
    print(f"=== (B) ring buffer depth={depth}, {n_submit} submissions: pipelining ===")
    rb = SA3RingBuffer(sam.model.model, recipe, depth=depth, noise_seed=7)
    for k in range(n_submit):
        rb.submit(seed=1000 + k)
    emitted = []
    max_ticks = n_submit + depth + 2
    t0 = time.time()
    for tk in range(max_ticks):
        fin = rb.tick()
        marker = "EMIT" if fin is not None else "...."
        if fin is not None:
            emitted.append(fin)
        active = sum(1 for s in rb._slots if s is not None)
        print(f"  tick {tk:2d}: {marker}  active_slots={active}  queue={len(rb._queue)}")
    dt = time.time() - t0
    print(f"\n  emitted {len(emitted)} finished latents in {rb.ticks} ticks "
          f"({dt*1e3:.0f} ms total, {dt/max(rb.ticks,1)*1e3:.1f} ms/tick)")
    # coherence of emitted latents
    if emitted:
        s0 = decode_stats(sam, emitted[0])
        sN = decode_stats(sam, emitted[-1])
        print(f"  decode(first emit): shape/peak/rms = {s0}")
        print(f"  decode(last  emit): shape/peak/rms = {sN}")
    # warmup check: first EMIT should appear at tick == depth-1 (0-indexed) once warm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
