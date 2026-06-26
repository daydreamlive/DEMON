"""Standalone Stable Audio 3 streaming ringbuffer (spike / test harness).

This module is intentionally SA3-specific. It mirrors the fundamental
StreamDiffusion/ringbuffer shape used by ACE-Step, but it does NOT route
SA3 through ACE's production ``StreamPipeline`` or the ModelAdapter seam —
production drives the shared pipeline via ``acestep.streaming.sa3_backend``.
``SA3StreamPipeline`` survives here as the reference ringbuffer exercised
by ``tests/unit/test_sa3_stream_pipeline.py`` and
``demos/test_stream_sa3_graph.py``.

The load-bearing runtime helpers (conditioning, source encode, SAME
windowed decode) now live in ``acestep.engine.sa3_stream_helpers`` and are
re-exported here so existing spike/test imports keep working.

Latents use SA3/SAME's native layout throughout: ``[B, C, T]``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Callable, Optional

import torch

# Re-exported for the spike/test importers that still do
# ``from sa3_stream_pipeline import ...``; the implementations live in the
# package now (production imports them directly, not via scripts/).
from acestep.engine.sa3_stream_helpers import (  # noqa: F401
    SA3Conditioning,
    SA3DecodeWindow,
    SA3WindowDecodeResult,
    decode_sa3_latent,
    decode_sa3_latent_window,
    encode_sa3_source,
    infer_sa3_decode_slice_alignment,
    prepare_sa3_conditioning,
    resolve_sa3_decode_window,
    sa3_decode_noise_mode,
    stack_sa3_cond_bundles,
)


@dataclass
class SA3Request:
    """One request submitted to :class:`SA3StreamPipeline`."""

    cond_bundle: dict
    latent_frames: int
    seed: Optional[int] = None
    source_latents: Optional[torch.Tensor] = None  # [1, 256, T]
    denoise: float = 1.0


@dataclass
class _SA3Slot:
    request: SA3Request
    xt: torch.Tensor
    schedule: torch.Tensor
    generator: Optional[torch.Generator] = None
    step_idx: int = 0


class SA3StreamPipeline:
    """SA3-only real-time ringbuffer.

    Each tick emits at most one finished latent, fills empty slots from the
    queue, and advances every active slot with one batched SA3 DiT forward.
    After warmup, submitting one request per tick produces one finished latent
    per tick, matching ACE's continuous streaming contract.
    """

    latent_channels: int = 256

    def __init__(
        self,
        dit,
        *,
        schedule_builder: Callable[[float], torch.Tensor],
        depth: int,
        steps: int,
        device: torch.device | str,
        dtype: torch.dtype,
        sampler: str = "pingpong",
        batch_active: bool = True,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if sampler not in {"ode", "sde", "pingpong"}:
            raise ValueError(f"Unsupported SA3 sampler {sampler!r}")

        self.dit = dit
        self.schedule_builder = schedule_builder
        self.depth = int(depth)
        self.steps = int(steps)
        self.device = torch.device(device)
        self.dtype = dtype
        self.sampler = "sde" if sampler == "pingpong" else sampler
        self.batch_active = bool(batch_active)

        self._slots: list[Optional[_SA3Slot]] = [None] * self.depth
        self._queue: list[SA3Request] = []
        self._schedule_cache: OrderedDict[float, torch.Tensor] = OrderedDict()
        self._schedule_cache_max = 64
        self.ticks = 0
        self._last_tick_ms = 0.0

    @classmethod
    def from_sched_args(
        cls,
        dit,
        sched_args: dict,
        *,
        depth: int,
        steps: int,
        device: torch.device | str,
        dtype: torch.dtype,
        sampler: str = "pingpong",
        batch_active: bool = True,
    ) -> "SA3StreamPipeline":
        """Build a pipeline from SA3 ``build_schedule`` arguments."""

        prepared = dict(sched_args)
        esl = prepared.get("effective_seq_len")
        if torch.is_tensor(esl):
            prepared["effective_seq_len"] = esl.detach().cpu()

        def _builder(denoise: float) -> torch.Tensor:
            import stable_audio_3.inference.sampling as sampling

            schedule = sampling.build_schedule(
                steps=prepared["steps"],
                sigma_max=float(denoise),
                dist_shift=prepared["dist_shift"],
                effective_seq_len=prepared["effective_seq_len"],
                fallback_seq_len=prepared["fallback_seq_len"],
                include_endpoint=True,
                device="cpu",
            )
            if schedule.dim() == 2:
                schedule = schedule[0]
            return schedule.detach().float().cpu()

        return cls(
            dit,
            schedule_builder=_builder,
            depth=depth,
            steps=steps,
            device=device,
            dtype=dtype,
            sampler=sampler,
            batch_active=batch_active,
        )

    @property
    def active_slots(self) -> int:
        return sum(1 for slot in self._slots if slot is not None)

    @property
    def is_warmed_up(self) -> bool:
        return all(slot is not None for slot in self._slots)

    def stats(self) -> dict:
        return {
            "backend": "sa3",
            "sampler": self.sampler,
            "batch_active": self.batch_active,
            "depth": self.depth,
            "steps": self.steps,
            "active_slots": self.active_slots,
            "queued": len(self._queue),
            "ticks": self.ticks,
            "last_tick_ms": self._last_tick_ms,
        }

    def submit(self, request: SA3Request) -> None:
        if request.source_latents is not None:
            self._validate_latent(request.source_latents, request.latent_frames)
        if len(self._queue) >= self.depth:
            self._queue.pop(0)
        self._queue.append(request)

    def _validate_latent(self, latent: torch.Tensor, T: int) -> None:
        if latent.ndim != 3:
            raise ValueError(f"SA3 latent must be [B,C,T], got {tuple(latent.shape)}")
        if latent.shape[0] != 1:
            raise ValueError("SA3 streaming requests are batch-1 per slot")
        if latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"SA3 latent channel mismatch: expected {self.latent_channels}, got {latent.shape[1]}"
            )
        if latent.shape[2] != T:
            raise ValueError(
                f"SA3 latent frame mismatch: request T={T}, latent T={latent.shape[2]}"
            )

    def _schedule(self, denoise: float) -> torch.Tensor:
        key = round(float(denoise), 6)
        cached = self._schedule_cache.get(key)
        if cached is not None:
            self._schedule_cache.move_to_end(key)
            return cached
        schedule = self.schedule_builder(float(denoise)).detach().float().cpu()
        if schedule.ndim != 1:
            raise ValueError(f"SA3 schedule must be 1-D, got {tuple(schedule.shape)}")
        if schedule.numel() != self.steps + 1:
            raise ValueError(
                f"SA3 schedule length mismatch: expected {self.steps + 1}, got {schedule.numel()}"
            )
        self._schedule_cache[key] = schedule
        while len(self._schedule_cache) > self._schedule_cache_max:
            self._schedule_cache.popitem(last=False)
        return schedule

    def _make_generator(self, request: SA3Request) -> Optional[torch.Generator]:
        if request.seed is None:
            return None
        return torch.Generator(device=self.device).manual_seed(int(request.seed))

    def _make_noise(
        self,
        request: SA3Request,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        return torch.randn(
            1,
            self.latent_channels,
            request.latent_frames,
            device=self.device,
            generator=generator,
        ).to(self.dtype)

    def _init_slot(self, request: SA3Request) -> _SA3Slot:
        schedule = self._schedule(request.denoise)
        generator = self._make_generator(request)
        noise = self._make_noise(request, generator)
        sigma_max = float(request.denoise)

        if request.source_latents is not None and sigma_max < 1.0:
            source = request.source_latents.to(device=self.device, dtype=self.dtype)
            xt = source * (1.0 - sigma_max) + noise * sigma_max
        else:
            xt = noise
        return _SA3Slot(request=request, xt=xt, schedule=schedule, generator=generator)

    @torch.no_grad()
    def tick(self) -> Optional[torch.Tensor]:
        t0 = time.time()

        if self._queue:
            target_T = self._queue[-1].latent_frames
            self._queue = [r for r in self._queue if r.latent_frames == target_T]
            for idx, slot in enumerate(self._slots):
                if slot is not None and slot.request.latent_frames != target_T:
                    self._slots[idx] = None

        finished = None
        for idx, slot in enumerate(self._slots):
            if slot is not None and slot.step_idx >= len(slot.schedule) - 1:
                finished = slot.xt
                self._slots[idx] = None
                break

        for idx, slot in enumerate(self._slots):
            if slot is None and self._queue:
                self._slots[idx] = self._init_slot(self._queue.pop(0))

        active = [
            slot for slot in self._slots
            if slot is not None and slot.step_idx < len(slot.schedule) - 1
        ]
        if not active:
            self._last_tick_ms = (time.time() - t0) * 1000
            self.ticks += 1
            return finished

        if self.batch_active:
            xt_batch = torch.cat([slot.xt for slot in active], dim=0)
            t_batch = torch.tensor(
                [float(slot.schedule[slot.step_idx]) for slot in active],
                device=self.device,
                dtype=self.dtype,
            )
            cond = stack_sa3_cond_bundles([slot.request.cond_bundle for slot in active])
            vt_batch = self.dit(xt_batch, t_batch, **cond)
        else:
            vt_batch = torch.cat([
                self.dit(
                    slot.xt,
                    torch.tensor(
                        [float(slot.schedule[slot.step_idx])],
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    **slot.request.cond_bundle,
                )
                for slot in active
            ], dim=0)

        for row, slot in enumerate(active):
            t_curr = slot.schedule[slot.step_idx].to(device=self.device, dtype=self.dtype)
            t_next = slot.schedule[slot.step_idx + 1].to(device=self.device, dtype=self.dtype)
            xt = slot.xt
            vt = vt_batch[row:row + 1]
            if self.sampler == "ode":
                slot.xt = xt + (t_next - t_curr) * vt
            else:
                x0 = xt - t_curr * vt
                noise = torch.randn(
                    xt.shape,
                    device=xt.device,
                    dtype=xt.dtype,
                    generator=slot.generator,
                )
                slot.xt = (1.0 - t_next) * x0 + t_next * noise
            slot.step_idx += 1

        self._last_tick_ms = (time.time() - t0) * 1000
        self.ticks += 1
        return finished

    def drain_one(self, request: SA3Request) -> torch.Tensor:
        """Submit one request and tick until its finished latent emits."""
        self.submit(request)
        for _ in range(self.steps + self.depth + 2):
            out = self.tick()
            if out is not None:
                return out
        raise RuntimeError("SA3StreamPipeline.drain_one produced no latent")
