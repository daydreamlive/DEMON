"""YuE2Context: the process-cached YuE2 model and its conditioning path.

One context per weights root per process (``acestep.streaming.
yue2_session.get_yue2_context``), shared by every session: the ~7 GB
model, the FP32 VAE decoder, the NAR / VAE TRT engines when built, and
a single conditioning worker thread.

The conditioning path turns (style, lyrics, seed) into a song:

1. :meth:`compose`: score plan (ABC, AR) then semantic tokens (AR), the
   expensive part (seconds), upstream's ``plan`` / ``generate_semantic``
   with CUDA-graph AR on cuDNN attention as the spike ran it;
2. :meth:`bundle`: the token prefix for a style, the AR-prefix KV
   prefill (``CachedNAR``) and the stacked KV for TRT
   (:class:`~acestep.engine.yue2_velocity.YuE2Bundle`); a style change
   on a fixed composition is only this step (fast restyle);
3. :meth:`solve`: one full 32-step midpoint solve (the song anchor);
4. :meth:`decode_full` / :meth:`decode_window`: FP32 VAE decode.

The ring itself never calls into here except through the velocity
backend (:attr:`velocity`) and the window decoder.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import patch

import torch

from acestep.engine.obs import logger
from acestep.engine.yue2_runtime import (
    MODEL_DIR,
    VAE_DIR,
    ensure_import_paths,
    load_verified,
)
from acestep.engine.yue2_trt import (
    FLEX_FRAMES,
    TRTVelocity,
    TRTWindowVAE,
    decode_span,
    find_flexible_engine,
    find_vae_window_engine,
)
from acestep.engine.yue2_velocity import (
    RELEASED_STEPS,
    EagerVelocity,
    YuE2Bundle,
    bundle_from_cached_nar,
    midpoint_velocity,
    truncated_grid,
    upstream_noise,
)

#: An instrumental song: the section skeleton with no lyric lines.
DEFAULT_LYRICS = "[Verse]\n\n[Outro]\n"

#: Score-plan token budget (the spike's).
ABC_MAX_TOKENS = 4096

#: Upstream reserves its CUDA memory fraction from this budget at
#: pipeline construction; 30 GiB is what the spike ran under on a 5090.
MEMORY_BUDGET_GIB = 30


@dataclass
class Composition:
    """One composed song: the score plan and its semantic tokens."""

    plan: object                 # upstream SymbolicPlan
    tokens: list                 # semantic codec ids (one per latent frame)
    seed: int
    style: str
    lyrics: str
    truncated: bool
    timings_ms: dict = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return len(self.tokens)


class YuE2Context:
    """See module docstring."""

    def __init__(self, root: Path, *, trt_dir: Optional[Path] = None, device: str = "cuda"):
        ensure_import_paths()
        t0 = time.perf_counter()
        self.root = Path(root)
        self.trt_dir = Path(trt_dir) if trt_dir else None
        self.device = torch.device(device)
        self.model, self.model_identity = load_verified(self.root / MODEL_DIR, device=device)
        self.vae, self.vae_identity = load_verified(self.root / VAE_DIR, vae=True, device=device)
        self.pipe = self._make_pipeline()
        self.eager_velocity = EagerVelocity()
        self.velocity = self._make_velocity()
        self.window_vae = self._make_window_vae()
        # One conditioning job at a time (the AR stages own the model's
        # static caches while they run).
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yue2-cond")
        self._model_lock = threading.Lock()
        self.load_s = time.perf_counter() - t0
        logger.info(
            "yue2_context_loaded root={} nar={} vae_window={} load_s={:.1f}",
            self.root, "trt" if self.velocity is not self.eager_velocity else "eager",
            "trt" if self.window_vae is not None else "eager", self.load_s,
        )

    # ---- assembly ------------------------------------------------------------

    def _make_pipeline(self):
        from yue2 import YuE2Pipeline

        identities = {MODEL_DIR: self.model_identity, VAE_DIR: self.vae_identity}
        # Hashes were verified by load_verified; do not re-read 7 GB.
        with patch("yue2.pipeline.model_identity",
                   side_effect=lambda path, verify: identities[Path(path).name]):
            pipe = YuE2Pipeline(self.root / MODEL_DIR, self.root / VAE_DIR, device=str(self.device),
                                progress=False, memory_budget_gib=MEMORY_BUDGET_GIB)
        pipe._model = self.model
        return pipe

    def _make_velocity(self):
        engine_dir = find_flexible_engine(self.trt_dir)
        if engine_dir is None:
            return self.eager_velocity
        try:
            return TRTVelocity(engine_dir, self.eager_velocity, device=str(self.device))
        except Exception as exc:  # engine/runtime mismatch: degrade loudly
            logger.warning("yue2_nar_trt_unavailable dir={} error={!r} using=eager", engine_dir, exc)
            return self.eager_velocity

    def _make_window_vae(self):
        path = find_vae_window_engine(self.trt_dir)
        if path is None:
            return None
        try:
            return TRTWindowVAE(path, device=str(self.device))
        except Exception as exc:
            logger.warning("yue2_vae_trt_unavailable path={} error={!r} using=eager", path, exc)
            return None

    @property
    def has_trt_nar(self) -> bool:
        return self.velocity is not self.eager_velocity

    # ---- conditioning ----------------------------------------------------------

    @contextmanager
    def _ar_graphs(self):
        """Upstream CUDA-graph AR with cuDNN attention (Windows torch has
        no FlashAttention), captured in thread-local mode so a capture
        on the conditioning worker cannot fail other threads' CUDA calls."""
        from yue2.cuda_graph import GraphAR

        original_capture = GraphAR._capture

        def capture(graph_ar):
            original_graph = torch.cuda.graph

            def thread_local_graph(cuda_graph, **kwargs):
                kwargs.setdefault("capture_error_mode", "thread_local")
                return original_graph(cuda_graph, **kwargs)

            with patch("torch.cuda.graph", thread_local_graph):
                return original_capture(graph_ar)

        def make(*args, **kwargs):
            return GraphAR(*args, **dict(kwargs, attention_backend="cudnn"))

        with patch("yue2.cuda_graph.GraphAR", side_effect=make), \
                patch.object(GraphAR, "_capture", capture):
            yield

    def semantic_budget(self, max_frames: int) -> dict:
        """Semantic sampling bounds for a song of at most ``max_frames``.
        With the flexible NAR engine, songs are held to its 1000-frame
        floor (40 s) when the budget allows, so they land on TRT instead
        of the several-times-slower eager NAR."""
        floor = FLEX_FRAMES[0] if self.has_trt_nar and max_frames >= FLEX_FRAMES[0] else 1
        return {"min_tokens": floor, "max_tokens": int(max_frames)}

    def compose(self, *, style: str, lyrics: str, seed: int, max_frames: int) -> Composition:
        """Score plan + semantic tokens for one song (seconds of AR)."""
        lyrics = lyrics or DEFAULT_LYRICS
        with self._model_lock, torch.inference_mode(), self._ar_graphs():
            t0 = time.perf_counter()
            plan = self.pipe.plan(style=style, lyrics=lyrics, cot="full", seed=int(seed),
                                  abc_sampling={"min_tokens": 1, "max_tokens": ABC_MAX_TOKENS})
            t1 = time.perf_counter()
            semantic = self.pipe.generate_semantic(plan, sampling=self.semantic_budget(max_frames))
            t2 = time.perf_counter()
        if not semantic.tokens:
            raise RuntimeError("yue2 semantic stage returned no tokens")
        return Composition(
            plan=plan, tokens=list(semantic.tokens), seed=int(seed), style=style, lyrics=lyrics,
            truncated=bool(plan.truncated or semantic.truncated),
            timings_ms={"plan_ms": (t1 - t0) * 1000, "semantic_ms": (t2 - t1) * 1000},
        )

    def prefix_for(self, composition: Composition, style: Optional[str] = None) -> list:
        """The AR token prefix for ``composition`` under ``style`` (the
        composition's own style when None): same score, new ``[Tags]``."""
        from yue2.protocol import token_prefixes

        request = composition.plan.request
        if style is not None and style != request.style:
            request = dataclasses.replace(request, style=style)
        return token_prefixes(request, self.pipe.tokenizer, composition.plan.abc_ids)

    def bundle(self, composition: Composition, *, style: Optional[str] = None,
               epoch: int = 0) -> YuE2Bundle:
        """Prefill the AR-prefix KV for ``composition`` under ``style``."""
        from yue2.nar import CachedNAR, song_chunks

        prefix = self.prefix_for(composition, style)
        chunks = song_chunks(prefix, composition.tokens, composition.seed)
        if len(chunks) != 1:
            raise ValueError(f"yue2 song splits into {len(chunks)} acoustic contexts; one is supported")
        with self._model_lock, torch.inference_mode():
            nar = CachedNAR(self.model, chunks[0])
        return bundle_from_cached_nar(
            nar, seed=composition.seed, prefix=prefix, codec=composition.tokens,
            tags=style if style is not None else composition.style, lyrics=composition.lyrics,
            truncated=composition.truncated, epoch=epoch,
        )

    # ---- acoustic --------------------------------------------------------------

    @torch.inference_mode()
    def solve(self, bundle: YuE2Bundle, *, seed: Optional[int] = None,
              steps: int = RELEASED_STEPS) -> torch.Tensor:
        """One full midpoint solve from the seed's noise: ``[1, T, 64]`` bf16."""
        seed = bundle.seed if seed is None else int(seed)
        x = upstream_noise(seed, bundle.frames).to(device=self.device, dtype=torch.bfloat16)[None]
        h = 1.0 / steps
        for s in truncated_grid(steps, 1.0)[:-1].tolist():
            x = x - midpoint_velocity(self.velocity, bundle, x, [s], h) * h
        return x

    @torch.inference_mode()
    def decode_full(self, latent_btc: torch.Tensor) -> torch.Tensor:
        """Full FP32 decode: ``[2, 1920*T - 64]`` on the device, unclipped."""
        return self.vae.decode(latent_btc[0].T[None].float().contiguous())[0]

    @torch.inference_mode()
    def decode_window(self, latent_btc: torch.Tensor, start_frame: int, n_frames: int) -> torch.Tensor:
        """``[2, n_frames*1920]`` for frames ``[start, start+n)`` through the
        37-frame window decoder (TRT when built, else the eager VAE on the
        same windows, so both paths share the edge handling)."""
        decode_ctx = self.window_vae if self.window_vae is not None else self.vae.decode
        return decode_span(decode_ctx, latent_btc, start_frame, n_frames)

    # ---- worker ----------------------------------------------------------------

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Run a conditioning job on the context's worker thread."""
        return self._worker.submit(fn, *args, **kwargs)

    def release_bundle(self, bundle: YuE2Bundle) -> None:
        release = getattr(self.velocity, "release", None)
        if release is not None:
            release(bundle)
        bundle.close()

    def close(self) -> None:
        self._worker.shutdown(wait=True, cancel_futures=True)
        self.pipe = None
        self.model = self.vae = None
        self.velocity = self.eager_velocity = self.window_vae = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
