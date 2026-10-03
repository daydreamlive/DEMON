"""YuE2Context: the process-cached YuE2 model and its conditioning path.

One context per weights root per process (``acestep.streaming.
yue2_session.get_yue2_context``), shared by every session: the ~7 GB
model, the FP32 VAE decoder, the NAR / VAE TRT engines when built, and
a single conditioning worker thread.

The conditioning path turns (style, lyrics, seed) into a song:

1. :meth:`compose`: score plan (ABC, AR) then semantic tokens (AR), the
   expensive part (seconds), upstream's ``plan`` / ``generate_semantic``
   with CUDA-graph AR on cuDNN attention as the spike ran it;
2. :meth:`bundle`: the token prefix, the AR-prefix KV prefill
   (``CachedNAR``) and the stacked KV for TRT
   (:class:`~acestep.engine.yue2_velocity.YuE2Bundle`). Its ``style``
   argument (same score, new ``[Tags]``) is experiment-only: with frozen
   semantics a tag change is nearly inaudible (E3), so ``set_prompt``
   re-composes instead (:mod:`acestep.streaming.yue2_recompose`);
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

import torch

from acestep.engine.obs import logger
from acestep.engine.yue2_runtime import (
    MODEL_DIR,
    VAE_DIR,
    ensure_import_paths,
    load_verified,
)
from acestep.engine.yue2_trt import (
    VAE_CTX_FRAMES,
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


@contextmanager
def swapped_attr(owner, name: str, value):
    """Set ``owner.name`` to ``value`` for the block, then restore it.

    Used for the two upstream module attributes YuE2 cannot be told about
    through an argument: ``yue2.cuda_graph.GraphAR`` (``sampling`` looks
    it up at call time) and ``yue2.pipeline.model_identity``. The swap is
    process-wide, so callers hold :attr:`YuE2Context._model_lock` (or
    run at construction) and nothing else imports those names meanwhile.
    """
    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, original)


def gated_graph_ar(gate: threading.Lock):
    """Upstream ``GraphAR`` on cuDNN attention (Windows torch has no
    FlashAttention) whose capture runs in thread-local error mode AND
    under ``gate``. Thread-local mode alone did not stop the ring's CUDA
    work on the runner thread from failing a capture on the worker
    ("operation not permitted when stream is capturing", M4), so the
    ring holds the same gate around its GPU work and a capture (warmup +
    record, tens of ms) briefly excludes it. ``_capture`` is upstream's
    (``yue2/cuda_graph.py`` @ CODE_REVISION) with only the capture mode
    added."""
    from yue2.cuda_graph import GraphAR

    class GatedGraphAR(GraphAR):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **dict(kwargs, attention_backend="cudnn"))

        @torch.inference_mode()
        def _capture(self):
            with gate, torch.cuda.device(self.device):
                current = torch.cuda.current_stream(self.device)
                warmup = torch.cuda.Stream(device=self.device)
                warmup.wait_stream(current)
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        self.positions.copy_(self.initial_positions)
                        self._decode()
                    self.positions.copy_(self.initial_positions)
                current.wait_stream(warmup)
                torch.cuda.synchronize(self.device)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
                    self.output = self._decode()
                self.positions.copy_(self.initial_positions)

    return GatedGraphAR


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
        #: Held by an AR graph capture and by the ring around its GPU work.
        self.gpu_gate = threading.Lock()
        self._graph_ar = None  # gated_graph_ar(gpu_gate), built on first compose
        self.load_s = time.perf_counter() - t0
        logger.info(
            "yue2_context_loaded root={} nar={} vae_window={} load_s={:.1f}",
            self.root, "trt" if self.velocity is not self.eager_velocity else "eager",
            "trt" if self.window_vae is not None else "eager", self.load_s,
        )

    # ---- assembly ------------------------------------------------------------

    def _make_pipeline(self):
        import yue2.pipeline
        from yue2 import YuE2Pipeline

        identities = {MODEL_DIR: self.model_identity, VAE_DIR: self.vae_identity}
        # Hashes were verified by load_verified; do not re-read 7 GB.
        with swapped_attr(yue2.pipeline, "model_identity",
                          lambda path, verify: identities[Path(path).name]):
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

    def _ar_graphs(self):
        """Upstream's AR stages on :func:`gated_graph_ar` for the length of
        a compose (caller holds ``_model_lock``)."""
        import yue2.cuda_graph

        if self._graph_ar is None:
            self._graph_ar = gated_graph_ar(self.gpu_gate)
        return swapped_attr(yue2.cuda_graph, "GraphAR", self._graph_ar)

    def semantic_budget(self, max_frames: int, exact_frames: Optional[int] = None) -> dict:
        """Semantic sampling bounds for a song of at most ``max_frames``.
        With the flexible NAR engine, songs are held to the floor of the
        largest engine profile the budget reaches (1000 frames = 40 s when
        the budget allows; 250 = 10 s for shorter budgets on an engine
        with the short profile), so they land on TRT instead of the
        several-times-slower eager NAR; never below the window decoder's
        37 frames. ``exact_frames`` forces the length (a re-compose keeps
        the session geometry)."""
        if exact_frames is not None:
            return {"min_tokens": int(exact_frames), "max_tokens": int(exact_frames)}
        floor = self.velocity.frames_floor(max_frames) if self.has_trt_nar else 1
        floor = max(floor, VAE_CTX_FRAMES)
        return {"min_tokens": floor, "max_tokens": int(max_frames)}

    def compose(self, *, style: str, lyrics: str, seed: int, max_frames: int,
                exact_frames: Optional[int] = None) -> Composition:
        """Score plan + semantic tokens for one song (seconds of AR);
        exactly ``exact_frames`` semantic tokens when given."""
        lyrics = lyrics or DEFAULT_LYRICS
        with self._model_lock, torch.inference_mode(), self._ar_graphs():
            t0 = time.perf_counter()
            plan = self.pipe.plan(style=style, lyrics=lyrics, cot="full", seed=int(seed),
                                  abc_sampling={"min_tokens": 1, "max_tokens": ABC_MAX_TOKENS})
            t1 = time.perf_counter()
            semantic = self.pipe.generate_semantic(
                plan, sampling=self.semantic_budget(max_frames, exact_frames))
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
