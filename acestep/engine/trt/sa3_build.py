#!/usr/bin/env python3
"""Build TensorRT engines for the SA3 (Stable Audio 3) family.

Single entry point for SA3 engine creation, holding the same shape as the
ACE-Step builder (:mod:`acestep.engine.trt.build`): env preflight, sidecar
``.metadata.json`` skip/rebuild gates, a canonical ``--all`` matrix with
``--dry-run`` / ``--force-rebuild``, and the per-engine layout
``<engines_dir>/<name>/<name>.trt`` under ``<models>/sa3/trt_engines/``
(the directory :func:`acestep.engine.sa3_trt.find_dit_engine` discovers).

Unlike the ACE builder there is no local ONNX export step: both engines
compile Stability's OFFICIAL ONNX exports from
``stabilityai/stable-audio-3-optimized``, fetched via ``huggingface_hub``
on first use.

* **sa3-m DiT**: the pre-surgered FP16-mixed graph (``dit_fp16.onnx``,
  FP16 trunk with FP32 islands around RMSNorm, attention softmax, and
  RoPE) compiled as a STRONGLY_TYPED network with no builder precision
  flags. This is upstream's canonical recipe. The BF16 recipe
  (``dit.onnx`` + ``BuilderFlag.BF16``) is explicitly rejected upstream:
  its quantization error compounds over the 8 pingpong steps (final-latent
  cos drifts to ~0.81 vs torch fp32), which reproduced here as the
  real-cond parity gap (cos 0.80-0.97/step). The fp16mixed build measures
  cos >= 0.9998/step vs eager on real conditioning
  (``scripts/sa3/sa3_trt_dit_cond_parity.py``).
* **sa3-m DiT (FP8, opt-in)**: ``dit_fp8.onnx`` (ModelOpt FP8 GEMM trunk on
  top of the fp16mixed graph) compiled to ``sa3_m_dit_fp8_l*`` engines,
  ~1.8x/step at compounded-euler cos ~0.976 vs fp16mixed. Built only with
  ``--fp8`` (additive to the fp16mixed DiT). The HF pair is not fetchable
  yet (see :data:`DIT_FP8_ONNX_FILES`); pass ``--fp8-onnx`` a
  producer-built graph until it is.
  :func:`acestep.engine.sa3_trt.find_dit_engine` prefers an fp8 engine when one
  covers the window, else fp16mixed.
* **sa3-sm-music DiT** (``--model small-music``): upstream's
  ``onnx/sa3-sm-music/dit_fp16.onnx`` (same fp16mixed recipe, same IO
  contract as sa3-m; a single proto, the ~0.9 GB of weights inline)
  compiled to ``sa3_sm_dit_l*`` engines. fp16mixed only: the fp8 and
  refit variants stay medium-only. Small decodes with SAME-S, so a small
  build never builds the SAME-L engine; it builds the SAME-S decoder.
* **SAME-S full decoder** (``--model small-music``, or ``--same-s-decode``):
  upstream publishes the SAME-S decoder only as an FP32 graph
  (``onnx/same-s/dec_bf16.onnx``; "bf16" names upstream's retired BF16
  builder-flag recipe, which crackles on long outputs, and is never used
  here). The vendored ``build_same_s_dec_fp16.py`` converts it to
  upstream's canonical fp16mixed graph (FP16 trunk, FP32 islands around
  the tanh norms, softmax, differential-attention Sub and RoPE), which is
  compiled STRONGLY_TYPED with no precision flags to
  ``same_s_decode_<recipe tag>_t{min}_{opt}_{max}``
  (:func:`acestep.engine.sa3_trt.same_s_decode_build_tag`). The converted
  graph is cached under ``<engines_dir>/_onnx/``.
* **SAME-L window decoder**: ``dec_fp16.onnx`` (upstream renamed it from
  ``dec_dynamic_triton_swa.onnx`` on 2026-08-28, same bytes),
  STRONGLY_TYPED; needs the ``samel::diff_attn_swa`` plugin registered
  before the ONNX parse (vendored tree, via
  :func:`acestep.engine.sa3_trt._register_same_plugin`).

Usage:
    # Canonical matrix (DiT latent profiles 324 + 646, SAME-L window):
    python -m acestep.engine.trt.sa3_build --all
    python -m acestep.engine.trt.sa3_build --all --dry-run
    python -m acestep.engine.trt.sa3_build --all --force-rebuild

    # small-music matrix (DiT profiles 324 + 646 + 1292, SAME-S decoder):
    python -m acestep.engine.trt.sa3_build --model small-music --all

    # SAME-S full decoder only (defaults t32_646_1292):
    python -m acestep.engine.trt.sa3_build --same-s-decode

    # Single DiT engine sized for a padded latent window:
    python -m acestep.engine.trt.sa3_build --dit --seconds 60
    python -m acestep.engine.trt.sa3_build --dit --opt-latents 324 --max-latents 324

    # SAME-L window decoder (defaults t32_56_96):
    python -m acestep.engine.trt.sa3_build --same-l-window

    # Canonical matrix plus the FP8 DiT variants (producer-built ONNX until
    # dit_fp8.onnx is published to HF):
    python -m acestep.engine.trt.sa3_build --all --fp8 \
        --fp8-onnx /path/to/dit_fp8.onnx

Requirements:
    - tensorrt (uv pip install tensorrt; version-gated by the shared
      preflight in acestep.engine.trt.build)
    - network access to huggingface.co on first build (ONNX cache after)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from ._engine_metadata import (
    expected_metadata as _expected_metadata,
    metadata_matches as _metadata_matches,
    write_metadata as _write_metadata,
)
from .build import _preflight, _save_build_report, _verify_engines

from acestep.engine.sa3_helpers import require_sa3_vendor
from acestep.engine.sa3_trt import (
    COND_DIM,
    IO_CHANNELS,
    SA3_SAMPLE_RATE,
    SAMPLES_PER_LATENT,
    T5_TOKENS,
    SAME_S_FP32_ISLANDS,
    SAME_S_UPSTREAM_DECODER_SHA256,
    _register_same_plugin,
    same_l_plugin_build_tag,
    same_s_decode_build_tag,
    trt_engines_dir,
)
from acestep.engine.sa3_helpers import sa3_vendor_dir

HF_REPO = "stabilityai/stable-audio-3-optimized"
# The medium DiT ONNX exceeds 2 GB, so the weights travel in an
# external-data sidecar next to the proto.
# Upstream renamed these `dit_fp16mixed.*` -> `dit_fp16.*` on 2026-08-02
# (HF commits 3f29675c "Rename fp16mixed->fp16 ... (copies; deletes follow
# after code merge)" then 48e34e2d, which deleted the old names 20 min
# later — so every SA3 bake since 404s on the old path). The weights
# sidecar is byte-identical across the rename; only the proto changed, and
# only where it names its sidecar. That still moves its sha256, which the
# engine metadata hashes, so the first build after this lands rebuilds the
# two DiT engines once (~5 min) for a functionally identical result. The
# SAME-L decoder ONNX is untouched and still skips.
DIT_ONNX_FILES = (
    "onnx/sa3-m/dit_fp16.onnx",
    "onnx/sa3-m/dit_fp16.onnx.data",
)
# FP8-trunk DiT (opt-in, ~1.8x/step). Upstream published `dit_fp8.onnx` in
# the 2026-08-02 sweep, but its sidecar landed as `dit_fp8lin.onnx.data`
# (sa3-sm-* got matching `dit_fp8.onnx.data` names) — so this pair still
# 404s on the second entry and --fp8 still needs a graph passed via
# --fp8-onnx. Switching the sidecar to the `fp8lin` name is a one-liner
# once someone builds and parity-checks an engine from it, and upstream's
# own consumer recipe (`build_from_onnx.py sa3-m-fp8`) confirms that pair
# is what HF now serves.
# Producing that graph locally is NOT a vendored-tree operation at the
# pinned revision: the ModelOpt PTQ builder (`build_dit_fp8.py`) lives in
# Stability PR #47 and is not merged into the pinned tree. What IS vendored
# is the rest of the chain — `make_calib.py` (real-conditioning capture),
# `build_dit_bf16.py` (shared RoPE-baker) and `transplant_scales.py` (grafts
# #47's calibrated scales onto the baked graph) — but the transplant still
# consumes a `dit_fp8_calib.onnx` that only #47 produces. See
# optimized/tensorRT/build/README.md in the vendored tree for the full flow.
# acestep.engine.sa3_trt does the runtime selection.
DIT_FP8_ONNX_FILES = (
    "onnx/sa3-m/dit_fp8.onnx",
    "onnx/sa3-m/dit_fp8.onnx.data",
)
# Upstream renamed this on 2026-08-28 ("Rename the autoencoder ONNX to match
# the engine naming scheme"); the bytes are unchanged (same LFS object as
# the old dec_dynamic_triton_swa.onnx), so engines keyed on the ONNX hash
# still match.
SAME_L_ONNX_FILES = ("onnx/same-l/dec_fp16.onnx",)
# SAME-S decoder: upstream's FP32 graph (the "bf16" suffix names the
# retired builder-flag recipe, not the graph's dtypes); converted locally
# to fp16mixed by the vendored build_same_s_dec_fp16.py before the build.
SAME_S_ONNX_FILES = ("onnx/same-s/dec_bf16.onnx",)
# sa3-sm-music DiT: the same fp16mixed recipe as sa3-m, published as a
# single proto (weights inline, under the 2 GB protobuf limit).
DIT_SMALL_ONNX_FILES = ("onnx/sa3-sm-music/dit_fp16.onnx",)

# Canonical DiT latent profiles for --all. min=1 keeps short windows
# on-engine; the names must keep the sa3_m_dit_l{min}_{opt}_{max} shape
# that acestep.engine.sa3_trt discovery matches. (1, 324, 324) covers
# the 24 s streaming session (30 s padded window; its range also covers
# the L=323 rounding variant), (1, 646, 646) covers the default 54 s
# session (60 s padded window). Upstream ships one (1, 1292, 4096)
# profile instead and notes TRT picks identical tactics across the
# range; we size engines per session shape to keep activation workspace
# small on streaming hosts.
CANONICAL_DIT_PROFILES: tuple[tuple[int, int, int], ...] = (
    (1, 324, 324),
    (1, 646, 646),
)
# small-music adds a full-range profile: its sessions run up to the 120 s
# window (SA3_MAX_DURATION_S; sample_size 5292032 / 4096 = 1292 frames),
# and without an engine covering that, the create path's TRT duration
# clamp would cut every >54 s small session down to the 646 engine.
# Medium never gets it (its sessions are clamped to 646 by design).
SMALL_EXTRA_DIT_PROFILES: tuple[tuple[int, int, int], ...] = ((1, 1292, 1292),)
# SAME-L windowed decode profile: DEMON decodes ~1 s windows with 2 s of
# context, so T stays inside [32, 96] with the steady state at 56.
CANONICAL_SAME_L_WINDOW: tuple[int, int, int] = (32, 56, 96)
# SAME-S full-decode profile: small decodes the whole canvas latent per
# fresh generation (no windowing), so the profile spans the session
# windows: opt 646 (the 54 s canvas class; production L = 614), max 1292
# (the 120 s small window), min 32 (upstream's floor).
CANONICAL_SAME_S_DECODE: tuple[int, int, int] = (32, 646, 1292)


def latents_for_seconds(seconds: float) -> int:
    """Padded-window seconds to latent frames (ceil, 4096 samples/latent)."""
    return max(1, int(math.ceil(seconds * SA3_SAMPLE_RATE / SAMPLES_PER_LATENT)))


@dataclass
class SA3DiTBuildConfig:
    """Build parameters for one sa3-m DiT engine (metadata identity)."""

    min_latents: int
    opt_latents: int
    max_latents: int
    workspace_gb: float = 16.0
    # list, not tuple: the config dict is JSON round-tripped by the
    # metadata skip gate, and JSON has no tuples.
    onnx_files: list[str] = field(default_factory=lambda: list(DIT_ONNX_FILES))

    def engine_name(self) -> str:
        return f"sa3_m_dit_l{self.min_latents}_{self.opt_latents}_{self.max_latents}"


@dataclass
class SA3SmallDiTBuildConfig:
    """Build parameters for one sa3-sm-music fp16mixed DiT engine.

    A separate dataclass (same rationale as the fp8 one): the metadata
    skip gate hashes the whole config, so a ``model`` field on
    :class:`SA3DiTBuildConfig` would change the medium engines' identity
    and force a needless rebuild. Same profile inputs, the ``sa3_sm_dit``
    name prefix :data:`acestep.engine.sa3_trt.DIT_ENGINE_PREFIX` maps
    ``small-music`` to, and the small ONNX.
    """

    min_latents: int
    opt_latents: int
    max_latents: int
    workspace_gb: float = 16.0
    onnx_files: list[str] = field(default_factory=lambda: list(DIT_SMALL_ONNX_FILES))

    def engine_name(self) -> str:
        return f"sa3_sm_dit_l{self.min_latents}_{self.opt_latents}_{self.max_latents}"


# Per-model fp16mixed DiT build: (config class, metadata component, label).
DIT_MODELS: dict[str, tuple[type, str, str]] = {
    "medium": (SA3DiTBuildConfig, "sa3_m_dit", "SA3-M"),
    "small-music": (SA3SmallDiTBuildConfig, "sa3_sm_dit", "SA3-SM"),
}


@dataclass
class SA3DiTFp8BuildConfig:
    """Build parameters for one sa3-m FP8-trunk DiT engine.

    A separate dataclass from :class:`SA3DiTBuildConfig` on purpose: the
    metadata skip gate hashes the whole config, so folding precision into one
    class would change the fp16mixed engines' identity and force a needless
    rebuild. Same profile inputs, ``_fp8`` engine name, fp8 ONNX files.
    """

    min_latents: int
    opt_latents: int
    max_latents: int
    workspace_gb: float = 16.0
    onnx_files: list[str] = field(default_factory=lambda: list(DIT_FP8_ONNX_FILES))

    def engine_name(self) -> str:
        return (
            f"sa3_m_dit_fp8_l{self.min_latents}"
            f"_{self.opt_latents}_{self.max_latents}"
        )


@dataclass
class SA3DiTRefitBuildConfig:
    """Build parameters for one REFITTABLE sa3-m DiT engine
    (notes/SA3_LORA_PLAN.md D6b: the LoRA refit endgame).

    A separate dataclass (same rationale as the fp8 one: the metadata
    skip gate hashes the whole config, so folding a ``refit`` field into
    :class:`SA3DiTBuildConfig` would change the existing engines'
    identity and force a needless rebuild). Same fp16mixed ONNX, the
    ``BuilderFlag.REFIT`` builder flag, and the ``_refit`` name marker
    that :func:`acestep.engine.sa3_trt.is_refittable_engine_path` (and
    the exclusive-ownership deserialization policy) keys on.
    """

    min_latents: int
    opt_latents: int
    max_latents: int
    workspace_gb: float = 16.0
    onnx_files: list[str] = field(default_factory=lambda: list(DIT_ONNX_FILES))

    def engine_name(self) -> str:
        return (
            f"sa3_m_dit_refit_l{self.min_latents}"
            f"_{self.opt_latents}_{self.max_latents}"
        )


@dataclass
class SameLWindowBuildConfig:
    """Build parameters for the SAME-L window decoder engine."""

    min_latents: int
    opt_latents: int
    max_latents: int
    workspace_gb: float = 16.0
    onnx_files: list[str] = field(default_factory=lambda: list(SAME_L_ONNX_FILES))
    plugin_build_tag: str = field(default_factory=same_l_plugin_build_tag)

    def engine_name(self) -> str:
        return (
            f"same_l_decode_window_{self.plugin_build_tag}_t{self.min_latents}"
            f"_{self.opt_latents}_{self.max_latents}"
        )


@dataclass
class SameSDecodeBuildConfig:
    """Build parameters for the SAME-S full decoder engine."""

    min_latents: int
    opt_latents: int
    max_latents: int
    workspace_gb: float = 16.0
    onnx_files: list[str] = field(default_factory=lambda: list(SAME_S_ONNX_FILES))
    recipe_tag: str = field(default_factory=same_s_decode_build_tag)
    # The codec weights upstream's graph carries (selection key; see
    # acestep.engine.sa3_trt._SAME_S_DIR_RE).
    weights_sha256: str = SAME_S_UPSTREAM_DECODER_SHA256

    def engine_name(self) -> str:
        return (
            f"same_s_decode_{self.recipe_tag}_w{self.weights_sha256[:12]}"
            f"_t{self.min_latents}_{self.opt_latents}_{self.max_latents}"
        )


def _fetch_onnx(rel_paths: list[str] | tuple[str, ...]) -> str:
    """Fetch the ONNX files from HF (cached); return the proto path.

    The returned path is what gets hashed into the engine metadata. For
    the DiT that is the proto only; the weights live in the external
    ``.data`` sidecar, but any upstream re-export rewrites the proto's
    external-data offsets too, so the proto hash tracks recipe identity.
    """
    from huggingface_hub import hf_hub_download

    local_paths = []
    for rel in rel_paths:
        logger.info("ONNX fetch (cached after first use): {}/{}", HF_REPO, rel)
        local_paths.append(hf_hub_download(repo_id=HF_REPO, filename=rel))
    return local_paths[0]


def _engine_survives_missing_onnx(
    *,
    engine_path: str,
    component: str,
    config,
    env: dict,
    force_rebuild: bool,
) -> str | None:
    """Reason the engine on disk stands in for an unfetchable ONNX, else None.

    The ONNX is fetched on every run, including runs with nothing to
    build, purely to hash it into the freshness check. That makes an
    upstream artifact rename fatal to a bake that had zero engines to
    build — which is exactly what Stability's 2026-08-02
    ``dit_fp16mixed`` -> ``dit_fp16`` rename did to bake-warm #87. An
    engine already built and current in every other respect doesn't need
    the graph it came from, so a failed fetch is a reason to keep it, not
    to fail the run.

    Deliberately narrow: TRT version, compute capability and build config
    are still compared, so a genuinely stale engine still demands the
    ONNX and still surfaces the fetch error. What this cannot see is an
    upstream *re-export* published under a new name — the engine would be
    silently kept — hence the WARNING the caller logs.
    """
    if force_rebuild or not os.path.exists(engine_path):
        return None
    expected = _expected_metadata(
        component=component, onnx_path=None, config=config, env=env,
    )
    matches, reason = _metadata_matches(
        engine_path, expected, ignore=("onnx_sha256",),
    )
    if not matches:
        return None
    size_mb = os.path.getsize(engine_path) / 1e6
    return (
        f"{os.path.basename(engine_path)} ({size_mb:.0f} MB, {reason} "
        "apart from the graph hash)"
    )


def _build_strongly_typed_engine(
    *,
    onnx_path: str,
    engine_path: str,
    workspace_gb: float,
    profile_shapes: dict[str, tuple[tuple, tuple, tuple]],
    refit: bool = False,
    python_plugin_preference: str | None = None,
) -> None:
    """Parse + build one STRONGLY_TYPED engine and serialize it to disk.

    Shared by both SA3 engine kinds: the fp16mixed ONNX graphs carry
    per-tensor dtypes (the FP32 islands), so the network must be
    STRONGLY_TYPED with no builder precision flags for TRT to honor
    them instead of auto-promoting.
    """
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    if python_plugin_preference == "aot":
        network_flags |= 1 << int(
            trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS
        )
    elif python_plugin_preference == "jit":
        network_flags |= 1 << int(
            trt.NetworkDefinitionCreationFlag.PREFER_JIT_PYTHON_PLUGINS
        )
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: {}", parser.get_error(i))
        raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)),
    )
    if refit:
        config.set_flag(trt.BuilderFlag.REFIT)
    profile = builder.create_optimization_profile()
    for input_name, (lo, opt, hi) in profile_shapes.items():
        profile.set_shape(input_name, lo, opt, hi)
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("Failed to add optimization profile")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build failed")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)


def _build_dit_engine(
    *,
    output_dir: str,
    config,
    env: dict,
    force_rebuild: bool = False,
    component: str = "sa3_m_dit",
    precision_label: str = "fp16mixed",
    local_onnx: str | None = None,
    refit: bool = False,
    model_label: str = "SA3-M",
) -> tuple[str, str, float, str]:
    """Build one sa3-m DiT engine. Returns (label, path, elapsed, status).

    ``config`` is an :class:`SA3DiTBuildConfig` (fp16mixed) or
    :class:`SA3DiTFp8BuildConfig` (fp8); they are duck-compatible. ``local_onnx``
    compiles a producer-built ONNX from disk (its ``.onnx.data`` sidecar must
    sit alongside) instead of fetching from HF, used for fp8 before the artifact
    is published."""
    name = config.engine_name()
    engine_path = os.path.join(output_dir, name, f"{name}.trt")
    label = (
        f"{model_label} DiT {precision_label} "
        f"l{config.min_latents}_{config.opt_latents}_{config.max_latents}"
        f" (~{config.max_latents * SAMPLES_PER_LATENT / SA3_SAMPLE_RATE:.0f}s window)"
    )

    if local_onnx:
        onnx_path = local_onnx
        logger.info("Using local ONNX: {}", onnx_path)
    else:
        try:
            onnx_path = _fetch_onnx(config.onnx_files)
        except Exception as exc:
            kept = _engine_survives_missing_onnx(
                engine_path=engine_path, component=component, config=config,
                env=env, force_rebuild=force_rebuild,
            )
            if kept:
                logger.warning(
                    "ONNX fetch failed ({}) but nothing needed building — "
                    "keeping {}. If upstream re-exported the graph under a "
                    "new name, update the *_ONNX_FILES paths and rebuild.",
                    exc, kept,
                )
                return (label, engine_path, 0.0, "SKIPPED")
            if precision_label == "fp8":
                raise RuntimeError(
                    f"dit_fp8.onnx is not fetchable from HF ({HF_REPO}) under "
                    "the names this builder expects — upstream's sidecar is "
                    "published as dit_fp8lin.onnx.data (see "
                    "DIT_FP8_ONNX_FILES). Pass an already-built graph with "
                    "--fp8-onnx <dit_fp8.onnx>. It cannot be produced from "
                    "the vendored tree alone at the pinned revision: the "
                    "ModelOpt PTQ builder (build_dit_fp8.py) lives in "
                    "Stability PR #47, not in the pin, and the vendored "
                    "transplant_scales.py needs that builder's "
                    "dit_fp8_calib.onnx. See <MODELS_DIR>/sa3/vendor/"
                    "stable-audio-3/optimized/tensorRT/build/README.md "
                    "('Medium fp8')."
                ) from exc
            raise
    expected = _expected_metadata(
        component=component, onnx_path=onnx_path, config=config, env=env,
    )

    if not force_rebuild and os.path.exists(engine_path):
        matches, reason = _metadata_matches(engine_path, expected)
        if matches:
            size_mb = os.path.getsize(engine_path) / 1e6
            logger.info("SKIP {} ({:.0f} MB, {})", name, size_mb, reason)
            return (label, engine_path, 0.0, "SKIPPED")
        logger.info("REBUILD {} ({})", name, reason)

    logger.info("=" * 60)
    logger.info(
        "SA3 DiT TRT BUILD: {} ({}, STRONGLY_TYPED, workspace {:.0f} GB)",
        name, precision_label, config.workspace_gb,
    )
    logger.info("=" * 60)

    lo, opt, hi = config.min_latents, config.opt_latents, config.max_latents
    t0 = time.time()
    _build_strongly_typed_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        workspace_gb=config.workspace_gb,
        profile_shapes={
            "x": ((1, IO_CHANNELS, lo), (1, IO_CHANNELS, opt), (1, IO_CHANNELS, hi)),
            "t": ((1,), (1,), (1,)),
            "t5_hidden": ((1, T5_TOKENS, COND_DIM),) * 3,
            "t5_mask": ((1, T5_TOKENS),) * 3,
            "seconds_total": ((1,), (1,), (1,)),
            "local_add_cond": ((1, 257, lo), (1, 257, opt), (1, 257, hi)),
        },
        refit=refit,
    )
    _write_metadata(engine_path=engine_path, expected=expected, env=env)
    elapsed = time.time() - t0
    logger.info("Built in {:.0f}s", elapsed)
    return (label, engine_path, elapsed, "OK")


def _build_same_l_window_engine(
    *,
    output_dir: str,
    config: SameLWindowBuildConfig,
    env: dict,
    force_rebuild: bool = False,
) -> tuple[str, str, float, str]:
    """Build the SAME-L window decoder. Returns (label, path, elapsed, status)."""
    name = config.engine_name()
    engine_path = os.path.join(output_dir, name, f"{name}.trt")
    label = (
        f"SAME-L window decoder t{config.min_latents}"
        f"_{config.opt_latents}_{config.max_latents}"
    )

    try:
        onnx_path = _fetch_onnx(config.onnx_files)
    except Exception as exc:
        kept = _engine_survives_missing_onnx(
            engine_path=engine_path, component="same_l_decode_window",
            config=config, env=env, force_rebuild=force_rebuild,
        )
        if not kept:
            raise
        logger.warning(
            "ONNX fetch failed ({}) but nothing needed building — keeping "
            "{}. If upstream re-exported the graph under a new name, update "
            "the *_ONNX_FILES paths and rebuild.",
            exc, kept,
        )
        return (label, engine_path, 0.0, "SKIPPED")
    expected = _expected_metadata(
        component="same_l_decode_window", onnx_path=onnx_path, config=config, env=env,
    )

    if not force_rebuild and os.path.exists(engine_path):
        matches, reason = _metadata_matches(engine_path, expected)
        if matches:
            size_mb = os.path.getsize(engine_path) / 1e6
            logger.info("SKIP {} ({:.0f} MB, {})", name, size_mb, reason)
            return (label, engine_path, 0.0, "SKIPPED")
        logger.info("REBUILD {} ({})", name, reason)

    logger.info("=" * 60)
    logger.info(
        "SAME-L TRT BUILD: {} (Triton SWA plugin, STRONGLY_TYPED, "
        "workspace {:.0f} GB)",
        name, config.workspace_gb,
    )
    logger.info("=" * 60)

    # The plugin must be registered before the ONNX parse or TRT can't
    # resolve the samel::diff_attn_swa node.
    _register_same_plugin()

    lo, opt, hi = config.min_latents, config.opt_latents, config.max_latents
    t0 = time.time()
    _build_strongly_typed_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        workspace_gb=config.workspace_gb,
        profile_shapes={
            "latent": ((1, IO_CHANNELS, lo), (1, IO_CHANNELS, opt), (1, IO_CHANNELS, hi)),
        },
        python_plugin_preference=(
            "jit" if config.plugin_build_tag.startswith("jit_") else "aot"
        ),
    )
    _write_metadata(engine_path=engine_path, expected=expected, env=env)
    elapsed = time.time() - t0
    logger.info("Built in {:.0f}s", elapsed)
    return (label, engine_path, elapsed, "OK")


def _convert_same_s_fp16mixed(source_onnx: str, output_dir: str, tag: str) -> str:
    """Upstream FP32 SAME-S decoder ONNX -> fp16mixed ONNX via the vendored
    converter (``build_same_s_dec_fp16.convert_to_fp16``), cached under
    ``<output_dir>/_onnx/``. Returns the converted graph's path."""
    import importlib.util

    out = Path(output_dir) / "_onnx" / f"same_s_dec_{tag}.onnx"
    if out.is_file():
        logger.info("SAME-S fp16mixed ONNX cached: {}", out)
        return str(out)
    build_dir = sa3_vendor_dir() / "optimized" / "tensorRT" / "build"
    script = build_dir / "build_same_s_dec_fp16.py"
    if not script.is_file():
        raise ImportError(
            f"SAME-S fp16 converter not found at {script}; the vendored "
            "stable_audio_3 tree must include optimized/tensorRT/build. "
            "Run `uv run demon-setup` to fetch the pinned SA3 source."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    before = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("_sa3_same_s_dec_fp16", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # inserts its own dir for build_dit_fp16
        tmp = out.with_name(out.stem + ".partial.onnx")
        module.convert_to_fp16(source_onnx, str(tmp), mode=SAME_S_FP32_ISLANDS)
    finally:
        sys.path[:] = before
    import onnx

    model = onnx.load(str(tmp))
    _pin_same_s_pcm_tail_fp32(model)
    onnx.save(model, str(tmp))
    os.replace(tmp, out)
    return str(out)


def _pin_same_s_pcm_tail_fp32(model) -> None:
    """Run the SAME-S PCM tail in FP32 (in place on the converted graph).

    Upstream's graph ends ``Cast(FP32) -> Clip(-1, 1) -> Mul(32767) ->
    Clip(+-32767) -> Cast(INT32) -> Transpose -> pcm``. The converter
    strips the leading FP32 Cast as a no-op and retargets only the first
    Clip's bounds to FP16, so the second Clip mixes FP16 data with FP32
    bounds (TensorRT refuses to parse it) and the x32767 product would be
    FP16 (11-bit mantissa: steps of 16 near full scale). Re-insert the
    FP32 Cast before the first Clip and keep every tail constant FP32.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    by_out = {o: n for n in graph.node for o in n.output}
    transpose = by_out.get("pcm")
    to_int = by_out.get(transpose.input[0]) if transpose is not None else None
    clip_pcm = by_out.get(to_int.input[0]) if to_int is not None else None
    mul = by_out.get(clip_pcm.input[0]) if clip_pcm is not None else None
    clip_unit = by_out.get(mul.input[0]) if mul is not None else None
    chain = (transpose, to_int, clip_pcm, mul, clip_unit)
    ops = tuple(n.op_type if n is not None else None for n in chain)
    if ops != ("Transpose", "Cast", "Clip", "Mul", "Clip"):
        raise RuntimeError(f"unexpected SAME-S PCM tail {ops}; upstream graph changed")

    def _as_fp32(name: str) -> None:
        node = by_out.get(name)
        if node is None:
            for init in graph.initializer:
                if init.name == name and init.data_type != TensorProto.FLOAT:
                    arr = numpy_helper.to_array(init).astype(np.float32)
                    init.CopyFrom(numpy_helper.from_array(arr, name))
            return
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    attr.i = TensorProto.FLOAT
        elif node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type != TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))

    for clip in (clip_unit, clip_pcm):
        for bound in clip.input[1:]:
            _as_fp32(bound)
    _as_fp32(mul.input[1])
    src = clip_unit.input[0]
    cast_name = "/demon_pcm_tail_fp32"
    clip_unit.input[0] = cast_name + "_output_0"
    cast = helper.make_node(
        "Cast", [src], [cast_name + "_output_0"], name=cast_name, to=TensorProto.FLOAT,
    )
    from onnx import NodeProto

    idx = next(i for i, n in enumerate(graph.node) if n.name == clip_unit.name)
    tail = []
    for node in graph.node[idx:]:
        copy = NodeProto()
        copy.CopyFrom(node)
        tail.append(copy)
    del graph.node[idx:]
    graph.node.append(cast)
    graph.node.extend(tail)


def _build_same_s_decode_engine(
    *,
    output_dir: str,
    config: SameSDecodeBuildConfig,
    env: dict,
    force_rebuild: bool = False,
) -> tuple[str, str, float, str]:
    """Build the SAME-S full decoder. Returns (label, path, elapsed, status)."""
    name = config.engine_name()
    engine_path = os.path.join(output_dir, name, f"{name}.trt")
    label = (
        f"SAME-S decoder fp16mixed t{config.min_latents}"
        f"_{config.opt_latents}_{config.max_latents}"
    )
    try:
        onnx_path = _fetch_onnx(config.onnx_files)
    except Exception as exc:
        kept = _engine_survives_missing_onnx(
            engine_path=engine_path, component="same_s_decode",
            config=config, env=env, force_rebuild=force_rebuild,
        )
        if not kept:
            raise
        logger.warning(
            "ONNX fetch failed ({}) but nothing needed building — keeping "
            "{}. If upstream re-exported the graph under a new name, update "
            "the *_ONNX_FILES paths and rebuild.",
            exc, kept,
        )
        return (label, engine_path, 0.0, "SKIPPED")
    # Identity = the upstream source graph + the recipe tag in the config
    # (converter revision + island mode); the converted graph is derived.
    expected = _expected_metadata(
        component="same_s_decode", onnx_path=onnx_path, config=config, env=env,
    )
    if not force_rebuild and os.path.exists(engine_path):
        matches, reason = _metadata_matches(engine_path, expected)
        if matches:
            size_mb = os.path.getsize(engine_path) / 1e6
            logger.info("SKIP {} ({:.0f} MB, {})", name, size_mb, reason)
            return (label, engine_path, 0.0, "SKIPPED")
        logger.info("REBUILD {} ({})", name, reason)

    logger.info("=" * 60)
    logger.info(
        "SAME-S TRT BUILD: {} (fp16mixed, STRONGLY_TYPED, workspace {:.0f} GB)",
        name, config.workspace_gb,
    )
    logger.info("=" * 60)
    t0 = time.time()
    fp16_onnx = _convert_same_s_fp16mixed(onnx_path, output_dir, config.recipe_tag)
    lo, opt, hi = config.min_latents, config.opt_latents, config.max_latents
    _build_strongly_typed_engine(
        onnx_path=fp16_onnx,
        engine_path=engine_path,
        workspace_gb=config.workspace_gb,
        profile_shapes={
            "latent": ((1, IO_CHANNELS, lo), (1, IO_CHANNELS, opt), (1, IO_CHANNELS, hi)),
        },
    )
    _write_metadata(engine_path=engine_path, expected=expected, env=env)
    elapsed = time.time() - t0
    logger.info("Built in {:.0f}s", elapsed)
    return (label, engine_path, elapsed, "OK")


# ------------------------------------------------------------------
# Batch mode (--all)
# ------------------------------------------------------------------


def _resolve_dit_profiles(args) -> tuple:
    """DiT ``(lo, opt, hi)`` profiles for this invocation: per-duration
    when ``--duration`` is given, else the canonical set. Single source
    for both the ``--all`` matrix preview (``_matrix_jobs``) and the
    actual build loop in ``main`` so the dry-run can't lie about what
    ``--all`` will build."""
    if args.duration:
        return tuple(
            (1, latents_for_seconds(s), latents_for_seconds(s))
            for s in args.duration
        )
    if getattr(args, "model", "medium") == "small-music":
        return CANONICAL_DIT_PROFILES + SMALL_EXTRA_DIT_PROFILES
    return CANONICAL_DIT_PROFILES


def _matrix_jobs(args) -> list[tuple[str, str]]:
    """(label, engine_dir_name) pairs for the --all matrix."""
    dit_profiles = _resolve_dit_profiles(args)

    cfg_cls, _component, model_label = DIT_MODELS[args.model]
    jobs = []
    if not args.same_l_only:
        for lo, opt, hi in dit_profiles:
            cfg = cfg_cls(lo, opt, hi)
            jobs.append((
                f"{model_label} DiT fp16mixed l{lo}_{opt}_{hi}"
                f" (~{hi * SAMPLES_PER_LATENT / SA3_SAMPLE_RATE:.0f}s window)",
                cfg.engine_name(),
            ))
        if args.fp8:
            for lo, opt, hi in dit_profiles:
                cfg = SA3DiTFp8BuildConfig(lo, opt, hi)
                jobs.append((
                    f"SA3-M DiT fp8 l{lo}_{opt}_{hi}"
                    f" (~{hi * SAMPLES_PER_LATENT / SA3_SAMPLE_RATE:.0f}s window)",
                    cfg.engine_name(),
                ))
        if args.refit:
            for lo, opt, hi in dit_profiles:
                cfg = SA3DiTRefitBuildConfig(lo, opt, hi)
                jobs.append((
                    f"SA3-M DiT refit l{lo}_{opt}_{hi}"
                    f" (~{hi * SAMPLES_PER_LATENT / SA3_SAMPLE_RATE:.0f}s window)",
                    cfg.engine_name(),
                ))
    if not args.dit_only:
        if args.model == "medium":
            lo, opt, hi = CANONICAL_SAME_L_WINDOW
            cfg = SameLWindowBuildConfig(lo, opt, hi)
            jobs.append((f"SAME-L window decoder t{lo}_{opt}_{hi}", cfg.engine_name()))
        else:
            lo, opt, hi = CANONICAL_SAME_S_DECODE
            cfg = SameSDecodeBuildConfig(lo, opt, hi)
            jobs.append((f"SAME-S decoder fp16mixed t{lo}_{opt}_{hi}", cfg.engine_name()))
    return jobs


def _print_matrix(jobs: list[tuple[str, str]], output_dir: str) -> None:
    to_build = to_skip = 0
    lines = []
    for label, dir_name in jobs:
        engine_file = os.path.join(output_dir, dir_name, f"{dir_name}.trt")
        if os.path.exists(engine_file):
            size_mb = os.path.getsize(engine_file) / 1e6
            lines.append(f"  [exists]  {label}  ({size_mb:.0f} MB)")
            to_skip += 1
        else:
            lines.append(f"  [build]   {label}")
            to_build += 1
    print(f"\nSA3 build matrix: {to_build} to build, {to_skip} existing")
    for line in lines:
        print(line)
    print()


def _print_summary(results, output_dir: str) -> int:
    print(f"\n{'=' * 60}")
    print("SA3 BUILD SUMMARY")
    print(f"{'=' * 60}")
    for label, path, elapsed, status in results:
        print(f"  {status:7s} {elapsed:6.0f}s  {label}")

    failures = sum(1 for _, _, _, s in results if s == "FAILED")
    if failures:
        print(f"\n{failures} build(s) FAILED")
    else:
        active = sum(1 for _, _, _, s in results if s != "SKIPPED")
        skipped = sum(1 for _, _, _, s in results if s == "SKIPPED")
        parts = [f"{active} built"]
        if skipped:
            parts.append(f"{skipped} skipped")
        print(f"\nAll done ({', '.join(parts)}).")

    trt_dir = Path(output_dir)
    if trt_dir.is_dir():
        print(f"\nEngines in {trt_dir}:")
        for d in sorted(trt_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            engine_file = d / f"{d.name}.trt"
            if engine_file.exists():
                size_mb = engine_file.stat().st_size / 1e6
                print(f"  {d.name + '/':50s} {size_mb:8.1f} MB")
    return failures


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build SA3 (Stable Audio 3) TRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    batch = parser.add_argument_group("batch mode (--all)")
    batch.add_argument("--all", action="store_true",
                       help="Build the canonical SA3 engine matrix "
                            "(DiT latent profiles + SAME-L window decoder)")
    batch.add_argument("--duration", nargs="*", type=float, default=None,
                       help="Padded-window duration(s) in seconds for --all "
                            "DiT engines (default: the canonical latent "
                            "profiles 324 and 646)")
    batch.add_argument("--force-rebuild", "--force", action="store_true",
                       help="Rebuild engines even when the metadata sidecar "
                            "matches (default: skip up-to-date engines)")
    batch.add_argument("--dry-run", action="store_true",
                       help="Print the build matrix without building")
    batch.add_argument("--dit-only", action="store_true",
                       help="Only build DiT engines (skip SAME-L)")
    batch.add_argument("--same-l-only", action="store_true",
                       help="Only build the SAME-L window decoder (skip DiT)")
    batch.add_argument("--same-s-only", action="store_true",
                       help="small-music: only build the SAME-S decoder (skip DiT)")

    single = parser.add_argument_group("single mode / shared options")
    single.add_argument("--model", choices=sorted(DIT_MODELS), default="medium",
                        help="Which SA3 checkpoint's DiT to build "
                             "(default: medium). small-music builds only "
                             "the fp16mixed DiT and the SAME-S decoder: no "
                             "fp8/refit variants, no SAME-L.")
    single.add_argument("--output-dir", default=str(trt_engines_dir()),
                        help="Engine output directory "
                             "(default: <models>/sa3/trt_engines)")
    single.add_argument("--dit", action="store_true",
                        help="Build one DiT engine (size via --seconds or "
                             "the latent flags)")
    single.add_argument("--same-l-window", action="store_true",
                        help="Build the SAME-L window decoder (size via the "
                             "latent flags; defaults t32_56_96)")
    single.add_argument("--same-s-decode", action="store_true",
                        help="Build the SAME-S full decoder (size via the "
                             "latent flags; defaults t32_646_1292)")
    single.add_argument("--seconds", type=float, default=60.0,
                        help="Padded-window seconds for a single --dit build "
                             "(default: 60 = the 54s session + 6s padding)")
    single.add_argument("--min-latents", type=int, default=None)
    single.add_argument("--opt-latents", type=int, default=None)
    single.add_argument("--max-latents", type=int, default=None)
    single.add_argument("--workspace-gb", type=float, default=16.0,
                        help="TRT builder workspace in GB (default: 16)")
    single.add_argument("--fp8", action="store_true",
                        help="Also build the FP8-trunk DiT variant(s) "
                             "(~1.8x/step; preferred at runtime when present, "
                             "fp16mixed fallback). Needs the published "
                             "dit_fp8.onnx or --fp8-onnx.")
    single.add_argument("--refit", action="store_true",
                        help="Also build the REFITTABLE fp16mixed DiT "
                             "variant(s) (BuilderFlag.REFIT; the LoRA "
                             "in-place-refit engines, preferred by "
                             "LoRA-enabled sessions and exclusively owned "
                             "at runtime — never process-cached).")
    single.add_argument("--fp8-onnx", default=None,
                        help="Path to a producer-built dit_fp8.onnx (with its "
                             ".onnx.data sidecar alongside) to compile instead "
                             "of fetching from HF; implies --fp8.")

    args = parser.parse_args()
    if args.fp8_onnx:
        args.fp8 = True
    if not (args.all or args.dit or args.same_l_window or args.same_s_decode):
        parser.error("nothing to build: pass --all, --dit, --same-l-window "
                     "or --same-s-decode")
    if sum(map(bool, (args.dit, args.same_l_window, args.same_s_decode))) > 1:
        parser.error("--dit, --same-l-window and --same-s-decode share the "
                     "latent flags; build them in separate invocations or "
                     "use --all")
    if args.dit_only and (args.same_l_only or args.same_s_only):
        parser.error("--dit-only and --same-l-only/--same-s-only are mutually exclusive")
    if args.model != "medium":
        if args.fp8 or args.refit:
            parser.error(f"--fp8/--refit are medium-only (got --model {args.model})")
        if args.same_l_window or args.same_l_only:
            parser.error(f"SAME-L is the medium codec; --model {args.model} "
                         "decodes with SAME-S and has no window engine")
    elif args.same_s_only:
        parser.error("--same-s-only needs --model small-music (medium decodes "
                     "with SAME-L)")
    # The decoder-only switches share one meaning in the matrix: skip DiT.
    args.same_l_only = args.same_l_only or args.same_s_only
    dit_cls, dit_component, dit_label = DIT_MODELS[args.model]

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all:
        jobs = _matrix_jobs(args)
        _print_matrix(jobs, args.output_dir)
        if args.dry_run:
            return 0
        # SAME-L parse needs the vendored Triton plugin; fail fast with the
        # actionable remedy BEFORE the (minutes-long) DiT builds rather than
        # after, when the matrix includes a SAME-L job.
        if not args.dit_only:
            require_sa3_vendor()
        env = _preflight("cuda")
        results = []
        dit_profiles = _resolve_dit_profiles(args)
        if not args.same_l_only:
            for lo, opt, hi in dit_profiles:
                results.append(_build_dit_engine(
                    output_dir=args.output_dir,
                    config=dit_cls(lo, opt, hi, workspace_gb=args.workspace_gb),
                    env=env,
                    force_rebuild=args.force_rebuild,
                    component=dit_component,
                    model_label=dit_label,
                ))
            if args.fp8:
                for lo, opt, hi in dit_profiles:
                    results.append(_build_dit_engine(
                        output_dir=args.output_dir,
                        config=SA3DiTFp8BuildConfig(
                            lo, opt, hi, workspace_gb=args.workspace_gb),
                        env=env,
                        force_rebuild=args.force_rebuild,
                        component="sa3_m_dit_fp8",
                        precision_label="fp8",
                        local_onnx=args.fp8_onnx,
                    ))
            if args.refit:
                for lo, opt, hi in dit_profiles:
                    results.append(_build_dit_engine(
                        output_dir=args.output_dir,
                        config=SA3DiTRefitBuildConfig(
                            lo, opt, hi, workspace_gb=args.workspace_gb),
                        env=env,
                        force_rebuild=args.force_rebuild,
                        component="sa3_m_dit_refit",
                        precision_label="fp16mixed refit",
                        refit=True,
                    ))
        if not args.dit_only and args.model == "medium":
            lo, opt, hi = CANONICAL_SAME_L_WINDOW
            results.append(_build_same_l_window_engine(
                output_dir=args.output_dir,
                config=SameLWindowBuildConfig(lo, opt, hi, workspace_gb=args.workspace_gb),
                env=env,
                force_rebuild=args.force_rebuild,
            ))
        elif not args.dit_only:
            lo, opt, hi = CANONICAL_SAME_S_DECODE
            results.append(_build_same_s_decode_engine(
                output_dir=args.output_dir,
                config=SameSDecodeBuildConfig(lo, opt, hi, workspace_gb=args.workspace_gb),
                env=env,
                force_rebuild=args.force_rebuild,
            ))
        failures = _print_summary(results, args.output_dir)
        _save_build_report(results, args.output_dir)
        return 1 if failures else 0

    # Single mode
    if args.same_l_window or args.same_s_decode:
        # Same vendor-tree requirement as the --all path; fail fast.
        require_sa3_vendor()
    env = _preflight("cuda")
    built = []
    if args.dit:
        profile_l = latents_for_seconds(args.seconds)
        config = dit_cls(
            min_latents=args.min_latents or 1,
            opt_latents=args.opt_latents or profile_l,
            max_latents=args.max_latents or args.opt_latents or profile_l,
            workspace_gb=args.workspace_gb,
        )
        if not (0 < config.min_latents <= config.opt_latents <= config.max_latents):
            parser.error("require 0 < min <= opt <= max latent frames")
        result = _build_dit_engine(
            output_dir=args.output_dir, config=config, env=env,
            force_rebuild=args.force_rebuild,
            component=dit_component, model_label=dit_label,
        )
        built.append(result)
        if args.fp8:
            fp8_cfg = SA3DiTFp8BuildConfig(
                min_latents=config.min_latents,
                opt_latents=config.opt_latents,
                max_latents=config.max_latents,
                workspace_gb=args.workspace_gb,
            )
            built.append(_build_dit_engine(
                output_dir=args.output_dir, config=fp8_cfg, env=env,
                force_rebuild=args.force_rebuild,
                component="sa3_m_dit_fp8", precision_label="fp8",
                local_onnx=args.fp8_onnx,
            ))
        if args.refit:
            refit_cfg = SA3DiTRefitBuildConfig(
                min_latents=config.min_latents,
                opt_latents=config.opt_latents,
                max_latents=config.max_latents,
                workspace_gb=args.workspace_gb,
            )
            built.append(_build_dit_engine(
                output_dir=args.output_dir, config=refit_cfg, env=env,
                force_rebuild=args.force_rebuild,
                component="sa3_m_dit_refit",
                precision_label="fp16mixed refit",
                refit=True,
            ))
    if args.same_l_window:
        d_lo, d_opt, d_hi = CANONICAL_SAME_L_WINDOW
        config = SameLWindowBuildConfig(
            min_latents=args.min_latents or d_lo,
            opt_latents=args.opt_latents or d_opt,
            max_latents=args.max_latents or d_hi,
            workspace_gb=args.workspace_gb,
        )
        if not (0 < config.min_latents <= config.opt_latents <= config.max_latents):
            parser.error("require 0 < min <= opt <= max latent frames")
        result = _build_same_l_window_engine(
            output_dir=args.output_dir, config=config, env=env,
            force_rebuild=args.force_rebuild,
        )
        built.append(result)
    if args.same_s_decode:
        d_lo, d_opt, d_hi = CANONICAL_SAME_S_DECODE
        config = SameSDecodeBuildConfig(
            min_latents=args.min_latents or d_lo,
            opt_latents=args.opt_latents or d_opt,
            max_latents=args.max_latents or d_hi,
            workspace_gb=args.workspace_gb,
        )
        if not (0 < config.min_latents <= config.opt_latents <= config.max_latents):
            parser.error("require 0 < min <= opt <= max latent frames")
        built.append(_build_same_s_decode_engine(
            output_dir=args.output_dir, config=config, env=env,
            force_rebuild=args.force_rebuild,
        ))

    fresh = [(label, path) for label, path, _, status in built if status == "OK"]
    if fresh:
        logger.info("=" * 60)
        logger.info("VERIFICATION")
        logger.info("=" * 60)
        _verify_engines(fresh)
    _print_summary(built, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
