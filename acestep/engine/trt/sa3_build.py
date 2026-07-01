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

* **sa3-m DiT**: the pre-surgered FP16-mixed graph (``dit_fp16mixed.onnx``,
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
  ``--fp8`` (additive to the fp16mixed DiT). The ONNX is not yet on HF; pass
  ``--fp8-onnx`` a producer-built graph until it is.
  :func:`acestep.engine.sa3_trt.find_dit_engine` prefers an fp8 engine when one
  covers the window, else fp16mixed.
* **SAME-L window decoder**: ``dec_dynamic_triton_swa.onnx``,
  STRONGLY_TYPED; needs the ``samel::diff_attn_swa`` plugin registered
  before the ONNX parse (vendored tree, via
  :func:`acestep.engine.sa3_trt._register_same_plugin`).

Usage:
    # Canonical matrix (DiT latent profiles 324 + 646, SAME-L window):
    python -m acestep.engine.trt.sa3_build --all
    python -m acestep.engine.trt.sa3_build --all --dry-run
    python -m acestep.engine.trt.sa3_build --all --force-rebuild

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
    _register_same_plugin,
    trt_engines_dir,
)

HF_REPO = "stabilityai/stable-audio-3-optimized"
# The medium DiT ONNX exceeds 2 GB, so the weights travel in an
# external-data sidecar next to the proto.
DIT_ONNX_FILES = (
    "onnx/sa3-m/dit_fp16mixed.onnx",
    "onnx/sa3-m/dit_fp16mixed.onnx.data",
)
# FP8-trunk DiT (opt-in, ~1.8x/step). Not yet published to HF, so the fetch
# 404s until the upstream artifact upload; build it locally with the vendored
# producer (optimized/tensorRT/build: make_calib.py then build_dit_fp8.py) and
# pass --fp8-onnx. acestep.engine.sa3_trt does the runtime selection.
DIT_FP8_ONNX_FILES = (
    "onnx/sa3-m/dit_fp8.onnx",
    "onnx/sa3-m/dit_fp8.onnx.data",
)
SAME_L_ONNX_FILES = ("onnx/same-l/dec_dynamic_triton_swa.onnx",)

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
# SAME-L windowed decode profile: DEMON decodes ~1 s windows with 2 s of
# context, so T stays inside [32, 96] with the steady state at 56.
CANONICAL_SAME_L_WINDOW: tuple[int, int, int] = (32, 56, 96)


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
class SameLWindowBuildConfig:
    """Build parameters for the SAME-L window decoder engine."""

    min_latents: int
    opt_latents: int
    max_latents: int
    workspace_gb: float = 16.0
    onnx_files: list[str] = field(default_factory=lambda: list(SAME_L_ONNX_FILES))

    def engine_name(self) -> str:
        return (
            f"same_l_decode_window_t{self.min_latents}"
            f"_{self.opt_latents}_{self.max_latents}"
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


def _build_strongly_typed_engine(
    *,
    onnx_path: str,
    engine_path: str,
    workspace_gb: float,
    profile_shapes: dict[str, tuple[tuple, tuple, tuple]],
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
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: {}", parser.get_error(i))
        raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)),
    )
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
        f"SA3-M DiT {precision_label} "
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
            if precision_label == "fp8":
                raise RuntimeError(
                    f"dit_fp8.onnx is not on HF ({HF_REPO}) yet. Build it with "
                    "the managed vendored producer "
                    "(<MODELS_DIR>/sa3/vendor/stable-audio-3/optimized/"
                    "tensorRT/build: make_calib.py then build_dit_fp8.py) and "
                    "pass --fp8-onnx <dit_fp8.onnx>, or wait for the upstream "
                    "artifact upload."
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

    onnx_path = _fetch_onnx(config.onnx_files)
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
    return CANONICAL_DIT_PROFILES


def _matrix_jobs(args) -> list[tuple[str, str]]:
    """(label, engine_dir_name) pairs for the --all matrix."""
    dit_profiles = _resolve_dit_profiles(args)

    jobs = []
    if not args.same_l_only:
        for lo, opt, hi in dit_profiles:
            cfg = SA3DiTBuildConfig(lo, opt, hi)
            jobs.append((
                f"SA3-M DiT fp16mixed l{lo}_{opt}_{hi}"
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
    if not args.dit_only:
        lo, opt, hi = CANONICAL_SAME_L_WINDOW
        cfg = SameLWindowBuildConfig(lo, opt, hi)
        jobs.append((f"SAME-L window decoder t{lo}_{opt}_{hi}", cfg.engine_name()))
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

    single = parser.add_argument_group("single mode / shared options")
    single.add_argument("--output-dir", default=str(trt_engines_dir()),
                        help="Engine output directory "
                             "(default: <models>/sa3/trt_engines)")
    single.add_argument("--dit", action="store_true",
                        help="Build one DiT engine (size via --seconds or "
                             "the latent flags)")
    single.add_argument("--same-l-window", action="store_true",
                        help="Build the SAME-L window decoder (size via the "
                             "latent flags; defaults t32_56_96)")
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
    single.add_argument("--fp8-onnx", default=None,
                        help="Path to a producer-built dit_fp8.onnx (with its "
                             ".onnx.data sidecar alongside) to compile instead "
                             "of fetching from HF; implies --fp8.")

    args = parser.parse_args()
    if args.fp8_onnx:
        args.fp8 = True
    if not (args.all or args.dit or args.same_l_window):
        parser.error("nothing to build: pass --all, --dit, or --same-l-window")
    if args.dit and args.same_l_window:
        parser.error("--dit and --same-l-window share the latent flags; "
                     "build them in separate invocations or use --all")
    if args.dit_only and args.same_l_only:
        parser.error("--dit-only and --same-l-only are mutually exclusive")

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
                    config=SA3DiTBuildConfig(lo, opt, hi, workspace_gb=args.workspace_gb),
                    env=env,
                    force_rebuild=args.force_rebuild,
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
        if not args.dit_only:
            lo, opt, hi = CANONICAL_SAME_L_WINDOW
            results.append(_build_same_l_window_engine(
                output_dir=args.output_dir,
                config=SameLWindowBuildConfig(lo, opt, hi, workspace_gb=args.workspace_gb),
                env=env,
                force_rebuild=args.force_rebuild,
            ))
        failures = _print_summary(results, args.output_dir)
        _save_build_report(results, args.output_dir)
        return 1 if failures else 0

    # Single mode
    if args.same_l_window:
        # Same vendor-plugin requirement as the --all path; fail fast.
        require_sa3_vendor()
    env = _preflight("cuda")
    built = []
    if args.dit:
        profile_l = latents_for_seconds(args.seconds)
        config = SA3DiTBuildConfig(
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
