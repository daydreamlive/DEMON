"""Model management nodes."""

from __future__ import annotations

from typing import Any, ClassVar

from .base import BaseNode, NodeDefinition, NodeParam, NodePort, NodeRegistry
from .types import CLIPHandle, ModelHandle, VAEHandle


@NodeRegistry.register
class LoadModel(BaseNode):
    """Load the ACE-Step checkpoint and return model/clip/vae handles.

    Node parameters (passed via execute kwargs):
        project_root: Path to project root or checkpoints directory.
            When omitted, resolves to ``acestep.paths.checkpoints_dir()``.
        config_path: Model config directory name (e.g. "acestep-v15-turbo").
        device: Device string ("auto", "cuda", "cpu").
        use_flash_attention: Whether to use flash attention.
        decoder_backend: "eager" | "compile" | "tensorrt" (default "eager").
        vae_backend:     "eager" | "compile" | "tensorrt" (default "eager").
        trt_engines: Optional dict with "decoder", "vae_encode", "vae_decode"
            keys mapping to engine file paths. When a backend is "tensorrt"
            and no dict is provided, ``acestep.paths.default_trt_engines()``
            is used.
        offload_to_cpu: Whether to offload models when not in use.
        offload_text_encoder: Whether to offload the text encoder between
            prompt encodes to reduce steady VRAM at the cost of prompt latency.
        quantization: Quantization type (None, "int8_weight_only", etc.).

    The ``compile_decoder`` / ``compile_vae`` flags are still accepted for
    backwards compatibility but are superseded by ``*_backend``.
    """

    node_type_id: ClassVar[str] = "acestep.LoadModel"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Load ACE-Step Model",
            category="model",
            description="Load checkpoint and return MODEL, CLIP, VAE handles.",
            inputs=(),
            outputs=(
                NodePort(name="model", type="MODEL"),
                NodePort(name="clip", type="CLIP"),
                NodePort(name="vae", type="VAE"),
            ),
            params=(
                NodeParam(
                    name="config_path", type="select",
                    default="acestep-v15-turbo",
                    description="Model config",
                    options=("acestep-v15-turbo", "acestep-v15-base"),
                ),
                NodeParam(
                    name="device", type="select", default="auto",
                    description="Device",
                    options=("cuda", "cpu", "auto"),
                ),
                NodeParam(
                    name="use_flash_attention", type="boolean", default=False,
                    description="Flash Attention",
                ),
                NodeParam(
                    name="decoder_backend", type="select", default="eager",
                    description="Decoder backend",
                    options=("eager", "compile", "tensorrt"),
                ),
                NodeParam(
                    name="vae_backend", type="select", default="eager",
                    description="VAE backend",
                    options=("eager", "compile", "tensorrt"),
                ),
                NodeParam(
                    name="project_root", type="string", default="",
                    description="Project root (empty = acestep.paths.checkpoints_dir())",
                ),
                NodeParam(
                    name="offload_to_cpu", type="boolean", default=False,
                    description="Offload to CPU when idle",
                    hidden=True,
                ),
                NodeParam(
                    name="offload_text_encoder", type="boolean", default=False,
                    description="Offload text encoder between prompt encodes",
                    hidden=True,
                ),
                NodeParam(
                    name="quantization", type="any", default=None,
                    description="Quantization type (None, 'int8_weight_only', ...)",
                    hidden=True,
                ),
                NodeParam(
                    name="compile_decoder", type="boolean", default=False,
                    description="[legacy] use decoder_backend",
                    hidden=True,
                ),
                NodeParam(
                    name="compile_vae", type="boolean", default=False,
                    description="[legacy] use vae_backend",
                    hidden=True,
                ),
                NodeParam(
                    name="trt_engines", type="any", default=None,
                    description="Optional TRT engines dict {decoder, vae_encode, vae_decode}",
                    hidden=True,
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        from acestep.engine.model_context import ModelContext
        from acestep.engine.runtime_init import (
            apply_trt_backends,
            backends_to_model_context_flags,
            validate_backends,
        )
        from acestep.paths import default_trt_engines

        # Backwards-compat: translate legacy compile_* flags into backends
        # only when the caller didn't specify a backend explicitly.
        decoder_backend = kwargs.get("decoder_backend")
        vae_backend = kwargs.get("vae_backend")
        if decoder_backend is None:
            decoder_backend = "compile" if kwargs.get("compile_decoder") else "eager"
        if vae_backend is None:
            vae_backend = "compile" if kwargs.get("compile_vae") else "eager"

        # Auto-fill default TRT engine paths when tensorrt is selected and
        # the caller didn't provide explicit paths. Session keeps its stricter
        # "must pass trt_engines" contract; the node-level API is friendlier.
        trt_engines = kwargs.get("trt_engines")
        if trt_engines is None and "tensorrt" in (decoder_backend, vae_backend):
            trt_engines = default_trt_engines()

        trt_engines = validate_backends(
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            trt_engines=trt_engines,
        )

        ctx_flags = backends_to_model_context_flags(
            decoder_backend=decoder_backend, vae_backend=vae_backend
        )

        # Empty-string project_root (scope widget default) means
        # "unset" — let ModelContext fall back to acestep.paths.
        project_root = kwargs.get("project_root") or None

        ctx = ModelContext(
            project_root=project_root,
            config_path=kwargs.get("config_path", "acestep-v15-turbo"),
            device=kwargs.get("device", "auto"),
            use_flash_attention=kwargs.get("use_flash_attention", False),
            offload_to_cpu=kwargs.get("offload_to_cpu", False),
            offload_text_encoder=kwargs.get("offload_text_encoder", False),
            quantization=kwargs.get("quantization", None),
            **ctx_flags,
        )

        apply_trt_backends(
            ctx,
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            trt_engines=trt_engines,
            device=ctx.device,
        )

        return {
            "model": ModelHandle(handler=ctx),
            "clip": CLIPHandle(handler=ctx),
            "vae": VAEHandle(handler=ctx),
        }
