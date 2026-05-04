"""Lightweight model container for the ACE-Step engine.

Loads the DiT model, VAE, text encoder, and tokenizer from a checkpoint
directory and exposes the attributes/methods that the engine and node
subsystems need.  This replaces AceStepHandler as the backing object for
ModelHandle / CLIPHandle / VAEHandle when running standalone (without
the Gradio UI).

AceStepHandler is left untouched; Gradio users keep using it.  The two
classes expose the same duck-typed surface so every node works with
either backend transparently.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
from loguru import logger

# Persist torch.compile kernel cache across sessions
if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        os.path.expanduser("~"), ".cache", "torchinductor"
    )


class ModelContext:
    """Persistent container for loaded ACE-Step models.

    Holds the DiT model, VAE, text encoder, tokenizer, and silence
    latent.  Provides device management, text/lyric embedding inference,
    tiled VAE encode/decode, and the composable diffusion engine.

    This is the engine/nodes replacement for ``AceStepHandler``.  The
    handler stays for Gradio users; ``ModelContext`` is the standalone
    path.

    Parameters mirror the useful subset of ``AceStepHandler.initialize_service``.
    """

    def __init__(
        self,
        *,
        project_root: Optional[str] = None,
        config_path: str = "acestep-v15-turbo",
        device: str = "auto",
        compile_decoder: bool = False,
        compile_vae: bool = False,
        use_flash_attention: bool = False,
        offload_to_cpu: bool = False,
        offload_dit_to_cpu: bool = False,
        quantization: Optional[str] = None,
        prefer_source: Optional[str] = None,
        skip_decoder: bool = False,
        skip_vae: bool = False,
    ):
        if project_root is None:
            from acestep.paths import checkpoints_dir
            project_root = str(checkpoints_dir())

        # Attributes that must exist before _load_models
        self.model = None
        self.config = None
        self.vae = None
        self.text_encoder = None
        self.text_tokenizer = None
        self.silence_latent = None
        self.sample_rate = 48000
        self.offload_to_cpu = offload_to_cpu
        self.offload_dit_to_cpu = offload_dit_to_cpu
        self.quantization = quantization
        self._compile_decoder = compile_decoder
        self._compile_vae = compile_vae
        self._offload_text_encoder = False
        self._diffusion_engine = None

        self._load_models(
            project_root=project_root,
            config_path=config_path,
            device=device,
            compile_decoder=compile_decoder,
            compile_vae=compile_vae,
            use_flash_attention=use_flash_attention,
            prefer_source=prefer_source,
            skip_decoder=skip_decoder,
            skip_vae=skip_vae,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _load_models(
        self,
        *,
        project_root: str,
        config_path: str,
        device: str,
        compile_decoder: bool,
        compile_vae: bool,
        use_flash_attention: bool,
        prefer_source: Optional[str],
        skip_decoder: bool,
        skip_vae: bool,
    ) -> None:
        from transformers import AutoTokenizer, AutoModel
        from diffusers.models import AutoencoderOobleck

        # --- Device / dtype ------------------------------------------------
        if device == "auto":
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            elif torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.dtype = torch.bfloat16 if device in ("cuda", "xpu") else torch.float32

        if self.quantization is not None:
            assert compile_decoder, "Quantization requires compile_decoder=True"
            try:
                import torchao  # noqa: F401
            except ImportError:
                raise ImportError(
                    "torchao is required for quantization but is not installed."
                )

        # --- Resolve checkpoint directory ----------------------------------
        checkpoint_dir = self._resolve_checkpoint_dir(project_root)

        # --- Auto-download if missing --------------------------------------
        self._ensure_downloaded(checkpoint_dir, config_path, prefer_source)

        # --- 1. Load DiT model ---------------------------------------------
        dit_path = os.path.join(checkpoint_dir, config_path)
        if not os.path.exists(dit_path):
            raise FileNotFoundError(
                f"ACE-Step checkpoint not found at {dit_path}"
            )

        attn_impl = self._choose_attn_impl(use_flash_attention)
        self.model = self._load_dit(dit_path, attn_impl)
        self.config = self.model.config

        if skip_decoder:
            import torch.nn as nn
            self.model.decoder = nn.Module()
            logger.info("Decoder weights discarded (TRT engine will be used)")

        self._place_dit(skip_decoder)

        if compile_decoder and not skip_decoder:
            self._apply_decoder_compile()

        if self.quantization is not None:
            self._quantize()

        # Silence latent
        silence_path = os.path.join(dit_path, "silence_latent.pt")
        if not os.path.exists(silence_path):
            raise FileNotFoundError(f"Silence latent not found at {silence_path}")
        self.silence_latent = (
            torch.load(silence_path, weights_only=True)
            .transpose(1, 2)
            .to(self.device)
            .to(self.dtype)
        )

        # --- 2. Load VAE ---------------------------------------------------
        if skip_vae:
            self.vae = None
            logger.info("VAE weights skipped (TRT engines will be used)")
        else:
            vae_path = os.path.join(checkpoint_dir, "vae")
            if not os.path.exists(vae_path):
                raise FileNotFoundError(f"VAE checkpoint not found at {vae_path}")
            self.vae = AutoencoderOobleck.from_pretrained(vae_path)
            vae_dtype = self._get_vae_dtype()
            if not self.offload_to_cpu:
                self.vae = self.vae.to(device).to(vae_dtype)
            else:
                self.vae = self.vae.to("cpu").to(vae_dtype)
            self.vae.eval()

        if compile_vae and not skip_vae:
            self.vae = torch.compile(self.vae, dynamic=True)

        # --- 3. Load text encoder / tokenizer ------------------------------
        self._offload_text_encoder = skip_decoder
        text_enc_path = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")
        if not os.path.exists(text_enc_path):
            raise FileNotFoundError(f"Text encoder not found at {text_enc_path}")
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_enc_path)
        self.text_encoder = AutoModel.from_pretrained(text_enc_path)
        if not self.offload_to_cpu and not skip_decoder:
            self.text_encoder = self.text_encoder.to(device).to(self.dtype)
        else:
            self.text_encoder = self.text_encoder.to("cpu").to(self.dtype)
        self.text_encoder.eval()

        actual_attn = getattr(self.config, "_attn_implementation", "eager")
        logger.info(
            f"ModelContext ready | device={device} attn={actual_attn} "
            f"compile_decoder={compile_decoder} compile_vae={compile_vae} "
            f"decoder={'TRT' if skip_decoder else 'PT'} "
            f"vae={'TRT' if skip_vae else 'PT'}"
        )

    # --- Private init helpers ----------------------------------------------

    @staticmethod
    def _resolve_checkpoint_dir(project_root: str) -> str:
        from pathlib import Path
        from acestep.model_downloader import check_main_model_exists

        # Preferred layout: <project_root>/checkpoints/
        candidate = os.path.join(project_root, "checkpoints")
        if os.path.isdir(candidate):
            return candidate

        # Fallback: project_root *is* the checkpoints dir (the canonical
        # path returned by acestep.paths.checkpoints_dir() works this way).
        # Only allow this when the directory already looks like a populated
        # checkpoints dir, so a misconfigured project_root never silently
        # triggers a ~10 GB download into a random location.
        if os.path.isdir(project_root) and check_main_model_exists(Path(project_root)):
            return project_root

        raise FileNotFoundError(
            f"Cannot locate a populated checkpoints directory from "
            f"project_root={project_root!r}. Either pass an explicit "
            f"checkpoints path or call Session() with no arguments to use "
            f"acestep.paths.checkpoints_dir() "
            f"(default: ~/.daydream-scope/models/demon/checkpoints)."
        )

    @staticmethod
    def _ensure_downloaded(
        checkpoint_dir: str,
        config_path: str,
        prefer_source: Optional[str],
    ) -> None:
        from pathlib import Path
        from acestep.model_downloader import (
            ensure_main_model,
            ensure_dit_model,
            check_main_model_exists,
            check_model_exists,
        )

        cp = Path(checkpoint_dir)
        if not check_main_model_exists(cp):
            logger.info("Main model not found, starting auto-download...")
            ok, msg = ensure_main_model(cp, prefer_source=prefer_source)
            if not ok:
                raise RuntimeError(f"Failed to download main model: {msg}")
            logger.info(msg)

        if not check_model_exists(config_path, cp):
            logger.info(f"DiT model '{config_path}' not found, downloading...")
            ok, msg = ensure_dit_model(config_path, cp, prefer_source=prefer_source)
            if not ok:
                raise RuntimeError(f"Failed to download DiT model '{config_path}': {msg}")
            logger.info(msg)

    @staticmethod
    def _choose_attn_impl(use_flash: bool) -> str:
        if use_flash:
            try:
                import flash_attn  # noqa: F401
                return "flash_attention_2"
            except ImportError:
                pass
        return "sdpa"

    # Maps the upstream `auto_map["AutoModel"]` string in a checkpoint's
    # config.json to the vendored class DEMON ships. Both turbo and XL turbo
    # checkpoints declare architectures=["AceStepConditionGenerationModel"],
    # so the auto_map module path is what actually distinguishes them.
    _VENDORED_DIT_CLASSES = {
        "modeling_acestep_v15_turbo.AceStepConditionGenerationModel":
            ("acestep.models.modeling_acestep_v15_turbo", "AceStepConditionGenerationModel"),
        "modeling_acestep_v15_xl_turbo.AceStepConditionGenerationModel":
            ("acestep.models.modeling_acestep_v15_xl_turbo", "AceStepConditionGenerationModel"),
        # Non-distilled 2B variants (acestep-v15-sft, acestep-v15-base) ship
        # the same modeling file under this auto_map module path. Forward
        # signature on the inner DiT matches what stream.py:_decoder_forward
        # calls, so the per-step path works without an adapter.
        "modeling_acestep_v15_base.AceStepConditionGenerationModel":
            ("acestep.models.modeling_acestep_v15_base", "AceStepConditionGenerationModel"),
    }

    def _load_dit(self, path: str, attn_impl: str):
        # Load the vendored class directly. Avoids `trust_remote_code=True`,
        # which would let arbitrary .py files in the checkpoint dir execute
        # in our process — exactly the failure mode that bit us when a
        # mid-session pod sync rewrote the checkpoint's modeling file.
        import importlib
        import json

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        architectures = cfg.get("architectures", [])
        auto_model = cfg.get("auto_map", {}).get("AutoModel", "")

        if architectures != ["AceStepConditionGenerationModel"]:
            raise NotImplementedError(
                f"Unexpected architectures={architectures} in {config_path}; "
                f"the vendored DiT path only handles AceStepConditionGenerationModel."
            )

        target = self._VENDORED_DIT_CLASSES.get(auto_model)
        if target is None:
            raise NotImplementedError(
                f"No vendored class for auto_map AutoModel={auto_model!r} "
                f"(checkpoint at {path}). Vendor its modeling file into "
                f"acestep/models/ and add it to ModelContext._VENDORED_DIT_CLASSES."
            )
        module_name, class_name = target
        ModelClass = getattr(importlib.import_module(module_name), class_name)

        try:
            logger.info(f"Loading {class_name} from {module_name} with attention={attn_impl}")
            model = ModelClass.from_pretrained(
                path,
                attn_implementation=attn_impl,
                dtype="bfloat16",
            )
        except Exception as exc:
            if attn_impl == "sdpa":
                logger.info("Falling back to eager attention")
                model = ModelClass.from_pretrained(
                    path,
                    attn_implementation="eager",
                )
                attn_impl = "eager"
            else:
                raise exc

        model.config._attn_implementation = attn_impl
        model.eval()
        return model

    def _place_dit(self, skip_decoder: bool) -> None:
        if not self.offload_to_cpu:
            self.model = self.model.to(self.device).to(self.dtype)
        elif not self.offload_dit_to_cpu:
            logger.info(f"Keeping main model on {self.device} (persistent)")
            self.model = self.model.to(self.device).to(self.dtype)
        else:
            self.model = self.model.to("cpu").to(self.dtype)

    def _apply_decoder_compile(self) -> None:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True

        mode = "max-autotune-no-cudagraphs"
        if self.quantization is not None:
            mode = "default"

        self.model.decoder = torch.compile(
            self.model.decoder, backend="inductor", dynamic=True, mode=mode,
        )

    def _quantize(self) -> None:
        from torchao.quantization import quantize_

        q = self.quantization
        if q == "int8_weight_only":
            from torchao.quantization import Int8WeightOnlyConfig
            cfg = Int8WeightOnlyConfig()
        elif q == "fp8_weight_only":
            from torchao.quantization import Float8WeightOnlyConfig
            cfg = Float8WeightOnlyConfig()
        elif q == "fp8_dynamic":
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
            cfg = Float8DynamicActivationFloat8WeightConfig()
        elif q == "w8a8_dynamic":
            from torchao.quantization import Int8DynamicActivationInt8WeightConfig, MappingType
            cfg = Int8DynamicActivationInt8WeightConfig(
                act_mapping_type=MappingType.ASYMMETRIC,
            )
        else:
            raise ValueError(f"Unsupported quantization type: {q}")

        quantize_(self.model, cfg)
        logger.info(f"DiT quantized with: {q}")

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def _get_vae_dtype(self, device: Optional[str] = None) -> torch.dtype:
        device = device or self.device
        return torch.bfloat16 if device in ("cuda", "xpu") else self.dtype

    @staticmethod
    def _is_on_target_device(tensor, target_device) -> bool:
        if tensor is None:
            return True
        target_type = "cpu" if target_device == "cpu" else "cuda"
        return tensor.device.type == target_type

    def _ensure_silence_latent_on_device(self) -> None:
        if self.silence_latent is not None:
            if not self._is_on_target_device(self.silence_latent, self.device):
                self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)

    def _recursive_to_device(self, model, device, dtype=None) -> None:
        target = torch.device(device) if isinstance(device, str) else device
        model.to(target)
        if dtype is not None:
            model.to(dtype)

        wrong = [n for n, p in model.named_parameters()
                 if not self._is_on_target_device(p, device)]
        if wrong and device != "cpu":
            logger.warning(f"{len(wrong)} params on wrong device, using state_dict move")
            sd = {
                k: (v.to(target).to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() and dtype
                     else v.to(target) if isinstance(v, torch.Tensor) else v)
                for k, v in model.state_dict().items()
            }
            model.load_state_dict(sd)

        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def _load_model_context(self, model_name: str):
        """Move *model_name* to GPU for the duration of the block."""
        if (
            model_name == "text_encoder"
            and self._offload_text_encoder
        ):
            pass  # fall through to offload logic
        elif not self.offload_to_cpu:
            yield
            return

        if model_name == "model" and not self.offload_dit_to_cpu:
            model = getattr(self, model_name, None)
            if model is not None:
                try:
                    param = next(model.parameters())
                    if param.device.type == "cpu":
                        logger.info(f"Moving {model_name} to {self.device} (persistent)")
                        self._recursive_to_device(model, self.device, self.dtype)
                        if self.silence_latent is not None:
                            self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)
                except StopIteration:
                    pass
            yield
            return

        model = getattr(self, model_name, None)
        if model is None:
            yield
            return

        logger.info(f"Loading {model_name} to {self.device}")
        t0 = time.time()
        if model_name == "vae":
            self._recursive_to_device(model, self.device, self._get_vae_dtype())
        else:
            self._recursive_to_device(model, self.device, self.dtype)

        if model_name == "model" and self.silence_latent is not None:
            self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)

        logger.info(f"Loaded {model_name} in {time.time() - t0:.3f}s")

        try:
            yield
        finally:
            logger.info(f"Offloading {model_name} to CPU")
            t0 = time.time()
            self._recursive_to_device(model, "cpu")
            torch.cuda.empty_cache()
            logger.info(f"Offloaded {model_name} in {time.time() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Text embedding inference
    # ------------------------------------------------------------------

    def infer_text_embeddings(self, text_token_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.text_encoder(
                input_ids=text_token_ids, lyric_attention_mask=None,
            ).last_hidden_state

    def infer_lyric_embeddings(self, lyric_token_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.text_encoder.embed_tokens(lyric_token_ids)

    # ------------------------------------------------------------------
    # VAE encode / decode (tiled for long audio)
    # ------------------------------------------------------------------

    @staticmethod
    def is_silence(audio: torch.Tensor) -> bool:
        return torch.all(audio.abs() < 1e-6).item()

    def _encode_audio_to_latents(self, audio: torch.Tensor) -> torch.Tensor:
        was_2d = audio.dim() == 2
        if was_2d:
            audio = audio.unsqueeze(0)
        with torch.no_grad():
            latents = self.tiled_encode(audio, offload_latent_to_cpu=True)
        latents = latents.to(self.device).to(self.dtype).transpose(1, 2)
        if was_2d:
            latents = latents.squeeze(0)
        return latents

    def tiled_encode(self, audio, chunk_size=None, overlap=None,
                     offload_latent_to_cpu=True):
        from acestep.gpu_config import get_gpu_memory_gb

        if chunk_size is None:
            gpu_mem = get_gpu_memory_gb()
            chunk_size = 48000 * 15 if gpu_mem <= 8 else 48000 * 30
        if overlap is None:
            overlap = 48000 * 2

        was_2d = audio.dim() == 2
        if was_2d:
            audio = audio.unsqueeze(0)
        B, C, S = audio.shape

        if S <= chunk_size:
            vae_input = audio.to(self.device).to(self.vae.dtype)
            with torch.no_grad():
                latents = self.vae.encode(vae_input).latent_dist.sample()
            if was_2d:
                latents = latents.squeeze(0)
            return latents

        stride = chunk_size - 2 * overlap
        if stride <= 0:
            raise ValueError(f"chunk_size {chunk_size} must be > 2*overlap {overlap}")
        num_steps = math.ceil(S / stride)

        if offload_latent_to_cpu:
            result = self._tiled_encode_offload(audio, B, S, stride, overlap, num_steps)
        else:
            result = self._tiled_encode_gpu(audio, B, S, stride, overlap, num_steps)
        if was_2d:
            result = result.squeeze(0)
        return result

    def _tiled_encode_gpu(self, audio, B, S, stride, overlap, num_steps):
        parts, ds = [], None
        for i in range(num_steps):
            cs, ce = i * stride, min(i * stride + stride, S)
            ws, we = max(0, cs - overlap), min(S, ce + overlap)
            chunk = audio[:, :, ws:we].to(self.device).to(self.vae.dtype)
            with torch.no_grad():
                lat = self.vae.encode(chunk).latent_dist.sample()
            if ds is None:
                ds = chunk.shape[-1] / lat.shape[-1]
            ts = int(round((cs - ws) / ds))
            te = int(round((we - ce) / ds))
            end = lat.shape[-1] - te if te > 0 else lat.shape[-1]
            parts.append(lat[:, :, ts:end])
            del chunk
        return torch.cat(parts, dim=-1)

    def _tiled_encode_offload(self, audio, B, S, stride, overlap, num_steps):
        ce0 = min(stride, S)
        we0 = min(S, ce0 + overlap)
        c0 = audio[:, :, :we0].to(self.device).to(self.vae.dtype)
        with torch.no_grad():
            l0 = self.vae.encode(c0).latent_dist.sample()
        ds = c0.shape[-1] / l0.shape[-1]
        total_T = int(round(S / ds))
        out = torch.zeros(B, l0.shape[1], total_T, dtype=l0.dtype, device="cpu")
        te0 = int(round((we0 - ce0) / ds))
        end0 = l0.shape[-1] - te0 if te0 > 0 else l0.shape[-1]
        pos = end0
        out[:, :, :pos] = l0[:, :, :end0].cpu()
        del c0, l0
        for i in range(1, num_steps):
            cs, ce = i * stride, min(i * stride + stride, S)
            ws, we = max(0, cs - overlap), min(S, ce + overlap)
            chunk = audio[:, :, ws:we].to(self.device).to(self.vae.dtype)
            with torch.no_grad():
                lat = self.vae.encode(chunk).latent_dist.sample()
            ts = int(round((cs - ws) / ds))
            te = int(round((we - ce) / ds))
            end = lat.shape[-1] - te if te > 0 else lat.shape[-1]
            core = lat[:, :, ts:end]
            clen = core.shape[-1]
            out[:, :, pos:pos + clen] = core.cpu()
            pos += clen
            del chunk, lat, core
        return out[:, :, :pos]

    def tiled_decode(self, latents, chunk_size=512, overlap=64,
                     offload_wav_to_cpu=True):
        B, C, T = latents.shape
        if T <= chunk_size:
            out = self.vae.decode(latents)
            result = out.sample
            del out
            return result
        stride = chunk_size - 2 * overlap
        if stride <= 0:
            raise ValueError(f"chunk_size {chunk_size} must be > 2*overlap {overlap}")
        num_steps = math.ceil(T / stride)
        if offload_wav_to_cpu:
            return self._tiled_decode_offload(latents, B, T, stride, overlap, num_steps)
        return self._tiled_decode_gpu(latents, B, T, stride, overlap, num_steps)

    def _tiled_decode_gpu(self, latents, B, T, stride, overlap, num_steps):
        parts, us = [], None
        for i in range(num_steps):
            cs, ce = i * stride, min(i * stride + stride, T)
            ws, we = max(0, cs - overlap), min(T, ce + overlap)
            chunk = latents[:, :, ws:we]
            out = self.vae.decode(chunk)
            audio = out.sample
            del out
            if us is None:
                us = audio.shape[-1] / chunk.shape[-1]
            ts = int(round((cs - ws) * us))
            te = int(round((we - ce) * us))
            end = audio.shape[-1] - te if te > 0 else audio.shape[-1]
            parts.append(audio[:, :, ts:end])
        return torch.cat(parts, dim=-1)

    def _tiled_decode_offload(self, latents, B, T, stride, overlap, num_steps):
        ce0 = min(stride, T)
        we0 = min(T, ce0 + overlap)
        c0 = latents[:, :, :we0]
        out0 = self.vae.decode(c0)
        a0 = out0.sample
        del out0
        us = a0.shape[-1] / c0.shape[-1]
        total_S = int(round(T * us))
        result = torch.zeros(B, a0.shape[1], total_S, dtype=a0.dtype, device="cpu")
        te0 = int(round((we0 - ce0) * us))
        end0 = a0.shape[-1] - te0 if te0 > 0 else a0.shape[-1]
        pos = end0
        result[:, :, :pos] = a0[:, :, :end0].cpu()
        del a0, c0
        for i in range(1, num_steps):
            cs, ce = i * stride, min(i * stride + stride, T)
            ws, we = max(0, cs - overlap), min(T, ce + overlap)
            chunk = latents[:, :, ws:we]
            out = self.vae.decode(chunk)
            audio = out.sample
            del out
            ts = int(round((cs - ws) * us))
            te = int(round((we - ce) * us))
            end = audio.shape[-1] - te if te > 0 else audio.shape[-1]
            core = audio[:, :, ts:end]
            clen = core.shape[-1]
            result[:, :, pos:pos + clen] = core.cpu()
            pos += clen
            del audio, core, chunk
        return result[:, :, :pos]

    # ------------------------------------------------------------------
    # Reference audio (timbre) encoding
    # ------------------------------------------------------------------

    def infer_refer_latent(self, refer_audioss):
        self._ensure_silence_latent_on_device()
        order_mask, latent_parts = [], []

        def _norm2d(a):
            if a.dim() == 3 and a.shape[0] == 1:
                a = a.squeeze(0)
            if a.dim() == 1:
                a = a.unsqueeze(0)
            if a.dim() != 2:
                raise ValueError(f"Expected 1D/2D audio, got shape={tuple(a.shape)}")
            if a.shape[0] == 1:
                a = torch.cat([a, a], dim=0)
            return a[:2]

        def _ensure3d(z):
            if z.dim() == 4 and z.shape[0] == 1:
                z = z.squeeze(0)
            if z.dim() == 2:
                z = z.unsqueeze(0)
            return z

        for batch_idx, audios in enumerate(refer_audioss):
            if len(audios) == 1 and torch.all(audios[0] == 0.0):
                latent_parts.append(_ensure3d(self.silence_latent[:, :750, :]))
                order_mask.append(batch_idx)
            else:
                for aud in audios:
                    aud = _norm2d(aud)
                    with torch.no_grad():
                        lat = self.tiled_encode(aud, offload_latent_to_cpu=True)
                    lat = lat.to(self.device).to(self.dtype)
                    if lat.dim() == 2:
                        lat = lat.unsqueeze(0)
                    latent_parts.append(_ensure3d(lat.transpose(1, 2)))
                    order_mask.append(batch_idx)

        packed = torch.cat(latent_parts, dim=0)
        mask = torch.tensor(order_mask, device=self.device, dtype=torch.long)
        return packed, mask

    def encode_reference_from_audio(self, audio_tensor):
        with self._load_model_context("vae"):
            return self.infer_refer_latent([[audio_tensor]])

    # ------------------------------------------------------------------
    # Condition building
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_condition(self, *, temporal_weight=None, step_range=None, **tensors):
        from acestep.engine.conditions import ConditionBuilder
        self._ensure_silence_latent_on_device()
        builder = ConditionBuilder(self.model)
        with self._load_model_context("model"):
            return builder.build(
                silence_latent=self.silence_latent,
                temporal_weight=temporal_weight,
                step_range=step_range, **tensors,
            )

    # ------------------------------------------------------------------
    # Utility predicates
    # ------------------------------------------------------------------

    def is_turbo_model(self) -> bool:
        if self.config is None:
            return False
        return getattr(self.config, "is_turbo", False)
