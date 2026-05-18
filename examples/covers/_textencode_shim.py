"""Compat shim: legacy ``TextEncode`` node.

The examples in this folder were written against an older combined
node, ``TextEncode``, that wrapped both the text encoder pass and the
conditioning fusion (timbre_ref). The split into ``EncodeText`` and
``EncodeConditioning`` is the current API.

Importing this module monkey-patches ``acestep.nodes.cond_nodes`` to
expose a ``TextEncode`` symbol with the legacy signature, so the
existing example scripts keep working without source edits.
"""

from __future__ import annotations

import acestep.nodes.cond_nodes as _cn


class TextEncode:
    """One-call wrapper around the EncodeText + EncodeConditioning pair.

    Matches the legacy node signature the cover examples use.
    """

    def execute(self, *,
                clip, model,
                refer_latent=None,
                tags: str = "",
                lyrics: str = "",
                instruction=None,
                bpm: int = 120,
                duration: float = 60.0,
                key: str = "C major",
                time_signature: str = "4",
                language: str = "en"):
        text_embed = _cn.EncodeText().execute(
            clip=clip, tags=tags, lyrics=lyrics,
            instruction=instruction,
            bpm=bpm, duration=duration, key=key,
            time_signature=time_signature, language=language,
        )["text_embed"]
        cond = _cn.EncodeConditioning().execute(
            model=model, text_embed=text_embed,
            timbre_ref=refer_latent,
        )["conditioning"]
        return {"conditioning": cond}


if not hasattr(_cn, "TextEncode"):
    _cn.TextEncode = TextEncode


# ---------------------------------------------------------------------------
# LoadModel project_root compat: examples pass the repo root, but the
# resolver now expects either a checkpoints dir or ``None`` (defers to
# acestep.paths.checkpoints_dir()). Wrap ``LoadModel.execute`` so that a
# project-root-shaped argument is dropped before the call.
# ---------------------------------------------------------------------------

import acestep.nodes.model_nodes as _mn

if not getattr(_mn.LoadModel, "_dj_compat_patched", False):
    _orig_execute = _mn.LoadModel.execute

    def _patched_execute(self, **kwargs):
        pr = kwargs.get("project_root")
        if pr:
            from pathlib import Path as _Path
            p = _Path(pr)
            # Heuristic: if it doesn't contain a config dir, treat as repo
            # root and drop it so the resolver falls through to the
            # canonical checkpoints location.
            looks_like_ckpt = any(
                (p / name).is_dir()
                for name in ("acestep-v15-turbo", "acestep-v15-xl-turbo")
            )
            if not looks_like_ckpt:
                kwargs.pop("project_root", None)
        return _orig_execute(self, **kwargs)

    _mn.LoadModel.execute = _patched_execute
    _mn.LoadModel._dj_compat_patched = True


# ---------------------------------------------------------------------------
# VAEEncodeAudio compat: the legacy fixtures here are slightly under 60 s
# (~56.6 s) so VAE encode produces a latent whose T (sample axis) isn't
# divisible by 5, which is the current model's patch size. The examples
# don't see this error in their original era's checkpoint. Trim the
# waveform on the way in so T % 5 == 0.
# ---------------------------------------------------------------------------

import acestep.nodes.vae_nodes as _vn

# VAE pool factor: samples per latent frame. 25 fps latent at 48000 Hz
# = 1920 samples per frame.
_SAMPLES_PER_LATENT_FRAME = 1920
_PATCH_SIZE = 5
# Trim waveforms to a multiple of (patch_size * samples_per_frame) so
# the resulting latent T is divisible by the patch.
_TRIM_MULTIPLE = _PATCH_SIZE * _SAMPLES_PER_LATENT_FRAME  # = 9600


if not getattr(_vn.VAEEncodeAudio, "_dj_compat_patched", False):
    _orig_vae_exec = _vn.VAEEncodeAudio.execute

    def _patched_vae_exec(self, **kwargs):
        audio = kwargs.get("audio")
        if audio is not None and hasattr(audio, "waveform"):
            wf = audio.waveform
            n_samples = wf.shape[-1]
            trimmed = (n_samples // _TRIM_MULTIPLE) * _TRIM_MULTIPLE
            if trimmed != n_samples and trimmed > 0:
                import dataclasses
                kwargs["audio"] = dataclasses.replace(
                    audio,
                    waveform=wf[..., :trimmed].contiguous(),
                )
        return _orig_vae_exec(self, **kwargs)

    _vn.VAEEncodeAudio.execute = _patched_vae_exec
    _vn.VAEEncodeAudio._dj_compat_patched = True


# ---------------------------------------------------------------------------
# Generate compat: auto-build a default Solver and bundle loose
# modulation kwargs (x0_target, *_curve, velocity_scale, ...) into a
# Modulation. The legacy examples pass these as flat kwargs; the current
# Generate.execute requires Solver and Modulation explicitly.
# ---------------------------------------------------------------------------

import acestep.nodes.diffusion_nodes as _dn
from acestep.nodes.types import Modulation as _Modulation, Solver as _Solver

_MOD_FIELDS = (
    "velocity_scale",
    "initial_noise_curve",
    "chunk_mask",
    "x0_target",
    "x0_target_curve",
    "x0_target_strength",
    "x0_target_gate",
    "guidance_curve",
)


def _to_tensor(v):
    if v is None:
        return None
    return v.tensor if hasattr(v, "tensor") else v


def _maybe_modulation(kwargs):
    if "modulation" in kwargs and kwargs["modulation"] is not None:
        return kwargs.pop("modulation")
    if not any(k in kwargs for k in _MOD_FIELDS):
        return None
    return _Modulation(
        velocity_scale=_to_tensor(kwargs.pop("velocity_scale", None)),
        initial_noise_curve=_to_tensor(kwargs.pop("initial_noise_curve", None)),
        chunk_mask=_to_tensor(kwargs.pop("chunk_mask", None)),
        x0_target=kwargs.pop("x0_target", None),
        x0_target_curve=_to_tensor(kwargs.pop("x0_target_curve", None)),
        x0_target_strength=float(kwargs.pop("x0_target_strength", 0.0)),
        x0_target_gate=float(kwargs.pop("x0_target_gate", 0.0)),
        guidance_curve=_to_tensor(kwargs.pop("guidance_curve", None)),
    )


# ---------------------------------------------------------------------------
# VAEDecodeAudio compat: workflows do ``audio.waveform.cpu().numpy()``
# without detach(). Current model returns grad-tracked / bf16 tensors in
# eager mode, so the legacy save path raises. Detach + float on the way
# out so the workflow's own save_audio works unchanged.
# ---------------------------------------------------------------------------

import dataclasses as _dc

if not getattr(_vn.VAEDecodeAudio, "_dj_compat_patched", False):
    _orig_dec_exec = _vn.VAEDecodeAudio.execute

    def _patched_dec_exec(self, **kwargs):
        result = _orig_dec_exec(self, **kwargs)
        audio = result.get("audio")
        if audio is not None and hasattr(audio, "waveform"):
            wf = audio.waveform
            result["audio"] = _dc.replace(
                audio,
                waveform=wf.detach().float().contiguous(),
            )
        return result

    _vn.VAEDecodeAudio.execute = _patched_dec_exec
    _vn.VAEDecodeAudio._dj_compat_patched = True


if not getattr(_dn.Generate, "_dj_compat_patched", False):
    _orig_gen_exec = _dn.Generate.execute

    def _patched_gen_exec(self, **kwargs):
        # Solver auto-build (legacy examples don't construct it themselves).
        if "solver" not in kwargs or kwargs["solver"] is None:
            method = kwargs.pop("method", "ode")
            ode_curve = kwargs.pop("ode_noise_curve", None)
            sde_curve = kwargs.pop("sde_denoise_curve", None)
            if sde_curve is not None:
                method = "sde"
                noise = _to_tensor(sde_curve)
            elif ode_curve is not None:
                method = "ode"
                noise = _to_tensor(ode_curve)
            else:
                noise = None
            kwargs["solver"] = _Solver(method=method, noise_curve=noise)

        # Modulation auto-bundle (legacy examples pass x0_target_* /
        # *_curve as flat kwargs).
        mod = _maybe_modulation(kwargs)
        if mod is not None:
            kwargs["modulation"] = mod

        return _orig_gen_exec(self, **kwargs)

    _dn.Generate.execute = _patched_gen_exec
    _dn.Generate._dj_compat_patched = True
