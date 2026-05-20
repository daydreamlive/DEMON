"""ACE-Step Transcriber wrapper.

Thin loader for the Qwen2.5-Omni-7B fine-tune at
``ACE-Step/acestep-transcriber``. Exposes :class:`Transcriber` whose
:meth:`Transcriber.transcribe` takes an audio path and returns the raw
structured output produced by the model (sections like ``# Languages``
and ``# Lyrics`` with ``[Verse 1]`` / ``[Chorus]`` tags).

Loads only the *thinker* branch — :py:meth:`disable_talker` strips the
audio-output head we never use, freeing ~4 GB of VRAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np
import torch
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)


# Canonical Qwen2.5-Omni system prompt. The model was fine-tuned with
# this exact string — substituting another one degrades quality.
_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well "
    "as generating text and speech."
)

_TASK_PROMPT = "*Task* Transcribe this audio in detail"


def default_checkpoint() -> Path:
    from acestep.paths import checkpoints_dir
    return checkpoints_dir() / "acestep-transcriber"


class Transcriber:
    def __init__(
        self,
        checkpoint: Union[str, Path, None] = None,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
    ) -> None:
        path = Path(checkpoint) if checkpoint is not None else default_checkpoint()
        if not path.exists():
            raise FileNotFoundError(
                f"transcriber checkpoint not found at {path}. Run "
                f"`acestep-download --model acestep-transcriber`."
            )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(str(path))
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            str(path),
            torch_dtype=dtype,
            device_map=device,
            attn_implementation=attn_implementation,
        )
        self.model.disable_talker()
        self.model.eval()

        # Remember the dtype + the "active" device the model was loaded
        # on so to_device() can restore that exact configuration on
        # restore. Useful when a caller parks the model on CPU between
        # presses then re-homes it to the original CUDA device.
        self._dtype = dtype
        self._home_device = device

        fe = self.processor.feature_extractor
        self.audio_sr: int = int(getattr(fe, "sampling_rate", 16000))

    def to_device(self, device: str) -> None:
        """Move the model's parameters to ``device``. Cheap CPU↔CUDA
        round-trip (no re-deserialization) — the caller is expected to
        ``torch.cuda.empty_cache()`` after the move if the goal is to
        return the freed VRAM to the OS / sibling models.

        ``self.model.device`` reads back the new device after the call,
        which the ``_run_inference`` path uses to route input tensors.
        """
        self.model = self.model.to(device)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    @torch.inference_mode()
    def transcribe(
        self,
        audio_path: Union[str, Path],
        *,
        max_new_tokens: int = 4096,
    ) -> str:
        audio, _ = librosa.load(str(audio_path), sr=self.audio_sr, mono=True)
        return self._run_inference(audio, str(audio_path), max_new_tokens)

    @torch.inference_mode()
    def transcribe_waveform(
        self,
        waveform: np.ndarray,
        sr: int,
        *,
        max_new_tokens: int = 4096,
    ) -> str:
        """In-memory variant for callers that already have decoded audio
        in hand (live demo source swaps, etc.). Mono down-mix and
        resampling to ``self.audio_sr`` happen here; the chat template
        uses a sentinel string in the ``audio`` slot since the processor
        reads the real audio from the parallel ``audio=[...]`` kwarg."""
        wf = np.asarray(waveform, dtype=np.float32)
        if wf.ndim > 1:
            # AudioEngine stores interleaved [N, C]; mean across channels.
            # Single-channel arrays of shape (1, N) collapse correctly too.
            axis = 1 if wf.shape[0] > wf.shape[1] else 0
            wf = wf.mean(axis=axis)
        if sr != self.audio_sr:
            wf = librosa.resample(wf, orig_sr=sr, target_sr=self.audio_sr)
        return self._run_inference(wf, "<live-source>", max_new_tokens)

    def _run_inference(
        self,
        audio: np.ndarray,
        audio_marker: str,
        max_new_tokens: int,
    ) -> str:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_marker},
                    {"type": "text", "text": _TASK_PROMPT},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text,
            audio=[audio],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        for k, v in inputs.items():
            if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                inputs[k] = v.to(self.model.dtype)

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_audio=False,
            do_sample=False,
        )
        prompt_len = inputs["input_ids"].shape[-1]
        generated = out_ids[:, prompt_len:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]
