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

        fe = self.processor.feature_extractor
        self.audio_sr: int = int(getattr(fe, "sampling_rate", 16000))

    @torch.inference_mode()
    def transcribe(
        self,
        audio_path: Union[str, Path],
        *,
        max_new_tokens: int = 4096,
    ) -> str:
        audio, _ = librosa.load(str(audio_path), sr=self.audio_sr, mono=True)

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
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
