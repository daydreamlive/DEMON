"""5Hz LM semantic-hint generation.

Loads ACE-Step's Qwen3-based 5Hz LM, generates audio-code tokens
(``<|audio_code_N|>``) from a tags+lyrics+metadata prompt, and exposes
the resulting integer codes. Pair with
:meth:`acestep.engine.session.Session.generate_lm_hints` to dequantize
those codes through the DiT's FSQ codebook into a 25Hz hint latent
suitable for the decoder's ``context_latent``.

Single-pass design: rather than running the upstream two-phase
(CoT then codes) pipeline, we synthesize the ``<think>...</think>``
metadata block directly from BPM/key/duration we already know, so we
only need ONE forward pass through the LM. Quality may differ slightly
from the upstream two-phase generation; the trade-off is ~halving
latency and avoiding a second model loop.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


_LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"
_AUDIO_CODE_RE = re.compile(r"<\|audio_code_(\d+)\|>")


def default_lm_checkpoint() -> Path:
    from acestep.paths import checkpoints_dir
    return checkpoints_dir() / "acestep-5Hz-lm-1.7B"


def _build_cot_block(
    *, caption: str, bpm: int, key: str, duration: float,
    language: str, time_signature: str,
) -> str:
    """Synthesize the upstream CoT metadata block from known values.

    Format mirrors upstream's parse_lm_output expectation:
    ``<think>\\nbpm: ...\\ncaption: ...\\nduration: ...\\ngenres: ...\\n
    keyscale: ...\\nlanguage: ...\\ntimesignature: ...\\n</think>``.
    """
    return (
        "<think>\n"
        f"bpm: {int(bpm)}\n"
        f"caption: {caption}\n"
        f"duration: {int(round(duration))}\n"
        f"genres: {caption}\n"
        f"keyscale: {key}\n"
        f"language: {language}\n"
        f"timesignature: {time_signature}\n"
        "</think>"
    )


class LMHintGenerator:
    """Wraps the 5Hz LM. Loads on construction; generate() is reentrant."""

    def __init__(
        self,
        checkpoint: Union[str, Path, None] = None,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
    ) -> None:
        path = Path(checkpoint) if checkpoint is not None else default_lm_checkpoint()
        if not path.exists():
            raise FileNotFoundError(
                f"5Hz LM checkpoint not found at {path}. Run "
                f"`acestep-download --model acestep-5Hz-lm-1.7B`."
            )

        logger.info(f"Loading 5Hz LM from {path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        # Config lists architectures=['Qwen3Model'] but tie_word_embeddings=True,
        # so AutoModelForCausalLM auto-attaches a tied lm_head.
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path),
            torch_dtype=dtype,
            device_map=device,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.device = self.model.device
        self.dtype = dtype

    def _build_prompt(
        self, *, caption: str, lyrics: str, bpm: int, key: str, duration: float,
        language: str, time_signature: str,
    ) -> str:
        cot = _build_cot_block(
            caption=caption, bpm=bpm, key=key, duration=duration,
            language=language, time_signature=time_signature,
        )
        user = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
        formatted = self.tokenizer.apply_chat_template(
            [
                {"role": "system",
                 "content": f"# Instruction\n{_LM_INSTRUCTION}\n\n"},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Keep the assistant turn open and append the synthesized CoT block
        # so the LM continues with audio codes (matches training layout).
        formatted += cot + "\n\n"
        return formatted

    @torch.inference_mode()
    def generate_codes(
        self,
        *,
        tags: str,
        lyrics: str = "",
        bpm: int = 120,
        key: str = "C major",
        duration: float = 60.0,
        language: str = "en",
        time_signature: str = "4",
        temperature: float = 0.85,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate audio codes for ``duration`` seconds at 5Hz.

        Returns a ``[1, T_5Hz]`` long tensor of integer codes, where
        ``T_5Hz = int(round(duration * 5))``. If the LM emits fewer
        codes than expected, the tail is padded with the last code (or
        0 if none were emitted).
        """
        target_codes = int(round(duration * 5))
        if target_codes <= 0:
            raise ValueError(f"duration {duration} produced target_codes={target_codes}")

        prompt = self._build_prompt(
            caption=tags, lyrics=lyrics, bpm=bpm, key=key,
            duration=duration, language=language,
            time_signature=time_signature,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        if seed is not None:
            torch.manual_seed(int(seed))

        gen_kwargs = dict(
            max_new_tokens=target_codes + 32,  # small safety buffer
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-5),
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if top_p is not None and 0.0 < top_p < 1.0:
            gen_kwargs["top_p"] = top_p

        out_ids = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = out_ids[0, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

        codes = [int(m.group(1)) for m in _AUDIO_CODE_RE.finditer(text)]
        if len(codes) > target_codes:
            codes = codes[:target_codes]
        elif len(codes) < target_codes:
            pad = codes[-1] if codes else 0
            logger.warning(
                f"[LMHints] LM emitted {len(codes)} codes, "
                f"padding to {target_codes} with code {pad}"
            )
            codes = codes + [pad] * (target_codes - len(codes))

        return torch.tensor(codes, dtype=torch.long, device=self.device).unsqueeze(0)

    def free(self) -> None:
        """Drop weights from VRAM. Object is unusable afterward."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
