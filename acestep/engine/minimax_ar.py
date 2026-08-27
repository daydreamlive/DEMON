"""MiniMaxAR: the composition-capture stage of MiniMax-Music3.

MiniMax-Music3's flow-matching renderer has no text input. Its only
conditioning is ``encoder_hidden_states``, derived from the per-frame
hidden states of an 8.58B Qwen3 autoregressive LM. DEMON never streams
that LM: it runs it *once* per composition, keeps the tensor, and covers
it forever. This module is that one run — prompt + lyrics + frame count
in, fused per-frame hidden states out.

The stage is two nested language models:

* the **Global LM** (Qwen3, 36 layers, hidden 4096) emits one semantic
  code ``c0`` per 25 Hz audio frame, autoregressively over a KV cache;
* the **RVQ depth decoder** (4 layers, hidden 4096, at most 16
  positions) then emits that frame's seven residual codebook codes
  ``c1..c7``, conditioned on the Global LM's hidden state.

Both run classifier-free guidance at scale 1.5 against a batch row whose
prompt tokens have been replaced by ``<|audio_cfg|>``, which is why every
tensor in the loop carries a leading dimension of 2.

The frame's contribution to the capture is the Global LM's hidden state
concatenated with the seven depth hidden states: ``8 * 4096 = 32768``
values per frame, which the ConditionEncoder later mixes down with
learned softmax weights. That is the *only* reason the depth decoder is
run at inference — its codes are never decoded to audio here; DEMON's
renderer consumes the hidden states, not the tokens.

Two things about this file are deliberate and load-bearing:

**It reimplements the depth decoder.** It is not a ``transformers``
architecture, and the only reference implementation lives in a
``diffusers`` version this repo cannot carry. Forty-seven safetensors
keys, four blocks, learned position embeddings, no RoPE. Reimplementing
it here costs less than a dependency.

**It normalizes the Qwen3 config.** ``language_model/config.json`` was
written by ``transformers`` 5.13.0.dev0 and the repo pins 4.57.x for
ACE-Step. See :func:`load_qwen3_config` — the failure mode this guards
against is silent, not loud.

Pure torch + transformers. No diffusers import.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

try:  # pragma: no cover - the CLI runs this module outside the server
    from acestep.engine.obs import logger
except Exception:  # pragma: no cover
    import logging as _logging

    class _Shim:
        _log = _logging.getLogger("minimax_ar")

        def _emit(self, level, msg, *args):
            self._log.log(level, msg.format(*args) if args else msg)

        def info(self, msg, *args):
            self._emit(_logging.INFO, msg, *args)

        def warning(self, msg, *args):
            self._emit(_logging.WARNING, msg, *args)

    logger = _Shim()


# ---------------------------------------------------------------------------
# Checkpoint contract
# ---------------------------------------------------------------------------
# These ids are not configuration. They are baked into the released weights
# and were verified against the shipped tokenizer: <|audio_cfg|> is 151654,
# <|audio_end|> is 151670, and 151675 is one past the end of the text vocab,
# where the 16384 semantic audio codes begin.

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
CAPTION_START, CAPTION_END = "<|caption_start|>", "<|caption_end|>"
LYRICS_START, LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
AUDIO_START = "<|audio_start|>"

AUDIO_END_TOKEN_ID = 151670
AUDIO_CFG_TOKEN_ID = 151654
AUDIO_CODE_OFFSET = 151675
SEMANTIC_VOCAB_SIZE = 16384

MAX_PROMPT_TOKENS = 5_000
MAX_AUDIO_FRAMES = 9_000
AR_FRAME_RATE_HZ = 25.0

# Fixed by the reference inference recipe, not tunable.
AR_CFG_SCALE = 1.5
AR_CFG_TOP_K = 50
AR_SAMPLING_TOP_K = 50

# Upstream's reference server exposes `seed` with a default of 0 — omitting
# it there means zero, not "random". Match that: a capture is an artifact and
# an artifact that cannot be reproduced is a liability.
DEFAULT_SEED = 0

FUSED_HIDDEN_DIM = 8 * 4096

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------
# Whitespace here is a checkpoint contract. Changing it changes the audio.


def clean_caption(caption: str) -> str:
    """Strip the markdown forms the checkpoint's input contract accepts.

    Also rewrites stray ``<|key value|>`` tags into ``key is value`` prose:
    captions written against the model's tag vocabulary would otherwise
    tokenize into unknown-special-token territory.
    """

    def _rewrite_special_tag(match: "re.Match[str]") -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)
    lines_out = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def normalize_lyrics(lyrics: str) -> str:
    """Put every structure tag on its own line and prepend ``[start]``.

    Text sharing a line with a leading ``[verse]``-style tag is DROPPED,
    not moved — that is the checkpoint's contract, and it is a good way to
    lose a whole line of lyrics without noticing.
    """
    output = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    text = "\n".join(output)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def build_prompt_text(prompt: str, lyrics: str) -> str:
    """The exact string the Global LM is prefilled with."""
    return (
        f"{IM_START}{CAPTION_START}{clean_caption(prompt)}{CAPTION_END}"
        f"{LYRICS_START}{normalize_lyrics(lyrics)}{LYRICS_END}{IM_END}{AUDIO_START}"
    )


# ---------------------------------------------------------------------------
# transformers-version shim
# ---------------------------------------------------------------------------

# What each v5 key is remapped to, and why. Kept as data so the migration is
# auditable and so the shim can no-op itself once the pin moves forward.
_V5_CONFIG_REMAPS = {
    "rope_parameters": (
        "rope_theta / rope_scaling",
        "v5 folded RoPE settings into a nested dict; 4.57 reads flat keys, "
        "and — the dangerous part — silently falls back to rope_theta=10000 "
        "instead of the checkpoint's 1000000, which changes every position "
        "encoding in the model without raising anything.",
    ),
    "dtype": (
        "torch_dtype",
        "v5 renamed the field. 4.57.x already accepts `dtype`, so this is a "
        "no-op on the current pin and a safety net on older ones.",
    ),
    "transformers_version": (
        "(dropped)",
        "provenance only; never read at construction time.",
    ),
}


def _accepted_config_keys() -> frozenset:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    return frozenset(inspect.signature(Qwen3Config.__init__).parameters)


def load_qwen3_config(config_dir: Path):
    """Build a ``Qwen3Config`` from a config.json written by transformers v5.

    ``AutoConfig.from_pretrained`` does not fail on this file. That is the
    problem. Unknown keys land in ``**kwargs`` and are stashed as inert
    attributes, so ``rope_parameters={"rope_theta": 1000000}`` is accepted
    and ``rope_theta`` quietly keeps its 4.x default of 10000.0. The model
    then loads, runs, and produces confidently wrong audio.

    So: read the JSON ourselves, remap the v5 keys onto whatever the
    installed version actually accepts, and assert the result afterwards.
    Every remap is a no-op on a version new enough to understand the key,
    which makes this shim self-retiring.
    """
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    raw = json.loads((Path(config_dir) / "config.json").read_text(encoding="utf-8"))
    accepted = _accepted_config_keys()
    applied = []

    kwargs = dict(raw)
    kwargs.pop("transformers_version", None)

    rope = kwargs.pop("rope_parameters", None)
    if rope is not None:
        if "rope_parameters" in accepted:
            kwargs["rope_parameters"] = rope
        else:
            rope = dict(rope)
            rope_type = rope.pop("rope_type", "default")
            if "rope_theta" in rope:
                kwargs["rope_theta"] = float(rope.pop("rope_theta"))
            # "default" means no scaling; anything else has to survive as a
            # rope_scaling dict or the positions come out wrong.
            if rope_type not in ("default", None) or rope:
                kwargs["rope_scaling"] = {"rope_type": rope_type, **rope}
            applied.append("rope_parameters")

    if "dtype" in kwargs and "dtype" not in accepted:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        applied.append("dtype")

    # layer_types landed in 4.55; on anything older it is inert but harmless,
    # since every entry in this checkpoint is "full_attention" anyway.
    if "layer_types" in kwargs and "layer_types" not in accepted:
        kwargs.pop("layer_types")
        applied.append("layer_types")

    config = Qwen3Config(**kwargs)

    # Assert the remap actually took. A shim that silently fails is worse
    # than no shim, because it looks like it worked.
    expected_theta = float((raw.get("rope_parameters") or {}).get("rope_theta", 0) or 0)
    if expected_theta:
        got = getattr(config, "rope_theta", None)
        if got is None and getattr(config, "rope_parameters", None):
            got = config.rope_parameters.get("rope_theta")
        if got is None or abs(float(got) - expected_theta) > 1e-6:
            raise RuntimeError(
                "MiniMax Qwen3 config normalization failed: rope_theta resolved "
                f"to {got!r}, expected {expected_theta!r}. The installed "
                f"transformers ({_transformers_version()}) changed its RoPE "
                "config surface again; update _V5_CONFIG_REMAPS."
            )

    if applied:
        logger.info(
            "minimax_ar_config_shim transformers={} remapped={}",
            _transformers_version(),
            ",".join(applied),
        )
    return config


def _transformers_version() -> str:
    import transformers

    return transformers.__version__


# ---------------------------------------------------------------------------
# RVQ depth decoder (reimplemented; see module docstring)
# ---------------------------------------------------------------------------


class _DepthRMSNorm(nn.Module):
    """Matches the reference norm exactly: variance in fp32, then the
    normalized activation is cast back to the weight dtype *before* the
    affine multiply. Doing the affine in fp32 instead drifts."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states.to(self.weight.dtype) * self.weight


class _DepthAttention(nn.Module):
    """Causal self-attention over at most 8 positions. No RoPE, no q/k norm,
    no bias — position comes from a learned table on the way in."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        shape = (batch, seq, self.heads, self.head_dim)
        query = self.to_q(hidden_states).view(shape).transpose(1, 2)
        key = self.to_k(hidden_states).view(shape).transpose(1, 2)
        value = self.to_v(hidden_states).view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        out = out.transpose(1, 2).flatten(2, 3).to(query.dtype)
        return self.to_out(out)


class _DepthBlock(nn.Module):
    def __init__(self, dim: int, heads: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = _DepthRMSNorm(dim)
        self.attn = _DepthAttention(dim, heads)
        self.post_attention_layernorm = _DepthRMSNorm(dim)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.input_layernorm(hidden_states))
        norm_states = self.post_attention_layernorm(hidden_states)
        gated = F.silu(self.gate_proj(norm_states)) * self.up_proj(norm_states)
        return hidden_states + self.down_proj(gated)


class MiniMaxRVQDepthDecoder(nn.Module):
    """The local language model: within one audio frame it predicts codebooks
    ``c1..c7`` from the Global LM's hidden state and the frame's semantic
    code, and exposes the per-step hidden states that condition the renderer.

    It also owns the residual-codebook embedding table, which the AR loop
    needs to embed a completed frame for the Global LM's feedback step.

    Parameter names mirror the checkpoint's safetensors keys one-to-one, so
    the state dict loads ``strict=True`` with no remapping.
    """

    _SUBDIR = "rvq_depth_decoder"
    _CONFIG_KEYS = (
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "intermediate_size",
        "audio_vocab_size",
        "num_codebooks",
        "max_position_embeddings",
    )

    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 4,
        num_attention_heads: int = 16,
        intermediate_size: int = 6144,
        audio_vocab_size: int = 1024,
        num_codebooks: int = 8,
        max_position_embeddings: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.audio_vocab_size = audio_vocab_size
        self.num_codebooks = num_codebooks
        self.max_position_embeddings = max_position_embeddings

        self.audio_embeddings = nn.Embedding(
            audio_vocab_size * (num_codebooks - 1), hidden_size
        )
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList(
            [
                _DepthBlock(hidden_size, num_attention_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )
        self.norm = _DepthRMSNorm(hidden_size)
        self.audio_heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, audio_vocab_size, bias=False)
                for _ in range(num_codebooks - 1)
            ]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """``(batch, steps, hidden)`` -> ``(batch, steps, hidden)``."""
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.pos_embedding(positions).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)

    @classmethod
    def from_pretrained(
        cls, model_dir, *, dtype: torch.dtype = torch.bfloat16, device="cpu"
    ) -> "MiniMaxRVQDepthDecoder":
        directory = Path(model_dir)
        if (directory / cls._SUBDIR / "config.json").is_file():
            directory = directory / cls._SUBDIR
        raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        config = {k: raw[k] for k in cls._CONFIG_KEYS if k in raw}
        state = load_file(str(directory / "diffusion_pytorch_model.safetensors"))
        # Meta-init then assign: never materialize a second copy of the table.
        with torch.device("meta"):
            model = cls(**config)
        model.load_state_dict(state, strict=True, assign=True)
        model.to(device=device, dtype=dtype)
        model.eval()
        model.requires_grad_(False)
        return model


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _resolve_device(device) -> torch.device:
    """``"cuda"`` -> ``cuda:0``, so device equality actually works."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _sample_top_k(
    logits: torch.Tensor, generator: Optional[torch.Generator]
) -> torch.Tensor:
    """Top-k multinomial, reproducing the reference's numerics exactly —
    including its choice to map ``-inf`` to ``-1e9`` before the top-k so a
    fully masked row degrades to a uniform pick instead of a NaN."""
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_k = min(AR_SAMPLING_TOP_K, values.shape[-1])
    threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sample_device = generator.device if generator is not None else probs.device
    drawn = torch.multinomial(probs.to(sample_device), 1, generator=generator)
    return drawn.squeeze(-1).to(probs.device)


# ---------------------------------------------------------------------------
# MiniMaxAR
# ---------------------------------------------------------------------------


class MiniMaxAR:
    """The composition-capture stage. Prompt + lyrics + frames -> fused
    per-frame hidden states of shape ``[1, frames, 32768]``.

    Deliberately not an ``nn.Module``: this object owns a tokenizer and a
    KV-cache policy alongside two models, and DEMON pages the whole thing
    between host and device around each capture. :meth:`to` is the seam
    that makes that paging one call.
    """

    frame_rate = AR_FRAME_RATE_HZ

    def __init__(
        self,
        language_model,
        tokenizer,
        depth_decoder: MiniMaxRVQDepthDecoder,
        *,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = DEFAULT_SEED,
        sample_on_cpu: bool = False,
    ):
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.depth_decoder = depth_decoder
        self.dtype = dtype
        self.seed = int(seed)
        # A CPU generator makes a capture reproducible across devices (the
        # diffusers convention) at the cost of a device sync per sampled
        # code — eight per frame. Off by default; the capture is an
        # artifact on disk, so cross-device determinism is not what makes
        # it reusable.
        self.sample_on_cpu = bool(sample_on_cpu)
        self._vocab_mask: Optional[torch.Tensor] = None
        self.last_stats: Dict[str, float] = {}

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        root: Path,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device="cpu",
        seed: int = DEFAULT_SEED,
        sample_on_cpu: bool = False,
    ) -> "MiniMaxAR":
        """Load the AR stack from a diffusers-layout checkpoint directory.

        ``root/language_model``, ``root/tokenizer``, ``root/rvq_depth_decoder``.
        Weights land on the CPU first and are moved once, so loading with
        ``device="cpu"`` (DEMON's offload policy) never touches the GPU.
        """
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        root = Path(root)
        lm_dir = root / "language_model"
        tok_dir = root / "tokenizer"
        for path in (lm_dir, tok_dir, root / MiniMaxRVQDepthDecoder._SUBDIR):
            if not path.is_dir():
                raise FileNotFoundError(
                    f"MiniMax AR stage incomplete: {path} is missing. The "
                    "renderer can still stream from a saved capture."
                )

        config = load_qwen3_config(lm_dir)
        started = time.perf_counter()
        language_model = Qwen3ForCausalLM.from_pretrained(
            str(lm_dir),
            config=config,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        language_model.eval()
        language_model.requires_grad_(False)

        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        depth_decoder = MiniMaxRVQDepthDecoder.from_pretrained(
            root, dtype=dtype, device="cpu"
        )
        logger.info(
            "minimax_ar_loaded root={} dtype={} seconds={:.1f}",
            root,
            dtype,
            time.perf_counter() - started,
        )

        model = cls(
            language_model,
            tokenizer,
            depth_decoder,
            dtype=dtype,
            seed=seed,
            sample_on_cpu=sample_on_cpu,
        )
        return model.to(device)

    # ---- placement ----------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.language_model.parameters()).device

    def to(self, device) -> "MiniMaxAR":
        """Page the whole stack. Both models must land together — every frame
        touches both, so a split placement would thrash across PCIe 400 times
        a second."""
        # torch.device("cuda") carries no index but parameters always report
        # one, so compare resolved devices or the early-out never fires and
        # every page re-walks 400 tensors for nothing.
        device = _resolve_device(device)
        if self.device == device:
            return self
        self.language_model.to(device)
        self.depth_decoder.to(device)
        # The vocab mask is the only persistent buffer, and it is cheap
        # enough (200k bools) to simply follow rather than re-derive.
        if self._vocab_mask is not None:
            self._vocab_mask = self._vocab_mask.to(device)
        return self

    def cuda(self) -> "MiniMaxAR":
        return self.to("cuda")

    def cpu(self) -> "MiniMaxAR":
        return self.to("cpu")

    # ---- prompt -------------------------------------------------------------

    def tokenize(self, prompt: str, lyrics: str) -> torch.Tensor:
        """``[2, L]``: row 0 conditional, row 1 the classifier-free twin.

        Row 1 keeps only the first token and the trailing two structure
        tokens (``<|im_end|><|audio_start|>``); everything between becomes
        ``<|audio_cfg|>``. The CFG branch therefore still knows it is being
        asked for audio, it just does not know for what.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"`prompt` (the music description) must be a non-empty string, got {prompt!r}"
            )
        if not isinstance(lyrics, str) or not lyrics.strip():
            raise ValueError(f"`lyrics` must be a non-empty string, got {lyrics!r}")

        text = build_prompt_text(prompt, lyrics)
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"The assembled prompt has {input_ids.shape[1]} tokens; the "
                f"maximum is {MAX_PROMPT_TOKENS}"
            )
        unconditional_ids = input_ids.clone()
        unconditional_ids[:, 1:-2] = AUDIO_CFG_TOKEN_ID
        return torch.cat((input_ids, unconditional_ids), dim=0).to(self.device)

    def _vocab_mask_for(self, device: torch.device) -> torch.Tensor:
        """``True`` where sampling is forbidden: everything except the 16384
        semantic codes and the end token."""
        mask = self._vocab_mask
        if mask is None or mask.device != device:
            size = self.language_model.config.vocab_size
            mask = torch.ones(size, dtype=torch.bool, device=device)
            mask[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE] = False
            mask[AUDIO_END_TOKEN_ID] = False
            self._vocab_mask = mask
        return mask

    # ---- inner loops --------------------------------------------------------

    def _embed_audio_frame(self, frame_codes: torch.Tensor) -> torch.Tensor:
        """``[2, 8]`` codes -> ``[2, 1, hidden]`` feedback embedding.

        The semantic code is embedded from the Global LM's own token table;
        the seven residual codes come from the depth decoder's table, each
        offset into its own 1024-wide slice, summed. The ``K**-0.5`` keeps
        the sum's scale where the LM expects a single embedding.
        """
        depth = self.depth_decoder
        num_codebooks = depth.num_codebooks
        embeds = self.language_model.model.embed_tokens(
            frame_codes[:, :1] + AUDIO_CODE_OFFSET
        )
        offsets = (
            torch.arange(num_codebooks - 1, device=frame_codes.device)
            * depth.audio_vocab_size
        ).unsqueeze(0)
        extra = depth.audio_embeddings(frame_codes[:, 1:] + offsets).sum(
            dim=1, keepdim=True
        )
        embeds = embeds + extra.to(embeds.dtype)
        return embeds * num_codebooks**-0.5

    def _generate_depth_codes(
        self,
        last_hidden: torch.Tensor,
        semantic_code: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ``c1..c7`` for one frame; return ``(codes, hidden_parts)``.

        No KV cache here on purpose. The sequence is at most eight steps, so
        caching would save a handful of tiny GEMMs while the seven forwards
        still have to stream all 1.1 GB of depth weights either way — this
        loop is bandwidth- and launch-bound, not attention-bound.
        """
        depth = self.depth_decoder
        num_codebooks = depth.num_codebooks

        sequence = [depth.projection(last_hidden).unsqueeze(1)]
        code_embed = self.language_model.model.embed_tokens(
            semantic_code + AUDIO_CODE_OFFSET
        )
        sequence.append(depth.projection(code_embed).unsqueeze(1))

        codes = [semantic_code]
        hidden_parts = []
        for index in range(1, num_codebooks):
            hidden = depth(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden[:1])
            logits = depth.audio_heads[index - 1](hidden)
            conditional, unconditional = logits[:1].float(), logits[1:2].float()
            logits = unconditional + (conditional - unconditional) * AR_CFG_SCALE
            # Repeat the sampled code across both rows: the CFG twin must see
            # the same audio history, only a different text prefix.
            code = _sample_top_k(logits, generator).repeat(2)
            codes.append(code)
            if index < num_codebooks - 1:
                embed = depth.audio_embeddings(
                    code + (index - 1) * depth.audio_vocab_size
                )
                sequence.append(depth.projection(embed).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    # ---- capture ------------------------------------------------------------

    @torch.no_grad()
    def generate_frame_hiddens(
        self,
        *,
        prompt: str,
        lyrics: str,
        frames: int,
        seed: Optional[int] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        """Run the AR stage. Returns ``[1, frames, 8 * 4096]``.

        ``frames`` is an upper bound at 25 Hz, capped at 9000 (six minutes).
        The LM may stop earlier by emitting ``<|audio_end|>``, in which case
        the returned tensor is shorter than requested; the renderer takes
        its length from this tensor, so a short capture is a short song, not
        an error.
        """
        frames = int(frames)
        if frames < 1:
            raise ValueError(f"`frames` must be at least 1, got {frames}")
        max_frames = min(frames, MAX_AUDIO_FRAMES)
        if frames > MAX_AUDIO_FRAMES:
            logger.warning(
                "minimax_ar_frames_capped requested={} cap={}", frames, MAX_AUDIO_FRAMES
            )

        language_model = self.language_model
        depth = self.depth_decoder
        device = self.device
        seed = self.seed if seed is None else int(seed)

        text_ids = self.tokenize(prompt, lyrics)
        prompt_tokens = int(text_ids.shape[1])

        # RoPE does not clip, it extrapolates — running past the trained
        # window degrades quality without failing, so say so out loud.
        window = int(getattr(language_model.config, "max_position_embeddings", 0) or 0)
        if window and prompt_tokens + max_frames > window:
            logger.warning(
                "minimax_ar_over_window prompt={} frames={} window={}",
                prompt_tokens,
                max_frames,
                window,
            )

        gen_device = torch.device("cpu") if self.sample_on_cpu else device
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        started = time.perf_counter()

        # Prefill. The reference embeds by hand rather than passing input_ids
        # so the prefill and the per-frame feedback take the identical path.
        text_embeds = language_model.model.embed_tokens(text_ids)
        output = language_model.model(inputs_embeds=text_embeds, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]

        vocab_mask = self._vocab_mask_for(device)
        frame_hiddens = []
        stopped_early = False

        # The first decode step only advances past <|audio_start|>; its
        # hidden state describes the prompt, not a frame, so it is consumed
        # for feedback and thrown away. Hence max_frames + 1 iterations.
        for frame_index in range(max_frames + 1):
            logits = language_model.lm_head(last_hidden).float()
            logits = logits.masked_fill(vocab_mask, -float("inf"))
            conditional, unconditional = logits[0:1], logits[1:2]
            guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
            # Guidance on two -inf logits is NaN, so restrict to the
            # conditional branch's top candidates and then re-mask.
            threshold = torch.topk(conditional, AR_CFG_TOP_K, dim=-1).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
            sampled = _sample_top_k(guided, generator)
            if int(sampled.item()) == AUDIO_END_TOKEN_ID:
                stopped_early = True
                break

            semantic_code = sampled - AUDIO_CODE_OFFSET
            frame_codes, depth_hidden = self._generate_depth_codes(
                last_hidden, semantic_code.repeat(2), generator
            )
            if frame_index > 0:
                frame_hiddens.append(
                    torch.cat((last_hidden[:1], depth_hidden), dim=-1)
                )
                if progress is not None:
                    progress(len(frame_hiddens), max_frames)
                if len(frame_hiddens) >= max_frames:
                    break

            feedback = self._embed_audio_frame(frame_codes)
            output = language_model.model(
                inputs_embeds=feedback,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]

        if not frame_hiddens:
            raise ValueError(
                "MiniMax Music 3 generated zero audio frames; the prompt ended "
                "generation immediately"
            )

        fused = torch.stack(frame_hiddens, dim=1)
        # One hidden state from the Global LM plus one per residual codebook
        # step: 8 * 4096 = 32768 on the released checkpoint. Derived, not
        # hardcoded, so the guard still means something on a stub stack.
        expected_width = depth.num_codebooks * depth.hidden_size
        if fused.shape[-1] != expected_width:
            raise RuntimeError(
                f"fused hidden width {fused.shape[-1]}, expected {expected_width}"
            )

        elapsed = time.perf_counter() - started
        produced = int(fused.shape[1])
        self.last_stats = {
            "frames": produced,
            "requested_frames": max_frames,
            "prompt_tokens": prompt_tokens,
            "seconds": elapsed,
            # One Global-LM step per frame; eight sampled codes per frame
            # (one semantic plus seven residual).
            "lm_tokens_per_s": produced / elapsed if elapsed else 0.0,
            "codes_per_s": produced * depth.num_codebooks / elapsed if elapsed else 0.0,
            "audio_seconds": produced / self.frame_rate,
            "realtime_factor": (produced / self.frame_rate) / elapsed if elapsed else 0.0,
            "stopped_early": stopped_early,
            "seed": seed,
        }
        logger.info(
            "minimax_ar_capture frames={} seconds={:.2f} lm_tok_s={:.1f} rtf={:.3f} early={}",
            produced,
            elapsed,
            self.last_stats["lm_tokens_per_s"],
            self.last_stats["realtime_factor"],
            stopped_early,
        )
        return fused

    def generate_for_duration(
        self, *, prompt: str, lyrics: str, seconds: float, **kwargs
    ) -> torch.Tensor:
        """Convenience wrapper: seconds of audio at 25 Hz."""
        if seconds <= 0:
            raise ValueError(f"`seconds` must be positive, got {seconds}")
        frames = int(round(seconds * self.frame_rate))
        if frames < 1:
            raise ValueError(
                f"`seconds` {seconds} is shorter than one frame "
                f"(1 / {self.frame_rate} s)"
            )
        return self.generate_frame_hiddens(
            prompt=prompt, lyrics=lyrics, frames=frames, **kwargs
        )
