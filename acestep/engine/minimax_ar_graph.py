"""The AR stage as one CUDA graph per frame.

Why this exists
---------------
The plain torch loop in :class:`~acestep.engine.minimax_ar.MiniMaxARStream`
is dispatch-bound: ``minimax_ar_bench.py --profile`` puts 22 ms of GPU
kernels inside a 52 ms frame, the other 30 ms being Python launching
~3900 kernels one at a time. A CUDA graph replays all of them from one
call. What a graph needs in return is fixed shapes and fixed addresses,
which the growing ``DynamicCache`` cannot give, so the session runs over
a ``StaticCache`` preallocated to its whole length.

Two things had to be true for that to pay, and both were measured
(``scripts/minimax/minimax_graph_spike.py``, 5090, bf16):

* **The static cache must not cost what the graph saves.** HF's sdpa
  path with a static cache spends ~3.7 ms per 1024 padded slots per
  frame: with a mask present it cannot use SDPA's native GQA, so
  ``repeat_kv`` materialises the padded K/V four times over in every
  layer, then a mask-bearing kernel reads all of it. SDPA's efficient
  kernel on the un-expanded K/V is no better at q_len 1 (0.64 ms per
  layer at 9600 slots against a 0.044 ms bandwidth floor): one CTA per
  (batch, head) walks every slot. :func:`decode_attention` folds the four
  query heads that share a KV head into the query's row axis and does
  the two matmuls as ``bmm`` -- 0.072 ms per layer, so the full 9600-slot
  cache costs ~3 ms per frame instead of ~23.
* **The result must match the dynamic path.** Teacher-forcing the
  graphed session's codes back through the dynamic cache and comparing
  LM hidden states: worst cos 0.99983 over 48 frames. HF's own eager
  attention against its sdpa on the same dynamic path: 0.99967. The
  graphed path is inside the noise of a kernel swap.

Measured: 25.7 ms per frame at the full 9600-slot cache, **1.56x
realtime**, flat in context length by construction. The dynamic path is
52 ms at the start of a piece and *not* flat: its LM forward alone goes
36.7 -> 58.6 -> 77.7 ms at 0 / 6000 / 9000 frames of history.

What changes, what does not
---------------------------
Sampling controls become device scalars, so a knob move is a buffer
write and never a recapture; ``top_k`` uses a sort-derived threshold
for the same reason. Temperature divides unconditionally (dividing by
exactly 1.0 is exact). The end-of-audio check is a flag read after the
frame rather than a ``.item()`` inside it. Everything else -- prompt
layout, CFG twin, depth-decoder order, feedback embedding, re-prompt
replay -- is the reference loop's.

Not preserved: the exact random draws. A captured ``multinomial`` runs
on graph-owned philox offsets, so a seed reproduces a graphed session
against another graphed session, not against the dynamic path. Captures
and parity fixtures written by the dynamic path stay valid; they are
produced by :meth:`MiniMaxAR.generate_frame_hiddens`, which does not use
this class.
"""

from __future__ import annotations

import contextlib
import time
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AttentionInterface, StaticCache
from transformers.masking_utils import AttentionMaskInterface

from acestep.engine.minimax_ar import (
    AR_CFG_TOP_K,
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    MAX_AUDIO_FRAMES,
    SEMANTIC_VOCAB_SIZE,
    ARControls,
    MiniMaxARStream,
)

__all__ = ["GraphedARStream", "decode_attention", "DECODE_ATTENTION"]

DECODE_ATTENTION = "demon_minimax_decode"
# The attention function receives the cache position under this kwarg.
# HF forwards unknown kwargs from `model.forward` through every layer
# to the attention interface, which keeps the function free of state.
POS_KWARG = "demon_pos"
# Cache slots reserved ahead of the audio for the text prefix. The
# prompt can change on a re-prompt, so this is a ceiling, not a fit.
PROMPT_SLOTS = 512
# Eager frames before capture, on a side stream, per torch's recipe:
# lazily-initialised kernels and workspaces must not land in the graph.
WARMUP_FRAMES = 2


# ---- decode attention ---------------------------------------------------------


def decode_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
):
    """Grouped-query attention over a static cache, at bandwidth.

    ``query`` is ``[B, Hq, Lq, D]``; ``key`` / ``value`` are the whole
    cache, ``[B, Hkv, S, D]``. The ``Hq / Hkv`` query heads that share a
    KV head are folded into the row axis so no head is repeated, scores
    come out of a bf16 x bf16 ``bmm`` as fp32 (no rounding before the
    softmax), and probabilities go back to bf16 for the value ``bmm``,
    which is where flash rounds too. Slot ``s`` is visible to the query
    at absolute position ``p`` iff ``s <= p``; positions come from
    ``kwargs[POS_KWARG]`` (``[1]``, the first query's position) on
    device, so the function is valid inside a graph.
    """
    pos = kwargs[POS_KWARG]
    batch, q_heads, q_len, dim = query.shape
    kv_heads, slots = key.shape[1], key.shape[2]
    group = q_heads // kv_heads
    rows = group * q_len
    if scaling is None:
        scaling = dim**-0.5

    q_pos = pos + torch.arange(q_len, device=query.device)
    visible = torch.arange(slots, device=query.device)[None, :] <= q_pos[:, None]
    # Row index within a KV head is (g, l) flattened, matching the
    # `repeat_kv` head order: q head h belongs to kv head h // group.
    mask = visible.unsqueeze(0).expand(group, q_len, slots).reshape(1, rows, slots)

    grouped = query.reshape(batch * kv_heads, rows, dim)
    keys = key.reshape(batch * kv_heads, slots, dim)
    values = value.reshape(batch * kv_heads, slots, dim)
    if grouped.is_cuda:
        scores = torch.bmm(grouped, keys.transpose(1, 2), out_dtype=torch.float32)
    else:  # `out_dtype` is CUDA-only; the CPU path exists for tests
        scores = torch.bmm(grouped.float(), keys.float().transpose(1, 2))
    scores = (scores * scaling).masked_fill(~mask, -float("inf"))
    probs = torch.softmax(scores, dim=-1).to(values.dtype)
    out = torch.bmm(probs, values)
    out = out.reshape(batch, q_heads, q_len, dim).transpose(1, 2).contiguous()
    return out, None


def _no_mask(*args, **kwargs):
    return None


AttentionInterface.register(DECODE_ATTENTION, decode_attention)
AttentionMaskInterface.register(DECODE_ATTENTION, _no_mask)


@contextlib.contextmanager
def _attention(lm, name: str):
    """Select the LM's attention function for the duration of a call.
    The model is shared with the capture path, so it is never left
    pointing at the decode function."""
    previous = lm.config._attn_implementation
    lm.config._attn_implementation = name
    try:
        yield
    finally:
        lm.config._attn_implementation = previous


# ---- the session --------------------------------------------------------------


class GraphedARStream(MiniMaxARStream):
    """:class:`MiniMaxARStream` with the frame captured as a CUDA graph.

    Same interface, same numerics up to kernel noise (see module doc),
    2x the frame rate. Opened through ``MiniMaxAR.stream(graph=True)``.

    Frame ``k``'s graph reads the feedback embedding frame ``k-1`` left
    in a static buffer, advances the LM one slot, samples the semantic
    code and the seven residuals, and writes the renderer's hidden, the
    frame's codes, the end flag, and frame ``k``'s feedback embedding
    into their buffers. The Python side then reads one flag, clones two
    small tensors, and replays again. Capture happens lazily after
    :data:`WARMUP_FRAMES` eager frames, which are real frames of the
    piece, not throwaways.
    """

    def __init__(
        self,
        ar,
        *,
        prompt: str,
        lyrics: str,
        seed: int = 0,
        max_frames: int = MAX_AUDIO_FRAMES,
        controls: Optional[ARControls] = None,
    ):
        device = ar.device
        if device.type != "cuda":
            raise ValueError("GraphedARStream needs the AR stage on a CUDA device")
        if ar.sample_on_cpu:
            raise ValueError(
                "GraphedARStream samples on device; `sample_on_cpu` is the "
                "capture path's option, not a streaming one"
            )
        lm = ar.language_model
        hidden = int(lm.config.hidden_size)
        codebooks = int(ar.depth_decoder.num_codebooks)
        initial = controls or ARControls()

        frames = max(1, min(int(max_frames), MAX_AUDIO_FRAMES))
        # Prefix, one slot per frame written (the warm-up frame's
        # feedback included), and a little slack.
        self._slots = PROMPT_SLOTS + frames + 8
        self._cache = StaticCache(config=lm.config, max_cache_len=self._slots)
        self._feedback = torch.zeros(2, 1, hidden, dtype=ar.dtype, device=device)
        self._pos = torch.zeros(1, dtype=torch.long, device=device)
        self._pos_host = 0
        self._guidance = torch.tensor(float(initial.guidance), device=device)
        self._temperature = torch.tensor(float(initial.temperature), device=device)
        self._top_k = torch.tensor([int(initial.top_k)], dtype=torch.long, device=device)
        self._mask_end = torch.tensor([bool(initial.mask_end)], device=device)
        self._codes_out = torch.zeros(2, codebooks, dtype=torch.long, device=device)
        self._hidden_out = torch.zeros(1, codebooks * hidden, dtype=ar.dtype, device=device)
        self._end = torch.zeros(1, dtype=torch.bool, device=device)
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._eager_frames = 0
        # True while the buffers hold a frame sampled eagerly by a
        # prefill (the warm-up frame, or the first frame after a
        # re-prompt) that `_step` has not consumed yet.
        self._ready = False
        self.capture_s = 0.0

        super().__init__(
            ar, prompt=prompt, lyrics=lyrics, seed=seed,
            max_frames=frames, controls=initial,
        )

    # ---- controls -----------------------------------------------------------

    def set_controls(self, controls: ARControls) -> None:
        """Publish new controls: three buffer writes on the default
        stream, ordered before the next replay. Safe from a control
        thread for the same reason the base class's swap is."""
        super().set_controls(controls)
        self._guidance.fill_(float(controls.guidance))
        self._temperature.fill_(float(controls.temperature))
        self._top_k.fill_(int(controls.top_k))
        self._mask_end.fill_(bool(controls.mask_end))

    # ---- pieces -------------------------------------------------------------

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """`_sample_top_k` with the controls read from device scalars.
        The threshold is the k-th largest value, from a sort rather than
        `topk` so that k can change without a recapture."""
        values = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        k = self._top_k.clamp(1, values.shape[-1]) - 1
        ordered = torch.sort(values, dim=-1, descending=True).values
        threshold = ordered.gather(-1, k.view(1, 1).expand(values.shape[0], 1))
        values = values.masked_fill(values < threshold, -float("inf"))
        values = values / self._temperature.clamp_min(1e-3)
        probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.multinomial(probs, 1, generator=self._generator).squeeze(-1)

    def _depth_codes(self, last_hidden: torch.Tensor, semantic_code: torch.Tensor):
        ar = self._ar
        depth = ar.depth_decoder
        num_codebooks = depth.num_codebooks
        sequence = [depth.projection(last_hidden).unsqueeze(1)]
        code_embed = ar.language_model.model.embed_tokens(semantic_code + AUDIO_CODE_OFFSET)
        sequence.append(depth.projection(code_embed).unsqueeze(1))
        codes = [semantic_code]
        hidden_parts = []
        for index in range(1, num_codebooks):
            hidden = depth(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden[:1])
            logits = depth.audio_heads[index - 1](hidden)
            conditional, unconditional = logits[:1].float(), logits[1:2].float()
            logits = unconditional + (conditional - unconditional) * self._guidance
            code = self._sample(logits).repeat(2)
            codes.append(code)
            if index < num_codebooks - 1:
                embed = depth.audio_embeddings(code + (index - 1) * depth.audio_vocab_size)
                sequence.append(depth.projection(embed).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    def _sample_frame(self, last_hidden: torch.Tensor) -> None:
        """From the LM's hidden state to the frame's codes, the
        renderer's hidden, the end flag, and the next feedback."""
        ar = self._ar
        lm = ar.language_model
        vocab_mask = ar._vocab_mask_for(last_hidden.device)
        logits = lm.lm_head(last_hidden).float().masked_fill(vocab_mask, -float("inf"))
        conditional, unconditional = logits[0:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * self._guidance
        threshold = torch.topk(conditional, AR_CFG_TOP_K, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(conditional < threshold, -float("inf"))
        guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
        # Static-shape column fill, so the mask can toggle per frame
        # without a recapture, like the other control scalars.
        guided[:, AUDIO_END_TOKEN_ID] = guided[:, AUDIO_END_TOKEN_ID].masked_fill(
            self._mask_end, -float("inf")
        )
        sampled = self._sample(guided)
        self._end.copy_(sampled == AUDIO_END_TOKEN_ID)
        # The end token sits below the code offset. Clamp so a frame that
        # sampled it never indexes an embedding table negatively; its
        # codes are discarded by `_step` on the flag.
        semantic = (sampled - AUDIO_CODE_OFFSET).clamp_(0, SEMANTIC_VOCAB_SIZE - 1)
        codes, depth_hidden = self._depth_codes(last_hidden, semantic.repeat(2))
        self._codes_out.copy_(codes)
        self._hidden_out.copy_(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
        self._feedback.copy_(ar._embed_audio_frame(codes))

    def _frame(self) -> None:
        """The captured body: feed the last frame, sample this one."""
        lm = self._ar.language_model
        with _attention(lm, DECODE_ATTENTION):
            out = lm.model(
                inputs_embeds=self._feedback,
                past_key_values=self._cache,
                use_cache=True,
                cache_position=self._pos,
                **{POS_KWARG: self._pos},
            )
        self._sample_frame(out.last_hidden_state[:, -1])
        self._pos += 1

    # ---- lifecycle ----------------------------------------------------------

    @torch.no_grad()
    def _prefill(self, prompt: str, lyrics: str, *, replay: bool = False) -> None:
        ar = self._ar
        lm = ar.language_model
        device = ar.device
        started = time.perf_counter()

        text_ids = ar.tokenize(prompt, lyrics).to(device)
        length = int(text_ids.shape[1])
        if length > PROMPT_SLOTS:
            raise ValueError(
                f"prompt is {length} tokens; the graphed session reserves "
                f"{PROMPT_SLOTS} cache slots for the text prefix"
            )
        window = int(getattr(lm.config, "max_position_embeddings", 0) or 0)
        if window and length + self.max_frames > window:
            logger.warning(
                "minimax_ar_over_window prompt={} frames={} window={}",
                length, self.max_frames, window,
            )

        # Prefill and replay run under HF's own sdpa: it builds the
        # causal mask from `cache_position` for a static cache, and a
        # one-off pass over a long history is not where the cost is.
        # Slots past the new end hold stale keys that the mask never
        # shows to a query, so a shorter re-prompt needs no clearing.
        with _attention(lm, "sdpa"):
            out = lm.model(
                inputs_embeds=lm.model.embed_tokens(text_ids),
                past_key_values=self._cache,
                use_cache=True,
                cache_position=torch.arange(length, device=device),
            )
            position = length
            if replay and self._code_history:
                for lo in range(0, len(self._code_history), self._REPLAY_BLOCK):
                    block = self._code_history[lo:lo + self._REPLAY_BLOCK]
                    embeds = torch.cat(
                        [ar._embed_audio_frame(codes) for codes in block], dim=1,
                    )
                    count = int(embeds.shape[1])
                    out = lm.model(
                        inputs_embeds=embeds,
                        past_key_values=self._cache,
                        use_cache=True,
                        cache_position=torch.arange(position, position + count, device=device),
                    )
                    position += count

        self._pos.fill_(position)
        self._pos_host = position
        self.prompt_tokens = length
        # The frame this hidden state implies is sampled now, eagerly:
        # the warm-up frame on a fresh prefill, the next real frame
        # after a re-prompt. `_step` consumes it before replaying.
        self._sample_frame(out.last_hidden_state[:, -1])
        self._ready = True
        self.last_prefill_s = time.perf_counter() - started

    def _capture(self) -> None:
        started = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        graph.register_generator_state(self._generator)
        # thread_local: a control thread writing a knob buffer on the
        # default stream during the ~0.2 s capture is legal and must not
        # abort it.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self._frame()
        self._graph = graph
        self.capture_s = time.perf_counter() - started
        logger.info(
            "minimax_ar_graph_captured slots={} seconds={:.2f}",
            self._slots, self.capture_s,
        )

    def _run_frame(self) -> None:
        if self._pos_host >= self._slots:
            raise RuntimeError("graphed AR session ran past its cache")
        if self._graph is not None:
            self._graph.replay()
        elif self._eager_frames < WARMUP_FRAMES:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self._frame()
            torch.cuda.current_stream().wait_stream(side)
            self._eager_frames += 1
        else:
            self._capture()
            self._graph.replay()
        self._pos_host += 1

    @torch.no_grad()
    def _step(self) -> Optional[torch.Tensor]:
        if not self._ready:
            self._run_frame()
        self._ready = False
        if bool(self._end.item()):
            self.finished = True
            self.stopped_early = True
            return None
        self._code_history.append(self._codes_out.clone())
        self._iteration += 1
        if self._iteration == 1:
            return None
        self.frames_emitted += 1
        if self.frames_emitted >= self.max_frames:
            self.finished = True
        return self._hidden_out.clone()

    def close(self) -> None:
        self._graph = None
        self._cache = None
        super().close()
