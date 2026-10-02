"""CUDA-graph feasibility spike for the MiniMax-Music3 AR stage.

This is the investigation that produced `acestep/engine/minimax_ar_graph.py`,
kept because its numbers are the ones the module's claims rest on.

The question. `minimax_ar_bench.py --profile` showed the AR decode loop
is dispatch-bound: 22 ms of GPU kernels inside a 53 ms frame, the rest
Python launching ~3900 kernels. A CUDA graph replays those launches
from one call. Does that recover the gap, and what does the static KV
cache a graph requires cost on its own?

Three paths over the same weights and prompt:

  A  dynamic cache, eager      the plain `MiniMaxARStream`
  B  static cache, eager       isolates the cache change from the graph
  C  static cache, CUDA graph  one `replay()` per frame

What it found (5090, bf16, ms per frame):

  static slots     512    1024    2048    4096    9600
  --attn hf       24.0    26.0    29.9    36.8    55.6   (HF sdpa, repeat_kv)
  --attn demon      --    22.7      --    32.4*   25.7   (*efficient kernel; bmm ships)
  plain loop, LM forward only: 36.7 / 40.0 / 58.6 / 77.7 at 0 / 3000 / 6000 / 9000 frames

So the graph pays (24 ms at 512 slots against a 22 ms kernel floor), HF's
attention over a padded static cache eats the gain back (3.7 ms per 1024
slots), and a grouped-query `bmm` decode attention removes that (0.072 ms
per layer at 9600 slots, 0.044 floor). The plain loop is not flat in
length either, which the docs had claimed from a 300-frame run.

The gate. RNG draws are not comparable across capture (the graph owns
its own philox offsets), so C's sampled codes are not expected to equal
A's. Instead C's codes are teacher-forced back through the dynamic-cache
path and the LM hidden states compared frame for frame, against HF's own
eager-vs-sdpa distance on the same path as the noise floor: worst cos
0.99983 graph-vs-sdpa against 0.99967 eager-vs-sdpa.

Run on an idle card with free VRAM: a co-resident process inflates the
dispatch gap, and a full card pages (32.1 of 32.6 GB in use took the
graphed frame from 23 to 69 ms with nothing in the logs to say why).

    .venv/Scripts/python.exe scripts/minimax/minimax_graph_spike.py --attn demon --cache-lens 1024,9600
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

# A sibling ACE-Step checkout shadows `acestep` otherwise.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from transformers import AttentionInterface, StaticCache  # noqa: E402
from transformers.cache_utils import StaticLayer  # noqa: E402
from transformers.masking_utils import AttentionMaskInterface  # noqa: E402

from acestep.engine.minimax_ar import (  # noqa: E402
    AR_CFG_TOP_K,
    AR_FRAME_RATE_HZ,
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    SEMANTIC_VOCAB_SIZE,
    ARControls,
    MiniMaxAR,
)
from acestep.engine.minimax_helpers import resolve_model_dir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimax_ar_bench import (  # noqa: E402
    DEFAULT_LYRICS,
    DEFAULT_PROMPT,
    STACK_VRAM_GB,
    _wait_for_vram,
)

FRAME_S = 1.0 / AR_FRAME_RATE_HZ

# ---- decode attention over a static cache ------------------------------------
#
# Decode attention over a static cache is a bandwidth problem that the
# library kernels get wrong at q_len 1 (measured per layer at 9600 slots,
# 0.044 ms bandwidth floor): HF sdpa with a mask 0.9 ms, since `repeat_kv`
# materialises the padded K/V four times over for GQA; SDPA's efficient
# kernel on the un-expanded K/V 0.64 ms, one CTA per (batch, head)
# walking every slot; two `bmm`s 0.072 ms. So: fold the four query heads
# that share a KV head into the query's row axis, scores as a bf16 x bf16
# bmm with an fp32 result (`out_dtype`, so no rounding before the
# softmax), softmax in fp32, probabilities back to bf16 for the value bmm,
# which is where flash rounds too. The mask comes from the cache position
# on device, so the function is valid inside a graph.

ATTN_NAME = "demon_decode"
_DECODE_STATE = {"pos": None}


def _decode_attention(module, query, key, value, attention_mask, dropout=0.0,
                      scaling=None, **kwargs):
    batch, q_heads, q_len, dim = query.shape
    kv_heads, slots = key.shape[1], key.shape[2]
    group = q_heads // kv_heads
    rows = group * q_len
    pos = _DECODE_STATE["pos"]
    q_pos = pos + torch.arange(q_len, device=query.device)
    visible = torch.arange(slots, device=query.device)[None, :] <= q_pos[:, None]
    mask = visible.unsqueeze(0).expand(group, q_len, slots).reshape(1, rows, slots)

    grouped = query.reshape(batch * kv_heads, rows, dim)
    keys = key.reshape(batch * kv_heads, slots, dim)
    values = value.reshape(batch * kv_heads, slots, dim)
    scores = torch.bmm(grouped, keys.transpose(1, 2), out_dtype=torch.float32)
    scores = (scores * scaling).masked_fill(~mask, -float("inf"))
    probs = torch.softmax(scores, dim=-1).to(values.dtype)
    out = torch.bmm(probs, values)
    out = out.reshape(batch, q_heads, q_len, dim).transpose(1, 2).contiguous()
    return out, None


def _no_mask(*args, **kwargs):
    return None


AttentionInterface.register(ATTN_NAME, _decode_attention)
AttentionMaskInterface.register(ATTN_NAME, _no_mask)


class _BucketLayer(StaticLayer):
    """A static layer that shows attention only its first `bucket` slots."""

    bucket: int | None = None

    def update(self, key_states, value_states, cache_kwargs=None):
        keys, values = super().update(key_states, value_states, cache_kwargs)
        n = self.bucket or self.max_cache_len
        return keys[:, :, :n], values[:, :, :n]

    def get_mask_sizes(self, cache_position):
        return (self.bucket or self.max_cache_len), 0


class BucketCache(StaticCache):
    def __init__(self, config, max_cache_len: int):
        super().__init__(config=config, max_cache_len=max_cache_len)
        self.layers = [_BucketLayer(max_cache_len=max_cache_len) for _ in self.layers]

    def set_bucket(self, n: int | None) -> None:
        for layer in self.layers:
            layer.bucket = n


def _set_attn(lm, name: str) -> None:
    try:
        lm.config._attn_implementation = name
    except Exception:
        lm.set_attn_implementation(name)

# Chunk render cost per 200-frame window on the TRT engine, from
# `minimax_stream_bench.py`: isolated, and co-resident with the AR stage
# at hop 100 where the render ran while the AR had been idle-gapping.
RENDER_ISOLATED_S = 0.518
RENDER_CORESIDENT_S = 1.03


class GraphedAR:
    """One AR frame as a fixed-shape function over static buffers.

    Everything the frame reads or writes lives in a buffer allocated
    once: the feedback embedding, the cache position, the sampling
    controls, and the outputs. Replaying the captured graph therefore
    advances the sequence with no Python between kernels.
    """

    def __init__(
        self,
        ar: MiniMaxAR,
        *,
        prompt: str,
        lyrics: str,
        seed: int,
        max_cache_len: int,
        controls: ARControls,
        attn: str = "hf",
    ):
        self.ar = ar
        self.lm = ar.language_model
        self.depth = ar.depth_decoder
        dev = ar.device
        hidden = int(self.lm.config.hidden_size)

        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.attn = attn
        if attn == "demon":
            self.cache = BucketCache(self.lm.config, max_cache_len)
            _DECODE_STATE["pos"] = self.pos
            _set_attn(self.lm, ATTN_NAME)
        else:
            self.cache = StaticCache(config=self.lm.config, max_cache_len=max_cache_len)
            _set_attn(self.lm, "sdpa")
        self.feedback = torch.zeros(2, 1, hidden, dtype=ar.dtype, device=dev)
        self.guidance = torch.tensor(float(controls.guidance), device=dev)
        self.temperature = torch.tensor(float(controls.temperature), device=dev)
        self.top_k = int(controls.top_k)

        self.codes_out = torch.zeros(2, self.depth.num_codebooks, dtype=torch.long, device=dev)
        self.hidden_out = torch.zeros(1, self.depth.num_codebooks * hidden, dtype=ar.dtype, device=dev)
        self.end_flag = torch.zeros(1, dtype=torch.bool, device=dev)

        self.generator = torch.Generator(device=dev).manual_seed(int(seed))
        self.vocab_mask = ar._vocab_mask_for(dev)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.sampler = "multinomial"

        # Every frame's codes and LM hidden, for the teacher-forced gate.
        self.history_codes: List[torch.Tensor] = []
        self.history_hidden: List[torch.Tensor] = []

        self._prefill(prompt, lyrics)

    # ---- pieces -------------------------------------------------------------

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """`_sample_top_k` with the controls read from device scalars, so a
        knob move is a buffer write rather than a recapture. Dividing by a
        temperature of exactly 1.0 is exact, so the reference's skip of
        that divide is not needed for bit-identity."""
        values = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        threshold = torch.topk(values, self.top_k, dim=-1).values[..., -1, None]
        values = values.masked_fill(values < threshold, -float("inf"))
        values = values / self.temperature.clamp_min(1e-3)
        probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.sampler == "multinomial":
            return torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
        # Gumbel-max: same distribution, and built from ops that are
        # capture-safe when `multinomial` is not.
        noise = torch.empty_like(probs).exponential_(generator=self.generator)
        return torch.argmax(probs / noise, dim=-1)

    def _depth_codes(
        self, last_hidden: torch.Tensor, semantic_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        depth = self.depth
        num_codebooks = depth.num_codebooks
        sequence = [depth.projection(last_hidden).unsqueeze(1)]
        code_embed = self.lm.model.embed_tokens(semantic_code + AUDIO_CODE_OFFSET)
        sequence.append(depth.projection(code_embed).unsqueeze(1))
        codes = [semantic_code]
        hidden_parts = []
        for index in range(1, num_codebooks):
            hidden = depth(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden[:1])
            logits = depth.audio_heads[index - 1](hidden)
            conditional, unconditional = logits[:1].float(), logits[1:2].float()
            logits = unconditional + (conditional - unconditional) * self.guidance
            code = self._sample(logits).repeat(2)
            codes.append(code)
            if index < num_codebooks - 1:
                embed = depth.audio_embeddings(code + (index - 1) * depth.audio_vocab_size)
                sequence.append(depth.projection(embed).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    def _sample_frame(self, last_hidden: torch.Tensor) -> None:
        """From the LM's hidden state to this frame's codes, the renderer's
        hidden, and the next feedback embedding. Mirrors `_step` after its
        LM forward, minus the `.item()`: the end flag is a buffer the
        caller reads after the frame."""
        logits = self.lm.lm_head(last_hidden).float()
        logits = logits.masked_fill(self.vocab_mask, -float("inf"))
        conditional, unconditional = logits[0:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * self.guidance
        threshold = torch.topk(conditional, AR_CFG_TOP_K, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(conditional < threshold, -float("inf"))
        guided = guided.masked_fill(self.vocab_mask.unsqueeze(0), -float("inf"))
        sampled = self._sample(guided)
        self.end_flag.copy_(sampled == AUDIO_END_TOKEN_ID)
        # The end token sits below the code offset; clamp so the frame
        # after it never indexes an embedding table negatively. Its codes
        # are discarded by the caller anyway.
        semantic = (sampled - AUDIO_CODE_OFFSET).clamp_(0, SEMANTIC_VOCAB_SIZE - 1)
        codes, depth_hidden = self._depth_codes(last_hidden, semantic.repeat(2))
        self.codes_out.copy_(codes)
        self.hidden_out.copy_(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
        self.feedback.copy_(self.ar._embed_audio_frame(codes))

    def _frame(self) -> None:
        """The captured body: feed last frame, sample this one."""
        out = self.lm.model(
            inputs_embeds=self.feedback,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=self.pos,
        )
        last_hidden = out.last_hidden_state[:, -1]
        self._sample_frame(last_hidden)
        self.pos += 1

    # ---- lifecycle ----------------------------------------------------------

    @torch.no_grad()
    def _prefill(self, prompt: str, lyrics: str) -> None:
        dev = self.ar.device
        ids = self.ar.tokenize(prompt, lyrics).to(dev)
        length = int(ids.shape[1])
        out = self.lm.model(
            inputs_embeds=self.lm.model.embed_tokens(ids),
            past_key_values=self.cache,
            use_cache=True,
            cache_position=torch.arange(length, device=dev),
        )
        self.pos.fill_(length)
        self.prompt_tokens = length
        # The warm-up frame: sampled from the prompt's own hidden state,
        # exactly as `_step` does when nothing is pending yet.
        self._sample_frame(out.last_hidden_state[:, -1])
        self._record()

    def _record(self) -> None:
        hidden = int(self.lm.config.hidden_size)
        self.history_codes.append(self.codes_out.clone())
        self.history_hidden.append(self.hidden_out[:, :hidden].clone())

    @torch.no_grad()
    def step_eager(self) -> None:
        self._frame()
        self._record()

    @torch.no_grad()
    def capture(self, warmup: int = 3) -> None:
        """Warm up on a side stream (cuBLAS workspaces, lazy kernels), then
        capture one frame. Capture records without executing, so the
        sequence state is where the warm-up left it."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._frame()
                self._record()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        if hasattr(graph, "register_generator_state"):
            graph.register_generator_state(self.generator)
        try:
            with torch.cuda.graph(graph):
                self._frame()
        except RuntimeError as exc:
            if self.sampler != "multinomial":
                raise
            print(f"  multinomial not capturable ({str(exc).splitlines()[0]}); "
                  "falling back to gumbel-max")
            self.sampler = "gumbel"
            graph = torch.cuda.CUDAGraph()
            if hasattr(graph, "register_generator_state"):
                graph.register_generator_state(self.generator)
            with torch.cuda.graph(graph):
                self._frame()
        self.graph = graph

    @torch.no_grad()
    def step_graph(self) -> None:
        assert self.graph is not None
        self.graph.replay()
        self._record()


# ---- measurement ------------------------------------------------------------


def _time_frames(step, frames: int, *, sync_each: bool) -> List[float]:
    per_frame = []
    torch.cuda.synchronize()
    total0 = time.perf_counter()
    for _ in range(frames):
        t0 = time.perf_counter()
        step()
        if sync_each:
            torch.cuda.synchronize()
            per_frame.append(time.perf_counter() - t0)
    torch.cuda.synchronize()
    total = time.perf_counter() - total0
    return per_frame if sync_each else [total / frames] * frames


def _report(label: str, per_frame: List[float]) -> float:
    ms = statistics.mean(per_frame) * 1000.0
    med = statistics.median(per_frame) * 1000.0
    rt = FRAME_S / statistics.mean(per_frame)
    print(f"{label:<34} {ms:7.2f} ms/frame  (median {med:6.2f})  {rt:5.2f}x realtime")
    return ms


@torch.no_grad()
def _gate(ar: MiniMaxAR, prompt: str, lyrics: str, gar: GraphedAR, frames: int) -> None:
    """Teacher-force the graphed path's codes through the dynamic cache and
    compare LM hidden states. Frame 0 is the prefill warm-up and took the
    same eager path in both, so it anchors the comparison."""
    lm = ar.language_model
    dev = ar.device
    # The reference is the shipping path: dynamic cache, HF sdpa.
    _set_attn(lm, "sdpa")
    ids = ar.tokenize(prompt, lyrics).to(dev)
    out = lm.model(inputs_embeds=lm.model.embed_tokens(ids), use_cache=True)
    past = out.past_key_values
    ref0 = out.last_hidden_state[:, -1][:1].float()
    got0 = gar.history_hidden[0].float()
    print(f"  frame 0 (prefill, eager both): max|d|={float((ref0 - got0).abs().max()):.3e}")

    # Noise floor: the same dynamic path under HF's eager attention (fp32
    # softmax). The graphed path is judged against the distance between
    # these two, since that is what a kernel swap alone costs.
    _set_attn(lm, "eager")
    out_e = lm.model(inputs_embeds=lm.model.embed_tokens(ids), use_cache=True)
    past_e = out_e.past_key_values
    _set_attn(lm, "sdpa")

    count = min(frames, len(gar.history_codes) - 1)
    worst = {"graph vs sdpa": [1.0, 0.0], "eager vs sdpa": [1.0, 0.0]}
    for i in range(count):
        feedback = ar._embed_audio_frame(gar.history_codes[i])
        out = lm.model(inputs_embeds=feedback, past_key_values=past, use_cache=True)
        past = out.past_key_values
        ref = out.last_hidden_state[:, -1][:1].float()
        _set_attn(lm, "eager")
        out_e = lm.model(inputs_embeds=feedback, past_key_values=past_e, use_cache=True)
        past_e = out_e.past_key_values
        _set_attn(lm, "sdpa")
        eager = out_e.last_hidden_state[:, -1][:1].float()
        got = gar.history_hidden[i + 1].float()
        for label, other in (("graph vs sdpa", got), ("eager vs sdpa", eager)):
            cos = float(F.cosine_similarity(ref, other, dim=-1))
            rel = float((ref - other).abs().max()) / float(ref.abs().max())
            worst[label][0] = min(worst[label][0], cos)
            worst[label][1] = max(worst[label][1], rel)
    for label, (cos, rel) in worst.items():
        print(f"  {count} frames teacher-forced, {label:<14}: worst cos {cos:.6f}, "
              f"worst max|d|/max|ref| {rel:.3e}")


@torch.no_grad()
def _dynamic_lm_at_length(ar: MiniMaxAR, prompt: str, lyrics: str, n_frames: int,
                          seed: int, timed: int = 50) -> List[float]:
    """The shipping path's single-token LM forward after `n_frames` of
    history, dynamic cache and HF sdpa. Only this forward depends on the
    cache length; the head, sampling and depth decoder do not. The
    history is random codes fed in blocks, the same batched prefill a
    live caption swap uses; the cost depends on the length, not on what
    the codes were. Sampling is not involved, so the end token cannot
    cut the run short."""
    _set_attn(ar.language_model, "sdpa")
    lm = ar.language_model
    dev = ar.device
    depth = ar.depth_decoder
    ids = ar.tokenize(prompt, lyrics).to(dev)
    out = lm.model(inputs_embeds=lm.model.embed_tokens(ids), use_cache=True)
    past = out.past_key_values
    draw = torch.Generator(device="cpu").manual_seed(seed)

    def random_codes(count: int) -> torch.Tensor:
        semantic = torch.randint(0, SEMANTIC_VOCAB_SIZE, (count, 1), generator=draw)
        residual = torch.randint(0, depth.audio_vocab_size, (count, depth.num_codebooks - 1), generator=draw)
        return torch.cat([semantic, residual], dim=1).to(dev)

    for lo in range(0, n_frames, 512):
        block = min(512, n_frames - lo)
        codes = random_codes(block)
        embeds = torch.cat(
            [ar._embed_audio_frame(codes[i:i + 1].repeat(2, 1)) for i in range(block)], dim=1,
        )
        out = lm.model(inputs_embeds=embeds, past_key_values=past, use_cache=True)
        past = out.past_key_values
    feedback = ar._embed_audio_frame(random_codes(1).repeat(2, 1))

    def step():
        nonlocal past
        result = lm.model(inputs_embeds=feedback, past_key_values=past, use_cache=True)
        past = result.past_key_values

    per_frame = _time_frames(step, timed, sync_each=True)
    del past, out
    torch.cuda.empty_cache()
    return per_frame


def _project(ar_ms: float) -> None:
    print("\nend-to-end projection (render cost per 200-frame window held at the "
          "measured values; every frame the AR writes is rendered once)")
    print(f"  {'hop':>4}  {'AR s/s':>7}  {'render isolated':>16}  {'render co-resident':>19}")
    for hop in (100, 50, 25):
        ar_s = AR_FRAME_RATE_HZ * ar_ms / 1000.0
        renders_per_s = AR_FRAME_RATE_HZ / hop
        iso = 1.0 / (ar_s + renders_per_s * RENDER_ISOLATED_S)
        cor = 1.0 / (ar_s + renders_per_s * RENDER_CORESIDENT_S)
        print(f"  {hop:>4}  {ar_s:7.3f}  {iso:11.2f}x       {cor:14.2f}x")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--gate-frames", type=int, default=48)
    ap.add_argument("--max-cache-len", type=int, default=9_600,
                    help="static cache slots: prompt + up to 9000 frames")
    ap.add_argument("--attn", choices=("hf", "demon"), default="hf",
                    help="hf: HF sdpa over the padded static cache; "
                         "demon: grouped-query decode attention over the bucket")
    ap.add_argument("--cache-lens", default=None,
                    help="comma list of static cache sizes to sweep, e.g. 512,2048,9600")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--lyrics", default=DEFAULT_LYRICS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--long-context", default="3000,6000",
                    help="also time the shipping dynamic path after this many "
                         "frames of history (comma list; empty to skip)")
    ap.add_argument("--skip-dynamic", action="store_true",
                    help="skip path A (use the published 52.6 ms)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    device = torch.device("cuda", 0)
    free = _wait_for_vram(0, STACK_VRAM_GB, 600.0)
    print(f"free VRAM      : {free:.1f} GB")
    root = resolve_model_dir()
    print(f"checkpoint     : {root}")
    print(f"torch          : {torch.__version__}")

    ar = MiniMaxAR.from_pretrained(root, dtype=torch.bfloat16, device="cpu", seed=args.seed)
    ar.to(device)
    torch.cuda.synchronize()
    controls = ARControls()

    print()
    a_ms = 52.6
    if not args.skip_dynamic:
        stream = ar.stream(prompt=args.prompt, lyrics=args.lyrics, seed=args.seed,
                           max_frames=args.frames + 8, controls=controls)
        stream.advance(1)  # warm-up frame

        def step_a():
            stream.advance(1)

        a_ms = _report("A  dynamic cache, eager", _time_frames(step_a, args.frames, sync_each=True))
        del stream
        torch.cuda.empty_cache()
    for n in [int(x) for x in args.long_context.split(",") if x.strip()]:
        per_frame = _dynamic_lm_at_length(ar, args.prompt, args.lyrics, n, args.seed)
        _report(f"A  LM forward only, {n} frames in", per_frame)

    lens = [int(x) for x in args.cache_lens.split(",")] if args.cache_lens else [args.max_cache_len]
    c_ms = None
    for max_len in lens:
        gar = GraphedAR(ar, prompt=args.prompt, lyrics=args.lyrics, seed=args.seed,
                        max_cache_len=max_len, controls=controls, attn=args.attn)
        print(f"\nstatic cache   : {max_len} slots, prompt {gar.prompt_tokens} tokens, attn={args.attn}")
        _report("B  static cache, eager", _time_frames(gar.step_eager, min(args.frames, 100), sync_each=True))

        t0 = time.perf_counter()
        gar.capture()
        print(f"capture        : {time.perf_counter() - t0:.2f}s, sampler={gar.sampler}")
        # Two timed passes share the cache; overrunning it is an
        # out-of-bounds index_copy_ inside the replay (an access
        # violation, not an exception).
        used = gar.prompt_tokens + len(gar.history_codes) + 4
        frames = max(8, min(args.frames, (max_len - used) // 2))
        c_ms = _report("C  static cache, graph (sync/frame)",
                       _time_frames(gar.step_graph, frames, sync_each=True))
        _report("C  static cache, graph (free-run)",
                _time_frames(gar.step_graph, frames, sync_each=False))
        print(f"peak VRAM      : {torch.cuda.max_memory_allocated(device) / 1024**3:.1f} GB allocated")

        print("gate: graphed LM hidden vs dynamic-cache teacher-forced")
        _gate(ar, args.prompt, args.lyrics, gar, args.gate_frames)
        del gar
        torch.cuda.empty_cache()

    print(f"\nAR stage: {a_ms:.1f} -> {c_ms:.1f} ms/frame "
          f"({a_ms / c_ms:.2f}x), {FRAME_S * 1000 / c_ms:.2f}x realtime")
    _project(c_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
