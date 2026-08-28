"""A local prompt enhancer, and a neighbourhood of variations around a prompt.

`/api/enhance` has always expanded a rough idea into a rich prompt by asking a
hosted LLM. That costs an API key, a network round trip, and a per-call fee for
an answer that is highly repetitive across a session.

This module offers a second backend for the same job: a small seq2seq model
(t5-small class) fine-tuned on that LLM's own output, running in-process on the
pod. It answers in roughly a tenth of the time, needs no key, and is
deterministic. It is NOT a general instruction model -- it was trained on
structured, composed prompt text and degrades on free-form input, so it is
opt-in per deployment and the hosted backend remains the default.

TWO ENTRY POINTS.

``enhance`` is the one-shot: a prompt in, a richer one out, greedy so the same
input always yields the same output.

``neighbourhood`` answers a different question -- "what else is near this
prompt?" -- and answers it for the WHOLE neighbourhood at once rather than one
point at a time. A client exploring nearby prompts interactively would
otherwise issue a request per interaction, which is a round trip per gesture.

It can answer wholesale because the reachable set is small and finite:

  * distance from the anchor collapses to an integer -- how many of the
    anchor's leading tokens are held fixed, bounded to MAX_FREE of the line;
  * variety at a given distance is LANES discrete sampling streams.

So one anchor has a few hundred reachable strings, all deterministic. Generate
them in one batched pass and the client can index the result locally.

WHY A FORCED PREFIX RATHER THAN A SEED. Seeds have no notion of "nearby": two
seeds are unrelated draws, so a seed cannot express distance. Prefix length
can. Holding the head fixed and resampling the tail is the smallest meaningful
change the model can make, and it grows monotonically as the prefix shortens.
"""

from __future__ import annotations

import functools
import os
import threading

#: Sampler freedom at the two ends of the travel. Both ramp with distance: near
#: the anchor the sampler is nearly greedy, far from it it is free to leave.
TOPK_MIN, TOPK_MAX = 4, 40
TEMP_MIN, TEMP_MAX = 0.35, 1.35
#: Most of the line always survives. Beyond roughly this fraction freed, the
#: model stops varying the brief and starts answering it again, so distance
#: past that point means nothing.
MAX_FREE = 0.55
#: Distinct sampling streams at a given distance. Enough that a deliberate move
#: lands somewhere else, few enough that each stays reachable and repeatable.
LANES = 12
#: Distance steps returned. More than this is wasted: the prefix length is an
#: integer token count, so adjacent steps start colliding on a short line.
STOPS = 16

_MODEL_ENV = "DEMON_ENHANCER_DIR"
_lock = threading.Lock()


def _model_dir() -> str:
    """Where the fine-tuned checkpoint lives.

    Under the models directory like every other checkpoint, never in the
    repository. ``DEMON_ENHANCER_DIR`` overrides for local experiments.
    """
    if os.environ.get(_MODEL_ENV):
        return os.environ[_MODEL_ENV]
    try:
        from acestep.paths import prompt_enhancer_dir

        return str(prompt_enhancer_dir())
    except Exception:
        base = os.environ.get(
            "ACESTEP_MODELS_DIR",
            os.path.expanduser("~/.daydream-scope/models/demon"),
        )
        return os.path.join(base, "PromptEnhancer")


@functools.lru_cache(maxsize=1)
def _load():
    """Tokenizer + model, once.

    Returns None when the checkpoint is absent or unloadable, so every caller
    degrades to the hosted backend instead of failing the request.
    """
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    path = _model_dir()
    if not os.path.isdir(path):
        return None
    try:
        tok = AutoTokenizer.from_pretrained(path, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        return tok, model, device
    except Exception:
        return None


def available() -> bool:
    """True when the local checkpoint is present and loadable."""
    return _load() is not None


def _prefix_for(anchor_ids: list[int], amount: float) -> list[int]:
    """How much of the anchor survives at `amount`.

    The rewrite eats forward from the TAIL. The opening phrases dominate how a
    prompt is interpreted, so holding them and resampling the end is what makes
    a small move read as a small move rather than a different brief.
    """
    n = len(anchor_ids)
    free = round(amount * MAX_FREE * n)
    return anchor_ids[: n - min(free, n)]


def _task(deck: str) -> str:
    """Task prefix the model was trained with, per prompt family."""
    return "enhance studio: " if deck == "sa3" else "enhance arranger: "


def _seed_for(text: str) -> int:
    """A seed that is stable across processes.

    Python's ``hash()`` is salted per interpreter (PYTHONHASHSEED), so using it
    would give a different neighbourhood after every restart -- and the whole
    contract here is that a variation you liked is still there when you come
    back to it.
    """
    import zlib

    return zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF


def enhance(text: str, deck: str = "sa3") -> str:
    """Expand `text` into a richer prompt. Greedy, so it is reproducible.

    Returns "" when the local checkpoint is unavailable or produced nothing,
    which the caller should treat as "fall back to the hosted backend".
    """
    loaded = _load()
    if loaded is None or not text.strip():
        return ""
    tok, model, device = loaded
    import torch

    with _lock, torch.inference_mode():
        enc = tok(_task(deck) + text, return_tensors="pt", max_length=160,
                  truncation=True).to(device)
        out = model.generate(**enc, max_new_tokens=128, num_beams=1,
                             do_sample=False, no_repeat_ngram_size=3)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def neighbourhood(text: str, deck: str = "sa3", lanes: int = LANES,
                  stops: int = STOPS) -> dict:
    """Every prompt reachable near `text`, in one batched pass.

    Returns ``{"anchor": str, "lanes": int, "stops": int, "grid": [[str]]}``
    where ``grid[lane][stop]`` is the prompt at that coordinate, stop 0 being
    the anchor itself and higher stops progressively further from it. Empty
    dict when the local checkpoint is unavailable.

    STOP 0 IS THE ANCHOR, EXACTLY, in every lane. Travelling away and back has
    to return precisely where you were, or the control is lossy and no client
    can offer it as navigation.

    BATCHED ALONG THE LANE AXIS. Every lane shares one distance, so the forced
    prefix is identical across the batch and needs no ragged decoder padding --
    and one sampling call over `lanes` identical rows draws `lanes` independent
    variants, which is what a lane is. Batching the other way (one lane, every
    distance) would need variable-length decoder inputs for no gain.

    ONE SEED FOR THE WHOLE TRAVEL, reset before each stop. Every distance then
    starts from an identical RNG state and differs only in how much of the line
    was freed, so moving outward DIVERGES from where you were instead of
    teleporting. Seeding per stop instead is still reproducible, but makes
    every distance an independent draw: measured, that roughly doubled the edit
    distance between adjacent steps, turning travel into a shuffle.
    """
    loaded = _load()
    if loaded is None or not text.strip():
        return {}
    tok, model, device = loaded
    import torch

    with _lock, torch.inference_mode():
        enc = tok(_task(deck) + text, return_tensors="pt", max_length=160,
                  truncation=True).to(device)
        anchor_out = model.generate(**enc, max_new_tokens=128, num_beams=1,
                                    do_sample=False, no_repeat_ngram_size=3)
        anchor_ids = [int(i) for i in anchor_out[0]
                      if int(i) not in (tok.pad_token_id, tok.eos_token_id)]
        anchor_text = tok.decode(anchor_out[0], skip_special_tokens=True).strip()
        if not anchor_ids:
            return {}

        anchor_seed = _seed_for(text)
        grid = [[anchor_text] for _ in range(lanes)]
        enc_b = {k: v.repeat(lanes, 1) for k, v in enc.items()}

        for s in range(1, stops):
            amount = s / (stops - 1)
            prefix = _prefix_for(anchor_ids, amount)
            dec = torch.tensor([[model.config.decoder_start_token_id] + prefix],
                               device=device).repeat(lanes, 1)
            topk = TOPK_MIN + round(amount * (TOPK_MAX - TOPK_MIN))
            temp = TEMP_MIN + amount * (TEMP_MAX - TEMP_MIN)
            torch.manual_seed(anchor_seed)
            out = model.generate(
                **enc_b,
                decoder_input_ids=dec,
                max_new_tokens=128,
                num_beams=1,
                do_sample=True,
                top_k=topk,
                temperature=temp,
                no_repeat_ngram_size=3,
            )
            for lane in range(lanes):
                txt = tok.decode(out[lane], skip_special_tokens=True).strip()
                grid[lane].append(txt or anchor_text)

    return {"anchor": anchor_text, "lanes": lanes, "stops": stops, "grid": grid}
