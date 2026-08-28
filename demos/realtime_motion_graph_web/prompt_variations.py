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

import contextlib
import os
import threading
import time

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
#: Serialises generation. Held only while a model is actually decoding.
_lock = threading.Lock()
#: Serialises LOADING, which _lock cannot: every caller does _load() before
#: taking _lock, and functools.lru_cache does NOT hold a lock across the wrapped
#: call -- N concurrent first requests run N full checkpoint loads.
_load_lock = threading.Lock()


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


def _resolve_device(torch) -> str:
    """CPU BY DEFAULT, even when a GPU is present.

    This model is small, but a pod's GPU is running a realtime audio pipeline
    and a few hundred milliseconds of contention there is audible. The work
    here is never on a latency path a listener can hear -- it happens while
    generated audio is already playing -- so it belongs on the idle CPU rather
    than in front of the thing that must not stutter.

    ``DEMON_ENHANCER_DEVICE=cuda`` opts in where that trade is worth making
    (a dedicated pod, or once contention has actually been measured);
    ``auto`` restores the usual take-the-GPU-if-present behaviour.
    """
    want = os.environ.get("DEMON_ENHANCER_DEVICE", "cpu").strip().lower()
    if want == "cpu":
        return "cpu"
    if want in ("cuda", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


_loaded = None
_load_failed = False
#: When a failed load may be retried. A failure that can resolve on its own -- a
#: half-staged checkpoint, a volume not mounted yet -- must not latch, or "auto"
#: becomes a one-shot decision taken by whoever sent the first request. But
#: retrying freely is worse: _load holds a BLOCKING lock across from_pretrained,
#: so N requests against a corrupt checkpoint serialise into N x load-time, in
#: front of the audio thread. Measured: 20 requests, 30s wall, 26s worst case,
#: repeating on every later burst. So it backs off instead of choosing.
_retry_after = 0.0
_RETRY_COOLDOWN_S = 60.0


def _load():
    """Tokenizer + model, once.

    Returns None when the checkpoint is absent or unloadable, so every caller
    degrades to the hosted backend instead of failing the request.

    NOT lru_cache: that memoises the return value but does not serialise
    concurrent misses, so the first N simultaneous requests to a cold process
    each load the checkpoint. It also would not let a failure be retried.
    """
    global _loaded, _load_failed, _retry_after
    if _loaded is not None or _load_failed:
        return _loaded
    if time.monotonic() < _retry_after:
        return None
    # CHEAP AND LOCK-FREE when there is no checkpoint. Otherwise every request
    # on a pod configured local-with-no-model queued on a blocking mutex around
    # a stat() -- the unbounded queue in front of the audio thread that the
    # non-blocking gate below exists to prevent, one function earlier.
    if not os.path.isdir(_model_dir()):
        return None
    with _load_lock:
        if _loaded is not None or _load_failed:
            return _loaded
        if time.monotonic() < _retry_after:
            return None   # another thread just failed; do not pile on
        try:
            # INSIDE the try. These used to sit above it, so a deployment
            # without torch/transformers raised ImportError out of the endpoint
            # as a 500 rather than degrading to the hosted backend, which is
            # the opposite of what every docstring here promises.
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration

            path = _model_dir()
            if not os.path.isdir(path):
                # NOT latched. A checkpoint staged after the first request --
                # a slow download, a volume mounted late -- must be picked up,
                # and "auto" is documented as "local when a checkpoint is
                # installed". Latching made that a one-shot decision taken by
                # whoever happened to send the first request.
                return None
            tok = AutoTokenizer.from_pretrained(path, legacy=False)
            model = T5ForConditionalGeneration.from_pretrained(path)
            device = _resolve_device(torch)
            model.to(device).eval()
            _loaded = (tok, model, device)
            return _loaded
        except ImportError:
            # Permanent: the packages are absent and will not appear.
            _load_failed = True
            return None
        except Exception:
            # Recoverable in principle, so no latch -- but backed off, because
            # every retry costs a full load attempt under the lock. Note this
            # catches ValueError/OSError/JSONDecodeError, which is what a
            # corrupt or half-written checkpoint actually raises; ImportError
            # above is the only failure that is genuinely permanent.
            _retry_after = time.monotonic() + _RETRY_COOLDOWN_S
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


class Busy(Exception):
    """Raised instead of queueing. See `_generating`."""


@contextlib.contextmanager
def _generating():
    """Admit one generation, or refuse.

    A blocking lock turns concurrent callers into an unbounded queue in front
    of multi-second work, in the same process (and GIL) as a live audio
    session -- the server already documents that starving that thread closes
    sessions with a keepalive timeout. Refusing immediately bounds the damage
    an unauthenticated caller can do to one in-flight generation.

    A CONTEXT MANAGER, not the Lock. This returned `_lock` itself, so
    `with _generating():` called Lock.__enter__ -- which IS acquire() -- on a
    non-reentrant lock the same thread had just taken. It deadlocked on the
    FIRST call, wedging that connection thread forever while holding the lock,
    so every later request got Busy for the life of the process. A gate meant
    to bound the damage instead guaranteed it.
    """
    if not _lock.acquire(blocking=False):
        raise Busy()
    try:
        yield
    finally:
        _lock.release()


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

    with _generating(), torch.inference_mode():
        enc = tok(_task(deck) + text, return_tensors="pt", max_length=160,
                  truncation=True).to(device)
        out = model.generate(**enc, max_new_tokens=128, num_beams=1,
                             do_sample=False, no_repeat_ngram_size=3)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def point(text: str, deck: str = "sa3", lane: int = 0, stop: int = 0,
          stops: int = STOPS, lanes: int = LANES) -> str:
    """ONE coordinate of the neighbourhood, for callers that cannot wait.

    ``neighbourhood`` is the efficient shape, but it answers in seconds. A
    client that starts exploring before its grid has arrived would otherwise
    have nothing to show, so this answers a single coordinate in roughly the
    time of one decode and the grid takes over silently once it lands.

    Consistent with ``neighbourhood``: same anchor, same lane, same stop gives
    the same string from either call, so the handover is invisible.
    """
    loaded = _load()
    if loaded is None or not text.strip():
        return ""
    stop, lane = clamp_coord(stop, lane, stops, lanes)
    if stop <= 0:
        return enhance(text, deck)
    tok, model, device = loaded
    import torch

    with _generating(), torch.inference_mode():
        enc = tok(_task(deck) + text, return_tensors="pt", max_length=160,
                  truncation=True).to(device)
        anchor_ids, anchor_text = _anchor(tok, model, enc)
        if not anchor_ids:
            return ""
        # THE FULL LANE BATCH, then index it -- not `lane + 1` rows.
        #
        # Sampling consumes the random stream per batch, so a narrower batch
        # gives a row a different draw than the same row inside the grid: two
        # of five test coordinates disagreed that way, which would make the pad
        # jump under the pointer the instant the grid replaced this answer.
        # Computing the whole stop costs one batch instead of a fraction of one
        # (~700 ms against ~300 ms here) and is exactly the work the grid does
        # for that stop, so the two cannot diverge.
        enc_b = {k: v.repeat(lanes, 1) for k, v in enc.items()}
        out = _sample(torch, model, enc_b, anchor_ids, stop, stops,
                      _seed_for(text), device, lanes)
        txt = tok.decode(out[lane], skip_special_tokens=True).strip()
    return txt or anchor_text


def _anchor(tok, model, enc):
    """The greedy decode: ids without specials, and the readable string."""
    out = model.generate(**enc, max_new_tokens=128, num_beams=1,
                         do_sample=False, no_repeat_ngram_size=3)
    ids = [int(i) for i in out[0]
           if int(i) not in (tok.pad_token_id, tok.eos_token_id)]
    return ids, tok.decode(out[0], skip_special_tokens=True).strip()


def route_query(params: dict) -> tuple[str, int, int]:
    """Turn a parsed query string into ("grid"|"point"|"reject", stop, lane).

    A pure function so it can be tested: this lived inside the request handler,
    where it was wrong in both directions at once. `?lane=5` with no stop ran
    the full grid -- roughly ten times the work, for a request that plainly
    names one coordinate -- and `?lane=abc` refused a grid, which never reads
    lane at all.

    Blank and absent mean the same thing (no coordinate asked for). Garbage
    does not: it is a client bug, and answering it with the most expensive
    thing on the endpoint is how a typo becomes a denial of service.
    """
    def read(name):
        raw = (params.get(name, [""])[0] or "").strip()
        if not raw:
            return None, False
        try:
            return int(raw), True
        except ValueError:
            return None, True

    stop, stop_given = read("stop")
    lane, lane_given = read("lane")
    if (stop_given and stop is None) or (lane_given and lane is None):
        return ("reject", 0, 0)
    if not stop_given and not lane_given:
        return ("grid", 0, 0)
    stop, lane = clamp_coord(stop or 0, lane or 0)
    return ("point", stop, lane)


def clamp_coord(stop: int, lane: int, stops: int = STOPS,
                lanes: int = LANES) -> tuple[int, int]:
    """Force a caller-supplied coordinate into the grid.

    Both arrive from a query string. Unclamped, `lane` indexes a tensor row
    (IndexError -> 500, raised AFTER the whole batch has been generated) and
    `stop` scales top_k/temperature without bound, so a large value asks for
    near-uniform sampling that never draws EOS -- a knob for making every
    request maximally expensive.
    """
    return (max(0, min(stops - 1, stop)), max(0, min(lanes - 1, lane)))


def _sample(torch, model, enc_b, anchor_ids, stop, stops, seed, device, rows):
    """One batched sampling pass at distance `stop`. Shared by both entry
    points so a coordinate cannot mean two different things.

    FORKS THE GLOBAL RNG. torch.manual_seed is process-wide and the audio
    engine draws from the same generator -- acestep/engine/stream.py seeds it
    and then immediately calls torch.randn for its noise. Seeding here without
    restoring would hand that slot noise derived from a prompt hash instead of
    the user's seed, silently breaking audio reproducibility. fork_rng puts the
    state back, so nothing outside this function can observe our seeding.

    The converse -- a stream reseeding mid-grid and perturbing OUR sampling --
    is not fixable from this side without a process-wide RNG lock. It costs
    determinism of the grid under concurrent load, which is the far smaller
    harm, and the 503 gate above makes it rare.
    """
    amount = stop / (stops - 1)
    prefix = _prefix_for(anchor_ids, amount)
    dec = torch.tensor([[model.config.decoder_start_token_id] + prefix],
                       device=device).repeat(rows, 1)
    # devices=[] restores the CPU generator ONLY, and torch.manual_seed is not
    # CPU-only -- _manual_seed_impl calls torch.cuda.manual_seed_all first. So
    # the empty list left the CUDA generator seeded from a prompt hash, which
    # is exactly the audio-seed corruption this was written to prevent, and
    # the fleet now defaults this model onto CUDA. Fork whatever devices exist.
    _devices = (list(range(torch.cuda.device_count()))
                if torch.cuda.is_available() else [])
    with torch.random.fork_rng(devices=_devices):
        torch.manual_seed(seed)
        return _generate(torch, model, enc_b, dec, amount)


def _generate(torch, model, enc_b, dec, amount):
    return model.generate(
        **enc_b, decoder_input_ids=dec, max_new_tokens=128, num_beams=1,
        do_sample=True,
        top_k=TOPK_MIN + round(amount * (TOPK_MAX - TOPK_MIN)),
        temperature=TEMP_MIN + amount * (TEMP_MAX - TEMP_MIN),
        no_repeat_ngram_size=3,
    )


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

    with _generating(), torch.inference_mode():
        enc = tok(_task(deck) + text, return_tensors="pt", max_length=160,
                  truncation=True).to(device)
        anchor_ids, anchor_text = _anchor(tok, model, enc)
        if not anchor_ids:
            return {}

        seed = _seed_for(text)
        grid = [[anchor_text] for _ in range(lanes)]
        enc_b = {k: v.repeat(lanes, 1) for k, v in enc.items()}

        for s in range(1, stops):
            out = _sample(torch, model, enc_b, anchor_ids, s, stops, seed,
                          device, lanes)
            for lane in range(lanes):
                txt = tok.decode(out[lane], skip_special_tokens=True).strip()
                grid[lane].append(txt or anchor_text)

    return {"anchor": anchor_text, "lanes": lanes, "stops": stops, "grid": grid}
