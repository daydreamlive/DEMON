"""Unit coverage for emit-trim region selection (_finalized_segments).

Trim mode transmits only the buffer region the write frontier just
finalized, instead of re-sending the whole ~9x-overlapping window. The
correctness requirement is: as the frontier sweeps a lap, every sample
is finalized exactly once (no gaps -> no raw-source bleed, no overlaps
-> no wasted re-sends), and a small frontier RETREAT (lead shrinking)
must not be mistaken for a loop WRAP (which would re-emit ~the whole
buffer).
"""
import numpy as np

from acestep.streaming.pipeline_runner import PipelineRunner, _finalized_segments


def test_first_call_anchors_without_emitting():
    # No high-water mark yet -> nothing finalized behind us; just anchor.
    assert _finalized_segments(None, 100, 1000) == ([], 100)


def test_no_advance_emits_nothing():
    # Gap-fill at the same frontier position.
    assert _finalized_segments(500, 500, 1000) == ([], 500)


def test_forward_finalizes_the_gap():
    assert _finalized_segments(500, 540, 1000) == ([(500, 540)], 540)


def test_loop_wrap_emits_tail_then_head():
    # win_start dropped by > half the buffer -> wrap.
    assert _finalized_segments(950, 20, 1000) == ([(950, 1000), (0, 20)], 20)


def test_small_retreat_skips_and_keeps_hwm():
    # Lead shrank a little: that region was already emitted. Skip, and
    # KEEP hwm so we don't re-emit and don't leave a gap.
    assert _finalized_segments(500, 480, 1000) == ([], 500)
    # Forward progress then resumes from the high-water mark, not 480.
    assert _finalized_segments(500, 540, 1000) == ([(500, 540)], 540)


def test_forward_sweep_tiles_contiguously_no_gaps_no_dups():
    n = 1000
    hwm = None
    covered = []
    # Includes a no-advance (40,40) and adjacent ticks (120,121).
    for ws in [10, 25, 40, 40, 73, 120, 121, 500, 999]:
        segs, hwm = _finalized_segments(hwm, ws, n)
        for s, e in segs:
            covered.extend(range(s, e))
    assert hwm == 999
    # First call anchored at 10 (no emit); everything since is finalized
    # exactly once, in ascending contiguous order.
    assert covered == list(range(10, 999))


def test_one_lap_with_wrap_covers_every_sample_exactly_once():
    n = 1000
    hwm = None
    count = [0] * n
    # Exactly one lap from the anchor (30): sweep forward to the end,
    # wrap, then advance back up to — but not past — the anchor. Going
    # past it would (correctly) start a second lap and re-finalize.
    seq = list(range(30, 1000, 37)) + [998]   # forward; anchor@30 (no emit)
    seq += [5, 30]                              # wrap, then back up to anchor
    for ws in seq:
        segs, hwm = _finalized_segments(hwm, ws, n)
        for s, e in segs:
            for i in range(s, e):
                count[i] += 1
    # Every sample finalized exactly once: no gaps (no raw bleed), no
    # overlaps (no wasted re-sends) across the lap.
    bad = [i for i, c in enumerate(count) if c != 1]
    assert not bad, f"{len(bad)} samples not covered exactly once, e.g. {bad[:10]}"


def test_trim_resume_after_untrimmed_fallback_reanchors_frontier():
    runner = PipelineRunner.__new__(PipelineRunner)
    runner._emit_hwm = 500
    calls = []
    runner.on_audio_ready = lambda wav, ss, se: calls.append((ss, se, wav.copy()))
    buf = np.arange(1000, dtype=np.float32).reshape(1000, 1)

    runner._reset_emit_trim_frontier()
    runner._emit_finalized(buf, 540)

    assert calls == []
    assert runner._emit_hwm == 540

    runner._emit_finalized(buf, 580)

    assert [(ss, se) for ss, se, _ in calls] == [(540, 580)]
    np.testing.assert_array_equal(calls[0][2], buf[540:580])


def test_a_session_asks_for_trim_in_its_config(monkeypatch):
    from acestep.streaming.audio_engine import AudioEngine
    from acestep.streaming.config import SessionConfig
    from acestep.streaming.generator_backend import LeadProfile

    class _Backend:
        name = "fake"
        playable_duration_s = staticmethod(lambda: 1.0)
        lead_profile = staticmethod(LeadProfile)

    class _State:
        running = False
        params = {}
        last_activity_ts = 0.0

    monkeypatch.delenv("DEMON_SLICE_EMIT_TRIM", raising=False)
    assert SessionConfig.from_dict({}).slice_emit_trim is False
    assert SessionConfig.from_dict({"slice_emit_trim": True}).slice_emit_trim is True

    def runner(**kw):
        eng = AudioEngine(np.zeros((48000, 2), dtype=np.float32), 48000)
        return PipelineRunner(_Backend(), eng, state=_State(), vae_window=0.36, **kw)

    assert runner()._emit_trim is False
    assert runner(emit_trim=True)._emit_trim is True
    monkeypatch.setenv("DEMON_SLICE_EMIT_TRIM", "1")
    assert runner()._emit_trim is True
