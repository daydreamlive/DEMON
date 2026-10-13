"""AudioEngine canvas revision, write listeners and snapshot."""

import threading

import numpy as np

from acestep.streaming.audio_engine import AudioEngine


def _eng(n=48000, ch=2):
    return AudioEngine(np.zeros((n, ch), dtype=np.float32), 48000)


def test_revision_bumps_once_per_write():
    eng = _eng()
    assert eng.revision == 0
    eng.patch_window(np.ones((100, 2), np.float32), 10)
    assert eng.revision == 1
    eng.patch(np.ones((400, 2), np.float32), 1000)
    assert eng.revision == 2
    eng.swap(np.zeros((24000, 2), np.float32))
    assert eng.revision == 3
    # A write that lands entirely past the end writes nothing and bumps nothing.
    eng.patch_window(np.ones((10, 2), np.float32), 10 ** 7)
    assert eng.revision == 3


def test_listener_gets_range_revision_kind_outside_the_lock():
    eng = _eng()
    seen = []

    def listener(start, end, revision, kind):
        # The buffer lock must be free: a listener can snapshot.
        assert eng._lock.acquire(blocking=False)
        eng._lock.release()
        seen.append((start, end, revision, kind))

    eng.add_write_listener(listener)
    eng.add_write_listener(listener)  # idempotent
    eng.patch_window(np.ones((100, 2), np.float32), 10)
    eng.patch(np.ones((48000, 2), np.float32), 47000)  # clipped at the end
    eng.swap(np.zeros((1000, 1), np.float32))
    assert seen == [
        (10, 110, 1, "patch_window"),
        (47000, 48000, 2, "patch"),
        (0, 1000, 3, "swap"),
    ]
    eng.remove_write_listener(listener)
    eng.patch_window(np.ones((10, 2), np.float32), 0)
    assert len(seen) == 3


def test_failing_listener_is_removed_and_write_lands():
    eng = _eng()
    calls = []

    def bad(*_a):
        calls.append(1)
        raise RuntimeError("boom")

    eng.add_write_listener(bad)
    eng.patch_window(np.ones((10, 2), np.float32), 0)
    eng.patch_window(np.ones((10, 2), np.float32), 0)
    assert calls == [1]
    assert eng.revision == 2
    assert float(eng.current[0, 0]) == 1.0


def test_snapshot_is_a_consistent_copy():
    eng = _eng()
    eng.patch_window(np.full((100, 2), 0.5, np.float32), 0)
    audio, rev = eng.snapshot(-10, 50)
    assert rev == 1 and audio.shape == (50, 2)
    audio[:] = 9.0
    assert float(eng.current[0, 0]) == 0.5
    audio, _ = eng.snapshot(47990, 10 ** 9)
    assert audio.shape == (10, 2)


def test_snapshot_never_tears_against_concurrent_writes():
    eng = _eng(n=4800, ch=1)
    stop = threading.Event()

    def writer():
        v = 0.0
        while not stop.is_set():
            v += 1.0
            eng.patch_window(np.full((4800, 1), v, np.float32), 0)

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(300):
            audio, _rev = eng.snapshot(0, 4800)
            assert float(audio.min()) == float(audio.max())
    finally:
        stop.set()
        t.join()
