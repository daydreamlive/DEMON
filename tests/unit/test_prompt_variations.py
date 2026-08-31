"""Checks for the variations grid that do not need the checkpoint.

The module is index arithmetic, RNG seeding and a batch/point equivalence, and
the equivalence has already been broken once: `point` built a narrower batch to
save work, sampling consumes the random stream per batch, and two of five
coordinates came back different from the grid's. A client indexing a cached
grid would have seen the answer change under it the moment the grid landed.

Everything here runs without torch or the weights, so it is a real gate in CI
rather than something that only runs on a pod.
"""

from __future__ import annotations

import threading

import pytest

from demos.realtime_motion_graph_web import prompt_variations as pv


def _enter_ok() -> bool:
    with pv._generating():
        return True


class TestClampCoord:
    """`stop` and `lane` arrive from a query string."""

    def test_lane_cannot_index_off_the_batch(self):
        # Unclamped this reached `out[lane]` on a 12-row tensor: IndexError,
        # 500, and raised only AFTER the whole batch had been generated.
        assert pv.clamp_coord(0, 999)[1] == pv.LANES - 1
        assert pv.clamp_coord(0, -5)[1] == 0

    def test_stop_cannot_run_the_sampler_off_its_range(self):
        # amount = stop/(stops-1) feeds top_k and temperature unbounded, so a
        # large stop asks for near-uniform sampling that rarely draws EOS --
        # every row then runs the full token budget.
        assert pv.clamp_coord(10**6, 0)[0] == pv.STOPS - 1
        assert pv.clamp_coord(-5, 0)[0] == 0

    def test_in_range_is_untouched(self):
        assert pv.clamp_coord(7, 3) == (7, 3)


class TestPrefix:
    """Distance is a count of anchor tokens held fixed."""

    def test_zero_distance_keeps_the_whole_anchor(self):
        ids = list(range(31))
        assert pv._prefix_for(ids, 0.0) == ids

    def test_travel_is_monotonic_and_bounded(self):
        ids = list(range(31))
        lengths = [len(pv._prefix_for(ids, s / (pv.STOPS - 1)))
                   for s in range(pv.STOPS)]
        assert lengths == sorted(lengths, reverse=True)
        # MAX_FREE is the point past which the model answers the brief again
        # instead of varying it, so the head must always survive.
        assert lengths[-1] >= len(ids) * (1 - pv.MAX_FREE) - 1

    def test_anchor_shorter_than_the_stop_count(self):
        for n in (1, 2, 3):
            ids = list(range(n))
            for s in range(pv.STOPS):
                got = pv._prefix_for(ids, s / (pv.STOPS - 1))
                assert 0 <= len(got) <= n
                assert got == ids[: len(got)]


class TestSeed:
    def test_stable_across_processes(self):
        # Python's hash() is salted per interpreter, so using it would give a
        # different neighbourhood after every pod restart -- and the pad's
        # whole contract is that a variation you liked is still there.
        assert pv._seed_for("techno") == pv._seed_for("techno")
        assert pv._seed_for("techno") != pv._seed_for("house")
        assert 0 <= pv._seed_for("x" * 500) <= 0x7FFFFFFF


class TestBusyGate:
    """One generation at a time, and a refusal rather than a queue.

    These must run under a TIMEOUT. The first version of `_generating`
    returned the Lock itself, so `with _generating():` re-acquired a
    non-reentrant lock on the same thread and deadlocked -- and a hanging test
    is not a failing test, which is exactly why it shipped. Every case here
    runs on a worker thread and asserts it finished.
    """

    @staticmethod
    def _run(fn, seconds=5.0):
        out = []
        t = threading.Thread(target=lambda: out.append(fn()), daemon=True)
        t.start()
        t.join(timeout=seconds)
        assert not t.is_alive(), "deadlocked -- _generating must not block"
        return out[0]

    def test_the_body_actually_runs(self):
        assert self._run(lambda: [1 for _ in [0] if _enter_ok()][0]) == 1

    def test_second_caller_is_refused_not_queued(self):
        def go():
            with pv._generating():
                try:
                    with pv._generating():
                        return "admitted"
                except pv.Busy:
                    return "refused"
        assert self._run(go) == "refused"

    def test_lock_is_released_for_the_next_caller(self):
        def go():
            with pv._generating():
                pass
            with pv._generating():
                pass
            return not pv._lock.locked()
        assert self._run(go) is True


class TestDegradation:
    def test_absent_checkpoint_returns_empty_not_an_exception(self, monkeypatch):
        # Every docstring promises callers fall back to the hosted backend
        # rather than seeing a 500.
        monkeypatch.setattr(pv, "_load", lambda: None)
        assert pv.point("techno", "sa3", lane=0, stop=3) == ""
        assert pv.enhance("techno", "sa3") == ""

    def test_empty_prompt_is_not_work(self, monkeypatch):
        monkeypatch.setattr(pv, "_load", lambda: ("tok", "model", "cpu"))
        assert pv.point("   ", "sa3", lane=0, stop=3) == ""
        assert pv.enhance("", "sa3") == ""


class TestRouteQuery:
    """How a query string becomes a decision.

    This lived inline in `_process_request`, where it could not be tested, and
    it was wrong in both directions at once: `?lane=5` with no stop served a
    full neighbourhood -- the expensive path, for a request that named a
    coordinate -- while `?lane=abc` refused one, for a field that path never
    read. Every shape now resolves to a coordinate or a refusal.
    """

    def test_no_params_is_the_origin(self):
        # Which is the anchor -- the cheapest possible answer, and the right
        # one. It used to be a full neighbourhood: the most expensive thing on
        # the endpoint, for a request that asked for nothing in particular.
        assert pv.route_query({}) == ("point", 0, 0)

    def test_stop_selects_a_point(self):
        assert pv.route_query({"stop": ["3"]}) == ("point", 3, 0)
        assert pv.route_query({"stop": ["3"], "lane": ["5"]}) == ("point", 3, 5)

    def test_lane_alone_names_a_coordinate(self):
        assert pv.route_query({"lane": ["5"]}) == ("point", 0, 5)

    def test_garbage_is_refused_not_escalated(self):
        assert pv.route_query({"stop": ["abc"]})[0] == "reject"
        assert pv.route_query({"lane": ["abc"]})[0] == "reject"
        assert pv.route_query({"stop": ["3"], "lane": ["abc"]})[0] == "reject"

    def test_absurd_numbers_clamp_rather_than_overflow(self):
        # A 400-digit stop is a valid integer. It used to reach
        # `amount = stop / (stops - 1)` and raise OverflowError -- an
        # unauthenticated 500. Clamping first makes it merely pointless.
        assert pv.route_query({"stop": ["9" * 400]}) == ("point", pv.STOPS - 1, 0)

    def test_blank_is_absent_not_garbage(self):
        assert pv.route_query({"stop": [""], "lane": [""]}) == ("point", 0, 0)

    def test_coordinates_are_clamped(self):
        assert pv.route_query({"stop": ["999"], "lane": ["999"]}) == (
            "point", pv.STOPS - 1, pv.LANES - 1)
        assert pv.route_query({"stop": ["-5"], "lane": ["-5"]}) == ("point", 0, 0)


class TestDownloadResidue:
    """A failed download must not leave something that reads as a checkpoint."""

    def test_import_failure_does_not_crash(self, monkeypatch, tmp_path):
        # `created` was assigned AFTER the lazy huggingface_hub import that the
        # same `except` catches, so a lean image got UnboundLocalError from the
        # function whose whole contract is to degrade. Forcing the ImportError
        # (a None entry in sys.modules makes the in-function import raise) is
        # what actually exercises that path -- letting snapshot_download fail
        # against a bogus repo only covered the download-error branch, and did
        # it with real network I/O from a unit test.
        import sys

        import acestep.model_downloader as md

        monkeypatch.setattr(md, "get_prompt_enhancer_repo", lambda: "x/y")
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        target = tmp_path / "PromptEnhancer"
        ok, msg = md.download_prompt_enhancer_model(target)
        assert ok is False and isinstance(msg, str)
        assert not target.exists(), "an empty dir reads as a staged checkpoint"

    def test_download_failure_removes_only_a_dir_we_created(self, monkeypatch, tmp_path):
        # The download-error branch, without network: snapshot_download raises,
        # and the empty directory the function just made must not survive to
        # read as a staged checkpoint.
        import sys
        import types

        import acestep.model_downloader as md

        def boom(**kwargs):
            raise RuntimeError("no network in unit tests")

        fake = types.SimpleNamespace(snapshot_download=boom)
        monkeypatch.setattr(md, "get_prompt_enhancer_repo", lambda: "x/y")
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
        target = tmp_path / "PromptEnhancer"
        ok, msg = md.download_prompt_enhancer_model(target)
        assert ok is False and "no network" in msg
        assert not target.exists(), "an empty dir reads as a staged checkpoint"
