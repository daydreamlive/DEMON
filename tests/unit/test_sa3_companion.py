"""A companion model in the same process: ``--companion-checkpoint``
entries, a companion context that loads beside the main one and survives a
main switch, session groups for preemption, a scoring request that still
clears everything, and one text encoder object for two loaded models whose
encoder weight files are equal."""

from __future__ import annotations

import gc
import threading
from types import SimpleNamespace

import pytest

import demos.realtime_motion_graph_web.ws_adapter as wa
from acestep.engine import sa3_context
from acestep.streaming import sa3_session
from acestep.streaming.families import FAMILY_SPECS, resolve_served
from demos.realtime_motion_graph_web.server import served_checkpoints_arg

MAIN_SFX = resolve_served(
    ["sa3-medium+plugin.ext", "sa3-medium"], companions=["sa3-sfx"],
)
MAIN_A, MAIN_B, SFX = MAIN_SFX


# ---- the flag and its entries ----


def test_companion_entries_follow_the_main_ones():
    assert [(e.name, e.model_id, e.extension, e.companion) for e in MAIN_SFX] == [
        ("sa3-medium+plugin.ext", "medium", "plugin.ext", False),
        ("sa3-medium", "medium", None, False),
        ("sa3-sfx", "small-sfx", None, True),
    ]


def test_without_companions_nothing_changes():
    entries = resolve_served(["sa3-medium", "sa3-sfx"])
    assert [e.companion for e in entries] == [False, False]
    assert served_checkpoints_arg(["--checkpoint", "sa3-medium,sa3-sfx"]) == entries


def test_companion_flag_parses_both_forms():
    for args in (
        ["--checkpoint", "sa3-medium", "--companion-checkpoint", "sa3-sfx"],
        ["--checkpoint=sa3-medium", "--companion-checkpoint=sa3-sfx"],
    ):
        entries = served_checkpoints_arg(args)
        assert [(e.name, e.companion) for e in entries] == [
            ("sa3-medium", False), ("sa3-sfx", True),
        ]


def test_companion_flag_takes_a_comma_list():
    entries = served_checkpoints_arg([
        "--checkpoint", "sa3-medium",
        "--companion-checkpoint", " sa3-sfx , sa3-small,sa3-sfx",
    ])
    assert [(e.name, e.companion) for e in entries] == [
        ("sa3-medium", False), ("sa3-sfx", True), ("sa3-small", True),
    ]


def test_a_name_in_both_lists_exits():
    with pytest.raises(SystemExit) as exc:
        served_checkpoints_arg([
            "--checkpoint", "sa3-medium,sa3-sfx",
            "--companion-checkpoint", "sa3-sfx",
        ])
    assert "'sa3-sfx'" in str(exc.value)


def test_companion_of_another_family_exits():
    with pytest.raises(SystemExit) as exc:
        served_checkpoints_arg([
            "--checkpoint", "sa3-medium", "--companion-checkpoint", "xl",
        ])
    assert "'xl'" in str(exc.value) and "'sa3-medium'" in str(exc.value)


def test_companion_without_checkpoint_exits():
    with pytest.raises(SystemExit):
        served_checkpoints_arg(["--companion-checkpoint", "sa3-sfx"])


def test_session_names_the_companion_like_any_entry():
    entry, err = wa._resolve_session_checkpoint(
        {"checkpoint": "sa3-sfx"}, default="medium", backend_family="sa3",
        served=MAIN_SFX,
    )
    assert (entry, err) == (SFX, None)


# ---- reached by path ----

BY_PATH = resolve_served(["sa3-medium"], companions=["sa3-sfx@/companion"])


def _by(config, path):
    return wa._resolve_session_checkpoint(
        config, default="medium", backend_family="sa3", served=BY_PATH,
        path=path,
    )


def test_path_entry_parses():
    assert [(e.name, e.model_id, e.companion, e.path) for e in BY_PATH] == [
        ("sa3-medium", "medium", False, None),
        ("sa3-sfx", "small-sfx", True, "/companion"),
    ]
    entries = served_checkpoints_arg([
        "--checkpoint", "sa3-medium",
        "--companion-checkpoint", "sa3-sfx@/companion",
    ])
    assert entries == BY_PATH


def test_path_picks_the_companion():
    assert _by({}, "/companion") == (BY_PATH[1], None)
    assert _by({"checkpoint": None}, "/companion") == (BY_PATH[1], None)


def test_a_named_checkpoint_wins_over_the_path():
    assert _by({"checkpoint": "sa3-medium"}, "/companion") == (BY_PATH[0], None)


def test_an_unknown_path_gets_the_default():
    assert _by({}, "/") == (BY_PATH[0], None)
    assert _by({}, "/other") == (BY_PATH[0], None)
    assert _by({}, None) == (BY_PATH[0], None)


def test_socket_path_drops_the_query():
    ws = SimpleNamespace(request=SimpleNamespace(path="/companion?token=x"))
    assert wa._socket_path(ws) == "/companion"
    assert wa._socket_path(SimpleNamespace()) is None


def test_a_path_without_a_slash_exits():
    with pytest.raises(SystemExit) as exc:
        served_checkpoints_arg([
            "--checkpoint", "sa3-medium",
            "--companion-checkpoint", "sa3-sfx@companion",
        ])
    assert "must start with /" in str(exc.value)


def test_two_companions_with_one_path_exit():
    with pytest.raises(SystemExit) as exc:
        served_checkpoints_arg([
            "--checkpoint", "sa3-medium",
            "--companion-checkpoint", "sa3-sfx@/fx,sa3-small@/fx",
        ])
    assert "'/fx'" in str(exc.value)


# ---- residency ----


@pytest.fixture
def _contexts(monkeypatch):
    import acestep.engine.sa3_trt as sa3_trt

    events = []

    class _Ctx:
        def __init__(self, model_id, *, companion=False, **_):
            self.model_id = model_id
            self.companion = companion
            events.append(("load", model_id))

        def close(self):
            events.append(("close", self.model_id))

    monkeypatch.setattr(sa3_context, "SA3Context", _Ctx)
    monkeypatch.setattr(sa3_session, "_CONTEXTS", {})
    monkeypatch.setattr(sa3_trt, "_ENGINE_CACHE", {"dit.trt": object()})
    return sa3_trt, events


def test_a_companion_loads_without_evicting(_contexts):
    sa3_trt, events = _contexts
    main = sa3_session.get_sa3_context("medium")
    sfx = sa3_session.get_sa3_context("small-sfx", companion=True)
    assert events == [("load", "medium"), ("load", "small-sfx")]
    assert sfx.companion and not main.companion
    assert sa3_session.get_sa3_context("medium") is main
    assert sa3_session.get_sa3_context("small-sfx", companion=True) is sfx
    assert sa3_trt._ENGINE_CACHE  # nothing evicted, nothing cleared


def test_a_main_switch_leaves_the_companion(_contexts):
    sa3_trt, events = _contexts
    sa3_session.get_sa3_context("medium")
    sfx = sa3_session.get_sa3_context("small-sfx", companion=True)
    sa3_session.get_sa3_context("small-music")
    assert events == [
        ("load", "medium"), ("load", "small-sfx"),
        ("close", "medium"), ("load", "small-music"),
    ]
    assert list(sa3_session._CONTEXTS) == [
        ("small-sfx", None, None), ("small-music", None, None),
    ]
    assert sa3_session.get_sa3_context("small-sfx", companion=True) is sfx
    assert sa3_trt._ENGINE_CACHE == {}  # the whole cache, as before


def test_a_main_load_beside_only_a_companion_evicts_nothing(_contexts):
    _, events = _contexts
    sa3_session.get_sa3_context("small-sfx", companion=True)
    sa3_session.get_sa3_context("medium")
    assert events == [("load", "small-sfx"), ("load", "medium")]


def test_evict_by_default_drops_companions_too(_contexts):
    _, events = _contexts
    sa3_session.get_sa3_context("medium")
    sa3_session.get_sa3_context("small-sfx", companion=True)
    assert sa3_session.evict_sa3_contexts() == 2
    assert sa3_session._CONTEXTS == {}


def test_release_sa3_evicts_both(_contexts):
    sa3_trt, events = _contexts
    sa3_session.get_sa3_context("medium")
    sa3_session.get_sa3_context("small-sfx", companion=True)
    assert sa3_session.release_sa3() == 2
    assert sorted(events[2:]) == [("close", "medium"), ("close", "small-sfx")]
    assert sa3_session._CONTEXTS == {}
    assert sa3_trt._ENGINE_CACHE == {}


def test_create_hands_companion_to_the_family_creator(monkeypatch):
    import acestep.streaming.session as session_mod

    seen = {}

    def _create(cls, **kwargs):
        seen.update(kwargs)
        return "session"

    monkeypatch.setattr(
        session_mod, "get_family",
        lambda family: SimpleNamespace(create_session=_create),
    )
    out = session_mod.StreamingSession.create(
        audio=None, config=SimpleNamespace(backend="sa3"),
        checkpoint="small-sfx", session_id="s", companion=True,
    )
    assert out == "session" and seen["companion"] is True


# ---- session groups ----


@pytest.fixture
def _active_slot():
    with wa._ACTIVE_SLOT_LOCK:
        prev = list(wa._ACTIVE_SESSIONS)
        wa._ACTIVE_SESSIONS.clear()
    with wa._SCORE_TURN:
        wa._SCORE_LINE.clear()
    yield wa._ACTIVE_SESSIONS
    with wa._SCORE_TURN:
        wa._SCORE_LINE.clear()
    with wa._ACTIVE_SLOT_LOCK:
        wa._ACTIVE_SESSIONS[:] = prev


def _live(session_id, entry):
    streaming = SimpleNamespace(
        state=SimpleNamespace(running=True), closed=threading.Event(),
    )
    streaming.closed.set()
    ws = SimpleNamespace(closes=[])
    ws.close = lambda code, reason="": ws.closes.append(code)
    with wa._ACTIVE_SLOT_LOCK:
        wa._ACTIVE_SESSIONS.append(
            wa._ActiveSession(session_id, streaming, ws, entry),
        )
    return streaming


def _ids():
    with wa._ACTIVE_SLOT_LOCK:
        return [a.session_id for a in wa._ACTIVE_SESSIONS]


def test_groups():
    assert wa._session_group(None) == wa._MAIN_GROUP
    assert wa._session_group(MAIN_A) == wa._session_group(MAIN_B) == wa._MAIN_GROUP
    assert wa._session_group(SFX) == SFX


def test_a_companion_beside_a_main_session_keeps_it(_active_slot):
    main = _live("main", MAIN_A)
    keep = wa._preempt_keep(4, SFX)
    assert keep == 3
    wa._preempt_active_session("new", keep=keep, group=wa._session_group(SFX))
    assert main.state.running is True and _ids() == ["main"]


def test_the_companion_cap_counts_only_companion_sessions(_active_slot):
    main = _live("main", MAIN_A)
    first = _live("sfx1", SFX)
    _live("sfx2", SFX)
    wa._preempt_active_session("new", keep=1, group=wa._session_group(SFX))
    assert main.state.running is True and first.state.running is False
    assert _ids() == ["main", "sfx2"]


def test_a_main_switch_preempts_only_main_sessions(_active_slot):
    main = _live("main", MAIN_A)
    sfx = _live("sfx", SFX)
    keep = wa._preempt_keep(1, MAIN_B)
    assert keep == 0
    wa._preempt_active_session("new", keep=keep, group=wa._session_group(MAIN_B))
    assert main.state.running is False and sfx.state.running is True
    assert _ids() == ["sfx"]


def test_a_main_session_ignores_a_live_companion_for_its_rule(_active_slot):
    _live("sfx", SFX)
    _live("small", MAIN_B)
    assert wa._preempt_keep(3, MAIN_B) == 2


def test_no_group_preempts_every_session(_active_slot):
    main = _live("main", MAIN_A)
    sfx = _live("sfx", SFX)
    wa._preempt_active_session("new")
    assert main.state.running is False and sfx.state.running is False
    assert _ids() == []


def test_a_score_request_preempts_both_groups(monkeypatch, _active_slot):
    released = []
    monkeypatch.setitem(
        FAMILY_SPECS, "fake-sa3",
        SimpleNamespace(release=lambda: released.append(_ids())),
    )
    main = _live("main", MAIN_A)
    sfx = _live("sfx", SFX)

    class _Ws:
        sent: list = []

        def send(self, text):
            self.sent.append(text)

        def close(self, code=1000, reason=""):
            pass

        def __iter__(self):
            return iter(())

    wa._handle_score_request(_Ws(), "fake-sa3")
    assert main.state.running is False and sfx.state.running is False
    assert released == [[], []]


# ---- one text encoder for both models ----


class T5GemmaConditioner:
    """Stands in for the vendor class: the encoder sits in ``__dict__``."""

    def __init__(self):
        self.__dict__["model"] = _Encoder()


class _Encoder:
    pass


def _sam():
    cond = T5GemmaConditioner()
    return SimpleNamespace(model=SimpleNamespace(
        conditioner=SimpleNamespace(conditioners={"prompt": cond}),
    )), cond


def _checkpoint(tmp_path, name, payload: bytes):
    folder = tmp_path / name / sa3_context.TEXT_ENCODER_SUBDIR
    folder.mkdir(parents=True)
    (folder / "model.safetensors").write_bytes(payload)
    return tmp_path / name


@pytest.fixture
def _registry(monkeypatch):
    import weakref

    monkeypatch.setattr(sa3_context, "_SHARED_TEXT_ENCODERS",
                        weakref.WeakKeyDictionary())
    monkeypatch.setattr(sa3_context, "_TEXT_ENCODER_SHA_CACHE", {})


def test_equal_files_share_one_encoder(tmp_path, _registry):
    music = _checkpoint(tmp_path, "medium", b"t5 weights")
    sfx = _checkpoint(tmp_path, "small-sfx", b"t5 weights")
    sam_a, cond_a = _sam()
    sam_b, cond_b = _sam()
    own_b = cond_b.model
    sa3_context._share_text_encoder(sam_a, music, "medium")
    sa3_context._share_text_encoder(sam_b, sfx, "small-sfx")
    assert cond_b.model is cond_a.model
    assert own_b is not cond_a.model


def test_different_files_keep_their_own(tmp_path, _registry):
    a = _checkpoint(tmp_path, "a", b"one")
    b = _checkpoint(tmp_path, "b", b"two")
    sam_a, cond_a = _sam()
    sam_b, cond_b = _sam()
    sa3_context._share_text_encoder(sam_a, a, "a")
    sa3_context._share_text_encoder(sam_b, b, "b")
    assert cond_a.model is not cond_b.model


def test_the_encoder_leaves_with_its_last_user(tmp_path, _registry):
    path = _checkpoint(tmp_path, "m", b"w")
    sam, cond = _sam()
    sa3_context._share_text_encoder(sam, path, "m")
    assert len(sa3_context._SHARED_TEXT_ENCODERS) == 1
    sam = cond = None
    gc.collect()
    assert len(sa3_context._SHARED_TEXT_ENCODERS) == 0


def test_a_single_context_never_hashes(tmp_path, _registry, monkeypatch):
    path = _checkpoint(tmp_path, "m", b"w")
    hashed = []
    monkeypatch.setattr(sa3_context, "text_encoder_weights_sha256",
                        lambda d: hashed.append(d) or "x")
    sam, cond = _sam()
    sa3_context._share_text_encoder(sam, path, "m")
    assert hashed == []
    assert list(sa3_context._SHARED_TEXT_ENCODERS.values()) == [path]


def test_a_second_context_hashes_both_files(tmp_path, _registry):
    a = _checkpoint(tmp_path, "a", b"w")
    b = _checkpoint(tmp_path, "b", b"w")
    sam_a, _ = _sam()
    sa3_context._share_text_encoder(sam_a, a, "a")
    assert sa3_context._TEXT_ENCODER_SHA_CACHE == {}
    sam_b, _ = _sam()
    sa3_context._share_text_encoder(sam_b, b, "b")
    assert len(sa3_context._TEXT_ENCODER_SHA_CACHE) == 2


def test_a_missing_file_shares_nothing(tmp_path, _registry):
    sam, cond = _sam()
    own = cond.model
    sa3_context._share_text_encoder(sam, tmp_path / "none", "m")
    assert cond.model is own
    assert len(sa3_context._SHARED_TEXT_ENCODERS) == 0
