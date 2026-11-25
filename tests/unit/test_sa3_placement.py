"""PyTorch parts wait in system memory while an engine serves them.

Covers :class:`acestep.engine.sa3_placement.CardPart` and every place the
SA3 context, codecs, backend and boot preload take or drop a hold. No GPU:
the "card" moves are recorded by the stand-in module instead of made.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from acestep.engine import sa3_context as ctx_mod
from acestep.engine import sa3_trt
from acestep.engine.sa3_adapter import SA3Adapter
from acestep.engine.sa3_placement import CardPart, pinned_part
from acestep.streaming.knobs import KnobState
from acestep.streaming.sa3_backend import SA3Backend, sa3_knob_specs
from tests.unit.test_sa3_backend import (
    SA3_SAMPLE_RATE,
    C,
    T,
    _FakeCodec,
    _FakeCond,
    _FakeLoraManager,
    _FakeMirror,
    _schedule_builder_factory,
    _ZeroDit,
)


class _Part(torch.nn.Module):
    """A module whose ``.to`` records the move instead of making it."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.moves: list = []
        self.where = "card"

    def to(self, device, *a, **k):  # noqa: D401 - recording stand-in
        dev = torch.device(device)
        self.where = "system" if dev.type == "cpu" else "card"
        self.moves.append(self.where)
        return self

    def forward(self, x, t, **kwargs):
        return torch.zeros_like(x)


def _part(name="dit"):
    mod = _Part()
    return CardPart(name, mod, "cuda"), mod


def _log_lines(monkeypatch):
    from acestep.engine import sa3_placement

    lines: list = []

    class _Log:
        def info(self, msg, *a):
            lines.append(("INFO", msg.split()[0]))

        def warning(self, msg, *a):
            lines.append(("WARNING", msg.split()[0]))

        def error(self, msg, *a):
            lines.append(("ERROR", msg.split()[0]))

    monkeypatch.setattr(sa3_placement, "logger", _Log())
    return lines


# ---- the holder ----------------------------------------------------------------


def test_nothing_moves_until_an_engine_parks_the_part():
    part, mod = _part()
    assert not part.parked and mod.moves == []
    part.park_if_free()
    assert part.parked and mod.where == "system"


def test_holds_count_and_the_last_release_parks(monkeypatch):
    lines = _log_lines(monkeypatch)
    part, mod = _part()
    part.park_if_free()
    part.hold("no_trt_engine")
    assert part.count == 1 and mod.where == "card" and not part.parked
    part.hold("lora_active")
    assert part.count == 2 and mod.moves == ["system", "card"]  # one move in
    part.park_if_free()                                          # held: stays
    assert mod.where == "card"
    part.release()
    assert part.count == 1 and mod.where == "card"
    part.release()
    assert part.count == 0 and mod.where == "system" and part.parked
    assert mod.moves == ["system", "card", "system"]
    assert lines == [
        ("INFO", "sa3_torch_part_parked"),
        ("WARNING", "sa3_torch_part_loaded"),
        ("INFO", "sa3_torch_part_parked"),
    ]


def test_a_hold_on_a_part_never_parked_moves_nothing():
    part, mod = _part()
    part.hold("backend_eager")
    part.release()
    # The last release parks it: a part waits in system memory whenever
    # its last holder lets go.
    assert mod.moves == ["system"]


def test_release_without_a_hold_is_refused_and_logged(monkeypatch):
    lines = _log_lines(monkeypatch)
    part, mod = _part()
    part.release()
    assert part.count == 0 and mod.moves == []
    assert lines == [("ERROR", "sa3_torch_part_release_unheld")]


def test_held_is_a_scoped_hold():
    part, mod = _part()
    part.park_if_free()
    with part.held("lora_refit") as m:
        assert m is mod and mod.where == "card" and part.count == 1
    assert mod.where == "system" and part.count == 0


def test_a_pinned_part_never_moves():
    mod = _Part()
    part = CardPart("dit", mod, "cuda", pinned=True)
    part.park_if_free()
    part.hold("x")
    part.release()
    part.release()
    with part.held("y"):
        pass
    assert mod.moves == [] and part.count == 0


# ---- the context -----------------------------------------------------------------


def _bare_context(model_id="small-sfx"):
    c = ctx_mod.SA3Context.__new__(ctx_mod.SA3Context)
    c.model_id = model_id
    c.extension_runtime = None
    c.torch_dit, c.dit_module = _part("dit")
    c.sam = SimpleNamespace(model=SimpleNamespace(
        model=c.dit_module,
        pretransform=SimpleNamespace(scale=1.0),
    ))
    c.torch_decoder, c.decoder_module = _part("decoder")
    return c


def test_engine_session_parks_the_transformer(monkeypatch):
    made = []

    class _TRTDit:
        def __init__(self, path, **kw):
            made.append(path)

    monkeypatch.setattr(sa3_trt, "find_dit_engine", lambda *a, **k: "/eng")
    monkeypatch.setattr(sa3_trt, "SA3TRTDit", _TRTDit)
    c = _bare_context()
    dit = c.make_dit(latent_frames=614, seconds_total=60.0, backend="tensorrt")
    assert isinstance(dit, _TRTDit)
    assert c.torch_dit.parked and c.torch_dit.count == 0
    assert c.dit_module.where == "system"


def test_no_engine_and_eager_backend_hold_the_transformer(monkeypatch):
    monkeypatch.setattr(sa3_trt, "find_dit_engine", lambda *a, **k: None)
    c = _bare_context()
    c.torch_dit.park_if_free()
    assert c.make_dit(latent_frames=613, seconds_total=60.0, backend="tensorrt") is c.dit
    assert c.torch_dit.count == 1 and c.dit_module.where == "card"
    assert c.make_dit(latent_frames=614, seconds_total=60.0, backend="eager") is c.dit
    assert c.torch_dit.count == 2


def test_an_extension_context_pins_both_parts(monkeypatch):
    """A real SA3Context built over a stand-in loader: with an extension
    both parts are pinned and never move; without one they are live
    holders over the transformer and the SAME decoder."""
    dit, decoder = _Part(), _Part()
    sam = SimpleNamespace(
        device="cpu",
        model=SimpleNamespace(
            model=dit,
            sample_rate=44100,
            io_channels=256,
            diffusion_objective="rf_denoiser",
            pretransform=SimpleNamespace(
                downsampling_ratio=4096,
                model=SimpleNamespace(decoder=decoder),
            ),
        ),
    )
    loader = SimpleNamespace(
        checkpoint_dir=lambda model_id: "/ckpt",
        load_local_model=lambda *a, **k: sam,
    )
    monkeypatch.setattr(ctx_mod, "require_sa3_vendor", lambda: None)
    monkeypatch.setattr(ctx_mod, "import_loader_helpers", lambda: loader)
    monkeypatch.setattr(ctx_mod, "import_stream_helpers", lambda: None)

    ext = SimpleNamespace(
        validate_base_model=lambda desc: None,
        install=lambda sam, cfg: object(),
        qualified_id="probe.ext",
    )
    c = ctx_mod.SA3Context("small-sfx", extension=ext)
    assert c.torch_dit.pinned and c.torch_decoder.pinned
    assert c.torch_dit.module is dit and c.torch_decoder.module is decoder
    c.torch_dit.park_if_free()
    c.torch_decoder.park_if_free()
    assert dit.moves == [] and decoder.moves == []

    stock = ctx_mod.SA3Context("small-sfx")
    assert not stock.torch_dit.pinned and not stock.torch_decoder.pinned
    assert stock.torch_decoder.module is decoder


# ---- the codecs ------------------------------------------------------------------


class _Helpers:
    def __init__(self):
        self.eager = 0

    def sa3_decode_rng(self, *_a, **_k):
        import contextlib

        return contextlib.nullcontext()

    def decode_sa3_latent(self, sam, latent):
        self.eager += 1
        return torch.zeros(1, 2, latent.shape[-1] * 4)


def _same_s_context(monkeypatch, engine):
    c = _bare_context()
    c._helpers = _Helpers()
    c.checkpoint_dir = "/ckpt"
    monkeypatch.setattr(sa3_trt, "same_decoder_weights_sha256", lambda d: "ab" * 32)
    monkeypatch.setattr(sa3_trt, "find_same_s_decode_engine", lambda sha: engine)
    monkeypatch.setattr(
        sa3_trt, "SameSTRTDecoder",
        lambda path: SimpleNamespace(decode=lambda x: torch.zeros(2, x.shape[-1] * 4)),
    )
    return c


def test_same_s_engine_parks_the_decoder_and_odd_frames_hold_it(monkeypatch):
    c = _same_s_context(monkeypatch, ("/same_s", 32, 1292))
    codec = ctx_mod.SA3SAMECodec(c, use_trt=True)
    assert c.torch_decoder.parked and c.torch_decoder.count == 0

    codec.decode_full(torch.zeros(1, 256, 614))         # engine serves
    assert c.torch_decoder.count == 0
    codec.decode_full(torch.zeros(1, 256, 613))         # odd: eager
    codec.decode_full(torch.zeros(1, 256, 611))
    assert c.torch_decoder.count == 1                   # one hold, kept
    assert c.decoder_module.where == "card"
    codec.close()
    codec.close()                                       # idempotent
    assert c.torch_decoder.count == 0 and c.torch_decoder.parked


def test_same_s_without_an_engine_holds_from_construction(monkeypatch):
    c = _same_s_context(monkeypatch, None)
    c.torch_decoder.park_if_free()
    codec = ctx_mod.SA3SAMECodec(c, use_trt=True)
    assert c.torch_decoder.count == 1 and c.decoder_module.where == "card"
    codec.close()
    eager = ctx_mod.SA3SAMECodec(c, use_trt=False)
    assert c.torch_decoder.count == 1
    eager.close()
    assert c.torch_decoder.count == 0


def test_window_codec_parks_with_engines_and_holds_on_first_eager_use(monkeypatch):
    c = _bare_context("medium")
    c._helpers = _Helpers()
    c.sample_rate, c.downsampling_ratio = 16, 4
    monkeypatch.setattr(sa3_trt, "find_same_l_engines", lambda: [("/w", 32, 96)])
    monkeypatch.setattr(sa3_trt, "find_same_l_window_engine", lambda: None)
    codec = ctx_mod.SA3SAMEWindowCodec(c, use_trt=True)
    assert c.torch_decoder.parked and c.torch_decoder.count == 0
    codec.decode_full(torch.zeros(1, 256, 500))         # no engine covers it
    codec.decode_full(torch.zeros(1, 256, 500))
    assert c._helpers.eager == 2
    assert c.torch_decoder.count == 1 and c.decoder_module.where == "card"
    codec.close()
    assert c.torch_decoder.parked

    eager = ctx_mod.SA3SAMEWindowCodec(c, use_trt=False)
    assert c.torch_decoder.count == 0                   # nothing until used
    eager.close()


# ---- the backend -----------------------------------------------------------------


def _backend(part, eager, accel=None, **kw):
    accel = accel if accel is not None else _ZeroDit()
    adapter = SA3Adapter(
        accel, schedule_builder=_schedule_builder_factory(3),
        device="cpu", dtype=torch.float32,
    )
    return SA3Backend(
        adapter=adapter,
        codec=kw.pop("codec", _FakeCodec()),
        cond=_FakeCond(),
        schedule_builder_factory=_schedule_builder_factory,
        knob_state=KnobState(sa3_knob_specs()),
        steps=3,
        depth=2,
        vae_window_s=0.1,
        eager_dit=eager,
        torch_dit=part,
        **kw,
    )


def test_engine_session_lora_swap_takes_and_drops_the_hold():
    part, mod = _part()
    part.park_if_free()
    mgr = _FakeLoraManager(ids=["x"])
    b = _backend(part, mod, lora_manager=mgr, use_lora=True)
    assert part.count == 0

    b.enable_lora("x", strength=1.0)
    assert b.adapter.dit is mod
    assert part.count == 1 and mod.where == "card"
    b.enable_lora("x", strength=0.5)                   # still one hold
    assert part.count == 1

    b.disable_lora("x")
    assert b.adapter.dit is not mod
    assert part.count == 0 and mod.where == "system"


def test_refit_sync_holds_the_transformer_for_the_merge():
    part, mod = _part()
    part.park_if_free()
    seen = []

    class _Mirror(_FakeMirror):
        def sync(self, *, reason=""):
            seen.append((reason, part.count, mod.where))
            return super().sync(reason=reason)

    mgr = _FakeLoraManager(ids=["x"])
    b = _backend(part, mod, lora_manager=mgr, use_lora=True, refit_mirror=_Mirror())
    b.enable_lora("x", strength=1.0)
    b.set_lora_strength("x", 0.5)
    b._pending_lora_strengths["x"] = 0.25
    b._apply_pending_lora_strengths()
    b.disable_lora("x")
    # While the LoRA is active the session's lora_active hold is there
    # too; the sync that restores base weights after the last disable
    # runs under the mirror's own hold.
    assert seen == [
        ("enable_lora", 2, "card"),
        ("set_lora_strength", 2, "card"),
        ("strength", 2, "card"),
        ("disable_lora", 1, "card"),
    ]
    assert part.count == 0 and mod.where == "system"


def test_a_refit_session_holds_while_a_lora_is_active_and_strength_moves_nothing():
    part, mod = _part()
    part.park_if_free()
    mgr = _FakeLoraManager(ids=["x"])
    b = _backend(part, mod, lora_manager=mgr, use_lora=True, refit_mirror=_FakeMirror())
    accel = b.adapter.dit

    b.enable_lora("x", strength=1.0)
    assert part.count == 1 and mod.where == "card"
    assert b.adapter.dit is accel                      # no swap under a mirror
    moves = list(mod.moves)
    b.set_lora_strength("x", 0.5)
    b._pending_lora_strengths["x"] = 0.25
    b._apply_pending_lora_strengths()
    assert mod.moves == moves and part.count == 1      # nothing moved

    b.disable_lora("x")
    assert part.count == 0 and mod.where == "system"


def test_close_releases_every_hold_and_closes_the_codec():
    part, mod = _part()
    part.hold("no_trt_engine")                         # make_dit's hold
    mgr = _FakeLoraManager(ids=["x"])
    codec = _FakeCodec()
    b = _backend(part, mod, accel=mod, lora_manager=mgr, use_lora=True, codec=codec)
    assert b._holds_torch_dit
    b.close()
    assert part.count == 0 and mod.where == "system"
    assert codec.closed
    b.close()                                          # nothing left to drop
    assert part.count == 0


def test_close_drops_the_lora_swap_hold_too():
    part, mod = _part()
    mgr = _FakeLoraManager(ids=["x"])
    b = _backend(part, mod, lora_manager=mgr, use_lora=True)
    b.enable_lora("x", strength=1.0)
    assert part.count == 1
    b.close()
    assert part.count == 0 and part.parked


class _Sized(_FakeCond):
    def __init__(self, frames):
        super().__init__()
        self.latent_frames = frames
        self.audio_sample_size = frames * 4096


def _resizing_backend(part, mod, accel, new_dit, *, encode_fails=False):
    def encoder(waveform, sample_rate, sample_size):
        if encode_fails:
            raise RuntimeError("encode failed")
        return torch.randn(1, C, int(sample_size) // 4096)

    def resizer(new_duration_s, tags_a, tags_b, steps):
        cond = _Sized(2 * T)
        dit = new_dit()
        return (
            cond.audio_sample_size / SA3_SAMPLE_RATE, cond, None, dit,
            _schedule_builder_factory,
        )

    return _backend(
        part, mod, accel=accel,
        source_latent_bct=torch.randn(1, C, T),
        source_encoder=encoder, resizer=resizer, prompt_tags="tags",
    )


def _eager_from_make_dit(part, mod):
    def make():
        part.hold("no_trt_engine")
        return mod
    return make


def test_resize_from_an_engine_to_eager_keeps_the_new_hold():
    part, mod = _part()
    part.park_if_free()
    b = _resizing_backend(part, mod, None, _eager_from_make_dit(part, mod))
    b.handle_swap_source(torch.zeros(2, 96000), 48000, duration_s=2 * T * 4096 / SA3_SAMPLE_RATE)
    assert b.adapter.dit is mod and b._holds_torch_dit
    assert part.count == 1 and mod.where == "card"
    b.close()
    assert part.count == 0 and part.parked


def test_resize_from_eager_to_an_engine_releases_the_old_hold():
    part, mod = _part()
    part.hold("no_trt_engine")
    b = _resizing_backend(part, mod, mod, _ZeroDit)
    assert b._holds_torch_dit
    b.handle_swap_source(torch.zeros(2, 96000), 48000, duration_s=2 * T * 4096 / SA3_SAMPLE_RATE)
    assert b.adapter.dit is not mod and not b._holds_torch_dit
    assert part.count == 0 and part.parked


def test_a_resize_that_fails_gives_back_the_hold_it_took():
    part, mod = _part()
    part.park_if_free()
    b = _resizing_backend(
        part, mod, None, _eager_from_make_dit(part, mod), encode_fails=True,
    )
    try:
        b.handle_swap_source(
            torch.zeros(2, 96000), 48000, duration_s=2 * T * 4096 / SA3_SAMPLE_RATE,
        )
    except RuntimeError:
        pass
    assert b.adapter.dit is not mod
    assert part.count == 0 and part.parked


# ---- boot preload ----------------------------------------------------------------


def test_preload_closes_its_codec_and_parks_the_transformer(monkeypatch):
    from acestep.streaming import sa3_session

    part, mod = _part()
    codec = _FakeCodec()

    class _Ctx:
        torch_dit = part

        def effective_dit_backend(self, requested):
            return requested, ""

        def effective_codec_backend(self, requested):
            return requested, ""

        def make_codec(self, *, backend):
            return codec

        def stream_layout(self):
            return None

    monkeypatch.setattr(sa3_session, "get_sa3_context", lambda *a, **k: _Ctx())
    monkeypatch.setattr(sa3_trt, "max_dit_engine_latents", lambda model_id: 646)
    monkeypatch.setattr(sa3_trt, "find_dit_engine", lambda *a, **k: "/stock.trt")
    monkeypatch.setattr(sa3_trt, "_deserialize_engine", lambda path: None)
    sa3_session.preload_sa3("small-sfx", decoder_backend="tensorrt", vae_backend="tensorrt")
    assert codec.closed
    assert part.parked and mod.where == "system"


def test_pinned_stand_in_needs_no_module():
    part = pinned_part("dit")
    part.hold("x")
    part.release()
    part.park_if_free()
    assert part.count == 0 and not part.parked


# ---- a transformer on demand under a model extension ---------------------------
#
# No second device here: "meta" stands in for the other side of a move
# (system memory on park, the card on load), so a moved tensor is visible
# by its device while a kept one stays on the CPU.


class _Table(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("values", torch.ones(8))


class _Trunk(torch.nn.Module):
    """A small transformer stand-in: two modules to keep, two to move."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 4)
        self.table = _Table()
        self.block = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("scale", torch.ones(2))


def _devices(trunk):
    return {
        "head": trunk.head.weight.device.type,
        "table": trunk.table.values.device.type,
        "block": trunk.block.weight.device.type,
        "scale": trunk.scale.device.type,
    }


def test_park_moves_everything_but_the_keep(monkeypatch):
    from acestep.engine import sa3_placement

    monkeypatch.setattr(sa3_placement, "_PARKED", torch.device("meta"))
    trunk = _Trunk()
    keep_weight = trunk.head.weight
    keep_buf = trunk.table.values
    part = CardPart("dit", trunk, "cpu")
    part.keep = (trunk.head, trunk.table)
    part.park_if_free()
    assert part.parked
    assert _devices(trunk) == {
        "head": "cpu", "table": "cpu", "block": "meta", "scale": "meta",
    }
    # The kept tensors are the very same objects, untouched.
    assert trunk.head.weight is keep_weight
    assert trunk.table.values is keep_buf


def test_a_keep_move_reports_only_what_moved_and_keeps_parameter_identity():
    trunk = _Trunk()
    weight = trunk.block.weight
    part = CardPart("dit", trunk, "cpu")
    part.keep = (trunk.head, trunk.table)
    # block (16 floats) and scale (2 floats) move; the keep does not count.
    assert part._move(torch.device("cpu")) == (16 + 2) * 4 / (1024 * 1024)
    assert trunk.block.weight is weight                # same Parameter object


def test_load_moves_everything_but_the_keep(monkeypatch):
    trunk = _Trunk()
    kept = trunk.table.values
    part = CardPart("dit", trunk, "meta")              # "the card"
    part.keep = (trunk.head, trunk.table)
    part._parked = True
    part.hold("on_demand")
    assert _devices(trunk) == {
        "head": "cpu", "table": "cpu", "block": "meta", "scale": "meta",
    }
    assert trunk.table.values is kept
    assert part.count == 1 and not part.parked


def test_an_empty_keep_is_module_to():
    part, mod = _part()
    assert part.keep == ()
    part.park_if_free()
    part.hold("x")
    assert mod.moves == ["system", "card"]             # Module.to, as before


def _sa3_context(monkeypatch, runtime):
    dit, decoder = _Trunk(), _Part()
    sam = SimpleNamespace(
        device="cpu",
        model=SimpleNamespace(
            model=dit,
            sample_rate=44100,
            io_channels=256,
            diffusion_objective="rf_denoiser",
            pretransform=SimpleNamespace(
                downsampling_ratio=4096,
                model=SimpleNamespace(decoder=decoder),
            ),
        ),
    )
    loader = SimpleNamespace(
        checkpoint_dir=lambda model_id: "/ckpt",
        load_local_model=lambda *a, **k: sam,
    )
    monkeypatch.setattr(ctx_mod, "require_sa3_vendor", lambda: None)
    monkeypatch.setattr(ctx_mod, "import_loader_helpers", lambda: loader)
    monkeypatch.setattr(ctx_mod, "import_stream_helpers", lambda: None)
    monkeypatch.setattr(ctx_mod, "_share_text_encoder", lambda *a, **k: None)
    ext = SimpleNamespace(
        validate_base_model=lambda desc: None,
        install=lambda sam, cfg: runtime(dit) if callable(runtime) else runtime,
        qualified_id="probe.ext",
    )
    return ctx_mod.SA3Context("small-sfx", extension=ext), dit, decoder


def test_an_on_demand_extension_parks_the_transformer_from_boot(monkeypatch):
    from acestep.engine import sa3_placement

    monkeypatch.setattr(sa3_placement, "_PARKED", torch.device("meta"))

    def runtime(trunk):
        return SimpleNamespace(
            transformer_on_demand=True,
            card_keep=lambda: [trunk.head, trunk.table],
        )

    c, dit, decoder = _sa3_context(monkeypatch, runtime)
    assert not c.torch_dit.pinned and c.torch_dit.parked
    assert c.torch_dit.keep == (dit.head, dit.table)
    assert _devices(dit) == {
        "head": "cpu", "table": "cpu", "block": "meta", "scale": "meta",
    }
    assert c.torch_decoder.pinned and decoder.moves == []


def test_an_on_demand_extension_without_a_keep_moves_the_whole_transformer(monkeypatch):
    from acestep.engine import sa3_placement

    monkeypatch.setattr(sa3_placement, "_PARKED", torch.device("meta"))
    c, dit, _ = _sa3_context(
        monkeypatch, SimpleNamespace(transformer_on_demand=True),
    )
    assert c.torch_dit.keep == () and c.torch_dit.parked
    assert set(_devices(dit).values()) == {"meta"}


def test_an_extension_that_does_not_ask_keeps_both_parts_pinned(monkeypatch):
    for runtime in (object(), SimpleNamespace(transformer_on_demand=False)):
        c, dit, decoder = _sa3_context(monkeypatch, runtime)
        assert c.torch_dit.pinned and c.torch_decoder.pinned
        assert c.torch_dit.keep == () and not c.torch_dit.parked
        assert set(_devices(dit).values()) == {"cpu"} and decoder.moves == []


def _layout_context(monkeypatch, layout_dit):
    c = _bare_context()
    layout = SimpleNamespace(
        n_streams=2, make_dit=lambda ctx, frames, secs, backend: layout_dit,
    )
    monkeypatch.setattr(c, "stream_layout", lambda: layout, raising=False)
    return c


def test_layout_eager_dit_takes_the_hold_the_backend_owns(monkeypatch):
    c = _layout_context(monkeypatch, None)
    c.torch_dit.park_if_free()
    assert c.make_dit(latent_frames=1228, seconds_total=60.0, backend="tensorrt") is c.dit
    assert c.torch_dit.count == 1 and c.dit_module.where == "card"


def test_layout_own_dit_lets_the_transformer_park(monkeypatch):
    own = object()
    c = _layout_context(monkeypatch, own)
    assert c.make_dit(latent_frames=1228, seconds_total=60.0, backend="tensorrt") is own
    assert c.torch_dit.count == 0 and c.torch_dit.parked


def test_layout_branch_is_a_no_op_while_pinned(monkeypatch):
    c = _layout_context(monkeypatch, None)
    c.torch_dit.pinned = True
    c.make_dit(latent_frames=1228, seconds_total=60.0, backend="tensorrt")
    assert c.torch_dit.count == 0 and c.dit_module.moves == []


class _SessionDit(_ZeroDit):
    """A session step object that counts ``release_session`` calls."""

    def __init__(self):
        super().__init__()
        self.released = 0

    def release_session(self):
        self.released += 1


def test_close_ends_the_session_dit_once():
    part, mod = _part()
    dit = _SessionDit()
    b = _backend(part, mod, accel=dit)
    b.close()
    b.close()
    assert dit.released == 1


def test_a_resize_ends_the_replaced_dit_and_close_the_new_one():
    part, mod = _part()
    old, made = _SessionDit(), []

    def new():
        made.append(_SessionDit())
        return made[-1]

    b = _resizing_backend(part, mod, old, new)
    b.handle_swap_source(torch.zeros(2, 96000), 48000, duration_s=2 * T * 4096 / SA3_SAMPLE_RATE)
    assert b.adapter.dit is made[0]
    assert old.released == 1 and made[0].released == 0
    b.close()
    assert old.released == 1 and made[0].released == 1


def test_a_failed_resize_ends_the_dit_it_made():
    part, mod = _part()
    old, made = _SessionDit(), []

    def new():
        made.append(_SessionDit())
        return made[-1]

    b = _resizing_backend(part, mod, old, new, encode_fails=True)
    try:
        b.handle_swap_source(
            torch.zeros(2, 96000), 48000, duration_s=2 * T * 4096 / SA3_SAMPLE_RATE,
        )
    except RuntimeError:
        pass
    assert b.adapter.dit is old
    assert made[0].released == 1 and old.released == 0
    b.close()
    assert made[0].released == 1 and old.released == 1


def test_a_session_that_fails_to_assemble_ends_its_dit():
    dit = _SessionDit()

    def no_schedule(*a, **k):
        raise RuntimeError("assembly failed")

    context = SimpleNamespace(
        stream_layout=lambda: None,
        make_dit=lambda **k: dit,
        cond_seconds_total=lambda d: d,
        make_schedule_builder=no_schedule,
    )
    try:
        SA3Backend.from_context(
            context, prompt="p", duration_s=1.0,
            knob_state=KnobState(sa3_knob_specs()), cond=_FakeCond(),
        )
    except RuntimeError:
        pass
    assert dit.released == 1
