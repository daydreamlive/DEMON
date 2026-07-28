"""Model-extension lifecycle wired into the SA3 backend (Tier 1, no GPU).

Two properties dominate:

* **Stock is untouched.** With no extension selected, every path must
  behave exactly as if the plugin system did not exist.
* **In-flight conditioning is never mutated.** A prompt or source swap
  rebuilds conditioning; requests already in the pipeline must keep
  seeing the bundle they were submitted with, or they finish against
  conditioning nobody asked for.
"""

from __future__ import annotations

import torch

from acestep.engine.sa3_adapter import SA3Adapter
from acestep.engine.sa3_stream_helpers import SA3Conditioning
from acestep.plugins import (
    BackendVerdict,
    ConfigField,
    ModelExtension,
    ModelExtensionRuntime,
    ModelExtensionSpec,
)
from acestep.plugins.api import LoadedPlugin, PluginManifest
from acestep.plugins.selection import select_model_extension
from acestep.streaming.knobs import KnobSpec, KnobState, coerce_knob_values
from acestep.streaming.sa3_backend import SA3Backend, sa3_knob_specs
from acestep.streaming.sa3_session import sa3_context_key

from tests.unit.test_sa3_backend import (  # reuse the established fakes
    C,
    CTX,
    N44,
    T,
    _FakeCodec,
    _FakeCond,
    _ZeroDit,
    _schedule_builder_factory,
)


def _cond():
    """A real SA3Conditioning: decoration rebuilds it with dataclass replace."""
    fake = _FakeCond()
    return SA3Conditioning(
        cond_bundle=fake.cond_bundle,
        sched_args={},
        latent_frames=T,
        audio_sample_size=N44,
    )

PLUGIN_ID = "probe_ext"
KNOB = f"plugin_{PLUGIN_ID}_strength"


class _ProbeRuntime(ModelExtensionRuntime):
    def __init__(self):
        self.controls = []
        self.decorations = []
        self.closed = 0

    def supports_dit_backend(self, backend):
        if backend == "tensorrt":
            return BackendVerdict(False, "no accelerated plan for this graph")
        return BackendVerdict(True)

    def decorate_conditioning(self, bundle, source_latent_bct=None):
        self.decorations.append(source_latent_bct)
        return {**bundle, "probe_cond": source_latent_bct}

    def apply_controls(self, values):
        self.controls.append(dict(values))

    def status(self):
        return {"installed": True}

    def close(self):
        self.closed += 1


class _ProbeExtension(ModelExtension):
    def __init__(self, context):
        self.context = context

    def config_fingerprint(self):
        return f"scale={self.context.config.get('scale')}"

    def install(self, model, model_config):
        return _ProbeRuntime()


def _selected(scale=1.0):
    spec = ModelExtensionSpec(
        name="probe",
        family="sa3",
        create=_ProbeExtension,
        config_schema={"scale": ConfigField(type="float", required=False, default=scale)},
        knob_specs=(KnobSpec(KNOB, default=1.0, min_val=0.0, max_val=3.0),),
    )
    plugins = {
        PLUGIN_ID: LoadedPlugin(
            manifest=PluginManifest(
                id=PLUGIN_ID, version="0.1.0", plugin_api=1,
            ),
            model_extensions={"probe": spec},
        )
    }
    return select_model_extension(
        f"{PLUGIN_ID}.probe", family="sa3", plugins=plugins,
    )


class _Context:
    """SA3Context stand-in carrying an installed extension runtime."""

    device = torch.device("cpu")
    dtype = torch.float32
    diffusion_objective = "rf_denoiser"

    def __init__(self, runtime=None):
        self.extension_runtime = runtime

    def make_dit(self, **_kw):
        return _ZeroDit()

    def make_schedule_builder(self, _cond, steps):
        return _schedule_builder_factory(steps)

    def make_codec(self, **_kw):
        return _FakeCodec()


def _backend(extension=None, runtime=None, source=None, cond=None, **kw):
    """Direct construction, mirroring test_sa3_backend's harness.

    Built directly rather than through ``from_context`` so individual
    tests can supply their own prompt_rebuilder / source_encoder without
    colliding with the closures from_context installs.
    """
    source = torch.randn(1, C, T) if source is None else source
    steps = kw.pop("steps", 2)
    adapter = SA3Adapter(
        _ZeroDit(),
        schedule_builder=_schedule_builder_factory(steps),
        device="cpu",
        dtype=torch.float32,
    )
    return SA3Backend(
        adapter=adapter,
        codec=_FakeCodec(),
        cond=_cond() if cond is None else cond,
        schedule_builder_factory=_schedule_builder_factory,
        knob_state=KnobState(
            sa3_knob_specs(
                extension_specs=extension.knob_specs if extension else (),
            ),
        ),
        source_latent_bct=source,
        steps=steps,
        depth=1,
        vae_window_s=0.1,
        model_extension=extension,
        extension_runtime=runtime,
        **kw,
    )


# ---- stock parity -----------------------------------------------------------


def test_stock_knob_manifest_has_no_extension_knobs():
    names = [s.name for s in sa3_knob_specs()]
    assert not [n for n in names if n.startswith("plugin_")]


def test_stock_backend_knob_specs_are_unchanged():
    backend = _backend()
    assert [s.name for s in backend.knob_specs()] == [
        s.name for s in sa3_knob_specs()
    ]


def test_stock_conditioning_bundle_is_passed_through_untouched():
    cond = _cond()
    original = cond.cond_bundle
    backend = _backend(cond=cond)
    # No extension means no copy, no new keys, same object identity.
    assert backend._cond.cond_bundle is original


def test_stock_context_key_does_not_change_shape():
    assert sa3_context_key("medium") == ("medium", None, None)


# ---- knob composition -------------------------------------------------------


def test_extension_knob_joins_the_family_manifest():
    extension = _selected()
    names = [s.name for s in sa3_knob_specs(extension_specs=extension.knob_specs)]
    assert KNOB in names


def test_backend_publishes_the_extension_knob():
    extension = _selected()
    backend = _backend(extension=extension, runtime=_ProbeRuntime())
    assert KNOB in [s.name for s in backend.knob_specs()]


def test_extension_knob_is_coerced_like_a_core_knob():
    # The property that matters: the composed spec map is what validation
    # runs against, so an out-of-range client value is clamped rather than
    # passed through raw.
    extension = _selected()
    backend = _backend(extension=extension, runtime=_ProbeRuntime())
    specs = {s.name: s for s in backend.knob_specs()}

    clean, errors = coerce_knob_values({KNOB: 99.0}, specs)

    assert clean[KNOB] == 3.0
    assert errors


# ---- conditioning decoration ------------------------------------------------


def test_conditioning_is_decorated_at_construction():
    runtime = _ProbeRuntime()
    source = torch.randn(1, C, T)
    backend = _backend(
        extension=_selected(), runtime=runtime, source=source,
    )

    assert "probe_cond" in backend._cond.cond_bundle
    assert torch.equal(backend._cond.cond_bundle["probe_cond"], source)


def test_decoration_does_not_mutate_the_supplied_bundle():
    cond = _cond()
    original_bundle = cond.cond_bundle

    backend = _backend(
        extension=_selected(), runtime=_ProbeRuntime(), cond=cond,
    )

    assert "probe_cond" not in original_bundle
    assert backend._cond.cond_bundle is not original_bundle


def test_from_context_installs_the_runtime_from_the_context():
    # Production assembly: the backend picks the installed runtime off the
    # context rather than being handed one separately.
    runtime = _ProbeRuntime()
    backend = SA3Backend.from_context(
        _Context(runtime),
        prompt="a",
        duration_s=1.0,
        knob_state=KnobState(sa3_knob_specs()),
        cond=_cond(),
        source_latent_bct=torch.randn(1, C, T),
        steps=2,
        depth=1,
        model_extension=_selected(),
    )
    assert backend._extension_runtime is runtime
    assert "probe_cond" in backend._cond.cond_bundle


def test_source_swap_rebuilds_conditioning_without_touching_the_old_bundle():
    runtime = _ProbeRuntime()
    initial = torch.randn(1, C, T)
    replacement = torch.randn(1, C, T)
    backend = _backend(
        extension=_selected(), runtime=runtime, source=initial,
        source_encoder=lambda *_a: replacement,
    )
    old_bundle = backend._active_bundle
    assert torch.equal(old_bundle["probe_cond"], initial)

    backend.handle_swap_source(torch.zeros(2, 48000), 48000)

    new_bundle = backend._active_bundle
    assert new_bundle is not old_bundle
    # An in-flight request holding the old bundle must still see the old
    # condition; nothing was edited underneath it.
    assert torch.equal(old_bundle["probe_cond"], initial)
    assert torch.equal(new_bundle["probe_cond"], replacement)


def test_source_swap_advances_the_cond_epoch():
    backend = _backend(
        extension=_selected(), runtime=_ProbeRuntime(),
        source_encoder=lambda *_a: torch.randn(1, C, T),
    )
    before = backend._cond_epoch
    backend.handle_swap_source(torch.zeros(2, 48000), 48000)
    assert backend._cond_epoch == before + 1


def test_stock_source_swap_leaves_conditioning_alone():
    backend = _backend(source_encoder=lambda *_a: torch.randn(1, C, T))
    before_bundle = backend._active_bundle
    before_epoch = backend._cond_epoch

    backend.handle_swap_source(torch.zeros(2, 48000), 48000)

    # No extension: the swap re-anchors the source only, exactly as before.
    assert backend._active_bundle is before_bundle
    assert backend._cond_epoch == before_epoch


def test_prompt_swap_decorates_the_new_conditioning():
    runtime = _ProbeRuntime()
    source = torch.randn(1, C, T)
    backend = _backend(
        extension=_selected(), runtime=runtime, source=source,
        prompt_rebuilder=lambda tags, steps: (
            _cond(), _schedule_builder_factory,
        ),
    )
    old_bundle = backend._active_bundle

    backend.handle_set_prompt("new tags")

    assert backend._active_bundle is not old_bundle
    assert torch.equal(backend._cond.cond_bundle["probe_cond"], source)
    assert "probe_cond" in old_bundle  # untouched, still attributable


# ---- controls ---------------------------------------------------------------


def test_extension_controls_are_applied_on_tick():
    runtime = _ProbeRuntime()
    extension = _selected()
    backend = _backend(extension=extension, runtime=runtime)

    knobs = {**backend.read_knobs(), "steps_override": 2, KNOB: 2.5}
    backend.produce(knobs, CTX, "generate")

    assert runtime.controls
    assert runtime.controls[-1] == {KNOB: 2.5}


def test_extension_receives_only_its_own_knobs():
    runtime = _ProbeRuntime()
    backend = _backend(extension=_selected(), runtime=runtime)

    knobs = {**backend.read_knobs(), "steps_override": 2, KNOB: 1.0}
    backend.produce(knobs, CTX, "generate")

    assert set(runtime.controls[-1]) == {KNOB}


def test_stock_backend_never_calls_control_hooks():
    runtime = _ProbeRuntime()
    backend = _backend()  # no extension
    backend.produce(
        {**backend.read_knobs(), "steps_override": 2}, CTX, "generate",
    )
    assert runtime.controls == []


# ---- cache keys -------------------------------------------------------------


def test_context_key_includes_extension_identity_and_config():
    stock = sa3_context_key("medium")
    with_ext = sa3_context_key("medium", extension=_selected(scale=1.0))
    other = sa3_context_key("medium", extension=_selected(scale=2.0))

    assert stock != with_ext
    assert with_ext != other
    assert with_ext == sa3_context_key("medium", extension=_selected(scale=1.0))
