# Writing a DEMON plugin

DEMON can be extended out-of-tree by an installed Python package. This
document is the contract for plugin authors and the operator guide for
running one.

Plugins are **trusted, in-process Python**. This is an extensibility
boundary, not a security sandbox. Plugins load only from operator-
controlled startup configuration — a browser client can never name a
plugin, a module path, a checkpoint, or a config file.

## Import tiers

| Tier | Where | Stability |
|---|---|---|
| **1** | `acestep.plugins` | Versioned by `PLUGIN_API_VERSION`. Changes only with a version bump. |
| **2** | `acestep.engine.sa3_internals` | The documented, tested subset of vendored SA3 model internals an extension may traverse when it must build modules against the trunk architecture. Versioned by its own `API_VERSION`. |
| **0** | everything else under `acestep.*` | Internal. Importing it is a bug in your plugin. |

Tier 2 exists because some extensions genuinely cannot be written against
an abstract API: if you are adding modules that participate in the
transformer's computation, you need the real architecture. Tier 2 does not
remove that coupling, it gives it one address — so a vendored model bump
breaks one public, tested file with a clear message instead of failing
silently inside your package. Assert `sa3_internals.API_VERSION` at install
and fail closed on mismatch.

## Anatomy of a plugin

A plugin is an installed distribution advertising a `demon.plugins` entry
point:

```toml
[project.entry-points."demon.plugins"]
my_extension = "my_extension.plugin:register"
```

The referenced module must define, at module scope:

- `PLUGIN_API` — the integer plugin API version you built against
  (required). DEMON reads this **before** running your `register()`, so an
  incompatible plugin never executes registration code against an API it
  does not understand.
- `register(registry)` — called with a `PluginRegistry` scoped to your
  plugin (required).
- `REQUIRES_DEMON`, `DESCRIPTION` — optional metadata.

`REQUIRES_DEMON` is currently **advisory** and recorded in logs only.
DEMON's distribution version tracks the ACE-Step model generation rather
than an API contract, so enforcing a specifier against it would look like a
guarantee while promising nothing. The real compatibility gates are
`PLUGIN_API` and, for Tier-2 consumers, `sa3_internals.API_VERSION`.

## Model extensions

A *model extension* is a persistent, startup-selected participant in a
loaded model. It is not a job or a recipe: it is part of the model for the
lifetime of the process.

```python
from acestep.plugins import ConfigField, ModelExtensionSpec
from acestep.streaming.knobs import KnobSpec

PLUGIN_API = 2

def register(registry):
    registry.add_model_extension(
        ModelExtensionSpec(
            name="my_capability",
            family="sa3",
            create=MyExtension,
            config_schema={
                "weights": ConfigField(type="path", required=True),
            },
            knob_specs=(
                KnobSpec(
                    "plugin_my_extension_amount",
                    default=1.0, min_val=0.0, max_val=2.0,
                ),
            ),
        )
    )
```

### Lifecycle

1. `create(ExtensionContext)` → your extension object. Configuration is
   already coerced against `config_schema`. This is the **earliest** hook
   and the right place for validation only you can do — see
   [Validating your own configuration](#validating-your-own-configuration).
2. `validate_base_model(descriptor)` → `None` to accept, or `Rejection` to
   abort boot. Runs **before** the weights load, so refuse cheaply. For
   facts about the base model only. You may veto one; you may never
   substitute one. Which weights load is the operator's decision.
3. `config_fingerprint()` → a stable, non-secret string folded into the
   model cache key.
4. `install(model, model_config)` → your runtime object. It **must**
   subclass `ModelExtensionRuntime`; DEMON checks at boot, because the
   alternative is a missing hook surfacing as an `AttributeError` inside
   the streaming runner. The base class defines a neutral default for
   every hook, so subclassing costs you nothing.
5. Per-session: `supports_dit_backend` / `supports_codec_backend`,
   `decorate_conditioning`, `apply_controls`, `status` / `metrics`.
6. `close()` at teardown.

### Validating your own configuration

`ConfigField(type="path")` is a **coercion, not a filesystem assertion**.
It validates that the operator supplied a string; nothing is stat'd. DEMON
cannot do better: it has no way to know whether your path is one you read
(must already exist) or one you create (must not have to). The same goes
for numeric ranges — `ConfigField` has no bounds, because a valid range is
a fact about your architecture, not about JSON.

So do it in your constructor, and raise `ExtensionConfigError`:

```python
class MyExtension(ModelExtension):
    def __init__(self, context):
        path = Path(str(context.config["weights"])).expanduser().resolve()
        if not path.is_file():
            raise ExtensionConfigError(f"weights not found: {path}")
        self.weights = path
```

The constructor runs before the model cache key is computed and long
before any weights load. Deferring the check costs the operator a
multi-second load before they learn they made a typo — and note that
`config_fingerprint()` is called *before* `validate_base_model`, so it must
tolerate a missing file rather than being the thing that catches one.

### Rules that are enforced, not merely documented

**Knobs must be namespaced** as `plugin_<plugin_id>_<control>`. Registration
fails otherwise. The namespace is also the whole anti-shadowing defense —
no core or family knob is named `plugin_*`, so a name matching the pattern
cannot collide with one, and DEMON keeps no separate blocklist of core names
to go stale. Knobs must additionally be `float` or `int`, and must declare a
default inside their own bounds (coercion clamps, so an out-of-range default
would silently become a different value than the one you declared).

`float`/`int` only is a real restriction: the browser control panel renders
numeric knobs generically from manifest bounds, but binds enum/bool knobs by
name and renders unknown ones disabled. An extension enum knob would appear
in the manifest and be unusable.

**Conditioning must not be mutated in place.** `decorate_conditioning` must
return a fresh mapping. The bundle you are handed may already be referenced
by requests in flight, and those requests must keep seeing the conditioning
they were submitted with. DEMON hands you a copy, so a mistake here cannot
reach anyone else's bundle — but returning a mutated input is still wrong.

**`close()` must fully detach.** The loaded model is process-cached and
shared across every session in the process. If you attached hooks or
replaced modules, remove and restore them here. `close()` must be
idempotent and must tolerate being called after a failed install.

**Install failures are fatal.** If your `install()` raises after it has
already mutated the model, DEMON calls `close()` on whatever was produced
and then aborts boot. It does **not** fall back to the stock model: the
operator explicitly asked for your extension, so running without it would
generate with a different model than the one requested, and the output would
sound entirely plausible.

### Controls, and which thread calls you

`apply_controls(values)` is called once per tick on the **streaming
runner**, unlocked, with **only your own** knobs. That is the same thread
that then runs your model code, so a value written here is read by that
tick with no synchronization needed. Do not block.

`decorate_conditioning` is called on the **command thread** (a prompt or
source swap). If it and `apply_controls` touch the same state, that is the
one race DEMON does not serialize for you, and you own it. Keeping controls
to plain scalar rebinds avoids the question entirely.

### Conditioning on the session source

`decorate_conditioning(bundle, source)` receives a `SourceView`, or `None`
when the session has no source at all:

```python
from acestep.plugins import ModelExtensionRuntime

class MyRuntime(ModelExtensionRuntime):
    def decorate_conditioning(self, bundle, source=None):
        if source is None or source.waveform is None:
            return bundle
        cond = my_analysis(source.waveform, source.sample_rate)
        return {**bundle, "my_cond": cond}
```

| Field | What it is |
|---|---|
| `latent` | The encoded source anchor in the model's own native layout (`[1, C, T]` for `sa3`). |
| `waveform` | The audio that latent was encoded from, or `None`. |
| `sample_rate` | Sample rate of `waveform`, or `None`. |

**Every field is optional.** A session created without a source has no
latent; a source re-anchored from a latent DEMON did not encode itself has
no waveform. Check before you use, and do not assume the shape of a session
you did not create.

The waveform is carried because DEMON already holds it. An extension whose
condition derives from the audio rather than from its latent would otherwise
have to decode the latent back to audio — real time spent per swap, to
recover something the encoder had already discarded.

**Treat both tensors as read-only.** They are the live session anchor
itself, not copies, and the streaming runner reads them concurrently.

**Size your condition from the conditioning bundle, never from a constant.**
The latent window's length is a property of the session (its duration, the
model's downsampling ratio, and the upstream alignment rules), and it is not
a round number. Read it off the tensors you are handed — `source.latent`'s
last axis, or the bundle's own geometry — and build to that. A hardcoded
length is correct for exactly one session configuration and silently
misaligned for every other.

Note also that the generated window is slightly longer than the requested
duration: upstream Stable Audio 3 adds a fixed span of duration **headroom**
(`duration_padding_sec`), which the padding mask marks as valid space rather
than masking off. It is a small fraction of a full-length window and a
larger one of a short session.

DEMON deliberately does not hold its control lock across your code: that
lock exists to order the command thread against the tick loop, and putting
arbitrary plugin code inside it, every tick, would put you in the path of
every swap.

Keep per-tick control state on the object you own (your model wrapper, for
example). Do not expect DEMON to carry your values through its hot path.

### Telemetry

`status()` is operator-visible and must contain no filesystem paths,
checkpoint metadata, training metrics, or proprietary architecture facts.
`metrics()` is operator-only and is polled on demand rather than per
generation, so it may be more expensive — but keep anything you expose on a
per-generation path cheap.

## Running one

```powershell
uv run python -u -m demos.realtime_motion_graph_web.run -- `
  --checkpoint sa3-medium `
  --model-extension my_extension.my_capability `
  --model-extension-config <path-to>\my_extension.json
```

The config file is local, outside any repository, and keyed by qualified
extension id so one file can configure several extensions and none can be
applied to the wrong one by accident:

```json
{
  "my_extension.my_capability": {
    "weights": "/path/to/my_weights.pt"
  }
}
```

A selected extension that is missing, incompatible, misconfigured, or whose
install fails **stops the server at boot**.

Discovery runs only when an extension is selected, so a stock server never
imports any plugin. Once one *is* selected, discovery imports **every**
installed `demon.plugins` module, not just the selected one — a plugin's
module-scope code runs whether or not you asked for it. Failures in an
unselected plugin are isolated and logged rather than fatal; failures in
the selected one are fatal.

### Acceleration

An extension that augments the transformer generally cannot run under a
prebuilt TensorRT plan, because the plan contains only the stock graph.
Return `BackendVerdict(False, reason)` from `supports_dit_backend` and DEMON
falls back to eager, **reporting the downgrade at boot** rather than leaving
the operator to discover it in a log line at session create. Duration
normalization then uses the effective backend, so a TensorRT duration cap is
not imposed on an eager graph.
