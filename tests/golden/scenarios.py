"""Declarative scenario table for the golden/latency harness.

Adding a regression case == adding one entry here. Each scenario drives
one streaming session: a fixture source, a comparison-region spec, and
a list of actions fired when the GENERATION FRONTIER (the furthest song
position slices have covered) crosses their position.

Why frontier-relative, twice over:

* The comparison artifact is a position-aligned region of the song
  buffer: ``[anchor + warmup_skip_s, + canonical_s]``, where the anchor
  is where coverage started (session start, or the swap point). The
  warm-up skip excludes the scheduling-dependent startup ramp; the
  settle margin (runner) lets the region finish its depth-refinement
  passes before the run stops.
* Actions fire on frontier positions so their effect lands at a bounded
  song position (frontier + in-flight pipeline depth) INSIDE the
  compared region on any machine, fast or slow. Playhead-relative
  triggers would land wherever the machine's realtime factor put the
  frontier, i.e. outside the region on a fast box.

Do not expect bit-exactness here: generation is playhead-paced and
windows are re-emitted as they refine through the pipeline depth, so
the wire-level audio carries small timing-coupled variance even on one
machine (engine-level seed determinism is covered separately by
``tests/test_stream.py``). The comparison contract is tier-2 metric
thresholds calibrated from the measured same-build noise floor
(``runner --repeat`` prints both the floor and suggested thresholds);
tier-1 hash identity remains as a short-circuit when it happens.
"""

from dataclasses import dataclass, field

# Built-in fixtures (acestep/fixtures.py KNOWN_FIXTURES, served from the
# pod's /fixtures cache: the harness never needs local copies except
# for the explicit upload-path scenario, which fetches over HTTP).
FIXTURE_A = "low_fi_Gm_loop_60s_gnm.wav"
FIXTURE_B = "prog_rock_loop_60s_enm.wav"

# Session config shared by every scenario: mirrors the web app's
# defaults (web/public/config.json) minus host-specific fields. sde off
# is deliberate: it keeps the lowest-variance generation path.
BASE_CONFIG: dict = {
    "sde": False,
    "lora": False,
    "depth": 4,
    "steps": 8,
    "fast_vae": False,
    "prompt": "lofi hip hop, mellow, instrumental",
    "use_server_fixture": True,
}


@dataclass
class Action:
    at_s: float          # generation-frontier song position (seconds)
    kind: str            # params | prompt | swap | enable_lora
    payload: dict = field(default_factory=dict)


@dataclass
class Scenario:
    name: str
    fixture: str = FIXTURE_A
    warmup_skip_s: float = 6.0      # scheduling-ramp region to exclude
    canonical_s: float = 20.0       # compared-region length
    settle_s: float = 5.0           # refinement margin past the region
    timeout_s: float = 240.0        # wall-clock abort
    config: dict = field(default_factory=dict)   # overrides on BASE_CONFIG
    actions: list = field(default_factory=list)
    upload: bool = False            # exercise the client-upload path
    notes: str = ""

    def session_config(self) -> dict:
        cfg = dict(BASE_CONFIG)
        cfg.update(self.config)
        cfg["fixture_name"] = self.fixture
        if self.upload:
            cfg["use_server_fixture"] = False
        return cfg


SCENARIOS: tuple = (
    Scenario(
        name="baseline_stream",
        notes="Untouched defaults: the lowest-variance reference and "
              "the pure-throughput latency baseline.",
    ),
    Scenario(
        name="knob_step",
        actions=[Action(12.0, "params", {"raw": {"denoise": 0.35}})],
        notes="Single knob step when the frontier crosses 12s; covers "
              "the params channel and measures knob->slice latency.",
    ),
    Scenario(
        name="prompt_change",
        actions=[Action(12.0, "prompt",
                        {"tags": "aggressive industrial techno, distorted"})],
        notes="Prompt re-encode mid-stream; acked by prompt_applied.",
    ),
    Scenario(
        name="swap_fixture",
        actions=[Action(15.0, "swap", {"fixture": FIXTURE_B})],
        notes="Mid-stream source swap via the server-side load path; "
              "covers swap_ready + replacement-buffer framing. The "
              "canonical region re-anchors at the swap point.",
    ),
    Scenario(
        name="upload_path",
        canonical_s=15.0,
        upload=True,
        notes="Same fixture but uploaded as PCM by the client: covers "
              "the binary upload handshake end to end.",
    ),
    Scenario(
        name="lora_enable",
        config={"lora": True},
        actions=[Action(12.0, "enable_lora", {"strength": 1.0})],
        notes="Enables the first LoRA in the server catalog; SKIPPED "
              "automatically when the pod ships no LoRAs.",
    ),
)

SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}
