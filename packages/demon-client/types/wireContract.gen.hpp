// AUTO-GENERATED — do not edit by hand.
//
// Projected from the Python wire-contract registry
//   demos/realtime_motion_graph_web/protocol.py :: wire_contract()
// by demos/realtime_motion_graph_web/scripts/gen_wire_types.py.
//
// Regenerate after any registry change:
//   python demos/realtime_motion_graph_web/scripts/gen_wire_types.py
// Drift-guarded by tests/unit/test_wire_contract.py
// (test_generated_wire_types_hpp_match_contract) — a stale copy fails CI.
//
// Contract-types-ONLY and JSON-library-agnostic: this header declares NO
// structs and pulls in NO JSON dependency. It provides the wire VOCABULARY as
// string constants — message "type" names, JSON field keys, and enum option
// values — so a C++ client (the rtmg-vst plugin) references generated names
// instead of hand-copied literals while keeping its own (de)serialization.
//
// The `params` command's `raw` payload (the knob set) is described separately
// by the /api/knobs manifest. Binary framing (PCM uploads, the float16 slice
// stream) is documented per-entry in the source registry, not encoded here.

#pragma once

namespace demon::wire {

inline constexpr int kProtocolVersion = 1;

// Knob-manifest schema version (the `version` field served by GET
// /api/knobs and the MCP list_knobs tool). Compare against the live
// manifest to detect a stale build, exactly like kProtocolVersion.
inline constexpr int kKnobSchemaVersion = 1;

// The JSON discriminator key carried by every command and event.
inline constexpr const char* kTypeKey = "type";

// ── Command payloads (client → server) ──

namespace command {

  namespace params {
    inline constexpr const char* kType = "params";
    /** Knob name -> value map. The payload schema is the separate /api/knobs manifest; values are clamped/validated server-side. */
    inline constexpr const char* kRaw = "raw";
    /** Playhead position in SECONDS (not a 0..1 ratio); used for time-keyed curve sampling. */
    inline constexpr const char* kPlaybackPos = "playback_pos";
    /** Optional stationary render placement in absolute buffer/song seconds. A finite value renders exactly at that position, bypassing transport lead and loop-band snapping. Absent retains the prior anchor; null clears it. */
    inline constexpr const char* kRenderAnchorS = "render_anchor_s";
    /** Optional BATCH of stationary render anchors (absolute buffer/song seconds), rendered back-to-back one window per tick whenever the scalar render_anchor_s is clear (the scalar preempts; the queue resumes when it clears). Each anchor is popped after its window emits. A list REPLACES the prior queue; absent retains it; null/empty clears it. Gated on ready.capabilities.render_anchor_queue — older servers ignore the field. Non-finite entries are dropped; length is capped server-side (1024). Like the scalar anchor, IGNORED in walk-window mode (sources longer than walk_window_s): the DiT holds one chunk and cannot render outside it — clients must not queue-warm walk sources. */
    inline constexpr const char* kRenderAnchorQueueS = "render_anchor_queue_s";
    /** Client monotonic send time in seconds (performance.now()/1000; arbitrary origin). Lets the server estimate how stale a playback_pos report is when messages queue (network congestion, recv backlog) and advance its playhead estimate accordingly. Optional: absent on older clients, which get the uncompensated behavior. */
    inline constexpr const char* kClientTime = "client_time";
    /** Flow-control ack: cumulative bytes of binary slice frames received on this connection. The server holds back slice emission while its sent-bytes minus this ack exceeds the in-flight window (DEMON_SLICE_WINDOW_BYTES, default 256 KiB) so a bandwidth-limited link receives fresh slices at link rate instead of an ever-staler buffered backlog. Optional; absent on older clients = no flow control. */
    inline constexpr const char* kSliceBytesRx = "slice_bytes_rx";
    /** Worst observed slice landing lead since the previous params message: how far AHEAD of the audible playhead the most-behind audio slice landed when the client applied it (negative = it landed in already-played audio and the raw source was heard). Folded modulo track duration. The server widens its playback lead to keep this positive — covering network transit and client main-thread scheduling (e.g. throttled background tabs). Optional; omitted when no slice arrived since the last report. */
    inline constexpr const char* kSliceLeadS = "slice_lead_s";
  }  // namespace params

  namespace loop_band {
    inline constexpr const char* kType = "loop_band";
    /** Loop start in seconds; null/degenerate clears. */
    inline constexpr const char* kStartSec = "start_sec";
    /** Loop end in seconds; null/degenerate clears. */
    inline constexpr const char* kEndSec = "end_sec";
  }  // namespace loop_band

  namespace prompt {
    inline constexpr const char* kType = "prompt";
    /** Prompt A (wire text; enabled-LoRA triggers are prepended client-side). */
    inline constexpr const char* kTags = "tags";
    /** Optional prompt B, cached for A/B blend. */
    inline constexpr const char* kTagsB = "tags_b";
    /** Optional BPM override for structured conditioning metadata. Does not retime or time-stretch the source audio. */
    inline constexpr const char* kBpm = "bpm";
    /** Musical key, e.g. "C major". */
    inline constexpr const char* kKey = "key";
    /** Meter numerator, e.g. "3"/"4"/"6". */
    inline constexpr const char* kTimeSignature = "time_signature";
  }  // namespace prompt

  namespace set_prompt_blend {
    inline constexpr const char* kType = "set_prompt_blend";
    /** 0.0 = A, 1.0 = B. Clamped to [0,1]. */
    inline constexpr const char* kValue = "value";
  }  // namespace set_prompt_blend

  namespace set_interp_method {
    inline constexpr const char* kType = "set_interp_method";
    /** Which live blend to retune. */
    inline constexpr const char* kPath = "path";
    /** Interpolation curve. */
    inline constexpr const char* kMethod = "method";

    namespace path {
      inline constexpr const char* kPrompt = "prompt";
      inline constexpr const char* kTimbre = "timbre";
      inline constexpr const char* kStructure = "structure";
      inline constexpr const char* kFeedback = "feedback";
    }  // namespace path

    namespace method {
      inline constexpr const char* kSlerp = "slerp";
      inline constexpr const char* kLinear = "linear";
    }  // namespace method
  }  // namespace set_interp_method

  namespace set_depth {
    inline constexpr const char* kType = "set_depth";
    /** Target ring depth; clamped to [1, max_pipeline_depth]. */
    inline constexpr const char* kValue = "value";
  }  // namespace set_depth

  namespace enable_lora {
    inline constexpr const char* kType = "enable_lora";
    /** LoRA id/stem (see /api/loras). */
    inline constexpr const char* kId = "id";
    /** Target strength the refit lands at. */
    inline constexpr const char* kStrength = "strength";
  }  // namespace enable_lora

  namespace disable_lora {
    inline constexpr const char* kType = "disable_lora";
    inline constexpr const char* kId = "id";
  }  // namespace disable_lora

  namespace manual_slot_add {
    inline constexpr const char* kType = "manual_slot_add";
  }  // namespace manual_slot_add

  namespace manual_slot_pop {
    inline constexpr const char* kType = "manual_slot_pop";
  }  // namespace manual_slot_pop

  namespace set_timbre_strength {
    inline constexpr const char* kType = "set_timbre_strength";
    /** 1.0 = full reference, 0.0 = silence baseline. Clamped to [0,1]. */
    inline constexpr const char* kValue = "value";
  }  // namespace set_timbre_strength

  namespace set_timbre_source {
    inline constexpr const char* kType = "set_timbre_source";
    /** Label echoed back in timbre_set. */
    inline constexpr const char* kName = "name";
  }  // namespace set_timbre_source

  namespace set_timbre_fixture {
    inline constexpr const char* kType = "set_timbre_fixture";
    /** Fixture name (see /api/fixtures). */
    inline constexpr const char* kName = "name";
  }  // namespace set_timbre_fixture

  namespace clear_timbre_source {
    inline constexpr const char* kType = "clear_timbre_source";
  }  // namespace clear_timbre_source

  namespace set_structure_source {
    inline constexpr const char* kType = "set_structure_source";
    /** Label echoed back in structure_set. */
    inline constexpr const char* kName = "name";
  }  // namespace set_structure_source

  namespace set_structure_fixture {
    inline constexpr const char* kType = "set_structure_fixture";
    /** Fixture name (see /api/fixtures). */
    inline constexpr const char* kName = "name";
  }  // namespace set_structure_fixture

  namespace clear_structure_source {
    inline constexpr const char* kType = "clear_structure_source";
  }  // namespace clear_structure_source

  namespace swap_source {
    inline constexpr const char* kType = "swap_source";
    /** Optional new prompt A. */
    inline constexpr const char* kTags = "tags";
    /** Optional BPM override for the swapped source's structured conditioning metadata. */
    inline constexpr const char* kBpm = "bpm";
    inline constexpr const char* kKey = "key";
    inline constexpr const char* kTimeSignature = "time_signature";
    /** Source label; for server-side loads, the fixture/upload name to read off the pod's disk. */
    inline constexpr const char* kFixtureName = "fixture_name";
    /** For uploads: which model-ripped stem feeds inference. */
    inline constexpr const char* kStemSourceMode = "stem_source_mode";
    /** When true, the server loads the named source off its own disk and NO binary frame is sent. */
    inline constexpr const char* kUseServerSource = "use_server_source";

    namespace stem_source_mode {
      inline constexpr const char* kFull = "full";
      inline constexpr const char* kVocals = "vocals";
      inline constexpr const char* kInstruments = "instruments";
    }  // namespace stem_source_mode
  }  // namespace swap_source

  namespace write_audio {
    inline constexpr const char* kType = "write_audio";
    /** Where the buffer's first sample lands on the source, in playback seconds (sample-exact; no frame or grid alignment required). Default 0. Audio past the source end is trimmed, never wrapped. */
    inline constexpr const char* kAtS = "at_s";
    /** replace = overwrite the span (declicked against the existing audio at the edges); sum = overdub on top of what's there. */
    inline constexpr const char* kMix = "mix";
    /** fill = treat the buffer as ONE period of a loop and lay it across the whole source, phase-anchored at at_s (sample-exact audio-domain tiling; any period length works). Default none = write once. */
    inline constexpr const char* kRepeat = "repeat";
    /** The source generation this write targets (from ready/swap_ready, bumped by every swap). A mismatch is rejected with audio_write_failed instead of splicing into the wrong source. Omit to write against whatever is live. */
    inline constexpr const char* kSourceEpoch = "source_epoch";
    /** Re-encode the self-timbre conditioning against the updated source (~+50 ms). Ignored when a timbre override is active. Default false. */
    inline constexpr const char* kRefreshTimbre = "refresh_timbre";

    namespace mix {
      inline constexpr const char* kReplace = "replace";
      inline constexpr const char* kSum = "sum";
    }  // namespace mix

    namespace repeat {
      inline constexpr const char* kNone = "none";
      inline constexpr const char* kFill = "fill";
    }  // namespace repeat
  }  // namespace write_audio

}  // namespace command

// ── Event payloads (server → client) ──

namespace event {

  namespace init_ack {
    inline constexpr const char* kType = "init_ack";
    /** Server-minted session id, sent as soon as log context binds so client startup failures correlate with pod logs. */
    inline constexpr const char* kSessionId = "session_id";
    /** The config client_id echoed back, or null when the client sent none. */
    inline constexpr const char* kClientId = "client_id";
  }  // namespace init_ack

  namespace ready {
    inline constexpr const char* kType = "ready";
    inline constexpr const char* kDuration = "duration";
    inline constexpr const char* kChannels = "channels";
    inline constexpr const char* kSampleRate = "sample_rate";
    inline constexpr const char* kLoraCatalog = "lora_catalog";
    inline constexpr const char* kLoraDir = "lora_dir";
    inline constexpr const char* kBpm = "bpm";
    inline constexpr const char* kKey = "key";
    inline constexpr const char* kTimeSignature = "time_signature";
    inline constexpr const char* kCheckpoint = "checkpoint";
    inline constexpr const char* kCheckpointScale = "checkpoint_scale";
    inline constexpr const char* kPipelineDepth = "pipeline_depth";
    inline constexpr const char* kMaxPipelineDepth = "max_pipeline_depth";
    /** LoRA ids the server will auto-enable on the first tick (from the session's initial enable set); empty when none. */
    inline constexpr const char* kLoraPendingEnable = "lora_pending_enable";
    /** Server-minted session id, echoed for client/analytics log correlation. */
    inline constexpr const char* kSessionId = "session_id";
    /** Source generation counter (0 at create, bumped by every swap). Echo it in write_audio to pin a write to the source it was computed against. */
    inline constexpr const char* kSourceEpoch = "source_epoch";
    /** Backend-declared audio geometry: {sample_rate, channels, chunk_rate_hz, duration_s|null}. chunk_rate_hz is the generation cadence (latent fps for diffusion, frame rate for AR models); duration_s null is reserved for endless streams. */
    inline constexpr const char* kGeometry = "geometry";
    /** Backend capability mask: {capability: bool} over the Capabilities fields (swap, timbre, structure, lora, ...). Client panels and MCP tools gate on it; commands tagged with a matching `requires` fail with command_failed when the bit is false. */
    inline constexpr const char* kCapabilities = "capabilities";
    /** Per-session knob manifest: the same {version, knobs} envelope GET /api/knobs serves, but backend-owned and session-resolved (SDE mode, enabled lora_str_<id> knobs). /api/knobs remains the static pre-session probe. */
    inline constexpr const char* kKnobManifest = "knob_manifest";
    /** Active manual steering slot count; drives the client's man_*_<N> row rendering. Updated live via the manual_slot_count event. */
    inline constexpr const char* kManualSlotCount = "manual_slot_count";
    /** Server-imposed ceiling on manual steering slots; gates the client's + button. */
    inline constexpr const char* kManualSlotCap = "manual_slot_cap";
    /** True when the session's checkpoint has a reachable steering-vector bundle; false hides the steering surface (the steer_*\/man_* knobs are absent from the manifest too). */
    inline constexpr const char* kSteeringAvailable = "steering_available";
  }  // namespace ready

  namespace error {
    inline constexpr const char* kType = "error";
    inline constexpr const char* kCode = "code";
    inline constexpr const char* kMessage = "message";
    inline constexpr const char* kBuildCommand = "build_command";
    /** Present only on the engine_not_built code: the source duration whose TRT profile is missing. */
    inline constexpr const char* kDurationS = "duration_s";
  }  // namespace error

  namespace params_update {
    inline constexpr const char* kType = "params_update";
    /** Applied params + runtime telemetry (num_gens, tick_ms, dec_ms). */
    inline constexpr const char* kParams = "params";
  }  // namespace params_update

  namespace params_echo {
    inline constexpr const char* kType = "params_echo";
    inline constexpr const char* kRaw = "raw";
  }  // namespace params_echo

  namespace prompt_blend_echo {
    inline constexpr const char* kType = "prompt_blend_echo";
    inline constexpr const char* kValue = "value";
  }  // namespace prompt_blend_echo

  namespace prompt_applied {
    inline constexpr const char* kType = "prompt_applied";
    inline constexpr const char* kTags = "tags";
  }  // namespace prompt_applied

  namespace lora_catalog {
    inline constexpr const char* kType = "lora_catalog";
    inline constexpr const char* kCatalog = "catalog";
  }  // namespace lora_catalog

  namespace swap_ready {
    inline constexpr const char* kType = "swap_ready";
    inline constexpr const char* kDuration = "duration";
    inline constexpr const char* kSampleRate = "sample_rate";
    inline constexpr const char* kChannels = "channels";
    inline constexpr const char* kBpm = "bpm";
    inline constexpr const char* kKey = "key";
    inline constexpr const char* kTimeSignature = "time_signature";
    inline constexpr const char* kFixtureName = "fixture_name";
    /** Source generation counter after this swap; write_audio sends targeting the old source are rejected. */
    inline constexpr const char* kSourceEpoch = "source_epoch";
  }  // namespace swap_ready

  namespace swap_failed {
    inline constexpr const char* kType = "swap_failed";
    inline constexpr const char* kError = "error";
    /** Present only when the swap failed on a missing TRT engine: the command to build the profile for the new source's duration. */
    inline constexpr const char* kBuildCommand = "build_command";
  }  // namespace swap_failed

  namespace stem_assets {
    inline constexpr const char* kType = "stem_assets";
    inline constexpr const char* kFixtureName = "fixture_name";
    inline constexpr const char* kSampleRate = "sample_rate";
    inline constexpr const char* kChannels = "channels";
    inline constexpr const char* kFrames = "frames";
    /** Ordered subset of ("vocals","instruments"). */
    inline constexpr const char* kStems = "stems";
    inline constexpr const char* kSourceMode = "source_mode";

    namespace source_mode {
      inline constexpr const char* kFull = "full";
      inline constexpr const char* kVocals = "vocals";
      inline constexpr const char* kInstruments = "instruments";
    }  // namespace source_mode
  }  // namespace stem_assets

  namespace stem_failed {
    inline constexpr const char* kType = "stem_failed";
    inline constexpr const char* kFixtureName = "fixture_name";
    inline constexpr const char* kError = "error";
  }  // namespace stem_failed

  namespace depth_applied {
    inline constexpr const char* kType = "depth_applied";
    /** The clamped applied depth. */
    inline constexpr const char* kValue = "value";
  }  // namespace depth_applied

  namespace manual_slot_count {
    inline constexpr const char* kType = "manual_slot_count";
    /** The live manual steering slot count after the command. */
    inline constexpr const char* kCount = "count";
  }  // namespace manual_slot_count

  namespace timbre_set {
    inline constexpr const char* kType = "timbre_set";
    inline constexpr const char* kName = "name";
    inline constexpr const char* kDuration = "duration";
  }  // namespace timbre_set

  namespace timbre_cleared {
    inline constexpr const char* kType = "timbre_cleared";
  }  // namespace timbre_cleared

  namespace timbre_failed {
    inline constexpr const char* kType = "timbre_failed";
    inline constexpr const char* kError = "error";
  }  // namespace timbre_failed

  namespace structure_set {
    inline constexpr const char* kType = "structure_set";
    inline constexpr const char* kName = "name";
    inline constexpr const char* kDuration = "duration";
  }  // namespace structure_set

  namespace structure_cleared {
    inline constexpr const char* kType = "structure_cleared";
  }  // namespace structure_cleared

  namespace structure_failed {
    inline constexpr const char* kType = "structure_failed";
    inline constexpr const char* kError = "error";
  }  // namespace structure_failed

  namespace audio_written {
    inline constexpr const char* kType = "audio_written";
    inline constexpr const char* kStartS = "start_s";
    inline constexpr const char* kEndS = "end_s";
    /** The source generation the write landed on (matches ready/swap_ready). */
    inline constexpr const char* kSourceEpoch = "source_epoch";
  }  // namespace audio_written

  namespace audio_write_failed {
    inline constexpr const char* kType = "audio_write_failed";
    inline constexpr const char* kError = "error";
  }  // namespace audio_write_failed

  namespace command_failed {
    inline constexpr const char* kType = "command_failed";
    /** The rejected command's wire name. */
    inline constexpr const char* kCommand = "command";
    /** The Capabilities field the command needs and the session's backend doesn't declare. */
    inline constexpr const char* kRequires = "requires";
    /** Human-readable reason. */
    inline constexpr const char* kError = "error";
  }  // namespace command_failed

}  // namespace event

// ── Session-init config (client → server, sent at handshake) ──

namespace config {
  inline constexpr const char* kSde = "sde";
  inline constexpr const char* kLora = "lora";
  inline constexpr const char* kVaeWindow = "vae_window";
  inline constexpr const char* kCrop = "crop";
  inline constexpr const char* kDepth = "depth";
  inline constexpr const char* kSteps = "steps";
  inline constexpr const char* kPrompt = "prompt";
  inline constexpr const char* kBpm = "bpm";
  inline constexpr const char* kPromptB = "prompt_b";
  inline constexpr const char* kFastVae = "fast_vae";
  inline constexpr const char* kWalkWindow = "walk_window";
  inline constexpr const char* kWalkWindowS = "walk_window_s";
  inline constexpr const char* kLeadFloorS = "lead_floor_s";
  inline constexpr const char* kLeadCeilingS = "lead_ceiling_s";
  inline constexpr const char* kLeadReleaseTauS = "lead_release_tau_s";
  inline constexpr const char* kFixtureName = "fixture_name";
  inline constexpr const char* kUseServerFixture = "use_server_fixture";
  inline constexpr const char* kStemSourceMode = "stem_source_mode";
  inline constexpr const char* kEnabledLoras = "enabled_loras";
  inline constexpr const char* kLoraStrengths = "lora_strengths";
  inline constexpr const char* kLoraPaths = "lora_paths";
  inline constexpr const char* kClientId = "client_id";
  inline constexpr const char* kBackend = "backend";
  inline constexpr const char* kSa3DurationS = "sa3_duration_s";
}  // namespace config

// ── Init-phase upload handshake ──

namespace handshake {

  namespace command {

    namespace upload_track {
      inline constexpr const char* kType = "upload_track";
      /** Requested track label; deduped server-side. */
      inline constexpr const char* kName = "name";
      /** Optional key override; forces a re-encode instead of the content-dedup fast path. */
      inline constexpr const char* kKey = "key";
      /** Optional meter override; same effect as key. */
      inline constexpr const char* kTimeSignature = "time_signature";
    }  // namespace upload_track

  }  // namespace command

  namespace event {

    namespace upload_ok {
      inline constexpr const char* kType = "upload_ok";
      /** Final persisted track name (may differ from the requested name after dedup/uniquify). */
      inline constexpr const char* kName = "name";
      inline constexpr const char* kBpm = "bpm";
      inline constexpr const char* kKey = "key";
      inline constexpr const char* kTimeSignature = "time_signature";
      inline constexpr const char* kDurationS = "duration_s";
      inline constexpr const char* kSamples = "samples";
      /** True when the vocal/instrument stem rip is still running on a background thread. The track is immediately swappable (full source); stems land later via a pushed stem_assets frame on the live session (or stem_failed). */
      inline constexpr const char* kStemsPending = "stems_pending";
    }  // namespace upload_ok

    namespace upload_failed {
      inline constexpr const char* kType = "upload_failed";
      inline constexpr const char* kError = "error";
    }  // namespace upload_failed

  }  // namespace event

}  // namespace handshake

}  // namespace demon::wire
