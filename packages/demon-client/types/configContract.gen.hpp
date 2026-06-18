// AUTO-GENERATED — do not edit by hand.
//
// Projected from the TypeScript config SDK
//   packages/demon-client/config/{defaults,types,enums}.ts + inputs.ts
// by packages/demon-client/scripts/genConfigTypes.mjs.
//
// Regenerate after any config change:
//   (cd packages/demon-client && npm run build && npm run gen:config)
// Drift-guarded by web/tests/unit/configContractDrift.test.ts
// (a stale copy fails CI).
//
// Constants-ONLY and JSON-library-agnostic: this header declares NO structs and
// pulls in NO JSON dependency. It provides the config VOCABULARY as string
// constants — JSON field keys, enum option values, and the schema version — so a
// C++ client (the rtmg-vst plugin) references generated names instead of
// hand-copied literals while keeping its own juce::var (de)serialization.
//
// The preset file IS a web DemonExport: an RtmgConfig plus an optional top-level
// `inputs` (SerializedInputs). Field-key coverage of NESTED objects mirrors
// DEFAULT_CONFIG; optional fields absent from defaults (engine._xl, prompt._xl,
// max_concurrent_loras*) are NOT emitted — they round-trip opaquely inside their
// parent object. The top-level key surface is complete (KNOWN_TOP_LEVEL_KEYS).

#pragma once

namespace demon::config {

// Shared-config schema version (RtmgConfig.version / DEFAULT_CONFIG.version).
// Bumped on a breaking shape change; compare against a loaded preset's
// `version` to detect a cross-frontend mismatch.
inline constexpr int kSchemaVersion = 1;

// ── Top-level keys ──
// A DemonExport is an RtmgConfig (these known keys) plus an optional
// top-level `inputs`. Unknown top-level keys round-trip untouched
// (preserve-unknown); kKnownTopLevelKeys is that boundary.
inline constexpr const char* kVersion = "version";
inline constexpr const char* kEngine = "engine";
inline constexpr const char* kPrompts = "prompts";
inline constexpr const char* kControls = "controls";
inline constexpr const char* kChannelRanges = "channel_ranges";
inline constexpr const char* kSeed = "seed";
inline constexpr const char* kSwapSourceMode = "swap_source_mode";
inline constexpr const char* kWeb = "web";
inline constexpr const char* kCurves = "curves";
inline constexpr const char* kLoop = "loop";
// DemonExport extension: serialized track/timbre/structure inputs.
inline constexpr const char* kInputs = "inputs";

// The preserve-unknown boundary: top-level keys the schema models. A
// loaded config's other top-level keys are re-emitted untouched on write.
inline constexpr const char* kKnownTopLevelKeys[] = {
    "version",
    "engine",
    "prompts",
    "controls",
    "channel_ranges",
    "seed",
    "swap_source_mode",
    "web",
    "curves",
    "loop",
};
inline constexpr int kKnownTopLevelKeyCount = 10;

// ── Nested object field keys (coverage == DEFAULT_CONFIG) ──

namespace engine {

  inline constexpr const char* kSde = "sde";
  inline constexpr const char* kLora = "lora";
  inline constexpr const char* kDepth = "depth";
  inline constexpr const char* kVaeWindow = "vae_window";
  inline constexpr const char* kCrop = "crop";
  inline constexpr const char* kSteps = "steps";
  inline constexpr const char* kFastVae = "fast_vae";
  inline constexpr const char* kWalkWindow = "walk_window";
  inline constexpr const char* kWalkWindowS = "walk_window_s";
  inline constexpr const char* kLeadFloorS = "lead_floor_s";
  inline constexpr const char* kLeadCeilingS = "lead_ceiling_s";
  inline constexpr const char* kLeadReleaseTauS = "lead_release_tau_s";
  inline constexpr const char* kMaxSourceDurationS = "max_source_duration_s";
  inline constexpr const char* kKey = "key";
  inline constexpr const char* kTimeSignature = "time_signature";
  inline constexpr const char* kEnabledLoras = "enabled_loras";
  inline constexpr const char* kAutoPrependLoraTriggers = "auto_prepend_lora_triggers";
  inline constexpr const char* kShowIncompatibleLoras = "show_incompatible_loras";

  // One enabled_loras[] entry. The array is empty in DEFAULT_CONFIG so its
  // element shape (EnabledLoraEntry's object form) is projected explicitly
  // from config/types.ts. A bare-string entry (enable at default strength)
  // has no keys.
  namespace enabled_lora {

    inline constexpr const char* kName = "name";
    inline constexpr const char* kStrength = "strength";

  }  // namespace enabled_lora

}  // namespace engine

namespace prompts {

  inline constexpr const char* kA = "a";
  inline constexpr const char* kB = "b";
  inline constexpr const char* kBlend = "blend";

}  // namespace prompts

// controls.* — per-knob initial values; keys are the engine param ids
// (also the MIDI-map / config-file names). The VST APVTS param IDs key
// off these. Friendly display names live in controlsContract.gen.hpp.

namespace controls {

  inline constexpr const char* kDenoise = "denoise";
  inline constexpr const char* kHintStrength = "hint_strength";
  inline constexpr const char* kFeedback = "feedback";
  inline constexpr const char* kFeedbackDepth = "feedback_depth";
  inline constexpr const char* kShift = "shift";
  inline constexpr const char* kChG0 = "ch_g0";
  inline constexpr const char* kChG1 = "ch_g1";
  inline constexpr const char* kChG2 = "ch_g2";
  inline constexpr const char* kChG3 = "ch_g3";
  inline constexpr const char* kChG4 = "ch_g4";
  inline constexpr const char* kChG5 = "ch_g5";
  inline constexpr const char* kChG6 = "ch_g6";
  inline constexpr const char* kChG7 = "ch_g7";
  inline constexpr const char* kCh13 = "ch13";
  inline constexpr const char* kCh14 = "ch14";
  inline constexpr const char* kCh19 = "ch19";
  inline constexpr const char* kCh23 = "ch23";
  inline constexpr const char* kCh29 = "ch29";
  inline constexpr const char* kCh56 = "ch56";
  inline constexpr const char* kDcwScaler = "dcw_scaler";
  inline constexpr const char* kDcwHighScaler = "dcw_high_scaler";
  inline constexpr const char* kDcwEnabled = "dcw_enabled";
  inline constexpr const char* kDcwMode = "dcw_mode";
  inline constexpr const char* kDcwWavelet = "dcw_wavelet";
  inline constexpr const char* kLoraDefaultStrength = "lora_default_strength";
  inline constexpr const char* kGuidanceScale = "guidance_scale";
  inline constexpr const char* kCfgRescale = "cfg_rescale";
  inline constexpr const char* kRcfgMode = "rcfg_mode";

}  // namespace controls

// channel_ranges.* — keyed by channel name (a subset of controls keys);
// each value is a { min, max, reverse } object (the `range` sub-keys).

namespace channel_ranges {

  inline constexpr const char* kChG0 = "ch_g0";
  inline constexpr const char* kChG1 = "ch_g1";
  inline constexpr const char* kChG2 = "ch_g2";
  inline constexpr const char* kChG3 = "ch_g3";
  inline constexpr const char* kChG4 = "ch_g4";
  inline constexpr const char* kChG5 = "ch_g5";
  inline constexpr const char* kChG6 = "ch_g6";
  inline constexpr const char* kChG7 = "ch_g7";
  inline constexpr const char* kCh13 = "ch13";
  inline constexpr const char* kCh14 = "ch14";
  inline constexpr const char* kCh19 = "ch19";
  inline constexpr const char* kCh23 = "ch23";
  inline constexpr const char* kCh29 = "ch29";
  inline constexpr const char* kCh56 = "ch56";

  namespace range {

    inline constexpr const char* kMin = "min";
    inline constexpr const char* kMax = "max";
    inline constexpr const char* kReverse = "reverse";

  }  // namespace range

}  // namespace channel_ranges

// web.* — the browser demo's own block. Other frontends ignore it but must
// round-trip it untouched (it is a known top-level key, not an unknown).

namespace web {

  inline constexpr const char* kEffects = "effects";
  inline constexpr const char* kAudio = "audio";
  inline constexpr const char* kResetSeconds = "reset_seconds";
  inline constexpr const char* kDenoiseSessionGate = "denoise_session_gate";
  inline constexpr const char* kRestartSongOnSwap = "restart_song_on_swap";

  namespace effects {

    inline constexpr const char* kParallaxStrength = "parallax_strength";
    inline constexpr const char* kBloomOnKick = "bloom_on_kick";
    inline constexpr const char* kBloomThreshold = "bloom_threshold";
    inline constexpr const char* kWarpStrength = "warp_strength";

  }  // namespace effects

  namespace audio {

    inline constexpr const char* kLufsEnabled = "lufs_enabled";
    inline constexpr const char* kLufsWindowSec = "lufs_window_sec";
    inline constexpr const char* kLufsMetric = "lufs_metric";
    inline constexpr const char* kLufsPeakHeadroom = "lufs_peak_headroom";
    inline constexpr const char* kLufsSilenceFloorDb = "lufs_silence_floor_db";
    inline constexpr const char* kLufsSilenceFloorHysteresisDb = "lufs_silence_floor_hysteresis_db";

  }  // namespace audio

  namespace denoise_session_gate {

    inline constexpr const char* kEnabled = "enabled";
    inline constexpr const char* kGlideMs = "glide_ms";

  }  // namespace denoise_session_gate

}  // namespace web

// ── Enum option values ──

namespace swap_source_mode {

  inline constexpr const char* kFull = "full";
  inline constexpr const char* kVocals = "vocals";
  inline constexpr const char* kInstruments = "instruments";

}  // namespace swap_source_mode

namespace time_signature {

  inline constexpr const char* k2 = "2";
  inline constexpr const char* k3 = "3";
  inline constexpr const char* k4 = "4";
  inline constexpr const char* k6 = "6";

}  // namespace time_signature

namespace dcw_mode {

  inline constexpr const char* kLow = "low";
  inline constexpr const char* kHigh = "high";
  inline constexpr const char* kDouble = "double";
  inline constexpr const char* kPix = "pix";

}  // namespace dcw_mode

namespace dcw_wavelet {

  inline constexpr const char* kHaar = "haar";
  inline constexpr const char* kDb4 = "db4";
  inline constexpr const char* kSym8 = "sym8";
  inline constexpr const char* kDb8 = "db8";

}  // namespace dcw_wavelet

namespace rcfg_mode {

  inline constexpr const char* kOff = "off";
  inline constexpr const char* kInitialize = "initialize";
  inline constexpr const char* kSelf = "self";

}  // namespace rcfg_mode

namespace loop_grid {

  inline constexpr const char* kBar = "bar";
  inline constexpr const char* kHalf = "half";
  inline constexpr const char* kBeat = "beat";
  inline constexpr const char* kEighth = "eighth";

}  // namespace loop_grid

// ── inputs.* — the DemonExport `inputs` block (SerializedInputs codec) ──

namespace inputs {

  // The three input axes.
  inline constexpr const char* kTrack = "track";
  inline constexpr const char* kTimbre = "timbre";
  inline constexpr const char* kStructure = "structure";

  namespace input {

    // SerializedInput fields. `kind` discriminates fixture vs clip;
    // a clip embeds trimmed PCM as a base64 16-bit WAV in `wavBase64`.
    inline constexpr const char* kKind = "kind";
    inline constexpr const char* kName = "name";
    inline constexpr const char* kSourceMode = "sourceMode";
    inline constexpr const char* kWavBase64 = "wavBase64";

    namespace kind_value {

      inline constexpr const char* kFixture = "fixture";
      inline constexpr const char* kClip = "clip";

    }  // namespace kind_value

    namespace source_mode_value {

      inline constexpr const char* kFull = "full";
      inline constexpr const char* kVocals = "vocals";
      inline constexpr const char* kInstruments = "instruments";

    }  // namespace source_mode_value

  }  // namespace input

}  // namespace inputs

}  // namespace demon::config
