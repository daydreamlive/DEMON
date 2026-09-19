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
// Constants-ONLY and JSON-library-agnostic: this header pulls in NO JSON
// dependency and declares no serialization machinery. It provides the config
// VOCABULARY as string constants — JSON field keys, enum option values, and the
// schema version — so a C++ client (the rtmg-vst plugin) references generated
// names instead of hand-copied literals while keeping its own juce::var
// (de)serialization. It also provides the DEFAULT_CONFIG VALUES as typed
// constexpr facts (demon::config::defaults), including two small constexpr POD
// tables for iteration, so C++ hosts consume the defaults instead of
// re-declaring them. The same value facts ship as machine-readable JSON in
// types/configDefaults.gen.json for JS/TS consumers.
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

// ── DEFAULT_CONFIG values (defaults::*) ──
// The VALUE facts of DEFAULT_CONFIG — control defaults, prompts, engine
// scalars, channel-range min/max/reverse — as typed constexpr constants,
// so a C++ host CONSUMES the defaults instead of re-declaring them.

namespace defaults {

  // Top-level scalar defaults.
  inline constexpr double kSeed = 0.0;
  inline constexpr const char* kSwapSourceMode = "instruments";

  // engine.* scalar defaults (enabled_loras is [] in DEFAULT_CONFIG; an
  // empty array has no value constant).
  namespace engine {

    inline constexpr bool kSde = false;
    inline constexpr bool kLora = true;
    inline constexpr double kDepth = 4.0;
    inline constexpr double kVaeWindow = 0.36;
    inline constexpr double kCrop = 0.0;
    inline constexpr double kSteps = 8.0;
    inline constexpr bool kFastVae = false;
    inline constexpr bool kWalkWindow = false;
    inline constexpr double kWalkWindowS = 60.0;
    inline constexpr double kLeadFloorS = 0.25;
    inline constexpr double kLeadCeilingS = 1.35;
    inline constexpr double kLeadReleaseTauS = 1.5;
    inline constexpr double kMaxSourceDurationS = 120.0;
    inline constexpr const char* kKey = "G# minor";
    inline constexpr const char* kTimeSignature = "4";
    inline constexpr bool kAutoPrependLoraTriggers = true;
    inline constexpr bool kShowIncompatibleLoras = false;

  }  // namespace engine

  namespace prompts {

    inline constexpr const char* kA = "heavy dubstep, deathstep, afxdump, growl heavy bass distortion";
    inline constexpr const char* kB = "daft punk style, beautiful, four to the floor, angelic";
    inline constexpr double kBlend = 0.4;

  }  // namespace prompts

  namespace controls {

    // DEFAULT_CONFIG.controls values, one typed constant per knob.
    inline constexpr double kDenoise = 0.7;
    inline constexpr double kHintStrength = 1.0;
    inline constexpr double kFeedback = 0.0;
    inline constexpr double kFeedbackDepth = 1.0;
    inline constexpr double kShift = 3.5;
    inline constexpr double kChG0 = 1.0;
    inline constexpr double kChG1 = 1.0;
    inline constexpr double kChG2 = 1.0;
    inline constexpr double kChG3 = 1.0;
    inline constexpr double kChG4 = 1.0;
    inline constexpr double kChG5 = 1.0;
    inline constexpr double kChG6 = 1.0;
    inline constexpr double kChG7 = 1.0;
    inline constexpr double kCh13 = 1.0;
    inline constexpr double kCh14 = 1.0;
    inline constexpr double kCh19 = 1.0;
    inline constexpr double kCh23 = 1.0;
    inline constexpr double kCh29 = 1.0;
    inline constexpr double kCh56 = 1.0;
    inline constexpr double kDcwScaler = 0.05;
    inline constexpr double kDcwHighScaler = 0.02;
    inline constexpr bool kDcwEnabled = true;
    inline constexpr const char* kDcwMode = "double";
    inline constexpr const char* kDcwWavelet = "haar";
    inline constexpr double kLoraDefaultStrength = 1.4;
    inline constexpr double kGuidanceScale = 7.0;
    inline constexpr double kCfgRescale = 0.0;
    inline constexpr const char* kRcfgMode = "off";

    // The same values as one ordered table (DEFAULT_CONFIG insertion order,
    // which is also JSON.stringify emission order), for consumers that
    // iterate the control set instead of naming each constant.
    enum class ValueKind { Number, Boolean, String };
    struct ControlValue {
      const char* key;
      ValueKind kind;
      double number;       // valid when kind == Number
      bool boolean;        // valid when kind == Boolean
      const char* string;  // valid when kind == String
    };
    inline constexpr ControlValue kValues[] = {
        { "denoise", ValueKind::Number, 0.7, false, nullptr },
        { "hint_strength", ValueKind::Number, 1.0, false, nullptr },
        { "feedback", ValueKind::Number, 0.0, false, nullptr },
        { "feedback_depth", ValueKind::Number, 1.0, false, nullptr },
        { "shift", ValueKind::Number, 3.5, false, nullptr },
        { "ch_g0", ValueKind::Number, 1.0, false, nullptr },
        { "ch_g1", ValueKind::Number, 1.0, false, nullptr },
        { "ch_g2", ValueKind::Number, 1.0, false, nullptr },
        { "ch_g3", ValueKind::Number, 1.0, false, nullptr },
        { "ch_g4", ValueKind::Number, 1.0, false, nullptr },
        { "ch_g5", ValueKind::Number, 1.0, false, nullptr },
        { "ch_g6", ValueKind::Number, 1.0, false, nullptr },
        { "ch_g7", ValueKind::Number, 1.0, false, nullptr },
        { "ch13", ValueKind::Number, 1.0, false, nullptr },
        { "ch14", ValueKind::Number, 1.0, false, nullptr },
        { "ch19", ValueKind::Number, 1.0, false, nullptr },
        { "ch23", ValueKind::Number, 1.0, false, nullptr },
        { "ch29", ValueKind::Number, 1.0, false, nullptr },
        { "ch56", ValueKind::Number, 1.0, false, nullptr },
        { "dcw_scaler", ValueKind::Number, 0.05, false, nullptr },
        { "dcw_high_scaler", ValueKind::Number, 0.02, false, nullptr },
        { "dcw_enabled", ValueKind::Boolean, 0.0, true, nullptr },
        { "dcw_mode", ValueKind::String, 0.0, false, "double" },
        { "dcw_wavelet", ValueKind::String, 0.0, false, "haar" },
        { "lora_default_strength", ValueKind::Number, 1.4, false, nullptr },
        { "guidance_scale", ValueKind::Number, 7.0, false, nullptr },
        { "cfg_rescale", ValueKind::Number, 0.0, false, nullptr },
        { "rcfg_mode", ValueKind::String, 0.0, false, "off" },
    };
    inline constexpr int kValueCount = 28;

  }  // namespace controls

  namespace channel_ranges {

    // DEFAULT_CONFIG.channel_ranges rows ({ min, max, reverse } per channel),
    // in DEFAULT_CONFIG insertion order.
    struct ChannelRange {
      const char* channel;
      double min;
      double max;
      bool reverse;
    };
    inline constexpr ChannelRange kRanges[] = {
        { "ch_g0", 0.0, 2.2, false },
        { "ch_g1", 0.0, 2.0, false },
        { "ch_g2", 0.0, 2.3, true },
        { "ch_g3", 0.0, 2.0, false },
        { "ch_g4", 0.0, 2.5, false },
        { "ch_g5", 0.0, 2.0, false },
        { "ch_g6", 0.0, 2.0, true },
        { "ch_g7", 0.0, 2.0, true },
        { "ch13", 0.0, 2.0, true },
        { "ch14", 0.0, 2.3, false },
        { "ch19", 0.0, 2.5, false },
        { "ch23", 0.0, 2.45, false },
        { "ch29", 0.0, 2.0, false },
        { "ch56", 0.0, 2.0, false },
    };
    inline constexpr int kRangeCount = 14;

  }  // namespace channel_ranges

}  // namespace defaults

}  // namespace demon::config
