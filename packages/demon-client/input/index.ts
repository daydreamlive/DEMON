// Live audio input: microphone capture and the 48 kHz framer that feeds
// `RemoteBackend.sendAudioIn` (the `audio_in` command of a `live_input`
// session).

export * from "./framer";
export * from "./MicCapture";
