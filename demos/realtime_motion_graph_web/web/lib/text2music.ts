// Text-to-music mode: generate music from the prompt alone, no input
// audio. Represented client-side as a sentinel "fixture" name so every
// existing source-selection surface (crate fan, CORE-tab picker, lite
// carousel) and the fixture-swap subscription work unchanged; the two
// wire send sites (session config in useStartSession, swap_source in
// useFixtureSwap) translate the sentinel into the contract's
// `text2music` flag instead of a fixture_name.

/** Sentinel value stored in usePerformanceStore.fixture. Never sent as a
 *  fixture_name on the wire. */
export const TEXT2MUSIC_SOURCE = "__text2music__";

/** Display label for the sentinel across pickers / placards. */
export const TEXT2MUSIC_LABEL = "Text to music";

export function isText2Music(name: string | null | undefined): boolean {
  return name === TEXT2MUSIC_SOURCE;
}
