// Host-injection seam.
//
// Each host wires its own URL builder, optional API key getter, and
// optional session id once at app boot. Module-level setters rather
// than React context so non-React readers (stores, workers) can read
// the values too.

export type EngineUrlBuilder = (path: string) => string;
export type ApiKeyGetter = () => string | null;

const NOT_CONFIGURED = (): never => {
  throw new Error(
    "URL builder not configured — call setEngineUrlBuilder() before any " +
      "engine fetch.",
  );
};

let _engineUrlBuilder: EngineUrlBuilder = NOT_CONFIGURED;
let _apiKey: ApiKeyGetter = () => null;
let _podSessionId: string | null = null;

/** Host wires this once at mount. The default throws to surface
 *  forgotten-configuration bugs loudly instead of producing silent 404s. */
export function setEngineUrlBuilder(fn: EngineUrlBuilder): void {
  _engineUrlBuilder = fn;
}

/** Host wires this once at mount. Default returns null (no auth). */
export function setApiKeyGetter(fn: ApiKeyGetter): void {
  _apiKey = fn;
}

/** Optional session id readable by the URL builder. */
export function setPodSessionId(id: string | null): void {
  _podSessionId = id;
}

export function getPodSessionId(): string | null {
  return _podSessionId;
}

/** Build a URL via the host-provided builder. */
export function podHttp(path: string): string {
  return _engineUrlBuilder(path);
}

/** Returns the host-provided API key, or null. */
export function getApiKey(): string | null {
  return _apiKey();
}
