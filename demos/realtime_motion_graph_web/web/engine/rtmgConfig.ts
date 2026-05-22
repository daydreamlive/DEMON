// Host-injection seam.
//
// Each host wires its own URL builder, optional API key getter, and
// optional session id once at app boot. Module-level setters rather
// than React context so non-React readers (stores, workers) can read
// the values too.

export type EngineUrlBuilder = (path: string) => string;
export type ApiKeyGetter = () => string | null;
export type ClientIdGetter = () => string | null;

const NOT_CONFIGURED = (): never => {
  throw new Error(
    "URL builder not configured — call setEngineUrlBuilder() before any " +
      "engine fetch.",
  );
};

let _engineUrlBuilder: EngineUrlBuilder = NOT_CONFIGURED;
let _apiKey: ApiKeyGetter = () => null;
let _clientId: ClientIdGetter = () => null;
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

/** Host wires this once at mount. Default returns null. The value lands
 *  in the WS handshake `config.client_id` and is bound into loguru's
 *  contextvars on the pod side (see acestep/engine/obs.py) so every log
 *  record on that connection carries it — useful for joining a browser
 *  trace to a pod-side log line. Standalone DEMON has no host-side
 *  identity, so the default no-op leaves the field absent. */
export function setClientIdGetter(fn: ClientIdGetter): void {
  _clientId = fn;
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

/** Returns the host-provided client id (e.g. an analytics distinct id),
 *  or null when the host hasn't wired one. */
export function getClientId(): string | null {
  return _clientId();
}
