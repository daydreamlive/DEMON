"""Download, verify, and materialize user-trained LoRAs.

The pod accepts a ``register_user_lora`` WS message (see ``ws_adapter.py``)
that carries presigned Tigris URLs for a ``.safetensors`` file and its
signature sidecar produced by the orchestrator. This module turns those
URLs into a verified file on disk that ``LoRAManager.register_lora`` can
add to the runtime catalog.

The trust chain — established by the orchestrator at training time and
verified here — is:

  1. Orchestrator signs a canonical manifest with Ed25519:
       { v:1, jobId, style, trigger, sha256, createdAt }
  2. The signature, base64-encoded manifest bytes, and signing key id
     (``kid``) ship in ``<style>.signature.json`` next to the safetensors.
  3. Pod fetches the signature sidecar, looks up the trusted public key
     by ``kid``, recomputes sha256 over the downloaded safetensors,
     verifies the signature against the manifest bytes.
  4. Only on full pass does the file land in :func:`acestep.paths.user_loras_dir`
     and the LoRA manager register it.

Public keys live in two layers (preferred order):
  a) ``GET https://app.daydream.live/api/loras/signing-public-key`` —
     fetched once at module init (5s timeout).
  b) ``LORA_SIGNING_PUBLIC_KEYS_PEM`` env (newline-separated
     ``kid<TAB>PEM`` pairs) — fallback when the fetch fails.

Both are cached at module load. Rotation is overlap-then-cutover: a new
``kid`` registered upstream / in env starts verifying immediately on
restart; old kids stay trusted until removed.

Pure module-level functions only — the session-side glue that calls
:func:`download_pack` → :func:`verify_pack` → :func:`materialize_pack`
lives in ``ws_adapter.py``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from acestep.paths import user_loras_dir

logger = logging.getLogger(__name__)

# Configuration knobs ---------------------------------------------------------

# Where the canonical public-key roster lives. Set via env so staging /
# preview / prod can each point at their own pipelines app instance
# without a code change.
_PUBLIC_KEY_REGISTRY_URL = os.environ.get(
    "LORA_PUBLIC_KEY_REGISTRY_URL",
    "https://app.daydream.live/api/loras/signing-public-key",
).strip() or None

# Newline-separated `kid<TAB>PEM` pairs. Fallback when the registry URL
# fetch fails (network glitch, app down, dev environment). Each PEM may
# span multiple lines as long as the leading ``kid<TAB>`` is on the
# header line of each entry; we split on ``\n---KEY---\n`` between entries.
_PUBLIC_KEYS_PEM_ENV = os.environ.get("LORA_SIGNING_PUBLIC_KEYS_PEM", "")

# Hard cap on safetensors download size to catch a misconfigured Tigris
# response that points at e.g. a multi-GB checkpoint. 250 MB covers the
# 168 MB v6 LoRAs the orchestrator produces with headroom.
_MAX_SAFETENSORS_BYTES = 250 * 1024 * 1024

# Per-request timeouts (seconds). Conservative — the rented GPU pod has
# enough headroom and a long-tail upload from Tigris under contention
# can hit a few seconds easily.
_DOWNLOAD_TIMEOUT_S = 60
_KEY_FETCH_TIMEOUT_S = 5


# Errors ----------------------------------------------------------------------


class UserLoraError(Exception):
    """Base for user-pack registration failures.

    Carries a stable ``code`` so the WS error response is grep-able
    from logs / dashboards without parsing the human message.
    """

    code = "user_lora_failed"

    def __init__(self, message: str, *, code: Optional[str] = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class DownloadError(UserLoraError):
    code = "download_failed"


class VerifyError(UserLoraError):
    code = "verify_failed"


# Trusted-keys cache ----------------------------------------------------------


@dataclass(frozen=True)
class _TrustedKey:
    kid: str
    public_key: Ed25519PublicKey


_keys_lock = threading.Lock()
_trusted_keys: Optional[Dict[str, _TrustedKey]] = None


def _parse_pem_entry(pem: str) -> Optional[Ed25519PublicKey]:
    """Parse a single PEM-encoded Ed25519 public key. Returns None on
    a bad/unsupported key so the caller can skip it without aborting
    the whole roster load."""
    try:
        key = serialization.load_pem_public_key(pem.encode("utf-8"))
    except Exception as exc:
        logger.warning("[user_loras] failed to parse PEM: %s", exc)
        return None
    if not isinstance(key, Ed25519PublicKey):
        logger.warning(
            "[user_loras] unsupported key type %s — only Ed25519 accepted",
            type(key).__name__,
        )
        return None
    return key


def _load_keys_from_registry() -> Dict[str, _TrustedKey]:
    """Fetch the public-key roster from ``LORA_PUBLIC_KEY_REGISTRY_URL``.

    The pipelines endpoint (``apps/streamdiffusion/src/app/api/loras/
    signing-public-key/route.ts``) returns ``{kid, alg, publicKey}``
    today — a single-key payload. We accept either that shape OR a
    list shape ``{keys: [{kid, alg, publicKey}, …]}`` so the endpoint
    can grow into multi-key without changing pod code.
    """
    if not _PUBLIC_KEY_REGISTRY_URL:
        return {}
    try:
        req = urllib.request.Request(
            _PUBLIC_KEY_REGISTRY_URL,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_KEY_FETCH_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        logger.warning(
            "[user_loras] could not fetch trusted-keys from %s: %s",
            _PUBLIC_KEY_REGISTRY_URL,
            exc,
        )
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("[user_loras] trusted-keys response not JSON: %s", exc)
        return {}

    entries = []
    if isinstance(payload, dict):
        if isinstance(payload.get("keys"), list):
            entries = payload["keys"]
        elif "kid" in payload and "publicKey" in payload:
            entries = [payload]

    out: Dict[str, _TrustedKey] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        kid = entry.get("kid")
        pem = entry.get("publicKey") or entry.get("public_key")
        if not isinstance(kid, str) or not isinstance(pem, str):
            continue
        key = _parse_pem_entry(pem)
        if key is None:
            continue
        out[kid] = _TrustedKey(kid=kid, public_key=key)
    if out:
        logger.info(
            "[user_loras] loaded %d trusted key(s) from registry: kids=%s",
            len(out),
            sorted(out.keys()),
        )
    return out


def _load_keys_from_env() -> Dict[str, _TrustedKey]:
    """Parse ``LORA_SIGNING_PUBLIC_KEYS_PEM`` (the env fallback). Each
    entry is ``kid<TAB>PEM``; entries separated by a blank line. We
    accept the ``-----BEGIN PUBLIC KEY-----`` PEM marker inside each
    entry so an operator can paste keys directly from openssl output.
    """
    raw = _PUBLIC_KEYS_PEM_ENV.strip()
    if not raw:
        return {}
    out: Dict[str, _TrustedKey] = {}
    # Split on blank-line-as-separator so a multi-line PEM can sit on
    # its own block. Then each block: first line is ``kid<TAB>...``.
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        # The kid can be tab- or space-separated from the start of the PEM.
        first, _, rest = block.partition("\n")
        kid_part, sep, pem_head = first.partition("\t")
        if not sep:
            # Fallback: space separator before the BEGIN line.
            kid_part, sep, pem_head = first.partition(" ")
        if not sep:
            logger.warning("[user_loras] env key entry has no kid separator")
            continue
        kid = kid_part.strip()
        pem = (pem_head.strip() + "\n" + rest).strip()
        if not pem.startswith("-----BEGIN"):
            logger.warning(
                "[user_loras] env key %r is missing PEM BEGIN header", kid,
            )
            continue
        key = _parse_pem_entry(pem)
        if key is None:
            continue
        out[kid] = _TrustedKey(kid=kid, public_key=key)
    if out:
        logger.info(
            "[user_loras] loaded %d trusted key(s) from env: kids=%s",
            len(out),
            sorted(out.keys()),
        )
    return out


def _ensure_keys() -> Dict[str, _TrustedKey]:
    """Lazy-init the trusted-keys cache. Tries the registry URL first,
    falls through to env. Result is cached for the process lifetime —
    rotation requires a pod restart, which matches the current
    operational model."""
    global _trusted_keys
    with _keys_lock:
        if _trusted_keys is not None:
            return _trusted_keys
        keys = _load_keys_from_registry()
        if not keys:
            keys = _load_keys_from_env()
        _trusted_keys = keys
        if not keys:
            logger.warning(
                "[user_loras] no trusted keys loaded — register_user_lora "
                "will reject all packs until configured",
            )
        return keys


def reset_trusted_keys_cache() -> None:
    """Drop the cached key roster. Tests + admin reloads only."""
    global _trusted_keys
    with _keys_lock:
        _trusted_keys = None


# Download --------------------------------------------------------------------


def _http_get_bytes(url: str, max_bytes: int) -> bytes:
    """GET ``url`` and return up to ``max_bytes``. Raises
    :class:`DownloadError` on non-2xx, oversized payloads, or
    network errors."""
    try:
        with urllib.request.urlopen(
            url, timeout=_DOWNLOAD_TIMEOUT_S,
        ) as resp:
            if resp.status < 200 or resp.status >= 300:
                raise DownloadError(
                    f"GET {url} returned status {resp.status}",
                )
            # Refuse early when Content-Length blows past the cap, to
            # avoid waiting on a multi-GB download we'll reject anyway.
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                try:
                    if int(cl) > max_bytes:
                        raise DownloadError(
                            f"content-length {cl} exceeds cap "
                            f"{max_bytes} bytes",
                        )
                except ValueError:
                    pass
            buf = bytearray()
            chunk = resp.read(1 << 20)
            while chunk:
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise DownloadError(
                        f"payload exceeds {max_bytes} bytes",
                    )
                chunk = resp.read(1 << 20)
            return bytes(buf)
    except DownloadError:
        raise
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise DownloadError(f"GET {url} failed: {exc}") from exc


def download_pack(
    safetensors_url: str,
    signature_url: str,
) -> Tuple[bytes, Dict[str, Any]]:
    """Download both files. Returns ``(safetensors_bytes, sidecar_dict)``.

    The sidecar is the parsed JSON object: ``{manifest_b64, sig_b64, kid}``.
    Caps the safetensors at :data:`_MAX_SAFETENSORS_BYTES`; the sidecar
    is tiny (a few hundred bytes) so we cap at 64 KiB.
    """
    sig_bytes = _http_get_bytes(signature_url, max_bytes=64 * 1024)
    try:
        sidecar = json.loads(sig_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownloadError(f"signature.json not valid JSON: {exc}") from exc
    if not isinstance(sidecar, dict):
        raise DownloadError("signature.json must be a JSON object")
    safetensors_bytes = _http_get_bytes(
        safetensors_url, max_bytes=_MAX_SAFETENSORS_BYTES,
    )
    return safetensors_bytes, sidecar


# Verify ----------------------------------------------------------------------


def verify_pack(
    safetensors_bytes: bytes,
    sidecar: Dict[str, Any],
    *,
    expected_sha256: Optional[str] = None,
    expected_kid: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify the signature chain. Returns the parsed manifest on success;
    raises :class:`VerifyError` otherwise.

    ``expected_sha256`` and ``expected_kid`` come from the WS message
    body and act as tripwires: we double-check that what the VST asked
    for matches what the orchestrator signed. Mismatches at this layer
    catch a mid-flight URL swap and key-rotation skew before the file
    ever lands in the catalog.
    """
    manifest_b64 = sidecar.get("manifest_b64")
    sig_b64 = sidecar.get("sig_b64")
    kid = sidecar.get("kid")
    if not all(isinstance(v, str) and v for v in (manifest_b64, sig_b64, kid)):
        raise VerifyError("sidecar missing manifest_b64 / sig_b64 / kid")
    try:
        manifest_bytes = base64.b64decode(manifest_b64, validate=True)
    except Exception as exc:
        raise VerifyError(f"manifest_b64 not valid base64: {exc}") from exc
    try:
        signature = base64.b64decode(sig_b64, validate=True)
    except Exception as exc:
        raise VerifyError(f"sig_b64 not valid base64: {exc}") from exc

    if expected_kid is not None and expected_kid != kid:
        raise VerifyError(
            f"kid mismatch: sidecar={kid!r}, expected={expected_kid!r}",
        )

    trusted = _ensure_keys()
    trusted_key = trusted.get(kid)
    if trusted_key is None:
        raise VerifyError(f"kid {kid!r} not in trusted set")

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifyError(f"manifest bytes not valid JSON: {exc}") from exc

    if manifest.get("v") != 1:
        raise VerifyError(
            f"unsupported manifest version: {manifest.get('v')!r}",
        )

    manifest_sha = manifest.get("sha256")
    if not isinstance(manifest_sha, str):
        raise VerifyError("manifest is missing sha256")

    actual_sha = hashlib.sha256(safetensors_bytes).hexdigest()
    if actual_sha != manifest_sha:
        raise VerifyError(
            f"sha256 mismatch: bytes={actual_sha}, manifest={manifest_sha}",
        )
    if expected_sha256 is not None and expected_sha256 != manifest_sha:
        raise VerifyError(
            f"sha256 mismatch vs request: manifest={manifest_sha}, "
            f"expected={expected_sha256}",
        )

    try:
        trusted_key.public_key.verify(signature, manifest_bytes)
    except InvalidSignature as exc:
        raise VerifyError(f"ed25519 signature invalid (kid={kid})") from exc

    return manifest


# Materialize -----------------------------------------------------------------


def _safe_id(lora_id: str) -> str:
    """Filename-safe stem. Strips any character outside [A-Za-z0-9._-]
    so an attacker can't traverse out of the user-loras dir via the
    WS message (defense-in-depth — the signed manifest already binds
    id/style, but we don't trust the WS body)."""
    out = "".join(c if (c.isalnum() or c in "._-") else "_" for c in lora_id)
    return out[:128] or "user_lora"


def materialize_pack(
    safetensors_bytes: bytes,
    manifest: Dict[str, Any],
    *,
    lora_id: str,
    display_name: str,
    trigger: Optional[str],
    target_dir: Optional[Path] = None,
) -> Path:
    """Write the verified safetensors + sidecars into ``target_dir`` and
    return the safetensors path. The path's stem becomes the LoRA id
    that ``LoRAManager.register_lora`` picks up.

    Layout (matches the stock catalog so :func:`acestep.lora_metadata.
    load_lora_metadata` picks the sidecars up automatically):

      ``{target_dir}/{stem}.safetensors``
      ``{target_dir}/{stem}.metadata.json``  (carries ``source="user_pack"``)
      ``{target_dir}/{stem}.trigger.txt``    (when trigger is present)

    Idempotent: re-running with the same bytes + id overwrites the
    existing files. The pod's :class:`LoRAManagerBase` is also
    idempotent on re-register, so calling this then ``register_lora``
    multiple times in a row is safe.
    """
    target = target_dir if target_dir is not None else user_loras_dir()
    target.mkdir(parents=True, exist_ok=True)
    stem = _safe_id(lora_id)
    safetensors_path = target / f"{stem}.safetensors"
    metadata_path = target / f"{stem}.metadata.json"
    trigger_path = target / f"{stem}.trigger.txt"

    # Write atomically: temp file + rename, so a crash mid-write doesn't
    # leave a half-written safetensors that the manager would then load.
    tmp_path = safetensors_path.with_suffix(".safetensors.tmp")
    tmp_path.write_bytes(safetensors_bytes)
    os.replace(tmp_path, safetensors_path)

    sidecar = {
        "schema_version": 1,
        "id": stem,
        "name": display_name or stem,
        "source": "user_pack",
        "inference": {},
        "classification": {},
        "model": {},
    }
    if trigger:
        sidecar["inference"]["primary_trigger_word"] = trigger
        sidecar["inference"]["trigger_words"] = [trigger]
    # Mirror the orchestrator-signed manifest under a known key so an
    # operator inspecting the pod's disk can trace any user pack back
    # to its training run without re-deriving from the registry.
    sidecar["provenance"] = {
        "manifest": manifest,
    }
    metadata_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8",
    )
    if trigger:
        trigger_path.write_text(str(trigger), encoding="utf-8")
    else:
        # Stale trigger from a prior register would mislead the metadata
        # loader's legacy fallback path; remove it.
        try:
            trigger_path.unlink()
        except FileNotFoundError:
            pass

    logger.info(
        "[user_loras] materialized %s at %s (%.1f MiB)",
        stem,
        safetensors_path,
        len(safetensors_bytes) / (1024 * 1024),
    )
    return safetensors_path
