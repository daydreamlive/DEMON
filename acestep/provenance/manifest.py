"""C2PA manifest construction + embedding for local file outputs.

Implements the manifest schema of spec 02 §3 for the DEMON local
file-writer path (§2 row 3): ``c2pa.created`` action with the IPTC
``digitalSourceType``, the CAWG training-and-data-mining assertion
(``notAllowed`` for all categories — the do-not-train default), and the
custom ``com.daydream.session`` assertion carrying session id,
model/LoRA identifiers, timeline summary counts and the session-log
hash. Signed with the local self-signed material from
:mod:`acestep.provenance.keys`.

Import-guarded: when ``c2pa-python`` (import name ``c2pa``) or
``cryptography`` is missing, :func:`embed_wav_manifest` degrades to a
no-op with a single logged warning. It also never raises — asset
writing must succeed whether or not it could be signed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from acestep.provenance import keys as provenance_keys

__all__ = [
    "TRAINED_ALGORITHMIC_MEDIA",
    "COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA",
    "DIGITAL_CAPTURE",
    "provenance_available",
    "build_manifest_definition",
    "embed_wav_manifest",
]

TRAINED_ALGORITHMIC_MEDIA = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
)
COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/"
    "compositeWithTrainedAlgorithmicMedia"
)
# Human-origin content (recorded / uploaded, not synthesized). Used for
# the user's own uploaded source track and separated stems: those are
# the dry input, and marking them trained/composite would be an
# affirmatively false synthetic claim (spec 02-architecture §4).
DIGITAL_CAPTURE = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
)

_CAWG_CATEGORIES = (
    "cawg.ai_generative_training",
    "cawg.ai_inference",
    "cawg.ai_training",
    "cawg.data_mining",
)

_warned_unavailable = False


def _import_c2pa() -> Any:
    global _warned_unavailable
    try:
        import c2pa
        return c2pa
    except ImportError:
        if not _warned_unavailable:
            _warned_unavailable = True
            logger.warning(
                "provenance manifests disabled: `c2pa` is not installed "
                "(install the `provenance` extra to embed Content "
                "Credentials in written audio)"
            )
        return None


def provenance_available() -> bool:
    """True when both the c2pa SDK and local signing material are
    usable. Cheap after the first call (imports are cached)."""
    return _import_c2pa() is not None and provenance_keys.signing_material() is not None


def _generator_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("demon")
    except Exception:  # noqa: BLE001 — running from a source tree
        return None


def build_manifest_definition(
    *,
    title: str,
    ingredient_fingerprint: str | None = None,
    session: dict | None = None,
    source_type: str | None = None,
) -> dict:
    """Pure manifest-definition builder (JSON dict for ``c2pa.Builder``).

    ``ingredient_fingerprint`` is the seed audio's ``waveform_sha256``
    fingerprint (:func:`acestep.track_assets.waveform_fingerprint`);
    when present the action's ``digitalSourceType`` switches to
    ``compositeWithTrainedAlgorithmicMedia`` per spec 02 §3. The seed
    reference lives inside ``com.daydream.session`` rather than a
    ``c2pa.ingredient`` because a hard ingredient assertion requires
    hashing the original asset stream, which these writers no longer
    have — the fingerprint is the identity the stem cache already uses.

    ``session`` is a :meth:`SessionLogTap.manifest_summary` dict; when
    ``None`` (no live session — offline precompute) the session
    assertion still ships with null identity fields so the do-not-train
    and AI-disclosure posture is never conditional on a session.

    ``source_type`` overrides the derived digitalSourceType for callers
    that know seed audio exists without holding its fingerprint (e.g.
    stems separated from a not-yet-persisted source).
    """
    if source_type is None:
        source_type = (
            COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA
            if ingredient_fingerprint
            else TRAINED_ALGORITHMIC_MEDIA
        )
    generator: dict = {"name": provenance_keys.CLAIM_GENERATOR_NAME}
    version = _generator_version()
    if version:
        generator["version"] = version
    session = session or {}
    session_data = {
        "session_id": session.get("session_id"),
        "model": session.get("model"),
        "loras": session.get("loras") or [],
        "timeline_summary": session.get("timeline_summary") or {},
        "session_log_sha256": session.get("session_log_sha256"),
    }
    if ingredient_fingerprint:
        session_data["seed_waveform_sha256"] = ingredient_fingerprint
    return {
        "claim_generator_info": [generator],
        "title": title,
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": source_type,
                            "softwareAgent": generator,
                        },
                    ],
                },
            },
            {
                # CAWG Training & Data Mining: notAllowed for every
                # category, on by default (spec 02 §3).
                "label": "cawg.training-mining",
                "data": {
                    "entries": {
                        c: {"use": "notAllowed"} for c in _CAWG_CATEGORIES
                    },
                },
            },
            {
                "label": "com.daydream.session",
                "data": session_data,
            },
        ],
    }


def _session_summary_for(session_id: str) -> Optional[dict]:
    try:
        from acestep.provenance.session_log import session_summary_for
        return session_summary_for(session_id)
    except Exception:  # noqa: BLE001
        return None


def embed_wav_manifest(
    path: Path,
    *,
    ingredient_fingerprint: str | None = None,
    session: dict | None = None,
    session_id: str | None = None,
    title: str | None = None,
    source_type: str | None = None,
) -> bool:
    """Embed a signed C2PA manifest into the WAV at ``path``, in place
    (write-temp-then-replace, so a failure leaves the unsigned WAV
    intact). Returns True on success; NEVER raises.

    The session assertion is bound to the SPECIFIC session that produced
    the asset (spec 06 §2.5): pass an explicit ``session`` summary dict,
    or a ``session_id`` to resolve the summary of that session's live tap.
    When neither is given (pure-local write, or the producing session is
    no longer live) the assertion ships with null session fields — the
    manifest never borrows an arbitrary "latest" session's identity
    (audit F7/G5).
    """
    c2pa = _import_c2pa()
    if c2pa is None:
        return False
    material = provenance_keys.signing_material()
    if material is None:
        return False
    path = Path(path)
    tmp = path.with_name(path.name + ".c2pa.tmp")
    try:
        if session is not None:
            resolved_session = session
        elif session_id is not None:
            resolved_session = _session_summary_for(session_id)
        else:
            resolved_session = None
        manifest = build_manifest_definition(
            title=title or path.name,
            ingredient_fingerprint=ingredient_fingerprint,
            session=resolved_session,
            source_type=source_type,
        )
        # The wrapper's __init__ rejects ta_url=None but the native
        # field is nullable; a NULL ta_url means "no timestamp
        # authority", which is right for offline local signing (an
        # empty-string URL makes the TSA fetch fail the whole sign).
        info = c2pa.C2paSignerInfo(
            alg=b"es256",
            sign_cert=material.cert_chain_pem,
            private_key=material.key_pem,
            ta_url=b"",
        )
        info.ta_url = None
        with c2pa.Signer.from_info(info) as signer:
            builder = c2pa.Builder(manifest)
            # Positional on purpose: Builder.sign's parameters are the
            # overload-style ``signer_or_format`` / ``format_or_source``
            # / ``source_or_dest`` names, so keyword calls TypeError.
            with open(path, "rb") as src, open(tmp, "wb") as dst:
                builder.sign(signer, "audio/wav", src, dst)
        os.replace(tmp, path)
        return True
    except Exception as exc:  # noqa: BLE001 — asset write must survive
        logger.warning(
            "c2pa manifest embed failed path={} error={}", path, exc,
        )
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
