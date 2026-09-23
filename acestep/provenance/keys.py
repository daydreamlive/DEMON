"""Local signing key management for self-signed C2PA manifests.

On first use, generates an ES256 (P-256) key plus a certificate chain
into :func:`acestep.provenance.keys_dir` and reuses it afterwards. The
claim generator identity is deliberately distinct ("DEMON (local,
self-signed)") so locally signed content can never be confused with
Daydream-signed content (spec 02 §8).

The chain is a throwaway local mini-CA plus one leaf rather than a
single self-issued cert because c2pa-rs enforces the C2PA claim-signing
certificate profile at sign time and rejects self-issued leafs. The CA
private key is discarded after issuing the leaf, so the "CA" can never
sign anything else; trust-wise this is still a self-signed identity.

Requires the optional ``cryptography`` dependency (``provenance``
extra); without it every entry point returns ``None`` after a single
logged warning.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from acestep.provenance import keys_dir

__all__ = [
    "CLAIM_GENERATOR_NAME",
    "SigningMaterial",
    "signing_material",
]

CLAIM_GENERATOR_NAME = "DEMON (local, self-signed)"

_KEY_FILENAME = "demon-local-es256.key.pem"
_CERTS_FILENAME = "demon-local-es256.certs.pem"
_VALIDITY_DAYS = 3650

_warned_unavailable = False


@dataclass(frozen=True)
class SigningMaterial:
    """PEM material ready to hand to a C2PA signer."""

    key_pem: bytes
    cert_chain_pem: bytes
    key_path: Path
    cert_path: Path


def _cryptography_available() -> bool:
    global _warned_unavailable
    try:
        import cryptography  # noqa: F401
        return True
    except ImportError:
        if not _warned_unavailable:
            _warned_unavailable = True
            logger.warning(
                "provenance signing disabled: `cryptography` is not "
                "installed (install the `provenance` extra to enable "
                "local C2PA signing)"
            )
        return False


def _generate(directory: Path) -> SigningMaterial:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    not_before = now - datetime.timedelta(days=1)
    not_after = now + datetime.timedelta(days=_VALIDITY_DAYS)

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "DEMON local provenance CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "DEMON local provenance"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=False,
                key_encipherment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=True, crl_sign=True,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf_name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, CLAIM_GENERATOR_NAME),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "DEMON local provenance"),
    ])
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(leaf_name)
        .issuer_name(ca_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=False,
                key_encipherment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            # The cert profile wants a non-empty EKU; emailProtection is
            # what the c2pa-rs test fixtures use for document signing.
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    key_pem = leaf_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    chain_pem = (
        leaf_cert.public_bytes(serialization.Encoding.PEM)
        + ca_cert.public_bytes(serialization.Encoding.PEM)
    )

    directory.mkdir(parents=True, exist_ok=True)
    key_path = directory / _KEY_FILENAME
    cert_path = directory / _CERTS_FILENAME
    key_path.write_bytes(key_pem)
    os.chmod(key_path, 0o600)
    cert_path.write_bytes(chain_pem)
    logger.info(
        "provenance signing material generated key={} certs={}",
        key_path, cert_path,
    )
    return SigningMaterial(
        key_pem=key_pem, cert_chain_pem=chain_pem,
        key_path=key_path, cert_path=cert_path,
    )


def _still_valid(cert_chain_pem: bytes) -> bool:
    from cryptography import x509

    try:
        leaf = x509.load_pem_x509_certificate(cert_chain_pem)
    except Exception:
        return False
    now = datetime.datetime.now(datetime.timezone.utc)
    return leaf.not_valid_before_utc <= now < leaf.not_valid_after_utc


def signing_material(directory: Path | None = None) -> SigningMaterial | None:
    """Return the local signing material, generating it on first use.

    ``None`` when ``cryptography`` is unavailable or generation failed;
    callers must treat that as "provenance off", never as an error.
    """
    if not _cryptography_available():
        return None
    d = Path(directory) if directory is not None else keys_dir()
    key_path = d / _KEY_FILENAME
    cert_path = d / _CERTS_FILENAME
    try:
        if key_path.is_file() and cert_path.is_file():
            key_pem = key_path.read_bytes()
            chain_pem = cert_path.read_bytes()
            if _still_valid(chain_pem):
                return SigningMaterial(
                    key_pem=key_pem, cert_chain_pem=chain_pem,
                    key_path=key_path, cert_path=cert_path,
                )
            logger.warning(
                "provenance cert at {} invalid or expired; regenerating",
                cert_path,
            )
        return _generate(d)
    except Exception as exc:  # noqa: BLE001 — never break a caller's write path
        logger.warning("provenance signing material unavailable: {}", exc)
        return None
