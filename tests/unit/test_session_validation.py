#!/usr/bin/env python3
"""Validation unit tests for the Session backend API.

These tests exercise only the constructor's validation logic. No model
is loaded, no GPU is required: each test patches ``ModelContext`` so
``Session.__init__`` raises before reaching the actual model load.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.engine.session import Session


def _stub_model_context():
    """Return a context manager that replaces ModelContext with a no-op stub."""
    class _StubCtx:
        def __init__(self, **kwargs):
            self.model = None
            self._diffusion_engine = None

    return patch("acestep.engine.model_context.ModelContext", _StubCtx)


class TestBackendStringValidation:
    def test_invalid_decoder_backend_raises(self):
        with pytest.raises(ValueError, match="decoder_backend"):
            Session(decoder_backend="quantized")

    def test_invalid_vae_backend_raises(self):
        with pytest.raises(ValueError, match="vae_backend"):
            Session(vae_backend="quantized")

    def test_eager_default_passes_validation(self):
        # No backend args + no trt_engines = eager defaults; validation
        # should not raise. Stub the model context so no model is loaded.
        with _stub_model_context():
            Session()


class TestTensorRTPairingValidation:
    def test_decoder_tensorrt_without_engine_raises(self):
        with pytest.raises(ValueError, match="trt_engines\\['decoder'\\]"):
            Session(decoder_backend="tensorrt")

    def test_decoder_tensorrt_with_only_vae_engines_raises(self):
        with pytest.raises(ValueError, match="trt_engines\\['decoder'\\]"):
            Session(
                decoder_backend="tensorrt",
                trt_engines={
                    "vae_encode": "/fake/path",
                    "vae_decode": "/fake/path",
                },
            )

    def test_vae_tensorrt_with_only_one_vae_key_raises(self):
        with pytest.raises(ValueError, match="vae_encode.*vae_decode"):
            Session(
                vae_backend="tensorrt",
                trt_engines={"vae_encode": "/fake/path"},
            )

    def test_vae_tensorrt_with_no_vae_keys_raises(self):
        with pytest.raises(ValueError, match="vae_encode.*vae_decode"):
            Session(
                vae_backend="tensorrt",
                trt_engines={"decoder": "/fake/path"},
            )


class TestEngineWithoutBackendValidation:
    def test_decoder_engine_without_tensorrt_backend_raises(self):
        # User passed an engine path but forgot to set the backend.
        with pytest.raises(ValueError, match="decoder_backend != 'tensorrt'"):
            Session(trt_engines={"decoder": "/fake/path"})

    def test_vae_engines_without_tensorrt_backend_raises(self):
        with pytest.raises(ValueError, match="vae_backend != 'tensorrt'"):
            Session(
                trt_engines={
                    "vae_encode": "/fake/path",
                    "vae_decode": "/fake/path",
                },
            )

    def test_decoder_engine_without_backend_raises_even_when_vae_set(self):
        with pytest.raises(ValueError, match="decoder_backend != 'tensorrt'"):
            Session(
                vae_backend="tensorrt",
                trt_engines={
                    "decoder": "/fake/path",
                    "vae_encode": "/fake/path",
                    "vae_decode": "/fake/path",
                },
            )


class TestTensorRTVAECleanup:
    def test_close_evicts_session_vae_trt_engines(self):
        engines = {
            "vae_encode": "/fake/vae_encode.engine",
            "vae_decode": "/fake/vae_decode.engine",
        }
        with (
            _stub_model_context(),
            patch("acestep.engine.runtime_init.apply_trt_backends"),
            patch("acestep.nodes.vae_nodes._evict_trt_vae", return_value=True) as evict,
        ):
            session = Session(vae_backend="tensorrt", trt_engines=engines)
            session.close()

        assert [call.args[0] for call in evict.call_args_list] == [
            "/fake/vae_encode.engine",
            "/fake/vae_decode.engine",
        ]

    def test_close_only_evicts_vae_engines_once(self):
        engines = {
            "vae_encode": "/fake/vae_encode.engine",
            "vae_decode": "/fake/vae_decode.engine",
        }
        with (
            _stub_model_context(),
            patch("acestep.engine.runtime_init.apply_trt_backends"),
            patch("acestep.nodes.vae_nodes._evict_trt_vae", return_value=True) as evict,
        ):
            session = Session(vae_backend="tensorrt", trt_engines=engines)
            session.close()
            session.close()

        assert evict.call_count == 2
