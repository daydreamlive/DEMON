"""Shared fixtures for Session API tests.

Session creation and source preparation are expensive (GPU model loading,
VAE encode, text encode). These fixtures are session-scoped so they load
once and are reused across all tests.
"""

import sys
from pathlib import Path

import pytest
import torch
from acestep.fixtures import audio_fixture

sys.path.insert(0, str(Path(__file__).parent.parent))
torch.set_grad_enabled(False)

PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_RATE = 48000
TEST_DURATION = 30.0  # seconds -- shorter for faster tests
TEST_AUDIO = audio_fixture("inside_confusion_loop_60s_gsm.wav")


def _find_trt_engines():
    """Return TRT engine dict if all engines exist, else None.

    Resolves through ``acestep.paths.default_trt_engines`` so the test
    suite picks up engines at the canonical models-dir layout
    (``$ACESTEP_MODELS_DIR`` or ``~/.daydream-scope/models/demon``)
    rather than a repo-local directory.
    """
    from acestep.paths import default_trt_engines

    paths = default_trt_engines()
    if all(Path(p).exists() for p in paths.values()):
        return paths
    return None


# ---------------------------------------------------------------------------
# Core fixtures (session-scoped, loaded once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def trt_engines():
    return _find_trt_engines()


@pytest.fixture(scope="session")
def session(trt_engines):
    from acestep.engine.session import Session
    return Session(
        decoder_backend="tensorrt" if trt_engines else "eager",
        vae_backend="tensorrt" if trt_engines else "eager",
        trt_engines=trt_engines,
    )


@pytest.fixture(scope="session")
def source_audio():
    import soundfile as sf
    from acestep.nodes.types import Audio

    data, sr = sf.read(str(TEST_AUDIO), dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    max_samples = int(TEST_DURATION * SAMPLE_RATE)
    waveform = waveform[:2, :max_samples]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]

    return Audio(waveform=waveform, sample_rate=SAMPLE_RATE)


@pytest.fixture(scope="session")
def prepared_source(session, source_audio):
    return session.prepare_source(source_audio)


@pytest.fixture(scope="session")
def conditioning(session, prepared_source):
    from acestep.constants import TASK_INSTRUCTIONS
    return session.encode_text(
        tags="electronic ambient, synthesizer",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=prepared_source.latent,
        bpm=120,
        duration=TEST_DURATION,
        key="C major",
    )


@pytest.fixture(scope="session")
def generated_latent(session, conditioning, prepared_source):
    """A single generation result, reused by decode tests."""
    return session.generate(
        conditioning=conditioning,
        context_latent=prepared_source.context_latent,
        source_latent=prepared_source.latent,
        seed=42,
        steps=4,
    )
