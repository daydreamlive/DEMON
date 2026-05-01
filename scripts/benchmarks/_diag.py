"""Quick diagnostic: check shapes and values for source preparation."""
import torch, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
torch.set_grad_enabled(False)

from acestep.engine.session import Session
from acestep.nodes.types import Audio
import soundfile as sf

session = Session(
    project_root="checkpoints",
    config_path="acestep-v15-base",
    use_flash_attention=True,
)

path = os.path.join("test_output", "base_tasks", "source_text2music.wav")
wav, sr = sf.read(path)
waveform = torch.from_numpy(wav.T).unsqueeze(0).float()
audio = Audio(waveform=waveform, sample_rate=sr)

source = session.prepare_source(audio)
print("latent shape:", source.latent.tensor.shape)
print("context_latent shape:", source.context_latent.tensor.shape)
print("latent D:", source.latent.tensor.shape[-1])
print("context_latent D:", source.context_latent.tensor.shape[-1])
print("latent mean:", source.latent.tensor.mean().item())
print("context mean:", source.context_latent.tensor.mean().item())

session.handler._ensure_silence_latent_on_device()
sil = session.handler.silence_latent
print("silence shape:", sil.shape)
print("silence D:", sil.shape[-1])
