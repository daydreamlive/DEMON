"""Diagnostic: test extract with and without CFG to isolate the issue."""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import soundfile as sf
torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes.types import Curve

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
out_dir = os.path.join(project_root, "test_output", "base_diag")
os.makedirs(out_dir, exist_ok=True)

# Use TRT
trt_engine = os.path.join(project_root, "trt_engines",
                          "decoder_base_mixed_b8_60s", "decoder_base_mixed_b8_60s.engine")
vae_enc = os.path.join(project_root, "trt_engines", "vae_encode_fp16_60s", "vae_encode_fp16_60s.engine")
vae_dec = os.path.join(project_root, "trt_engines", "vae_decode_fp16_60s", "vae_decode_fp16_60s.engine")

trt_engines = {"decoder": trt_engine}
has_vae_trt = os.path.isfile(vae_enc) and os.path.isfile(vae_dec)
if has_vae_trt:
    trt_engines["vae_encode"] = vae_enc
    trt_engines["vae_decode"] = vae_dec

print("Loading base model with TRT...")
session = Session(
    project_root=os.path.join(project_root, "checkpoints"),
    config_path="acestep-v15-base",
    decoder_backend="tensorrt",
    vae_backend="tensorrt" if has_vae_trt else "eager",
    use_flash_attention=True,
    trt_engines=trt_engines,
)

TAGS = "jazz piano trio, brushed drums, walking bass, 140 bpm"
DUR = 60.0
STEPS = 50
SHIFT = 1.0
SEED = 42
T = int(DUR * 25)

def save(audio, name):
    path = os.path.join(out_dir, name)
    wav = audio.waveform.squeeze(0).cpu().numpy().T
    sf.write(path, wav, audio.sample_rate)
    print(f"  -> {path}")

# 1. text2music WITHOUT CFG (baseline)
print("\n=== text2music, no CFG ===")
cond = session.encode_text(tags=TAGS, lyrics="[instrumental]", duration=DUR)
lat = session.generate(conditioning=cond, seed=SEED, steps=STEPS, shift=SHIFT)
save(session.decode(lat), "t2m_no_cfg.wav")

# 2. text2music WITH CFG (null_condition_emb)
print("\n=== text2music, CFG=7.5 (null_condition_emb) ===")
neg = session.null_conditioning(cond)
gc = Curve(tensor=torch.full((T,), 7.5, dtype=torch.bfloat16))
lat = session.generate(conditioning=cond, seed=SEED, steps=STEPS, shift=SHIFT,
                       negative=neg, guidance_curve=gc)
save(session.decode(lat), "t2m_cfg_null_emb.wav")

# 3. Prepare source from text2music output
print("\n=== Preparing source ===")
audio_src = session.decode(session.generate(conditioning=cond, seed=SEED, steps=STEPS, shift=SHIFT))
source = session.prepare_source(audio_src)
save(audio_src, "source.wav")
print(f"  latent shape: {source.latent.tensor.shape}")
print(f"  context_latent shape: {source.context_latent.tensor.shape}")

# 4. extract WITHOUT CFG
print("\n=== extract drums, no CFG ===")
instr = TASK_INSTRUCTIONS["extract"].replace("{TRACK_NAME}", "drums")
cond_ext = session.encode_text(tags=TAGS, lyrics="[instrumental]", duration=DUR,
                               instruction=instr, refer_latent=source.latent)
lat = session.generate(conditioning=cond_ext, seed=SEED, steps=STEPS, shift=SHIFT,
                       context_latent=source.context_latent)
save(session.decode(lat), "extract_no_cfg.wav")

# 5. extract WITH CFG (null_condition_emb)
print("\n=== extract drums, CFG=7.5 (null_condition_emb) ===")
neg_ext = session.null_conditioning(cond_ext)
lat = session.generate(conditioning=cond_ext, seed=SEED, steps=STEPS, shift=SHIFT,
                       context_latent=source.context_latent,
                       negative=neg_ext, guidance_curve=gc)
save(session.decode(lat), "extract_cfg_null_emb.wav")

# 6. lego WITHOUT CFG
print("\n=== lego bass, no CFG ===")
instr = TASK_INSTRUCTIONS["lego"].replace("{TRACK_NAME}", "bass")
cond_lego = session.encode_text(tags=TAGS, lyrics="[instrumental]", duration=DUR,
                                instruction=instr, refer_latent=source.latent)
lat = session.generate(conditioning=cond_lego, seed=SEED, steps=STEPS, shift=SHIFT,
                       context_latent=source.latent)
save(session.decode(lat), "lego_no_cfg.wav")

# 7. lego WITH CFG (null_condition_emb)
print("\n=== lego bass, CFG=7.5 (null_condition_emb) ===")
neg_lego = session.null_conditioning(cond_lego)
lat = session.generate(conditioning=cond_lego, seed=SEED, steps=STEPS, shift=SHIFT,
                       context_latent=source.latent,
                       negative=neg_lego, guidance_curve=gc)
save(session.decode(lat), "lego_cfg_null_emb.wav")

print("\nDone. Listen to all files in:", out_dir)
