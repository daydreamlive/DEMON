# Box status: vast 51480126 (SA3 steering bench)

## 2026-10-05 15:25Z READY
- ssh: `ssh -i <ssh-key> -p <box-port> root@<box-host>` (direct <box-ip>:<box-direct-port>)
- GPUs: 4x RTX 5090 32 GB, driver 610.43.02, CUDA UMD 13.3, idle. Ubuntu 24.04, 384 cores, 377 GB RAM.
- DISK CONSTRAINT: overlay / 3.8G free (shared box: /workspace = 487G of other projects). /dev/shm is noexec, 25G free (shared, shrinking). All our bulk lives in /dev/shm/steerbench (LOST ON REBOOT). Bench outputs must stay small or go to shm with care.

### Activation
- DEMON (both trees): `source /dev/shm/steerbench/env.sh; cd /root/DEMON-bench; /root/demonenv/bin/python ...` (or `source /root/demonenv/bin/activate`)
- Scoring: `source /dev/shm/steerbench/env.sh; cd /root/DEMON-bench; /root/evalenv/bin/python scripts/tada/sa3_tada_score.py ...`
- env.sh sets ACESTEP_MODELS_DIR, DEMON_STEERING_PACKS_DIR, HF_HOME=/root/hf, HF_TOKEN, TADA_ROOT=/root/tada, TADA_REF=/root/steer-audio.

### Paths (all /root/* are symlinks into /dev/shm/steerbench)
- /root/DEMON-steer = git archive of DEMON-steer HEAD 83f5b77b (feat/steering-generic)
- /root/DEMON-bench = git archive of DEMON-tada-sa3 974ab049, clean; scripts/tada_sae/ = DEMON-tada-sae-sa3 d5f4c63f scripts/tada (sa3_e5_capture/e5_vectors/e4_probe)
- models: /dev/shm/steerbench/models/demon/sa3/checkpoints/stable-audio-3-medium, vendor sa3/vendor/stable-audio-3 @960da1f8, packs steering_packs/sa3/medium (bright, density, percussive, rough, warm)
- /root/steer-audio (code; excluded res/ = only the 2.2G CLAP ckpt, now a symlink), /root/tada (TADA_ROOT; data -> /root/data, steer-audio link), /root/data (musiccaps-public.csv, patching_prompts.csv), /root/hf (MuQ-MuLan-large + xlm-roberta-base)
- TADA prompt JSONs present: DEMON-bench/acestep/tada/data/{benchmark_prompts,localization,steering_concepts}.json

### Pre-existing on box (reused read-only, not modified)
- SA3 medium weights + t5gemma: /workspace/accomp/data/models/stable-audio-3-medium (sampled md5 first/last 64MB == local medium); symlinked, our own model_config.json (box copy differs only in pretransform freeze=True)
- CLAP music_audioset_epoch_15_esc_90.14.pt: /root/.cache/audio_metrics/ (md5 c2220d1f == local)
- python3.12 /usr/bin, uv python 3.11 at /.uv/python_install, HF token in /root/.cache/huggingface/token
- base venvs layered via .pth (box's own pattern, as /dev/shm/musetimbre/env): /workspace/xattn_combined/venv (py3.12, torch 2.9.1+cu128, transformers 5.17) for demonenv; /workspace/venv (py3.11, torch 2.11+cu128) + /workspace/master_adapter/ab_p1b/venv (laion_clap, muq, audiobox_aesthetics) for evalenv
- No DEMON checkout, steer-audio, evalenv or packs existed before.

### Added by me
- /dev/shm/steerbench/* (code, models dir, data, hf, env.sh, logs/, fixdeps.sh, bundle/, old/ = discarded first attempts)
- demonenv (py3.12 overlay venv): + loguru 0.7.3; flash-attn 2.8.3 (cu128 torch2.9 wheel from pyproject) in /root/steerbench_exec/site (539M on overlay, needs exec)
- evalenv (py3.11 overlay venv): + laion-clap 1.1.6, fire 0.7.0, termcolor 3.3.0, timbral_models, pyloudnorm (all --no-deps)
- DEVIATIONS from the lock: python 3.12 (lock 3.11), transformers 5.17 (lock 4.57.6), numpy 2.2.6; evalenv torch 2.11 (pin 2.9.1), transformers 5.17 (pin 4.47). The pinned `uv sync` could not run: no exec space.

### Smoke (GPU 0)
- render: `sa3_tada_run.py swap --concepts tempo --n-prompts 2 --batch 2 --out /root/tada/smoke` rc 0, 12 s wall (load ~8 s, 8 steps ~19 it/s), flash-attn active
- score: `sa3_tada_score.py protocol --out /root/tada/smoke --sub calib --lpaps-only` (pos as alpha 0, neg as alpha 1) rc 0, 18 s; LPAPS 3.28 +- 1.41
- evalenv loads MuQ-MuLan-large, CLAP music ckpt, audiobox aesthetics (38 s); `import timbral_models` OK
- DEMON-steer imports from demonenv; packs dir and vendor resolve.
- Note: without flash-attn the vendor falls back to SDPA (the vendor flags SDPA as numerically wrong for its windowed attention); the first SDPA smoke was discarded.

## discriminators (box-prep, 2026-10-05 17:05Z, CPU-only smoke, no GPU used)
- Env for ALL scorers: evalenv. `source /dev/shm/steerbench/env.sh; export TORCH_HOME=/dev/shm/steerbench/torchhub; /root/evalenv/bin/python ...`
  New compiled/pure wheels went into /root/steerbench_exec/evalsite (overlay, exec, prepended via 00_sb_evalsite.pth), installed `uv pip install --python /root/evalenv/bin/python --target /root/steerbench_exec/evalsite --no-deps ...`. Caches on /dev/shm (uv_cache, pip_cache). No labelenv was needed.
- Versions: torch 2.11.0+cu128, torchvision 0.26.0, numpy 2.2.6, librosa 0.11.0, pandas 3.0.6, pyarrow 25.0.1 (new), soundfile 0.14.0, timm 1.0.30 (new), hear21passt 0.0.26 (new), essentia 2.1b6.dev1389 (new, 1 s install), laion_clap 1.1.6, transformers 4.47.1, panns_inference 0.1.0 (pre-existing, untested: PaSST works).
- Smoke wav: /dev/shm/steerbench/boxprep_smoke.wav (10 s, 44.1k, synth A-major chord + 120 bpm clicks). Script /dev/shm/steerbench/boxprep_smoke.py, results logs/boxprep_smoke.json.
- PaSST (AudioSet 527): `from hear21passt.base import get_basic_model; m=get_basic_model(mode="logits").eval(); p=torch.sigmoid(m(wav32k[None]))` (input mono 32 kHz float tensor [B,T]). Weights auto-download to $TORCH_HOME/hub/checkpoints/passt-s-f128-p16-s10-ap.476-swa.pt (329 MB, present). Model prints shape debug lines; harmless. Smoke: 527 classes, top3 Static .27 / Sine wave .13 / Music .07, 11 s CPU.
- AudioSet labels: /dev/shm/steerbench/ckpt/audioset/class_labels_indices.csv (527 rows: index,mid,display_name) + ontology.json.
- CLAP general: /dev/shm/steerbench/ckpt/clap/630k-audioset-best.pt (1.86 GB): `laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny"); m.load_ckpt(path)` with torch.load patched to weights_only=False (as sa3_tada_score._clap_model). Audio 48 kHz via get_audio_embedding_from_data. Smoke cos [chord+click .29, female singer .03].
- CLAP music: /root/.cache/audio_metrics/music_audioset_epoch_15_esc_90.14.pt, same call with amodel="HTSAT-base". Smoke cos [.30, -.03].
- MuQ-MuLan: `from muq import MuQMuLan; MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")` (HF_HOME=/root/hf), audio 24 kHz. Smoke sim [.44, .21].
- pyloudnorm: `pyln.Meter(44100).integrated_loudness(y)` -> -21.3 LUFS.
- librosa: beat_track tempo 120.2, centroid OK. essentia: `essentia.standard` MonoLoader + RhythmExtractor2013(method="multifeature") 119.9 bpm, KeyExtractor A major .76.
- parquet: pandas to_parquet/read_parquet roundtrip OK (pyarrow engine, dotted column names fine).
- timbral_models: BROKEN on numpy 2 / librosa 0.11 without a shim. Required before `import timbral_models`:
  `if not hasattr(np.lib,"pad"): np.lib.pad=np.pad`; wrap librosa.onset.onset_detect and onset_strength so positional (y, sr) become keywords. With the shim all 8 work (hardness 53.6, depth 57.6, brightness 53.4, roughness 50.0, warmth 47.3, sharpness 42.8, booming 29.0, reverb 1.0). Function names: timbral_{hardness,depth,brightness,roughness,warmth,sharpness,booming,reverb}.
- Not done: MuseTimbre (per instructions). efficientat not tried (PaSST works).
- Disk after: /dev/shm 7.0 GB free (was 11 GB at start; +2.2 GB mine = ckpt 1.8G + torchhub 329M; the rest is other jobs). Largest: hf 11G, out 8.3G, steer-bench-trash 7.2G (purge candidate), ckpt 1.8G. Overlay / 3.4 GB free.
- Re-verify 2026-10-05 (CPU only, CUDA_VISIBLE_DEVICES="", nothing new installed): evalenv imports `import hear21passt` 0.0.26, `import pyarrow` 25.0.1 (fastparquet absent, not needed), `import transformers` 4.47.1, `import timm` 1.0.30, numpy 2.2.6, torch 2.11.0+cu128. PaSST re-smoke `from hear21passt.base import get_basic_model; get_basic_model(mode="logits")` on boxprep_smoke.wav at 32 kHz -> (1, 527), max .273, 3.4 s CPU; AST fallback not needed.
- demucs: `import demucs` OK, 4.0.1, from the layered base venv /workspace/master_adapter/ab_p1b/venv (not installed by us).
- Skipped per instructions: beat_this (not importable, not installed), MuseTimbre (not installed). General CLAP 630k-audioset-best.pt was already downloaded by the earlier box-prep pass (1.86 GB in /dev/shm/steerbench/ckpt/clap); left in place, not deleted.
- Disk at re-verify: overlay / 3.4 GB free, /dev/shm 13 GB free.
