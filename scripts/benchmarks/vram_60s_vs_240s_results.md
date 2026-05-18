# 60s vs 240s TRT engine — VRAM at identical 60s input

Captured by `scripts/benchmarks/bench_vram_60s_vs_240s.py` on an RTX 5090.
Each engine ran in its own subprocess; VRAM was measured at the driver
level via `torch.cuda.mem_get_info()` so TRT's non-PyTorch allocations
are included. Inputs were sized to 60 seconds for every engine
(decoder: B=2 CFG, T=1500, enc_L=200; vae_decode: T=1500;
vae_encode: 60s @ 48 kHz).

## Peak VRAM (10 timed iters after 5 warmups)

| component       | 60s engine peak | 240s engine peak |          Δ |
| --------------- | --------------: | ---------------: | ---------: |
| decoder (refit) |       13,511 MB |        15,911 MB |  +2,400 MB |
| vae_decode      |       10,547 MB |        10,814 MB |    +267 MB |
| vae_encode      |        4,178 MB |        10,614 MB |  +6,436 MB |
| **total**       |   **28,236 MB** |    **37,339 MB** | **+9,103 MB** |

Switching the three default engines from 240s to 60s frees ~9.1 GB at a
60s workload. Most of the saving comes from the VAE encode engine.

The numbers track immediately after engine load (`post_load`), so this
is workspace reserved at context-creation time, not transient activations.

## How to reproduce

```
uv run python scripts/benchmarks/bench_vram_60s_vs_240s.py
```

Raw per-engine numbers are written to
`scripts/benchmarks/bench_vram_60s_vs_240s.json`.
