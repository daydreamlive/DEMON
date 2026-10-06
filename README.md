<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/img/DEMON-logo-dark.png">
    <img src="docs/assets/img/DEMON-logo-transparent.png" width="560" alt="DEMON">
  </picture>
</p>

<p align="center"><strong>Diffusion Engine for Musical Orchestrated Noise</strong></p>
<p align="center"><sub>A platform for real-time music generation</sub></p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.28657"><img alt="Paper on arXiv" src="https://img.shields.io/badge/paper-arXiv%3A2605.28657-b31b1b.svg"></a>
  <a href="https://daydreamlive.github.io/DEMON/"><img alt="Project page" src="https://img.shields.io/badge/project%20page-demos%20%26%20experiments-111111.svg"></a>
  <a href="https://daydream.live/#preview"><img alt="Try it" src="https://img.shields.io/badge/try%20it-daydream.live-ff7a00.svg"></a>
  <a href="LICENSE"><img alt="License: AGPL-3.0-or-later + MIT" src="https://img.shields.io/badge/license-AGPL--3.0--or--later%20%2B%20MIT-blue.svg"></a>
</p>

DEMON streams music from a generative model while you steer it. One streaming engine, one wire protocol and one client SDK sit in front of interchangeable model families: Stable Audio 3 and ACE-Step v1.5 are featured, Magenta RealTime 2, MiniMax-Music3 and YuE2 are experimental. Changes to a knob are heard within one tick, tens of milliseconds, without stopping the music.

**Hear it first.** The [project page](https://daydreamlive.github.io/DEMON/) has the video captures and audio experiments; the [paper](https://arxiv.org/abs/2605.28657) describes the engine; [daydream.live](https://daydream.live/#preview) runs it without a GPU.

## Model families

A server boots one family (`--checkpoint <alias>`). Every client speaks the same protocol whichever family is loaded and reads that family's capabilities and knobs from the handshake.

| Family | Status | Steer it with | Boot alias |
|---|---|---|---|
| [Stable Audio 3](https://huggingface.co/stabilityai/stable-audio-3-small-music) (small-music, medium) | Featured | Per-frame curves, prompt morphing, audio-to-audio from an uploaded source | `sa3-small`, `sa3-medium` |
| [ACE-Step v1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (turbo 2B, XL turbo 5B) | Featured, the default install | Per-frame curves on every solver knob, prompt A/B morphing, LoRA hot-swap, timbre and structure references, audio in | default; `xl` |
| Magenta RealTime 2 (`mrt2_small`, `mrt2_base`) | Experimental; JAX sidecar | Prompt A/B blend, sampling and guidance knobs; append-only | `mrt2-sidecar` |
| [MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) | Experimental; needs a 32 GB card | Style prompt and lyrics; append-only | `minimax-music3` |
| [YuE2](https://github.com/multimodal-art-projection/YuE) (3B) | Experimental; non-commercial weights | Style prompt and lyrics compose the song; denoise, step count, x0 target, feedback and seed reshape it live | `yue2-3b` |

Setup, knobs and known gaps per family: [docs/FAMILIES.md](docs/FAMILIES.md) (and [docs/MINIMAX.md](docs/MINIMAX.md)). ACE-Step operating detail: [docs/ACESTEP.md](docs/ACESTEP.md).

### Benchmarks (RTX 5090)

Latency depends on the length of the latent being generated, so each row states what it was measured on. "Knob to ear" is the wall time from a control change to the first output that carries it, measured at the engine's output (not through the browser).

| Family | Measured on | Throughput | Knob to ear | Cold start | VRAM |
|---|---|---|---|---|---|
| ACE-Step v1.5 turbo 2B, TensorRT, 8 steps | 60 s song, 1500 latent frames at 25 Hz | 11.3 generations/s at depth 4, 12.3 at depth 8 | Per-frame curves and other shared-state knobs: one tick, 14 ms at depth 1, 43 ms at depth 4, 81 ms at depth 8. Prompt, source or denoise change: 112 ms at depth 1, 471 ms at depth 4, 649 ms at depth 8 | ~15 s | ~12 GB working set |
| Stable Audio 3 medium, TensorRT, 8 steps | 54 s window, 646 latent frames | 6.3 generations/s, flat from depth 1 to 8 | 218 ms at depth 1, 1,202 ms at depth 4 (knob rides the next request) | 17-21 s | 8-11 GB |
| Magenta RealTime 2 | 60 s rolling window, 40 ms frames | 1.7x realtime (small), 0.93x (base) | 750 ms, the `mrt2_lead` default | ~30 s sidecar warmup, once | not measured |
| MiniMax-Music3, TensorRT renderer | rolling song, 4 s render chunks | 1.3x realtime | 3,300-3,600 ms to the delivery frontier, plus the playback lead | ~30 s | 28.7 GB of 32 GB |
| YuE2 3B, TensorRT acoustic stage | 60 s song | 70 ms tick, 32 ticks per full solve | 1,200 ms (denoise), 2,400-2,500 ms (seed, x0 target); a prompt change re-composes the song in 21-58 s | ~34 s | 21-22 GB |

Sources: ACE-Step from the paper (Tables 12, 14, 15); Stable Audio 3 from the maintainers' runs on the same machine; the rest from [docs/FAMILIES.md](docs/FAMILIES.md) and [docs/MINIMAX.md](docs/MINIMAX.md). For the diffusion families, ring depth trades control latency for smoother parameter glides.

## Quickstart

**You need:** an NVIDIA GPU (tested on RTX 3090 / 4090 / 5090; ACE-Step fits a 24 GB card), [uv](https://docs.astral.sh/uv/), Node.js 20+ (web demo only) and about 40 GB of disk.

```bash
git clone https://github.com/daydreamlive/DEMON.git
cd DEMON
uv sync
uv run demon-setup
```

`demon-setup` downloads the ACE-Step v1.5 checkpoints (~18 GB), the pinned Stable Audio 3 source checkout and a starter pack of LoRAs, and builds the minimal TensorRT engines. It is idempotent. Everything lands in `~/.daydream-scope/models/demon/` (override with `ACESTEP_MODELS_DIR`); directory layout, options and troubleshooting are in [docs/INSTALL.md](docs/INSTALL.md).

**Stable Audio 3.** The weights are a manual download, then boot with an SA3 alias:

```bash
# small-music (or stable-audio-3-medium for the medium checkpoint)
huggingface-cli download stabilityai/stable-audio-3-small-music \
  --local-dir ~/.daydream-scope/models/demon/sa3/checkpoints/stable-audio-3-small-music

uv run python -u -m demos.realtime_motion_graph_web.run -- --checkpoint sa3-small   # or sa3-medium
# open http://localhost:6660/sa3/
```

The variant is fixed at boot; opening `/sa3/` against an ACE-Step server does not switch models.

**ACE-Step v1.5.**

```bash
uv run python -u -m demos.realtime_motion_graph_web.run
# open http://localhost:6660
```

Click **Play**. The first start takes ~15 s while the model and engines load, then audio streams continuously and every control in the drawer is live. The bare launch runs all-TensorRT and needs the engines `demon-setup` built; after `demon-setup --skip-engines`, launch with `-- --accel compile` instead.

**Experimental families** boot with their own alias and serve a page on the backend port. Magenta RealTime 2 needs its sidecar running first (`python scripts/mrt2_sidecar.py --model mrt2_small`, in a Linux/WSL venv), MiniMax-Music3 needs the model under `DEMON_MINIMAX_DIR`, YuE2 needs the `DEMON_YUE2_*` environment; all three are spelled out in [docs/FAMILIES.md](docs/FAMILIES.md).

```bash
uv run python -u -m demos.realtime_motion_graph_web.run -- --checkpoint mrt2-sidecar     # http://localhost:1318/mrt2/
uv run python -u -m demos.realtime_motion_graph_web.run -- --checkpoint minimax-music3   # http://localhost:1318/minimax/
uv run python -u -m demos.realtime_motion_graph_web.run -- --checkpoint yue2-3b          # http://localhost:1318/yue2/
```

## The streaming engine

Stable Audio 3 and ACE-Step run through one streaming diffusion pipeline ([`acestep/engine/stream.py`](acestep/engine/stream.py)), and YuE2's acoustic stage runs in the same ring. A ring buffer holds several in-flight generations at different denoising stages and advances them together in one batched forward pass per tick. Each slot carries its own seed, denoise strength and schedule, so a change never has to drain the buffer. Per-frame curves (source preservation, velocity, guidance, noise injection, x0 target) are read from shared state on every solver step, which is why a write is heard one tick later at any depth. The DiT and VAE run end to end in TensorRT with a refit-enabled decoder, so a LoRA swap never rebuilds an engine, and the streaming output is bit-identical to a batch run.

The paper covers the mechanism in full. The Session API, acceleration backends, tuning, TensorRT builds and the web demo are in [docs/ACESTEP.md](docs/ACESTEP.md); the wire protocol and the client SDK are in [`packages/demon-client`](packages/demon-client/) and [AGENTS.md](AGENTS.md).

## Citation

```bibtex
@article{fosdick2026demon,
  title   = {DEMON: Diffusion Engine for Musical Orchestrated Noise},
  author  = {Fosdick, Ryan},
  journal = {arXiv preprint arXiv:2605.28657},
  year    = {2026}
}
```

Please also cite the model of the family you ran. For ACE-Step: Gong et al., *ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation*, arXiv:2602.00744.

## Acknowledgments

DEMON does not train or own the models it runs. Each family is its upstream team's work: **ACE-Step v1.5** by the [ACE-Step team](https://github.com/ace-step/ACE-Step) (MIT), where DEMON began; **Stable Audio 3** by [Stability AI](https://huggingface.co/stabilityai); **Magenta RealTime 2** by Google Magenta; **MiniMax-Music3** by [MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-Music3); **YuE2** by the m-a-p YuE team ([code](https://github.com/multimodal-art-projection/YuE), [weights](https://huggingface.co/m-a-p/YuE2-3B), CC BY-NC 4.0). The ring-buffer streaming pattern follows [StreamDiffusion](https://arxiv.org/abs/2312.12491).

## Authors and license

DEMON was created by Ryan Fosdick ([@RyanOnTheInside](https://ryanontheinside.com)) and is maintained by [Daydream Live](https://daydream.live) and contributors. Contributions are welcome; [AGENTS.md](AGENTS.md) is the developer guide and `uv run pytest tests/` runs the suite.

DEMON is distributed under the **GNU Affero General Public License v3.0 or later**; see [`LICENSE`](LICENSE). Portions derive from ACE-Step, released under MIT; that notice is preserved in [`LICENSE-MIT`](LICENSE-MIT).
