"""ACE-Step v1.5 runtime for the TADA replication (DEMON's real pipeline).

Everything generates through ``Session.stream`` / ``StreamPipeline`` in
drain mode (one request, all steps, synchronous), so steering reaches
the model exactly the way it does live: the pipeline's one steering
slot, eager hooks or the TensorRT ``steering_xattn`` input.

Text-to-music follows DEMON's text-only ACE session: a silent source
anchor of the clip length, denoise 1.0 (ACE v1.5 turbo has no CFG, so
TADA's conditional-pass-only rule is every row).

Hook points (TADA, Staniszewski et al., arXiv 2602.11910):

* steering / capture: ``decoder.layers[l].cross_attn`` output, before
  the residual add (v1 ``transformer_blocks.l.cross_attn``);
* patching: the cross-attention K/V of layer ``l``, implemented as the
  layer's cross-attention seeing the clean prompt's (embedded) text
  states and mask while every other layer sees the corrupted prompt.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
while str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

CHECKPOINT = "acestep-v15-turbo"
SR = 48000
TADA_ENGINE = "tada_decoder_mixed_refit_b8_60s"


def _silence(duration_s: float):
    from acestep.nodes.types import Audio

    n = int(round(duration_s * SR))
    pool = 1920 * 5
    n -= n % pool
    return Audio(waveform=torch.zeros(2, n), sample_rate=SR)


class AceTada:
    """One ACE session plus a silent anchor, ready for TADA runs."""

    def __init__(self, *, backend: str = "tensorrt", duration_s: float = 30.0,
                 steps: int = 8, shift: float = 3.0,
                 decoder_engine: str = TADA_ENGINE):
        from acestep.engine.session import Session
        from acestep.paths import checkpoints_dir, default_trt_engines

        self.backend = backend
        self.steps = int(steps)
        self.shift = float(shift)
        self.duration_s = float(duration_s)
        trt = default_trt_engines(decoder=decoder_engine)
        self.session = Session(
            project_root=str(checkpoints_dir()), config_path=CHECKPOINT,
            decoder_backend=backend, vae_backend="tensorrt", trt_engines=trt,
        )
        torch.manual_seed(0)
        self.source = self.session.prepare_source(_silence(self.duration_s))
        self._cond_cache: Dict[tuple, object] = {}
        self.handle = self.session.stream(
            source=self.source, conditioning=self.cond("music", ""),
            steps=self.steps, shift=self.shift, method="ode", pipeline_depth=1,
        )
        # Build the pipeline once so steering can be set before any
        # request of interest runs.
        self._raw_generate(self.cond("music", ""), seed=0)

    # ---- conditioning --------------------------------------------------

    def cond(self, prompt: str, lyrics: str = ""):
        key = (prompt, lyrics)
        if key not in self._cond_cache:
            if len(self._cond_cache) > 512:
                self._cond_cache.clear()
            self._cond_cache[key] = self.session.encode_text(
                tags=prompt, lyrics=lyrics, duration=self.duration_s,
            )
        return self._cond_cache[key]

    # ---- generation ----------------------------------------------------

    @property
    def pipeline(self):
        return self.handle.stream_node.pipeline

    @property
    def decoder(self):
        return self.pipeline.decoder

    def _raw_generate(self, cond, seed: int) -> torch.Tensor:
        lat = self.handle.tick(
            positive=cond, seed=int(seed), denoise=1.0, drain=True,
            source_latent=self.source.latent,
        )
        return lat.tensor

    def generate(self, prompt: str, *, seed: int, lyrics: str = "",
                 steering: Optional[list] = None) -> torch.Tensor:
        """Finished latent for one prompt; ``steering`` = set_steering
        configs for this generation (None or [] = unsteered)."""
        pipe = self.pipeline
        pipe.set_steering(list(steering or []))
        try:
            return self._raw_generate(self.cond(prompt, lyrics), seed)
        finally:
            pipe.set_steering([])

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """``[channels, samples]`` float32 at 48 kHz."""
        from acestep.nodes.types import Latent

        audio = self.handle.decode(Latent(tensor=latent))
        wav = audio.waveform
        if wav.dim() == 3:
            wav = wav[0]
        return wav.detach().float().cpu()

    # ---- capture (eager only) --------------------------------------------

    @contextmanager
    def capture_cross_attn(self, sink: Callable[[int, int, torch.Tensor], None],
                           layers: Optional[Sequence[int]] = None):
        """Call ``sink(layer, step, mean_over_time[H])`` for every row of
        every eager forward, from the cross-attention OUTPUT (before the
        residual add). Eager decoder only."""
        if self.backend != "eager":
            raise RuntimeError("activation capture needs the eager decoder")
        pipe = self.pipeline
        mods = self.decoder.layers
        idx = list(range(len(mods))) if layers is None else list(layers)
        handles = []
        for l in idx:
            def hook(_m, _i, out, l=l):
                h = out[0] if isinstance(out, tuple) else out
                rows = list(pipe._current_step_per_row)
                if pipe._steering_neg_pass or len(rows) != h.shape[0]:
                    return None
                m = h.detach().float().mean(dim=1)
                for r, s in enumerate(rows):
                    sink(l, int(s), m[r].cpu())
                return None
            handles.append(mods[l].cross_attn.register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    # ---- patching (eager only) -------------------------------------------

    @contextmanager
    def patch_cross_attn_text(self, layers: Sequence[int], clean_prompt: str,
                              lyrics: str = ""):
        """Inside the block, the cross-attention of ``layers`` sees the
        clean prompt's text states (TADA's K/V patch): every other layer
        keeps the prompt being generated."""
        if self.backend != "eager":
            raise RuntimeError("patching needs the eager decoder")
        mods = self.decoder.layers
        clean = self._embedded_text(clean_prompt, lyrics)
        handles = []
        for l in layers:
            def pre(_m, args, kwargs):
                B = kwargs["encoder_hidden_states"].shape[0]
                kwargs = dict(kwargs)
                kwargs["encoder_hidden_states"] = clean["enc"].expand(B, -1, -1)
                if clean["mask"] is not None:
                    m = clean["mask"]
                    kwargs["attention_mask"] = m.expand(B, *m.shape[1:])
                return args, kwargs
            handles.append(mods[l].cross_attn.register_forward_pre_hook(pre, with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _embedded_text(self, prompt: str, lyrics: str) -> dict:
        """The condition-embedded text states (what every cross_attn
        receives) for ``prompt``, captured from one eager forward."""
        got: dict = {}
        ca = self.decoder.layers[0].cross_attn

        def pre(_m, args, kwargs):
            if "enc" not in got:
                got["enc"] = kwargs["encoder_hidden_states"][:1].detach().clone()
                m = kwargs.get("attention_mask")
                got["mask"] = None if m is None else m[:1].detach().clone()
            return None

        h = ca.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            entry = self.cond(prompt, lyrics)
            # One full generation is the simplest path to a real forward.
            self._raw_generate(entry, seed=0)
        finally:
            h.remove()
        return got

    def close(self) -> None:
        try:
            self.handle.close()
        finally:
            self.session.close()
