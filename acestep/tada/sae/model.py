"""BatchTopK sparse autoencoder, ported from TADA's reference code.

TADA (Staniszewski, Zaleska, Modrzejewski, Deja, "TADA! Tuning Audio
Diffusion Models through Activation Steering", arXiv 2602.11910; code
github.com/luk-st/steer-audio, MIT) trains one BatchTopK SAE (Bussmann et
al. 2024; Gao et al. 2024) per localised cross-attention layer, on that
layer's output (paper App. I.1.4 Eq. 10-11, App. I.2):

    f = TopK(W_enc (h - b_pre)),     h_hat = W_dec f + b_pre

This module is a port of the reference ``Sae`` (from
``src/steering/methods/sae/lib/sae/sae.py``) and its config, keeping its
names, defaults, initialisation (encoder bias zero, decoder = a copy of
the encoder rows normalised to unit norm, ``b_dec`` zero until the trainer
sets it to the geometric median) and the AuxK auxiliary loss (Gao et al.
App. B.1: the top ``d_in // 2`` dead latents predict the residual). The
reference's Triton decoder is replaced by its own eager fallback, which
computes the same thing. ``save_to_disk`` / ``load_from_disk`` keep the
reference layout (``cfg.json`` + ``sae.safetensors``), so a released
reference SAE loads here unchanged.

``W_dec`` is stored ``[num_latents, d_in]``: row ``j`` is the paper's
decoder column ``W_dec[:, j]``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn


@dataclass
class SaeConfig:
    """Reference ``SaeConfig`` (the TADA training scripts override
    ``expansion_factor``, ``k`` and ``batch_topk``)."""

    expansion_factor: int = 32
    normalize_decoder: bool = True
    num_latents: int = 0
    k: int = 32
    batch_topk: bool = False
    sample_topk: bool = False
    input_unit_norm: bool = False
    multi_topk: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SaeConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    top_indices: Tensor


class ForwardOutput(NamedTuple):
    sae_out: Tensor
    latent_acts: Tensor
    latent_indices: Tensor
    fvu: Tensor
    l0_loss: Tensor
    l2_loss: Tensor
    auxk_loss: Tensor
    multi_topk_fvu: Tensor
    explained_variance: Tensor


def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor) -> Tensor:
    """Reference fallback decoder (``utils.eager_decode``); ``W_dec`` is
    ``[d_in, num_latents]`` here (the caller passes ``W_dec.mT``)."""
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ W_dec.mT


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5) -> Tensor:
    """Weiszfeld iteration; the reference initialises ``b_dec`` with it."""
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess
        weights = 1 / torch.norm(points - guess, dim=1)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


class Sae(nn.Module):
    """Reference ``Sae`` (BatchTopK / per-token TopK, AuxK, multi-TopK)."""

    def __init__(
        self,
        d_in: int,
        cfg: SaeConfig,
        device: str | torch.device = "cpu",
        dtype: Optional[torch.dtype] = None,
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = int(d_in)
        self.num_latents = cfg.num_latents or self.d_in * cfg.expansion_factor
        self.encoder = nn.Linear(self.d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()
        self.W_dec = (
            nn.Parameter(self.encoder.weight.data[:, : self.d_in].clone()) if decoder else None
        )
        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()
        self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype=dtype, device=device))

    # ---- persistence (reference layout) ---------------------------------

    def save_to_disk(self, path: Path | str) -> Path:
        from safetensors.torch import save_model

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_model(self, str(path / "sae.safetensors"))
        (path / "cfg.json").write_text(
            json.dumps({**self.cfg.to_dict(), "d_in": self.d_in}), encoding="utf-8"
        )
        return path

    @staticmethod
    def load_from_disk(
        path: Path | str, device: str | torch.device = "cpu", *, decoder: bool = True,
    ) -> "Sae":
        from safetensors.torch import load_model

        path = Path(path)
        cfg_dict = json.loads((path / "cfg.json").read_text(encoding="utf-8"))
        d_in = cfg_dict.pop("d_in")
        sae = Sae(d_in, SaeConfig.from_dict(cfg_dict), device=device, decoder=decoder)
        load_model(sae, str(path / "sae.safetensors"), device=str(device), strict=decoder)
        # The reference switches a loaded SAE to per-token TopK.
        sae.cfg.batch_topk = False
        sae.cfg.sample_topk = False
        return sae

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    # ---- encode / decode -------------------------------------------------

    def pre_acts(self, x: Tensor) -> Tensor:
        """``relu(W_enc (x - b_dec) + b_enc)`` (decoder bias removed first)."""
        sae_in = x.to(self.dtype) - self.b_dec
        return nn.functional.relu(self.encoder(sae_in))

    def select_topk(self, latents: Tensor, k: Optional[int] = None,
                    batch_size: Optional[int] = None) -> EncoderOutput:
        """Top-k latents: across the whole batch (BatchTopK, ``k`` per row
        on average), per sample, or per token (plain TopK)."""
        if k is None:
            k = self.cfg.k
        if self.cfg.batch_topk:
            flat = latents.flatten()
            total_k = k * latents.shape[0]
            acts_f, idx_f = flat.topk(total_k, sorted=False)
            top_acts = (
                torch.zeros_like(flat).scatter(-1, idx_f, acts_f).reshape(latents.shape)
            )
            top_indices = (idx_f % self.num_latents).reshape(latents.shape[0], k)
            return EncoderOutput(top_acts=top_acts, top_indices=top_indices)
        if self.cfg.sample_topk:
            sample = latents.view(batch_size, -1)
            acts_s, idx_s = sample.topk(k, sorted=False)
            top_acts = torch.zeros_like(sample).scatter_(-1, idx_s, acts_s).reshape(latents.shape)
            return EncoderOutput(top_acts=top_acts, top_indices=idx_s % self.num_latents)
        return EncoderOutput(*latents.topk(k, sorted=False))

    def encode(self, x: Tensor) -> EncoderOutput:
        b, s, e = x.shape
        return self.select_topk(self.pre_acts(x.reshape(b * s, e)))

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        if self.cfg.batch_topk or self.cfg.sample_topk:
            y = top_acts.to(self.dtype) @ self.W_dec
        else:
            y = eager_decode(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    def preprocess_input(self, x: Tensor):
        b, s, e = x.shape
        x = x.reshape(b * s, e)
        if self.cfg.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.input_unit_norm:
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    def forward(self, x: Tensor, dead_mask: Optional[Tensor] = None) -> ForwardOutput:
        """``x`` is ``[batch, sample_size, d_in]`` (one sample = all tokens
        of one cached activation); losses as the reference computes them."""
        batch_size, sample_size, emb = x.shape
        x, x_mean, x_std = self.preprocess_input(x)
        pre_acts = self.pre_acts(x)
        top_acts, top_indices = self.select_topk(pre_acts, batch_size=batch_size, k=self.cfg.k)
        sae_out = self.decode(top_acts, top_indices)
        e = (sae_out - x).float()
        total_variance = (x - x.mean(0)).float().pow(2).sum()
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = x.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = self.select_topk(auxk_latents, k=k_aux, batch_size=batch_size)
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = scale * (e_hat - e).float().pow(2).sum() / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)
        l0_loss = (pre_acts > 0).float().sum(-1).mean()
        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance
        per_tok = (sae_out - x).pow(2).sum(dim=-1).squeeze()
        per_tok_var = (x - x.mean(0)).pow(2).sum(-1)
        explained_variance = 1 - per_tok / per_tok_var
        if self.cfg.multi_topk:
            ta, ti = self.select_topk(pre_acts, k=4 * self.cfg.k, batch_size=batch_size)
            multi_topk_fvu = (self.decode(ta, ti) - x).float().pow(2).sum() / total_variance
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)
        sae_out = self.postprocess_output(sae_out, x_mean, x_std)
        return ForwardOutput(
            sae_out, top_acts, top_indices, fvu, l0_loss,
            l2_loss / (batch_size * sample_size * emb), auxk_loss, multi_topk_fvu,
            explained_variance,
        )

    # ---- decoder constraints --------------------------------------------

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self) -> None:
        assert self.W_dec is not None
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self) -> None:
        assert self.W_dec is not None and self.W_dec.grad is not None
        parallel = (self.W_dec.grad * self.W_dec.data).sum(dim=-1)
        self.W_dec.grad -= parallel[:, None] * self.W_dec.data
