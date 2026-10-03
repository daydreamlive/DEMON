"""TADA's sparse-autoencoder steering, ported for any DEMON family.

Staniszewski, Zaleska, Modrzejewski, Deja, "TADA! Tuning Audio Diffusion
Models through Activation Steering" (arXiv 2602.11910; reference code
github.com/luk-st/steer-audio, MIT), App. I.1.4 and I.2:

* :mod:`.cache`: activations of a localised layer's cross-attention output,
  every token, conditional pass, every ``k``-th step, recorded while the
  model generates from varied captions.
* :mod:`.model`: the reference BatchTopK SAE (AuxK, unit-norm decoder).
* :mod:`.train`: the reference trainer for one process, the ``(m, k)``
  sweep, held-out FVU per noise bucket with an absolute gate, and the
  configuration choice.
* :mod:`.scoring`: TF-IDF concept scores (Eq. 12), top-``k_c`` features,
  ``v_SAE`` (Eq. 13) per step or pooled, ``k_c`` chosen on held-out prompts.
* :mod:`.packs`: ``v_SAE`` as a ``cross_attn_output`` steering pack.

A family plugs in exactly as for CAA: an
:class:`~acestep.tada.target.ActivationTarget` over its cross-attention
modules, a call context saying which rows are the conditional pass, and a
generate callable. Nothing here names a model.
"""

from .cache import ActivationStore, TokenRecorder, cache_generations
from .model import Sae, SaeConfig
from .packs import METHOD_TADA_SAE, sae_pack, write_sae_pack
from .scoring import (
    K_GRID, FeatureMeanRecorder, concept_vectors, pooled_scores, sae_vector,
    score_tables, select_k, stack_blocks, tfidf, top_features,
)
from .train import (
    SWEEP_K, SWEEP_M, SaeTrainer, TrainConfig, absolute_bucket_gate, choose_config,
    config_name, fvu_by_bucket, sweep_configs,
)

__all__ = [
    "ActivationStore", "FeatureMeanRecorder", "K_GRID", "METHOD_TADA_SAE", "SWEEP_K",
    "SWEEP_M", "Sae", "SaeConfig", "SaeTrainer", "TokenRecorder", "TrainConfig",
    "absolute_bucket_gate", "cache_generations", "choose_config", "concept_vectors",
    "config_name", "fvu_by_bucket", "pooled_scores", "sae_pack", "sae_vector",
    "score_tables", "select_k", "stack_blocks", "sweep_configs", "tfidf", "top_features",
    "write_sae_pack",
]
