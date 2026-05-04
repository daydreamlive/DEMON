"""Host-side bank ring buffer + TRT engine binding helper.

Pairs with ``acestep.engine.trt.bank_export``: the engine takes 8
inputs (hidden_states, timestep, encoder_hidden_states,
context_latents, bank_k, bank_v, bank_bias, step_indices) and
returns 3 outputs (velocity, k_stack, v_stack).

Bank tensor layout is RANK-5 from the engine's perspective:

  bank_k / bank_v: ``[N_banked, num_steps, kv_heads, T_lat, head_dim]``
  k_stack / v_stack: ``[N_banked, B, kv_heads, T_lat, head_dim]``

Internal storage and engine I/O share the same rank-5 layout. An
earlier rank-4 collapse (stacking the leading two dims) was tried as
a workaround for a perceived Myelin issue under TRT strongly_typed,
but the rank-4 form actually broke the build at every seq length
while the rank-5 form has a confirmed working 10s engine artifact.

``bank_bias`` is a ``[num_steps]`` additive mask: ``log(strength)`` for
populated step slots, ``-inf`` for slots not yet written.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from loguru import logger


class BankBuffer:
    """Host-side bank state for the bank-aware TRT path."""

    def __init__(
        self,
        banked_layers: Sequence[int],
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        strength: float = 1.0,
    ):
        self.banked_layers: Tuple[int, ...] = tuple(int(i) for i in banked_layers)
        self.num_steps: int = int(num_steps)
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        # Rank-5 backing storage. Engine binds the same rank-5 tensors
        # directly (no flattened view).
        self._bank_k_5d: Optional[torch.Tensor] = None
        self._bank_v_5d: Optional[torch.Tensor] = None
        self.bank_k: Optional[torch.Tensor] = None  # alias of _bank_k_5d
        self.bank_v: Optional[torch.Tensor] = None

        self.bank_bias: torch.Tensor = self._make_bias(num_steps, strength, device, dtype)
        self._valid_mask: torch.Tensor = torch.zeros(
            num_steps, dtype=torch.bool, device=device,
        )
        self._strength: float = float(strength)

    @staticmethod
    def _make_bias(
        num_steps: int, strength: float, device, dtype,
    ) -> torch.Tensor:
        # All slots invalid initially -> very-negative bias on every
        # banked column (fmin in the dtype, not -inf, so bf16 arithmetic
        # doesn't churn out NaN).
        neg_inf = torch.finfo(dtype).min
        return torch.full((num_steps,), neg_inf, device=device, dtype=dtype)

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, s: float) -> None:
        self._strength = float(s)
        log_s = math.log(max(self._strength, 1e-12))
        log_s_t = torch.tensor(log_s, device=self.device, dtype=self.dtype)
        neg_inf_t = torch.tensor(
            torch.finfo(self.dtype).min, device=self.device, dtype=self.dtype,
        )
        self.bank_bias = torch.where(self._valid_mask, log_s_t, neg_inf_t)

    def ensure_alloc(self, kv_heads: int, T_lat: int, head_dim: int) -> None:
        N = len(self.banked_layers)
        target_5d = (N, self.num_steps, kv_heads, T_lat, head_dim)
        if self._bank_k_5d is not None:
            if tuple(self._bank_k_5d.shape) != target_5d:
                raise RuntimeError(
                    f"BankBuffer was allocated as {tuple(self._bank_k_5d.shape)} "
                    f"but a forward expects {target_5d}. Call reset() / "
                    f"reallocate."
                )
            return
        self._bank_k_5d = torch.zeros(
            *target_5d, device=self.device, dtype=self.dtype,
        )
        self._bank_v_5d = torch.zeros(
            *target_5d, device=self.device, dtype=self.dtype,
        )
        # Engine binds the rank-5 tensor directly.
        self.bank_k = self._bank_k_5d
        self.bank_v = self._bank_v_5d
        logger.info("BankBuffer allocated: %s (%s)", target_5d, self.dtype)

    def reset(self) -> None:
        if self._bank_k_5d is not None:
            self._bank_k_5d.zero_()
            self._bank_v_5d.zero_()
        self._valid_mask.zero_()
        self.bank_bias = self._make_bias(
            self.num_steps, self._strength, self.device, self.dtype,
        )

    def num_entries(self) -> int:
        return int(self._valid_mask.sum().item()) * len(self.banked_layers)

    def step_indices_tensor(
        self, step_indices: Sequence[int],
    ) -> torch.Tensor:
        return torch.tensor(
            list(step_indices), dtype=torch.long, device=self.device,
        )

    def ingest_outputs(
        self,
        k_stack: torch.Tensor,
        v_stack: torch.Tensor,
        step_indices: torch.Tensor,
    ) -> None:
        """Scatter engine outputs into the bank ring at step_indices.

        ``k_stack`` / ``v_stack``: rank-5 ``[N_banked, B, kv_heads,
        T_lat, head_dim]``. ``index_copy_`` on dim 1 scatters across
        all layers in one call.
        """
        if self._bank_k_5d is None:
            raise RuntimeError(
                "BankBuffer.ingest_outputs called before ensure_alloc"
            )
        self._bank_k_5d.index_copy_(1, step_indices, k_stack)
        self._bank_v_5d.index_copy_(1, step_indices, v_stack)
        self._valid_mask[step_indices] = True
        log_s = math.log(max(self._strength, 1e-12))
        self.bank_bias[step_indices] = torch.tensor(
            log_s, device=self.device, dtype=self.dtype,
        )
