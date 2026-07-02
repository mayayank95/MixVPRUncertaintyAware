"""Kappa-adaptive Multi-Similarity (Margin-MS) loss with vMF log-partition regularizer."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from losses.ms_constants import KAPPA_EPS, MS_NEG_WEIGHT, MS_POS_WEIGHT
from losses.vmf_loss import VMFLikelihood


class KappaMSLoss(torch.nn.Module):
    """In-batch MultiSimilarityLoss with per-anchor base = tanh(1/kappa_i), plus R(kappa).

    MS terms match pytorch_metric_learning MultiSimilarityLoss (_compute_loss):
      pos_exp = base - mat,  neg_exp = mat - base  (base was scalar 0 in MixVPR).
    """

    def __init__(self, d: int = 128):
        super().__init__()
        self.alpha = float(MS_POS_WEIGHT)
        self.beta = float(MS_NEG_WEIGHT)
        self.kappa_eps = float(KAPPA_EPS)
        self._vmf = VMFLikelihood(d=int(d), eps=self.kappa_eps)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        kappa: torch.Tensor,
        indices_tuple=None,
        reg_weight: float = 1.0,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = embeddings.new_zeros(())
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels)
        if all(len(x) <= 1 for x in indices_tuple):
            if return_parts:
                return zero, zero, zero
            return zero

        mat = embeddings @ embeddings.T
        a1, p, a2, n = indices_tuple
        pos_mask = torch.zeros_like(mat, dtype=torch.bool)
        neg_mask = torch.zeros_like(mat, dtype=torch.bool)
        if len(a1) > 0:
            pos_mask[a1, p] = True
        if len(a2) > 0:
            neg_mask[a2, n] = True

        kappa_i = kappa.reshape(-1, 1).to(dtype=mat.dtype, device=mat.device)
        inv_kappa = 1.0 / (kappa_i + self.kappa_eps)
        base = torch.tanh(inv_kappa)
        pos_exp = base - mat
        neg_exp = mat - base

        pos_loss = (1.0 / self.alpha) * lmu.logsumexp(
            self.alpha * pos_exp,
            keep_mask=pos_mask,
            add_one=True,
        )
        neg_loss = (1.0 / self.beta) * lmu.logsumexp(
            self.beta * neg_exp,
            keep_mask=neg_mask,
            add_one=True,
        )

        r_kappa = self._vmf.log_partition_function(kappa_i)
        ms_per_anchor = pos_loss + neg_loss
        reg_per_anchor = float(reg_weight) * r_kappa
        per_anchor = ms_per_anchor + reg_per_anchor

        active = torch.any(pos_mask, dim=1, keepdim=True) | torch.any(
            neg_mask, dim=1, keepdim=True
        )
        if not torch.any(active):
            if return_parts:
                return zero, zero, zero
            return zero

        active_flat = active.squeeze(-1)
        total = per_anchor.masked_select(active_flat).mean()
        if return_parts:
            ms = ms_per_anchor.masked_select(active_flat).mean()
            reg = reg_per_anchor.masked_select(active_flat).mean()
            return total, ms, reg
        return total
