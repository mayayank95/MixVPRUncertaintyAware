"""Batch place-centroid targets + vMF NLL (KappaPlace-style supervision)."""
import torch
import torch.nn.functional as F

from losses.vmf_loss import VMFLikelihood


def place_centroid_targets(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """z: [N, D] (already L2-normalized per image). labels: [N] place ids."""
    labels = labels.view(-1).long()
    uniq, inv = torch.unique(labels, return_inverse=True)
    c, _d = uniq.numel(), z.shape[1]
    sums = torch.zeros(c, _d, device=z.device, dtype=z.dtype)
    sums.index_add_(0, inv, z)
    gathered_sums = sums[inv]
    pure_positive_sums = gathered_sums - z
    return F.normalize(pure_positive_sums, dim=-1)
