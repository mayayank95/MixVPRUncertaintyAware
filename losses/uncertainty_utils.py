"""Shared uncertainty loss for train and validation (same criterion and scaling)."""
import math

import torch

from losses.gaussian_cosine_loss import GaussianCosineLoss
from losses.vmf_loss import VMFLikelihood


def compute_uncertainty_loss(
    input_norm: torch.Tensor,
    target_norm: torch.Tensor,
    variance: torch.Tensor,
    loss_type: str = "gaussian_nll",
    lambda_: float = 1.0,
    gnll_mu_scale_mode: str = "sqrt_dim",
    gnll_mu_scale_value: float = 1.0,
) -> torch.Tensor:
    """
    Uncertainty loss used by both training and validation.

    Args:
        input_norm: L2-normalized embeddings [N, D].
        target_norm: L2-normalized target vectors [N, D].
        variance: Predicted variance [N, D].
        loss_type: 'gaussian_nll', 'gaussian_cosine', or 'vmf'.
        lambda_: Scale factor applied to the loss.
        gnll_mu_scale_mode: GNLL scaling mode for mean vectors: sqrt_dim, none, custom.
        gnll_mu_scale_value: Custom GNLL scale when mode=custom.

    Returns:
        Tensor scalar: lambda_ * mean loss (use .backward() in train, .item() in val).
    """
    loss_type = loss_type.lower()
    if loss_type == "gaussian_cosine":
        criterion = GaussianCosineLoss()
        loss = criterion(input_norm, target_norm, variance)
    elif loss_type == "vmf":
        kappa = torch.mean(variance, dim=-1, keepdim=True)  # [B, 1]
        criterion = VMFLikelihood()
        loss = criterion(input_norm, kappa, target_norm)
    elif loss_type == "gaussian_nll":
        criterion = torch.nn.GaussianNLLLoss()
        mode = str(gnll_mu_scale_mode).lower()
        if mode == "none":
            scale = 1.0
        elif mode == "custom":
            scale = float(gnll_mu_scale_value)
        else:
            scale = math.sqrt(input_norm.shape[1])
        loss = criterion(input_norm * scale, target_norm * scale, variance)
    else:
        raise ValueError(
            f"Unknown uncertainty_loss '{loss_type}'. "
            "Expected one of: gaussian_nll, gaussian_cosine, vmf."
        )
    return lambda_ * loss

