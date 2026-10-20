"""SALAD encoder: DINOv2 ViT-B/14 + Sinkhorn aggregation (Serra & Trulls, CVPR 2024)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.dinov2 import DINOv2

# Released DINOv2-B SALAD at 322px (patch 14). Descriptor is m*l + g.
# dino_salad:        64×128 + 256 = 8448
# dino_salad_512_32: 16×32  +  32 = 544  (paper writes "512+32"; ckpt uses m=16 not 15)
_IMAGE_SIZE = 322
_PATCH = 14
_SALAD_ARCH = {
    8448: {"num_clusters": 64, "cluster_dim": 128, "token_dim": 256},
    544: {"num_clusters": 16, "cluster_dim": 32, "token_dim": 32},
}


def salad_arch(descriptors_dimension: int) -> dict:
    """Aggregator sizes for a released dino_salad descriptor width."""
    if descriptors_dimension not in _SALAD_ARCH:
        allowed = ", ".join(str(k) for k in sorted(_SALAD_ARCH))
        hint = ""
        if descriptors_dimension == 512:
            hint = " Compact dino_salad_512_32 is 544-d (16×32 + 32), not 512."
        raise ValueError(
            f"salad descriptors_dimension must be one of {{{allowed}}}, "
            f"got {descriptors_dimension}.{hint}"
        )
    return dict(_SALAD_ARCH[descriptors_dimension])


def log_otp_solver(log_a, log_b, cost, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Code adapted from OpenGlue, MIT license:
    https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        cost : torch.Tensor
            metric cost matrix
        num_iters : int, default=20
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    cost = cost / reg
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(cost + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(cost + u.unsqueeze(2), dim=1).squeeze()
    return cost + u.unsqueeze(2) + v.unsqueeze(1)


def get_matching_probs(scores, dustbin_score=1.0, num_iters=3, reg=1.0):
    """Sinkhorn matching probabilities. Adapted from OpenGlue, MIT license:
    https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
    """
    batch_size, n_clusters, n_features = scores.size()
    scores_aug = torch.empty(
        batch_size, n_clusters + 1, n_features, dtype=scores.dtype, device=scores.device
    )
    scores_aug[:, :n_clusters, :n_features] = scores
    scores_aug[:, n_clusters, :] = dustbin_score
    norm = -torch.tensor(math.log(n_features + n_clusters), device=scores.device)
    log_a, log_b = norm.expand(n_clusters + 1).contiguous(), norm.expand(n_features).contiguous()
    log_a[-1] = log_a[-1] + math.log(n_features - n_clusters)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_p = log_otp_solver(log_a, log_b, scores_aug, num_iters=num_iters, reg=reg)
    return log_p - norm


class SALAD(nn.Module):
    """
    Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD).

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        num_channels=1536,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim),
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        x (tuple): A tuple containing two elements, features and token.
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            (torch.Tensor): The global descriptor [B, m*l + g], before the final L2.
            The retrieval model applies that L2, same as MixVPR's final_l2.
        """
        features, token = x
        cluster = self.cluster_features(features).flatten(2)
        scores = self.score(features).flatten(2)
        token = self.token_features(token)
        scores = torch.exp(get_matching_probs(scores, self.dust_bin, 3))
        scores = scores[:, :-1, :]
        scores = scores.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        cluster = cluster.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        return torch.cat(
            [
                nn.functional.normalize(token, p=2, dim=-1),
                nn.functional.normalize((cluster * scores).sum(dim=-1), p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )


class SaladEncoder(nn.Module):
    """DINOv2 ViT-B/14 + SALAD, matching a released dino_salad checkpoint."""

    def __init__(self, opt):
        super().__init__()
        desc_dim = int(opt["descriptors_dimension"])
        arch = salad_arch(desc_dim)
        raw_size = opt.get("image_size")
        if raw_size is not None:
            image_size = int(raw_size[0] if isinstance(raw_size, (list, tuple)) else raw_size)
            if image_size % _PATCH != 0:
                raise ValueError(
                    f"salad image_size must be divisible by DINOv2 patch {_PATCH}, got {image_size}"
                )
            if image_size != _IMAGE_SIZE:
                raise ValueError(
                    f"salad image_size must be {_IMAGE_SIZE} (released dino_salad), got {image_size}"
                )
        self.backbone = DINOv2(
            model_name="dinov2_vitb14",
            num_trainable_blocks=4,
            norm_layer=True,
            return_token=True,
        )
        self.aggregator = SALAD(
            num_channels=768,
            num_clusters=arch["num_clusters"],
            cluster_dim=arch["cluster_dim"],
            token_dim=arch["token_dim"],
        )

    @property
    def aggregation_module(self):
        return self.aggregator

    def freeze_base(self):
        """Freeze backbone and aggregator. Returns remaining trainable params."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.aggregator.parameters():
            parameter.requires_grad = False
        return [parameter for parameter in self.parameters() if parameter.requires_grad]
