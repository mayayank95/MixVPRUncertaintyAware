# DINOv2 backbone wrapper used by SALAD (Serra & Trulls, CVPR 2024).
# Architecture is constructed via torch.hub facebookresearch/dinov2 (local cache if present).
import torch
import torch.nn as nn

DINOV2_ARCHS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """

    def __init__(
        self,
        model_name="dinov2_vitb14",
        num_trainable_blocks=2,
        norm_layer=False,
        return_token=False,
    ):
        super().__init__()
        if model_name not in DINOV2_ARCHS:
            raise ValueError(f"Unknown DINOv2 model name: {model_name}")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            features (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            token (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """
        batch_size, _channels, height, width = x.shape
        x = self.model.prepare_tokens_with_masks(x)
        with torch.no_grad():
            for blk in self.model.blocks[: -self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()
        for blk in self.model.blocks[-self.num_trainable_blocks :]:
            x = blk(x)
        if self.norm_layer:
            x = self.model.norm(x)
        token = x[:, 0]
        features = x[:, 1:]
        features = features.reshape(
            (batch_size, height // 14, width // 14, self.num_channels)
        ).permute(0, 3, 1, 2)
        if self.return_token:
            return features, token
        return features
