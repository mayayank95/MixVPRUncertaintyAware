# Model from "MixVPR: Feature Mixing for Visual Place Recognition" - https://arxiv.org/abs/2303.02190
# Parts of this code are from https://github.com/amaralibey/MixVPR

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Architecture settings per output descriptor dimension.
MIXVPR_ARCH = {
    128: {"out_channels": 64, "out_rows": 2},
    512: {"out_channels": 256, "out_rows": 2},
    4096: {"out_channels": 1024, "out_rows": 4},
}


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=512,
        mix_depth=1,
        mlp_ratio=1,
        out_rows=4,
    ) -> None:
        super().__init__()

        self.in_h = in_h  # height of input feature maps
        self.in_w = in_w  # width of input feature maps
        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio  # ratio of the mid projection layer in the mixer block

        hw = in_h * in_w
        self.mix = nn.Sequential(*[FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio) for _ in range(self.mix_depth)])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        return x.flatten(1)


class ResNet(nn.Module):
    """ResNet50 through layer3; layer4 removed (1024-d maps), as in Lightning/GSV MixVPR training."""

    def __init__(self, pretrained=True, layers_to_freeze=2):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.model = torchvision.models.resnet50(weights=weights)

        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        self.model.avgpool = None
        self.model.fc = None
        self.model.layer4 = None

        out_channels = 2048
        self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        self.out_channels = self.out_channels // 2 if self.model.layer3 is None else out_channels

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x


class MixVPRModel(torch.nn.Module):
    def __init__(self, agg_config={}):
        super().__init__()
        self.backbone = ResNet()
        self.aggregator = MixVPR(**agg_config)

    def forward(self, x):
        x = transforms.Resize([320, 320], antialias=True)(x)
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

    def freeze_base(self):
        """Freeze backbone and aggregator. Returns remaining trainable params."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.aggregator.parameters():
            p.requires_grad = False
        return [p for p in self.parameters() if p.requires_grad]


def mixvpr_agg_config(descriptors_dimension):
    if descriptors_dimension not in MIXVPR_ARCH:
        raise ValueError(
            f"Unsupported MixVPR descriptors_dimension={descriptors_dimension}. "
            f"Expected one of {sorted(MIXVPR_ARCH)}."
        )
    arch = MIXVPR_ARCH[descriptors_dimension]
    return {
        "in_channels": 1024,
        "in_h": 20,
        "in_w": 20,
        "out_channels": arch["out_channels"],
        "mix_depth": 4,
        "mlp_ratio": 1,
        "out_rows": arch["out_rows"],
    }
