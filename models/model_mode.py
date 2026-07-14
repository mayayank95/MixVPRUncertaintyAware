import copy
import math

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.layers import Flatten, GeM, L2Norm
from models.mixvpr import MixVPRModel, mixvpr_agg_config

MIXVPR_BACKBONE_CHANNELS = 1024


class GeneralModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            return output
        return output, torch.zeros_like(output)


def _stable_var_init(module, activation="softplus", target_variance=0.1):
    """Initialize variance head bias so activation(0) ≈ target_variance."""
    if activation == "softplus":
        bias_val = math.log(math.exp(target_variance) - 1)
    else:  # sigmoid
        bias_val = -math.log(1.0 / target_variance - 1)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_val)


def _make_var_head_cls(tag: str, linear_kind: str, act_name: str):
    """Subclass with a descriptive __name__ for Lightning model summaries."""
    type_name = f"VarHead({tag}, lin={linear_kind}, {act_name})"

    class VarHead(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.seq = nn.Sequential(*layers)

        def forward(self, x):
            return self.seq(x)

    VarHead.__name__ = type_name
    VarHead.__qualname__ = type_name
    return VarHead


def _build_gem_aggregation(in_channels: int, fc_output_dim: int) -> nn.Sequential:
    """CosPlace-style GeM aggregation (same block as KappaPlace/CosPlace GeoLocalizationNet)."""
    return nn.Sequential(
        L2Norm(),
        GeM(),
        Flatten(),
        nn.Linear(in_channels, fc_output_dim),
    )


def _resolve_var_head_type(opt) -> str:
    """Map CLI/config flags to a variance head type."""
    explicit = str(opt.get("var_head_type", "")).lower()
    if explicit in ("descriptor", "agg", "gem"):
        return explicit
    if opt.get("var_head_agg"):
        return "agg"
    return "descriptor"


def _build_var_head(opt, fc_output_dim, aggregation=None):
    """Build the variance head from orthogonal flags (type × linear × activation)."""
    head_type = _resolve_var_head_type(opt)
    linear_kind = str(opt.get("var_head_linear", "d")).lower()
    act_name = opt.get("variance_activation", "softplus") or "softplus"
    activation = nn.Sigmoid() if act_name == "sigmoid" else nn.Softplus()

    layers = []
    if head_type == "gem":
        layers.append(_build_gem_aggregation(MIXVPR_BACKBONE_CHANNELS, fc_output_dim))
        tag = "gem_agg"
        from_feature_map = True
    elif head_type == "agg":
        assert aggregation is not None, "var_head_agg=True requires the aggregation module"
        layers.append(copy.deepcopy(aggregation))
        tag = "agg"
        from_feature_map = True
    else:
        tag = "noagg"
        from_feature_map = False

    if linear_kind == "d":
        layers.append(nn.Linear(fc_output_dim, fc_output_dim))
    elif linear_kind == "1":
        layers.append(nn.Linear(fc_output_dim, 1))
    elif linear_kind != "none":
        raise ValueError(
            f"Unknown var_head_linear: {linear_kind!r}. Expected 'none', 'd', or '1'."
        )

    layers.append(activation)
    head = _make_var_head_cls(tag, linear_kind, act_name)(layers)

    if opt.get("var_init"):
        _stable_var_init(head, act_name)

    return head, from_feature_map


class MixVPREncoder(MixVPRModel):
    """MixVPR method encoder: backbone + aggregator (flatten in MixVPR.forward)."""

    def __init__(self, opt):
        super().__init__(agg_config=mixvpr_agg_config(opt["descriptors_dimension"]))

    @property
    def aggregation_module(self):
        return self.aggregator


_METHOD_ENCODERS = {
    "mixvpr": MixVPREncoder,
}


def build_method_encoder(opt):
    method = opt.get("method", "cosplace")
    encoder_cls = _METHOD_ENCODERS.get(method)
    if encoder_cls is None:
        raise ValueError(f"Unknown method: {method}")
    return encoder_cls(opt)


def _attach_encoder(model, encoder):
    """Register encoder weights at top level so checkpoints stay flat."""
    for name, module in encoder.named_children():
        setattr(model, name, module)
    model._uses_mixvpr = isinstance(encoder, MixVPREncoder)
    model._aggregation_module = encoder.aggregation_module
    model._freeze_base = encoder.freeze_base
    return encoder


def _encode_inputs(model, inputs):
    if model._uses_mixvpr:
        x = transforms.Resize([320, 320], antialias=True)(inputs)
        feat = model.backbone(x)
        return feat, model.aggregator(feat)
    feat = model.backbone(inputs)
    return feat, model.aggregation(feat)


class Basic(nn.Module):
    """Method-agnostic descriptor model with zero variance."""

    def __init__(self, opt):
        super().__init__()
        _attach_encoder(self, build_method_encoder(opt))
        self.id = "basic"
        self.final_l2 = L2Norm()

    def freeze_base(self):
        return self._freeze_base()

    def forward(self, inputs):
        _, desc = _encode_inputs(self, inputs)
        mu = self.final_l2(desc)
        return mu, torch.zeros_like(mu)


class Uncertainty(nn.Module):
    """Method-agnostic descriptor model with learned variance head."""

    def __init__(self, opt):
        super().__init__()
        _attach_encoder(self, build_method_encoder(opt))
        self.id = "uncertainty"
        self.final_l2 = L2Norm()
        fc_output_dim = opt.get("descriptors_dimension", 512)
        self.var_head, self._var_from_feature_map = _build_var_head(
            opt, fc_output_dim, aggregation=self._aggregation_module
        )

    def freeze_base(self):
        return self._freeze_base()

    def forward(self, inputs):
        feat, desc = _encode_inputs(self, inputs)
        mu = self.final_l2(desc)
        if self._var_from_feature_map:
            variance = self.var_head(feat)
        else:
            variance = self.var_head(desc)
        return mu, variance + 1e-6


def build_model_mode(opt):
    mode = opt["model_mode"]
    if mode == "basic":
        return Basic(opt)
    if mode == "uncertainty":
        return Uncertainty(opt)
    raise ValueError(f"Unknown model_mode: {mode}")
