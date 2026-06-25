"""Load VPR encoders for place-latent-space analysis."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import torch

_ROOT = Path(__file__).resolve().parents[2]
_UA_ROOT = Path("/home/dsi/mayayan/projects/UncertaintyAwareVPR")

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_ua_basic_class() -> Tuple[Type[torch.nn.Module], Dict[str, Any]]:
    """Load UncertaintyAwareVPR CosPlace Basic without namespace-package clashes."""
    ua = str(_UA_ROOT)
    if ua not in sys.path:
        sys.path.insert(0, ua)
    elif sys.path[0] != ua:
        sys.path.remove(ua)
        sys.path.insert(0, ua)

    to_restore: Dict[str, Any] = {}
    for key in list(sys.modules):
        if key == "models" or key.startswith("models."):
            to_restore[key] = sys.modules.pop(key)

    ua_models = types.ModuleType("models")
    ua_models.__path__ = [str(_UA_ROOT / "models")]
    sys.modules["models"] = ua_models

    ua_basic = importlib.import_module("models.model_mode").Basic
    return ua_basic, to_restore


def build_model_cfg(spec: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Return a config dict understood by the appropriate loader."""
    loader = spec["loader"]
    cfg: Dict[str, Any] = {
        "device": str(device),
        "model_mode": "basic",
        "descriptors_dimension": spec["dim"],
        "image_size": spec.get("image_size", 512),
        "resume_model": spec.get("ckpt"),
        "ckpt_state_dict_key": spec.get("ckpt_state_dict_key", "model_state_dict"),
    }
    if loader == "mixvpr":
        cfg["method"] = "mixvpr"
    elif loader == "cosplace_hub":
        cfg["method"] = "cosplace_pretrained"
        cfg["backbone"] = spec.get("backbone", "ResNet50")
        cfg["resume_model"] = None
    elif loader == "cosplace_local":
        cfg["backbone"] = spec.get("backbone", "ResNet50")
    else:
        raise ValueError(f"Unknown loader: {loader}")
    return cfg


def _load_cosplace_local(spec: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    ua_basic, to_restore = _load_ua_basic_class()
    try:
        opt = {
            "backbone": spec.get("backbone", "ResNet50"),
            "descriptors_dimension": spec["dim"],
        }
        model = ua_basic(opt)
        ckpt_path = spec["ckpt"]
        key = spec.get("ckpt_state_dict_key", "model_state_dict")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get(key, checkpoint)
        model.load_state_dict(state_dict, strict=False)
        return model.to(device).eval()
    finally:
        for key, module in to_restore.items():
            sys.modules[key] = module


def load_model(spec: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Instantiate and load weights for one model spec."""
    loader = spec["loader"]
    if loader == "cosplace_local":
        return _load_cosplace_local(spec, device)

    cfg = build_model_cfg(spec, device)
    from utils.runtime import init_model

    _device, model = init_model(cfg)
    return model.to(device).eval()
