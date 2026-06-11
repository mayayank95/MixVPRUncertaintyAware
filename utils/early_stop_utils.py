"""Early-stop metric parsing and reading values from eval_dataset W&B metric dicts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_ECE_RECALL_ALIAS = {"ece_recall": "ece_recall_01"}

_ALLOWED = frozenset({
    "recall",
    "ece_recall_01",
    "ece_recall_05",
    "ece_recall_10",
    "ece_ap",
})


def canonical_early_stop_metrics(raw: Any) -> List[str]:
    """Parse CLI/config value into ordered unique canonical metric ids."""
    if raw is None:
        return ["recall"]
    if isinstance(raw, list):
        tokens = [str(x).strip().lower().replace("-", "_") for x in raw if str(x).strip()]
    else:
        tokens = [s.strip().lower().replace("-", "_") for s in str(raw).split(",") if s.strip()]
    if not tokens:
        return ["recall"]
    out: List[str] = []
    for t in tokens:
        t = _ECE_RECALL_ALIAS.get(t, t)
        if t not in _ALLOWED:
            raise ValueError(
                f"Invalid early_stop_metric token {t!r}. "
                f"Allowed: {sorted(_ALLOWED)} (use ece_recall for R@1 ECE)."
            )
        if t not in out:
            out.append(t)
    return out


def recall_values_needed_for_metrics(metrics: List[str]) -> List[int]:
    """K values that must appear in recall_values for ECE @K early stops."""
    need: List[int] = []
    for m in metrics:
        if m.startswith("ece_recall_"):
            need.append(int(m.rsplit("_", 1)[-1]))
    return need


def best_model_filename(metric: str, metrics: List[str]) -> str:
    """Filename for a metric-specific best checkpoint (weights only)."""
    if len(metrics) == 1 and metrics[0] == "recall":
        return "best_model.pth"
    return f"best_model_{metric}.pth"


def get_metric_value(
    metric: str,
    *,
    recalls: Any,
    recall_values: List[int],
    eval_wandb_metrics: Optional[Dict[str, Any]] = None,
    panel_data: Optional[Dict[str, Any]] = None,
    dataset_name: str,
) -> Optional[float]:
    """Scalar for this epoch, or None if unavailable."""
    if metric == "recall":
        if recalls is None or len(recalls) == 0:
            return None
        return float(recalls[0])
    if metric.startswith("ece_recall_"):
        k = int(metric.rsplit("_", 1)[-1])
        if k not in recall_values:
            return None
        if panel_data:
            ece = panel_data.get("ece_recall", {}).get("kappa", {})
            if k in ece:
                return float(ece[k])
        if eval_wandb_metrics:
            flat = f"ece/recall_{k:02d}"
            if flat in eval_wandb_metrics:
                return float(eval_wandb_metrics[flat])
            pref = f"Eval_{dataset_name}/ece_kappa_recall_{k:02d}"
            if pref in eval_wandb_metrics:
                return float(eval_wandb_metrics[pref])
            legacy = f"Eval_{dataset_name}/ece_recall_{k:02d}"
            if legacy in eval_wandb_metrics:
                return float(eval_wandb_metrics[legacy])
        return None
    return None


def initial_best_for_metric(metric: str) -> float:
    if metric == "recall":
        return 0.0
    return float("inf")


def is_improvement(metric: str, current: float, best_so_far: float) -> bool:
    if metric == "recall":
        return current > best_so_far
    return current < best_so_far


def _first_val_set_name(cfg: dict) -> str:
    val_sets = cfg.get("mixvpr_val_sets") or []
    if isinstance(val_sets, str):
        val_sets = [s.strip() for s in val_sets.split(",") if s.strip()]
    else:
        val_sets = list(val_sets)
    return val_sets[0] if val_sets else "val"


def lightning_ckpt_metric_name(early_stop_metric: str, val_set_name: str) -> str:
    """Map canonical early-stop id to the scalar logged by ``panel_to_train_val_metrics``."""
    if early_stop_metric == "recall":
        return f"{val_set_name}/R1"
    if early_stop_metric.startswith("ece_recall_"):
        k = int(early_stop_metric.rsplit("_", 1)[-1])
        return f"{val_set_name}/ece_kappa_recall_{k:02d}"
    if early_stop_metric == "ece_ap":
        return f"{val_set_name}/ece_kappa_ap"
    raise ValueError(f"Unsupported early_stop metric for Lightning checkpoint: {early_stop_metric!r}")


def lightning_ckpt_filename_tag(early_stop_metric: str) -> str:
    if early_stop_metric == "recall":
        return "R1"
    if early_stop_metric.startswith("ece_recall_"):
        k = int(early_stop_metric.rsplit("_", 1)[-1])
        return f"ece_r{k:02d}"
    if early_stop_metric == "ece_ap":
        return "ece_ap"
    return early_stop_metric.replace("/", "_")


def resolve_lightning_ckpt_monitor(cfg: dict) -> tuple[str, str, str]:
    """Return ``(monitor, mode, filename_tag)`` for ``ModelCheckpoint``.

    Uses the first ``ece_recall_*`` in ``early_stop_metrics``, else ``recall``,
    else falls back to ``mixvpr_ckpt_monitor`` (legacy).
    """
    val_set = _first_val_set_name(cfg)
    early_stop_metrics = list(cfg.get("early_stop_metrics") or ["recall"])

    for metric in early_stop_metrics:
        if metric.startswith("ece_recall_"):
            return (
                lightning_ckpt_metric_name(metric, val_set),
                "min",
                lightning_ckpt_filename_tag(metric),
            )

    if "recall" in early_stop_metrics:
        return lightning_ckpt_metric_name("recall", val_set), "max", "R1"

    legacy = str(cfg.get("mixvpr_ckpt_monitor", f"{val_set}/R1"))
    if "/" in legacy:
        prefix, suffix = legacy.split("/", 1)
        if val_set and prefix not in (val_set,):
            legacy = f"{val_set}/{suffix}"
    mode = "max" if legacy.endswith("/R1") else "min"
    tag = legacy.split("/")[-1].replace("ece_kappa_recall_", "ece_r")
    return legacy, mode, tag


def resolve_recall_early_stop_ckpt_spec(cfg: dict) -> tuple[str, str, str, str]:
    """``(canonical_id, monitor, mode, filename_tag)`` for recall early-stop / checkpoint."""
    val_set = _first_val_set_name(cfg)
    return (
        "recall",
        lightning_ckpt_metric_name("recall", val_set),
        "max",
        lightning_ckpt_filename_tag("recall"),
    )


def resolve_ece_early_stop_ckpt_specs(cfg: dict) -> List[tuple[str, str, str, str]]:
    """``(canonical_id, monitor, mode, filename_tag)`` for each ECE early-stop metric."""
    val_set = _first_val_set_name(cfg)
    specs: List[tuple[str, str, str, str]] = []
    for metric in cfg.get("early_stop_metrics") or []:
        if metric.startswith("ece_recall_") or metric == "ece_ap":
            specs.append((
                metric,
                lightning_ckpt_metric_name(metric, val_set),
                "min",
                lightning_ckpt_filename_tag(metric),
            ))
    return specs
