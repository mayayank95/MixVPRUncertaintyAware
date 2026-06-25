#!/usr/bin/env python3
"""Build a supervisor-facing markdown report from GSV latent-space summaries."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

HEADLINE_ROWS = [
    ("mean_dist", "Mean intra-place distance", "Lower = tighter cluster around place centroid"),
    ("var_dist", "Variance of intra-place distances", "Lower = more consistent spread around centroid"),
    ("loo_mean_dist", "Mean leave-one-out distance", "Lower = images still close when one view is held out"),
    ("loo_var_dist", "Variance of leave-one-out distances", "Lower = more consistent LOO spread"),
    ("pairwise_mean_dist", "Mean pairwise image distance", "Lower = views within a place are more similar"),
    ("pairwise_var_dist", "Variance of pairwise distances", "Lower = more consistent pairwise spread"),
    ("nearest_other_centroid_dist", "Mean nearest-other-place centroid distance", "Higher = place centroids are farther from neighbors"),
    ("separation_ratio", "Separation ratio", "nearest_other / mean_dist; higher = better inter/intra separation"),
]


METRIC_GLOSSARY = """
## Metric definitions

All distances are **cosine distance** on **L2-normalized** global descriptors:
`distance = 1 - cosine_similarity`.

For each place (at least `min_img_per_place` images in GSV-Cities):

1. **Place centroid** — L2-normalized mean of all image descriptors in that place.
2. **mean_dist** — mean cosine distance from each image to the full-place centroid.
3. **var_dist** — variance of those image-to-centroid distances within the place.
4. **loo_mean_dist** — for each image, distance to the centroid built from all *other* images in the same place; then averaged over images in the place.
5. **loo_var_dist** — variance of the leave-one-out distances.
6. **pairwise_mean_dist** — mean cosine distance over all image pairs within the place (each pair counted once).
7. **pairwise_var_dist** — variance of those pairwise distances.
8. **nearest_other_centroid_dist** — cosine distance from this place's centroid to the closest centroid of any *other* place.
9. **separation_ratio** — `nearest_other_centroid_dist / mean_dist` (higher means the place is more separated from neighbors relative to its internal spread).
"""


def _load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _format_table(summary: Dict[str, Any]) -> List[str]:
    models = summary["models"]
    lines = [
        "| Metric | " + " | ".join(m["label"] for m in models) + " |",
        "| --- | " + " | ".join("---" for _ in models) + " |",
    ]
    for metric_key, title, _note in HEADLINE_ROWS:
        row = [title]
        for model in models:
            slug = model["slug"]
            val = model.get(f"{slug}_{metric_key}_over_places")
            row.append(f"{val:.4f}" if val is not None else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _models_section(summary: Dict[str, Any]) -> List[str]:
    lines = ["## Models", ""]
    models = summary["models"]
    dims = {m["dim"] for m in models}
    backbones = {m["backbone"] for m in models if m.get("backbone")}

    if len(backbones) == 1:
        lines.append(f"- **Backbone:** {next(iter(backbones))} (all models)")
    if len(dims) == 1:
        lines.append(f"- **Descriptor dimension:** {next(iter(dims))} (all models)")
    if len(backbones) == 1 or len(dims) == 1:
        lines.append("")

    for model in models:
        bits = [f"**{model['label']}** (`{model['slug']}`)"]
        if len(backbones) != 1 and model.get("backbone"):
            bits.append(f"- Backbone: {model['backbone']}")
        if len(dims) != 1:
            bits.append(f"- Descriptor dimension: {model['dim']}")
        if model.get("image_size"):
            bits.append(f"- Input resolution: {model['image_size']}×{model['image_size']}")
        if model.get("ckpt"):
            bits.append(f"- Checkpoint: `{model['ckpt']}`")
        elif model.get("loader") == "cosplace_hub":
            bits.append("- Weights: `torch.hub` (`gmberton/cosplace`)")
        lines.extend(bits)
        lines.append("")
    return lines


def _model_metric(summary: Dict[str, Any], slug: str, metric: str) -> float:
    for model in summary["models"]:
        if model["slug"] == slug:
            return float(model[f"{slug}_{metric}_over_places"])
    raise KeyError(f"{slug} not in summary {summary['tag']}")


def _conclusion_section(summaries: List[Dict[str, Any]]) -> List[str]:
    bits = []
    for summary in summaries:
        dim = summary["models"][0]["dim"]
        cos_intra = _model_metric(summary, "cosplace_hub", "mean_dist")
        mix_intra = _model_metric(summary, "mixvpr_official", "mean_dist")
        mix_inter = _model_metric(summary, "mixvpr_official", "nearest_other_centroid_dist")
        cos_inter = _model_metric(summary, "cosplace_hub", "nearest_other_centroid_dist")
        bits.append(
            f"at {dim}-d CosPlace mean intra-place distance is {cos_intra:.4f} vs MixVPR {mix_intra:.4f}, "
            f"while MixVPR nearest-other centroid distance is {mix_inter:.4f} vs CosPlace {cos_inter:.4f}"
        )
    detail = "; ".join(bits)
    return [
        "## Conclusion",
        "",
        "**Intra-place: CosPlace; inter-place: MixVPR.** "
        f"On GSV-Cities, CosPlace official checkpoints form tighter clusters within each place, "
        f"whereas MixVPR places centroids farther from their nearest neighbor place "
        f"({detail}).",
        "",
    ]


def _notes_section(summaries: List[Dict[str, Any]]) -> List[str]:
    lines = ["## Notes", ""]
    min_img = summaries[0]["min_img_per_place"]
    n_places = summaries[0]["n_places"]
    lines.append(
        f"- **Place filtering:** the raw GSV list contains 64,394 places; "
        f"{64_394 - n_places:,} places with fewer than {min_img} images are excluded "
        f"(pairwise and leave-one-out metrics need at least {min_img} views per place)."
    )
    lines.append(
        "- **MixVPR** uses contrastive training (official checkpoints from the MixVPR authors). "
        "Input size is 320×320 for MixVPR and 512×512 for CosPlace, following each method's standard setup."
    )
    lines.append(
        "- **CosPlace official** weights are loaded from `torch.hub` (`gmberton/cosplace`)."
    )
    if any(
        any(m.get("loader") == "cosplace_local" for m in s["models"])
        for s in summaries
    ):
        lines.append(
            "- **CosPlace local (KappaPlace-PT)** models are basic (classification) CosPlace models trained on SF-XL in our "
            "UncertaintyAwareVPR codebase; checkpoints are `best_model.pth` from the listed run folders."
        )
    lines.append("")
    return lines


def build_report(summaries: List[Dict[str, Any]], title: str) -> str:
    today = date.today().isoformat()
    parts = [
        f"# {title}",
        "",
        f"*Generated {today}*",
        "",
    ]
    parts.extend(_conclusion_section(summaries))
    parts.extend([
        "## Dataset and setup",
        "",
        "- **Dataset:** GSV-Cities (precomputed place list in `cache/gsv_preweights/places_all.csv`).",
        f"- **Places analyzed:** {summaries[0]['n_places']:,} (minimum {summaries[0]['min_img_per_place']} images per place).",
        f"- **Total images:** {summaries[0]['n_images']:,}.",
        "- Each model encodes every image independently; metrics are computed in that model's descriptor space.",
        "",
    ])

    parts.append(METRIC_GLOSSARY.strip())
    parts.append("")
    parts.extend(_notes_section(summaries))

    for summary in summaries:
        parts.append(f"## Headline results — {summary['tag']}")
        parts.append("")
        parts.extend(_format_table(summary))
        parts.append("")
        parts.extend(_models_section(summary))

    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown report for GSV latent-space analysis.")
    parser.add_argument(
        "--summary",
        type=Path,
        action="append",
        required=True,
        help="Summary JSON from gsv_multi_model.py (repeatable).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/place_latent_space/outputs/gsv/gsv_latent_space_supervisor_report.md"),
    )
    parser.add_argument(
        "--title",
        default="GSV-Cities Place Latent-Space Analysis",
    )
    args = parser.parse_args()

    summaries = [_load_summary(p) for p in args.summary]
    report = build_report(summaries, args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
