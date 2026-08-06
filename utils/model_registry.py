"""Single registry for proposed, comparison, and ablation experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch.nn as nn

from models.comparisons import build_comparison_deep_model
from models.landslidenet import build_landslidenet_variant


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    family: str
    implementation: str
    paper_role: str
    reference: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Stable insertion order controls both ``model=all`` and deterministic seeds.
MODEL_SPECS = {
    "logistic_regression": ModelSpec(
        "logistic_regression",
        "Logistic Regression",
        "classical",
        "native",
        "machine_learning_comparison",
    ),
    "catboost": ModelSpec(
        "catboost",
        "CatBoost",
        "classical",
        "library",
        "machine_learning_comparison",
        notes=(
            "Requires catboost; all inputs use the frozen fold-specific "
            "feature transformer."
        ),
    ),
    "extra_trees": ModelSpec(
        "extra_trees",
        "Extra Trees",
        "classical",
        "native",
        "machine_learning_comparison",
    ),
    "lightgbm": ModelSpec(
        "lightgbm",
        "LightGBM",
        "classical",
        "library",
        "machine_learning_comparison",
        notes="Requires the lightgbm package.",
    ),
    "random_forest": ModelSpec(
        "random_forest",
        "Random Forest",
        "classical",
        "native",
        "machine_learning_comparison",
    ),
    "baseline": ModelSpec(
        "baseline",
        "Baseline",
        "deep",
        "controlled_ablation",
        "ablation",
        notes="LandslideNet backbone with neither deformable SPB nor DCSE.",
    ),
    "only_dcse": ModelSpec(
        "only_dcse",
        "Only DCSE",
        "deep",
        "controlled_ablation",
        "ablation",
        notes="Same backbone; ordinary spatial convolution plus DCSE.",
    ),
    "only_spb": ModelSpec(
        "only_spb",
        "Only SPB",
        "deep",
        "controlled_ablation",
        "ablation",
        notes="Same backbone; deformable spatial block without DCSE.",
    ),
    "landslidenet": ModelSpec(
        "landslidenet",
        "LandslideNet",
        "deep",
        "study_model",
        "proposed_model",
        notes="Deformable spatial perception plus DCSE.",
    ),
    "dbpfnet": ModelSpec(
        "dbpfnet",
        "DBPFNet (task-adapted)",
        "deep",
        "task_adapter",
        "deep_learning_comparison",
        reference="https://doi.org/10.1080/17538947.2025.2499199",
        notes=(
            "Dense spatial/factor-sequence adapter. The source model also uses "
            "temporal InSAR inputs unavailable here; this is not official author code."
        ),
    ),
    "da_lsf": ModelSpec(
        "da_lsf",
        "DA-LSF (task-adapted)",
        "deep",
        "task_adapter",
        "deep_learning_comparison",
        reference="https://doi.org/10.1109/TGRS.2023.3323668",
        notes=(
            "Factor-attention/dilated-spatial task adapter; not an official "
            "drop-in semantic-segmentation reproduction."
        ),
    ),
    "lgc_net": ModelSpec(
        "lgc_net",
        "LGC-Net (task-adapted)",
        "deep",
        "paper_guided_task_adapter",
        "deep_learning_comparison",
        reference="https://doi.org/10.1016/j.engappai.2025.110924",
        notes=(
            "Local CNN/global Transformer adapter for aligned static factors; "
            "Gate-FFT/InSAR extensions are excluded."
        ),
    ),
}

MODEL_GROUPS = {
    "proposed": ["landslidenet"],
    "machine_learning": [
        "logistic_regression",
        "catboost",
        "extra_trees",
        "lightgbm",
        "random_forest",
    ],
    "deep_learning": ["dbpfnet", "da_lsf", "lgc_net"],
    "ablation": ["baseline", "only_dcse", "only_spb"],
}
MODEL_GROUPS["all"] = list(MODEL_SPECS)


_ALIASES = {
    "lr": "logistic_regression",
    "logistic": "logistic_regression",
    "extratrees": "extra_trees",
    "et": "extra_trees",
    "lgbm": "lightgbm",
    "rf": "random_forest",
    "unet": "baseline",
    "dcse": "only_dcse",
    "spb": "only_spb",
    "landsildenet": "landslidenet",
    "lgc-net": "lgc_net",
    "lgcnet": "lgc_net",
    "da-lsf": "da_lsf",
}


def canonical_model_name(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model {value!r}. Available models: {', '.join(MODEL_SPECS)}"
        )
    return key


def expand_model_selection(values) -> list[str]:
    """Expand model names, group names, or ``all`` while preserving order."""
    if isinstance(values, str):
        values = [values]
    result = []
    for value in values:
        token = str(value).strip().lower()
        expanded = (
            MODEL_GROUPS[token]
            if token in MODEL_GROUPS
            else [canonical_model_name(value)]
        )
        for raw_name in expanded:
            name = canonical_model_name(raw_name)
            if name not in result:
                result.append(name)
    return result


def model_specs(names) -> list[dict]:
    return [MODEL_SPECS[canonical_model_name(name)].to_dict() for name in names]


def build_deep_model(
    name: str, num_bands: int, num_classes: int = 2
) -> nn.Module:
    name = canonical_model_name(name)
    if MODEL_SPECS[name].family != "deep":
        raise ValueError(f"{name} is a classical model, not a torch model.")
    if name in {"landslidenet", "baseline", "only_dcse", "only_spb"}:
        return build_landslidenet_variant(name, num_bands, num_classes)
    return build_comparison_deep_model(name, num_bands, num_classes)
