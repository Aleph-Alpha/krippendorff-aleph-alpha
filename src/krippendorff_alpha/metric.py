import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, Union, List
from src.krippendorff_alpha.constants import ANNOTATOR_REGEX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def nominal_distance(a: Union[int, float, str], b: Union[int, float, str]) -> int:
    return 0 if a == b else 1


def ordinal_distance(
    a: Union[int, float, str], b: Union[int, float, str], scale: Optional[List[Union[int, float, str]]] = None
) -> int:
    if scale is None or a not in scale or b not in scale:
        return nominal_distance(a, b)
    return (scale.index(a) - scale.index(b)) ** 2


def interval_distance(a: float, b: float) -> float:
    return (a - b) ** 2


def ratio_distance(a: float, b: float) -> float:
    return ((a - b) / (a + b)) ** 2 if (a + b) != 0 else 0.0


def reverse_map(value: int, mapping: Dict[str, Dict[str, int]]) -> Union[str, int]:
    logger.debug(f"Attempting to reverse map value: {value}")
    for _, label_dict in mapping.items():
        for label, num in label_dict.items():
            if num == value:
                logger.debug(f"Mapped {value} -> {label}")
                return label
    logger.warning(f"Value {value} not found in mapping.")
    logger.debug(f"Reverse mapping check: {mapping}")
    return value


def parse_annotator_name(name: str) -> str:
    match = ANNOTATOR_REGEX.match(name)
    return match.group(0) if match else name


def krippendorff_alpha(
    df: pd.DataFrame,
    annotator_cols: List[str],
    metric: str = "nominal",
    weight_dict: Optional[Dict[str, float]] = None,
    ordinal_scale: Optional[List[Union[int, float, str]]] = None,
    nominal_mappings: Optional[Dict[str, Dict[str, int]]] = None,
    ordinal_mappings: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    logger.info("Starting Krippendorff's alpha calculation.")
    reliability_matrix = df.to_numpy(dtype=np.float64)
    k, n = reliability_matrix.shape  # Rows are annotators, columns are subjects
    logger.info(f"Reliability matrix size: {k} annotators, {n} subjects")

    if k < 3 or n < 3:
        raise ValueError("Reliability matrix must have at least three annotators and three subjects.")

    weight_vector = np.ones(k)
    if weight_dict:
        for i, annotator in enumerate(annotator_cols):
            parsed_name = parse_annotator_name(annotator)
            if parsed_name in weight_dict:
                weight_vector[i] = weight_dict[parsed_name]
    logger.info("Weight vector computed.")

    distance_fn = {
        "nominal": nominal_distance,
        "ordinal": lambda a, b: ordinal_distance(a, b, ordinal_scale),
        "interval": interval_distance,
        "ratio": ratio_distance,
    }[metric]

    observed_disagreement = 0.0
    expected_disagreement = 0.0

    unique_values, counts = np.unique(reliability_matrix[~np.isnan(reliability_matrix)], return_counts=True)
    category_frequencies = {v: c / counts.sum() for v, c in zip(unique_values, counts)}

    if metric in ["nominal", "ordinal"]:
        per_category_obs_dis = {v: 0.0 for v in unique_values}
        per_category_exp_dis = {v: 0.0 for v in unique_values}
        pairwise_counts = {v: 0 for v in unique_values}

    for j in range(k):
        for i in range(n):
            for idx_l in range(i + 1, n):
                d = weight_vector[j] * distance_fn(reliability_matrix[j, i], reliability_matrix[j, idx_l])
                observed_disagreement += d
                if metric in ["nominal", "ordinal"]:
                    per_category_obs_dis[reliability_matrix[j, i]] += d
                    pairwise_counts[reliability_matrix[j, i]] += 1
    logger.info(f"Observed disagreement: {observed_disagreement}")

    for v1 in unique_values:
        for v2 in unique_values:
            d = distance_fn(v1, v2) * category_frequencies.get(v1, 0) * category_frequencies.get(v2, 0)
            expected_disagreement += d
            if metric in ["nominal", "ordinal"]:
                per_category_exp_dis[v1] += d
    logger.info(f"Expected disagreement: {expected_disagreement}")

    observed_disagreement /= k * (n * (n - 1) / 2)
    expected_disagreement /= sum(category_frequencies.values())

    per_category_scores: Dict[str, Dict[str, float]] = {}
    if metric in ["nominal", "ordinal"] and (nominal_mappings or ordinal_mappings):
        correct_mapping = ordinal_mappings if metric == "ordinal" else nominal_mappings
        if correct_mapping is not None:  # Ensure correct_mapping is valid
            for category in unique_values:
                mapped_category = reverse_map(int(category), correct_mapping)
                logger.debug(f"Mapping category {category} -> {mapped_category}")
                per_category_scores[str(mapped_category)] = {
                    "observed_disagreement": per_category_obs_dis[category] / max(pairwise_counts.get(category, 1), 1),
                    "expected_disagreement": per_category_exp_dis[category],
                }

    overall_alpha = 1 - (observed_disagreement / expected_disagreement) if expected_disagreement > 0 else 1.0
    logger.info(f"Krippendorff's alpha: {overall_alpha}")

    return {
        "alpha": overall_alpha,
        "observed_disagreement": observed_disagreement,
        "expected_disagreement": expected_disagreement,
        "per_category_scores": per_category_scores,
    }
