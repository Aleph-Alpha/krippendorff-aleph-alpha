import numpy as np
from typing import Optional, Dict, Any, Union, List


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


def krippendorff_alpha(
    reliability_matrix: np.ndarray[Any, np.dtype[np.float64]],
    metric: str = "nominal",
    weight_dict: Optional[Dict[str, float]] = None,
    ordinal_scale: Optional[List[Union[int, float, str]]] = None,
) -> Dict[str, Any]:
    n, k = reliability_matrix.shape
    if n < 3 or k < 3:
        raise ValueError("Reliability matrix must have at least three annotators and three subjects.")

    weight_matrix = np.ones((k, k))
    if weight_dict:
        annotator_ids = list(weight_dict.keys())
        for i in range(k):
            for j in range(k):
                weight_matrix[i, j] = weight_dict.get(annotator_ids[i], 1) * weight_dict.get(annotator_ids[j], 1)

    distance_fn = {
        "nominal": nominal_distance,
        "ordinal": lambda a, b: ordinal_distance(a, b, ordinal_scale),
        "interval": interval_distance,
        "ratio": ratio_distance,
    }[metric]

    observed_disagreement = 0.0
    expected_disagreement = 0.0

    unique_values = np.unique(reliability_matrix[~np.isnan(reliability_matrix)])
    per_category_obs_dis = {v: 0.0 for v in unique_values}
    per_category_exp_dis = {v: 0.0 for v in unique_values}
    category_counts = {v: 0 for v in unique_values}

    for i in range(n):
        for j in range(k):
            for idx_l in range(k):
                d = weight_matrix[j, idx_l] * distance_fn(reliability_matrix[i, j], reliability_matrix[i, idx_l])
                observed_disagreement += d
                per_category_obs_dis[reliability_matrix[i, j]] += d
                category_counts[reliability_matrix[i, j]] += 1

    for v1 in unique_values:
        for v2 in unique_values:
            d = distance_fn(v1, v2)
            expected_disagreement += d
            per_category_exp_dis[v1] += d

    observed_disagreement /= n * k * (k - 1)
    expected_disagreement /= len(unique_values) ** 2

    per_category_scores = {
        category: {
            "observed_disagreement": per_category_obs_dis[category] / max(category_counts[category], 1),
            "expected_disagreement": per_category_exp_dis[category] / max(len(unique_values), 1),
        }
        for category in unique_values
    }

    overall_alpha = 1 - (observed_disagreement / expected_disagreement) if expected_disagreement > 0 else 1.0

    return {
        "alpha": overall_alpha,
        "observed_disagreement": observed_disagreement,
        "expected_disagreement": expected_disagreement,
        "per_category_scores": per_category_scores,
    }
