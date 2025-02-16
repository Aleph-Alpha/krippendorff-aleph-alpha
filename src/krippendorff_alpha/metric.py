import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, Union, List, Tuple, Callable

from krippendorff_alpha.schema import DataTypeEnum
from krippendorff_alpha.constants import ANNOTATOR_REGEX

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def nominal_distance(a: Union[int, float, str], b: Union[int, float, str]) -> float:
    return float(0 if a == b else 1)


def ordinal_distance(
    a: Union[int, float, str], b: Union[int, float, str], scale: Optional[List[Union[int, float, str]]] = None
) -> float:
    if scale is None or a not in scale or b not in scale:
        return float(nominal_distance(a, b))

    diff = scale.index(a) - scale.index(b)
    return float(diff**2)


def interval_distance(a: float, b: float) -> float:
    return (a - b) ** 2


def ratio_distance(a: float, b: float) -> float:
    if a == 0 and b == 0:
        return 0.0
    return (a - b) ** 2 / (a + b)


def reverse_map(
    value: Union[int, float, str], mapping: Optional[Dict[str, Union[int, float]]]
) -> Union[int, float, str]:
    if mapping is None:
        return value

    if not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in mapping.items()):
        raise TypeError("Mapping dictionary must have string keys and numeric (int or float) values.")

    reversed_mapping: Dict[Union[int, float], str] = {v: k for k, v in mapping.items()}

    if isinstance(value, (int, float)):
        return reversed_mapping.get(value, str(value))

    # If value is a string, return as-is (cannot be reversed)
    return value


def parse_annotator_name(name: str) -> str:
    match = ANNOTATOR_REGEX.match(name)
    return match.group(0) if match else name


def compute_weight_vector(
    df: pd.DataFrame, weight_dict: Optional[Dict[str, float]]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    k = len(df.index)
    weight_vector = np.ones(k)
    if weight_dict:
        for i, annotator in enumerate(df.index):
            parsed_name = parse_annotator_name(annotator)
            if parsed_name in weight_dict:
                weight_vector[i] = weight_dict[parsed_name]
    return weight_vector


def compute_observed_disagreement(
    reliability_matrix: np.ndarray[Any, np.dtype[np.float64]],
    weight_vector: np.ndarray[Any, np.dtype[np.float64]],
    distance_fn: Callable[[Any, Any], float],
    data_type: DataTypeEnum,
) -> Tuple[float, Dict[int, float], Dict[int, int]]:
    n, k = reliability_matrix.shape
    observed_disagreement = 0.0
    per_category_obs_dis: Dict[int, float] = {}
    pairwise_counts: Dict[int, int] = {}

    total_comparisons = 0  # Keep track of the number of comparisons

    for j in range(k):  # Iterate over items (columns)
        annotator_values = reliability_matrix[:, j]

        for a in range(n):
            for b in range(a + 1, n):
                d = weight_vector[a] * weight_vector[b] * distance_fn(annotator_values[a], annotator_values[b])

                observed_disagreement += d
                total_comparisons += 1  # Count valid comparisons

                if data_type in {DataTypeEnum.NOMINAL, DataTypeEnum.ORDINAL}:
                    cat = int(annotator_values[a])
                    per_category_obs_dis[cat] = per_category_obs_dis.get(cat, 0) + d
                    pairwise_counts[cat] = pairwise_counts.get(cat, 0) + 1

    # Normalize observed disagreement
    observed_disagreement /= total_comparisons if total_comparisons > 0 else 1
    return observed_disagreement, per_category_obs_dis, pairwise_counts


def compute_expected_disagreement(
    reliability_matrix: np.ndarray[Any, np.dtype[np.float64]],
    distance_fn: Callable[[Any, Any], float],
    data_type: DataTypeEnum,
) -> Tuple[float, Dict[int, float]]:
    expected_disagreement = 0.0
    per_category_exp_dis: Dict[int, float] = {}

    unique_values, counts = np.unique(reliability_matrix[~np.isnan(reliability_matrix)], return_counts=True)
    total_values = counts.sum()

    # Compute category probabilities
    category_frequencies = {int(v): c / total_values for v, c in zip(unique_values, counts)}

    # Compute expected disagreement
    for v1 in unique_values:
        for v2 in unique_values:
            d = distance_fn(v1, v2) * category_frequencies[v1] * category_frequencies[v2]
            expected_disagreement += d

            if data_type in {DataTypeEnum.NOMINAL, DataTypeEnum.ORDINAL}:
                per_category_exp_dis[int(v1)] = per_category_exp_dis.get(int(v1), 0) + d
    return expected_disagreement, per_category_exp_dis


def compute_per_category_scores(
    unique_values: np.ndarray[Any, np.dtype[np.number]],
    per_category_obs_dis: Dict[int, float],
    per_category_exp_dis: Dict[int, float],
    pairwise_counts: Dict[int, int],
    mapping: Optional[Dict[str, Union[int, float]]],
) -> Dict[Union[str, int], Dict[str, float]]:
    per_category_scores = {}
    for category in unique_values:
        category_value = category.item()

        if isinstance(category_value, complex):
            raise ValueError(f"Unexpected complex value: {category_value}")

        if isinstance(category_value, float) and category_value.is_integer():
            category_value = int(category_value)

        mapped_category = reverse_map(category_value, mapping) if mapping else str(category_value)

        # Ensure mapped_category is either str or int
        if isinstance(mapped_category, float) and mapped_category.is_integer():
            mapped_category = int(mapped_category)
        elif not isinstance(mapped_category, (str, int)):
            mapped_category = str(mapped_category)  # Convert any unexpected types to string

        observed_disagreement_value = per_category_obs_dis.get(int(category_value), 0) / max(
            pairwise_counts.get(int(category_value), 1), 1
        )
        expected_disagreement_value = per_category_exp_dis.get(int(category_value), 0)

        per_category_scores[mapped_category] = {
            "observed_disagreement": observed_disagreement_value,
            "expected_disagreement": expected_disagreement_value,
        }

    return per_category_scores


def krippendorff_alpha(
    df: pd.DataFrame,
    data_type: DataTypeEnum,
    ordinal_scale: Optional[List[Union[int, float, str]]] = None,
    mapping: Optional[Dict[str, Union[int, float]]] = None,
    weight_dict: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Computes Krippendorff's alpha reliability coefficient for assessing inter-annotator agreement.

    Args:
        df (pd.DataFrame): The input DataFrame containing annotations from multiple annotators.
        data_type (DataTypeEnum): The type of data (nominal, ordinal, interval, or ratio).
        ordinal_scale (Optional[List[Union[int, float, str]]]): The predefined scale for ordinal data (if applicable).
        mapping (Optional[Dict[str, Union[int, float]]]): A mapping of categorical labels to numeric values.
        weight_dict (Optional[Dict[str, float]]): An optional dictionary assigning weights to annotators.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "alpha": The Krippendorff's alpha value (float).
            - "observed_disagreement": The observed disagreement value (float).
            - "expected_disagreement": The expected disagreement value (float).
            - "per_category_scores": A breakdown of disagreements per category (dict) for nominal/ordinal data.
    """

    logger.info("Starting Krippendorff's alpha calculation.")
    reliability_matrix = df.to_numpy(dtype=np.float64)
    n, k = reliability_matrix.shape
    if k < 3 or n < 3:
        raise ValueError("Reliability matrix must have at least three annotators and three subjects.")

    weight_vector = compute_weight_vector(df, weight_dict)
    logger.info(f"Weight vector: {weight_vector}")

    distance_fn = {
        DataTypeEnum.NOMINAL: nominal_distance,
        DataTypeEnum.ORDINAL: lambda a, b: ordinal_distance(a, b, scale=ordinal_scale),
        DataTypeEnum.INTERVAL: interval_distance,
        DataTypeEnum.RATIO: ratio_distance,
    }.get(data_type)

    if distance_fn is None:
        raise ValueError(f"Unsupported data type: {data_type}")
    observed_disagreement, per_category_obs_dis, pairwise_counts = compute_observed_disagreement(
        reliability_matrix, weight_vector, distance_fn, data_type
    )
    expected_disagreement, per_category_exp_dis = compute_expected_disagreement(
        reliability_matrix, distance_fn, data_type
    )

    unique_values = np.unique(reliability_matrix[~np.isnan(reliability_matrix)])
    per_category_scores = compute_per_category_scores(
        unique_values, per_category_obs_dis, per_category_exp_dis, pairwise_counts, mapping
    )

    overall_alpha = 1 - (observed_disagreement / expected_disagreement) if expected_disagreement > 0 else 1.0
    logger.info(f"Krippendorff's alpha: {overall_alpha}")

    return {
        "alpha": round(float(overall_alpha), 3),
        "observed_disagreement": round(float(observed_disagreement), 3),
        "expected_disagreement": round(float(expected_disagreement), 3),
        "per_category_scores": (
            {
                k: {
                    "observed_disagreement": round(float(v["observed_disagreement"]), 3),
                    "expected_disagreement": round(float(v["expected_disagreement"]), 3),
                }
                for k, v in per_category_scores.items()
            }
            if data_type in {DataTypeEnum.NOMINAL, DataTypeEnum.ORDINAL}
            else None
        ),
    }
