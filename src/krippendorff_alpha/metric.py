import numpy as np
import numpy.typing as npt
import pandas as pd
import logging
from functools import partial
from typing import Any, Callable

from krippendorff_alpha.schema import DataTypeEnum
from krippendorff_alpha.constants import (
    ANNOTATOR_REGEX,
    SYMMETRIC_DISAGREEMENT_DIVISOR,
    DEFAULT_DECIMAL_PLACES,
    MIN_ANNOTATORS_REQUIRED,
    MIN_SUBJECTS_REQUIRED,
)

logger = logging.getLogger(__name__)


def nominal_distance(a: int | float | str, b: int | float | str) -> float:
    """
    Calculate distance between two nominal (categorical) values.

    For nominal data, distance is binary: 0 if values match, 1 if they don't.

    Args:
        a: First value
        b: Second value

    Returns:
        0.0 if values are equal, 1.0 otherwise
    """
    return float(0 if a == b else 1)


def ordinal_distance(a: int | float | str, b: int | float | str, scale: list[int | float | str] | None = None) -> float:
    """
    Calculate distance between two ordinal values.

    For ordinal data, distance is the squared difference in rank positions.
    Falls back to nominal distance if scale is not provided or values are not in scale.

    Args:
        a: First ordinal value
        b: Second ordinal value
        scale: Ordered list defining the ordinal scale

    Returns:
        Squared difference in rank positions, or 1.0 if values don't match (nominal fallback)
    """
    if scale is None or a not in scale or b not in scale:
        return float(nominal_distance(a, b))

    diff = scale.index(a) - scale.index(b)
    return float(diff**2)


def interval_distance(a: float, b: float) -> float:
    """
    Calculate distance between two interval values.

    For interval data, distance is the squared difference between values.

    Args:
        a: First interval value
        b: Second interval value

    Returns:
        Squared difference: (a - b)²
    """
    return (a - b) ** 2


def ratio_distance(a: float, b: float) -> float:
    """
    Calculate distance between two ratio values.

    For ratio data, distance is normalized by the sum of values to account for scale.
    Handles edge case where both values are zero.

    Args:
        a: First ratio value
        b: Second ratio value

    Returns:
        Normalized squared difference: (a - b)² / (a + b), or 0.0 if both are zero
    """
    if a == 0 and b == 0:
        return 0.0
    if a + b == 0:
        return float("inf") if a != b else 0.0
    return (a - b) ** 2 / (a + b)


def reverse_map(value: int | float | str, mapping: dict[str, int | float] | None) -> int | float | str:
    """
    Reverse map a numeric value back to its original categorical label.

    Args:
        value: Numeric value to reverse map
        mapping: Dictionary mapping labels to numeric values

    Returns:
        Original label if found in mapping, otherwise returns value as-is
    """
    if mapping is None:
        return value

    if not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in mapping.items()):
        raise TypeError("Mapping dictionary must have string keys and numeric (int or float) values.")

    reversed_mapping: dict[int | float, str] = {v: k for k, v in mapping.items()}

    if isinstance(value, (int, float)):
        return reversed_mapping.get(value, str(value))

    return value


def parse_annotator_name(name: str) -> str:
    """
    Parse annotator name using regex pattern to extract base name.

    Args:
        name: Annotator column name (e.g., "annotator1", "annotator_2")

    Returns:
        Parsed annotator name matching the regex pattern, or original name if no match
    """
    match = ANNOTATOR_REGEX.match(name)
    return match.group(0) if match else name


def compute_weight_vector(
    df: pd.DataFrame, weight_dict: dict[str, float] | None
) -> npt.NDArray[np.float64]:
    """
    Compute weight vector for annotators.

    Args:
        df: DataFrame with annotators as index
        weight_dict: Optional dictionary mapping annotator names to weights

    Returns:
        Array of weights, defaulting to 1.0 for all annotators if weight_dict is None
    """
    num_annotators = len(df.index)
    weight_vector = np.ones(num_annotators)
    if weight_dict:
        for i, annotator in enumerate(df.index):
            parsed_name = parse_annotator_name(annotator)
            if parsed_name in weight_dict:
                weight_vector[i] = weight_dict[parsed_name]
    return weight_vector


def _calculate_unit_weight(m_u: int) -> float:
    """
    Calculate weight for a unit according to Krippendorff's formula.

    Weight = m_u / P(m_u, 2) where P(m_u, 2) = m_u * (m_u - 1)
    This simplifies to 1 / (m_u - 1)
    """
    if m_u < 2:
        return 0.0
    return 1.0 / (m_u - 1)


def _process_unit_pairs(
    annotator_values: npt.NDArray[np.float64],
    weight_vector: npt.NDArray[np.float64],
    distance_fn: Callable[[float, float], float],
    unit_weight: float,
    data_type: DataTypeEnum,
) -> tuple[float, dict[int, float], dict[int, int]]:
    """
    Process all pairs of annotators for a single unit.

    Returns:
        Tuple of (total_weighted_disagreement, per_category_disagreement, pairwise_counts)
    """
    n = len(annotator_values)
    unit_disagreement = 0.0
    per_category_dis: dict[int, float] = {}
    pairwise_counts: dict[int, int] = {}

    non_nan_indices = [i for i in range(n) if not np.isnan(annotator_values[i])]

    for idx_a, a in enumerate(non_nan_indices):
        for b in non_nan_indices[idx_a + 1 :]:
            d = (
                weight_vector[a]
                * weight_vector[b]
                * distance_fn(float(annotator_values[a]), float(annotator_values[b]))
            )

            weighted_d = d * unit_weight
            unit_disagreement += weighted_d

            if data_type in {DataTypeEnum.NOMINAL, DataTypeEnum.ORDINAL}:
                cat_a = int(annotator_values[a])
                cat_b = int(annotator_values[b])

                half_weighted_d = weighted_d / SYMMETRIC_DISAGREEMENT_DIVISOR
                per_category_dis[cat_a] = per_category_dis.get(cat_a, 0.0) + half_weighted_d
                per_category_dis[cat_b] = per_category_dis.get(cat_b, 0.0) + half_weighted_d
                pairwise_counts[cat_a] = pairwise_counts.get(cat_a, 0) + 1
                pairwise_counts[cat_b] = pairwise_counts.get(cat_b, 0) + 1

    return unit_disagreement, per_category_dis, pairwise_counts


def compute_observed_disagreement(
    reliability_matrix: npt.NDArray[np.float64],
    weight_vector: npt.NDArray[np.float64],
    distance_fn: Callable[[float, float], float],
    data_type: DataTypeEnum,
) -> tuple[float, dict[int, float], dict[int, int]]:
    """
    Computes observed disagreement according to Krippendorff's formula:
    D_o = (1/n) * Σ_c Σ_k δ(c,k) * Σ_u m_u * (n_{cku} / P(m_u,2))

    Where:
    - n = total pairable values = Σ m_u across all units
    - m_u = number of coders who coded unit u (non-NaN values)
    - P(m_u, 2) = m_u * (m_u - 1) (permutations)
    - n_{cku} = number of (c,k) pairs in unit u

    Missing value strategy "IGNORE" is implemented by skipping comparisons where
    either value is NaN, ensuring only pairable values are included in the calculation.

    Args:
        reliability_matrix: Matrix with annotators as rows and units as columns
        weight_vector: Vector of weights for each annotator
        distance_fn: Function to calculate distance between two values
        data_type: Type of data (nominal, ordinal, interval, ratio)

    Returns:
        Tuple of (observed_disagreement, per_category_observed_disagreement, pairwise_counts)
    """
    num_units: int = reliability_matrix.shape[1]
    observed_disagreement = 0.0
    per_category_obs_dis: dict[int, float] = {}
    pairwise_counts: dict[int, int] = {}
    total_pairable_values = 0

    for unit_idx in range(num_units):
        annotator_values = reliability_matrix[:, unit_idx]

        non_nan_mask = ~np.isnan(annotator_values)
        num_coders_per_unit = int(np.sum(non_nan_mask))

        if num_coders_per_unit < 2:
            continue

        unit_weight = _calculate_unit_weight(num_coders_per_unit)
        total_pairable_values += num_coders_per_unit

        unit_dis, unit_per_cat, unit_counts = _process_unit_pairs(
            annotator_values, weight_vector, distance_fn, unit_weight, data_type
        )

        observed_disagreement += unit_dis

        for cat, dis in unit_per_cat.items():
            per_category_obs_dis[cat] = per_category_obs_dis.get(cat, 0.0) + dis
        for cat, count in unit_counts.items():
            pairwise_counts[cat] = pairwise_counts.get(cat, 0) + count

    if total_pairable_values > 0:
        observed_disagreement /= total_pairable_values
    else:
        observed_disagreement = 0.0

    return observed_disagreement, per_category_obs_dis, pairwise_counts


def compute_expected_disagreement(
    reliability_matrix: npt.NDArray[np.float64],
    distance_fn: Callable[[float, float], float],
    data_type: DataTypeEnum,
) -> tuple[float, dict[int, float]]:
    """
    Compute expected disagreement based on category frequencies.

    Expected disagreement is calculated as the sum of all pairwise distances
    weighted by their joint probability of occurrence.

    Args:
        reliability_matrix: Matrix with annotators as rows and units as columns
        distance_fn: Function to calculate distance between two values
        data_type: Type of data (nominal, ordinal, interval, ratio)

    Returns:
        Tuple of (expected_disagreement, per_category_expected_disagreement)
    """
    expected_disagreement = 0.0
    per_category_exp_dis: dict[int, float] = {}

    non_nan_values = reliability_matrix[~np.isnan(reliability_matrix)]
    if len(non_nan_values) == 0:
        return 0.0, {}

    unique_values, counts = np.unique(non_nan_values, return_counts=True)
    total_values = counts.sum()

    if total_values == 0:
        return 0.0, {}

    category_frequencies = {int(value): count / total_values for value, count in zip(unique_values, counts)}

    for value1 in unique_values:
        for value2 in unique_values:
            distance = distance_fn(float(value1), float(value2))
            prob_product = category_frequencies[int(value1)] * category_frequencies[int(value2)]
            disagreement_contribution = distance * prob_product
            expected_disagreement += disagreement_contribution

            if data_type in {DataTypeEnum.NOMINAL, DataTypeEnum.ORDINAL}:
                category1 = int(value1)
                category2 = int(value2)

                half_disagreement = disagreement_contribution / SYMMETRIC_DISAGREEMENT_DIVISOR
                per_category_exp_dis[category1] = per_category_exp_dis.get(category1, 0.0) + half_disagreement
                per_category_exp_dis[category2] = per_category_exp_dis.get(category2, 0.0) + half_disagreement
    return expected_disagreement, per_category_exp_dis


def compute_per_category_scores(
    unique_values: npt.NDArray[Any],
    per_category_obs_dis: dict[int, float],
    per_category_exp_dis: dict[int, float],
    pairwise_counts: dict[int, int],
    mapping: dict[str, int | float] | None,
) -> dict[str | int, dict[str, float]]:
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
            mapped_category = str(mapped_category)

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
    ordinal_scale: list[int | float | str] | None = None,
    mapping: dict[str, int | float] | None = None,
    weight_dict: dict[str, float] | None = None,
) -> dict[str, Any]:
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
    num_subjects, num_annotators = reliability_matrix.shape
    if num_annotators < MIN_ANNOTATORS_REQUIRED or num_subjects < MIN_SUBJECTS_REQUIRED:
        raise ValueError(
            f"Reliability matrix must have at least {MIN_ANNOTATORS_REQUIRED} annotators "
            f"and {MIN_SUBJECTS_REQUIRED} subjects."
        )

    weight_vector = compute_weight_vector(df, weight_dict)
    logger.info(f"Weight vector: {weight_vector}")

    distance_fn: Callable[[float, float], float]
    if data_type == DataTypeEnum.NOMINAL:
        distance_fn = nominal_distance
    elif data_type == DataTypeEnum.ORDINAL:
        distance_fn = partial(ordinal_distance, scale=ordinal_scale)
    elif data_type == DataTypeEnum.INTERVAL:
        distance_fn = interval_distance
    elif data_type == DataTypeEnum.RATIO:
        distance_fn = ratio_distance
    else:
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
        "alpha": round(float(overall_alpha), DEFAULT_DECIMAL_PLACES),
        "observed_disagreement": round(float(observed_disagreement), DEFAULT_DECIMAL_PLACES),
        "expected_disagreement": round(float(expected_disagreement), DEFAULT_DECIMAL_PLACES),
        "per_category_scores": (
            {
                category: {
                    "observed_disagreement": round(float(scores["observed_disagreement"]), DEFAULT_DECIMAL_PLACES),
                    "expected_disagreement": round(float(scores["expected_disagreement"]), DEFAULT_DECIMAL_PLACES),
                }
                for category, scores in per_category_scores.items()
            }
            if data_type in {DataTypeEnum.NOMINAL, DataTypeEnum.ORDINAL}
            else None
        ),
    }
