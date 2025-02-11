import pandas as pd
from typing import Optional, List, Dict, Union, Any
from src.krippendorff_alpha.preprocessing import preprocess_data
from src.krippendorff_alpha.reliability import compute_reliability_matrix
from src.krippendorff_alpha.metric import krippendorff_alpha


def compute_alpha(
    path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    text_col: Optional[str] = None,
    annotator_cols: Optional[List[str]] = None,
    weight_dict: Optional[Dict[str, float]] = None,
    metric: str = "nominal",
    ordinal_scale: Optional[List[Union[int, float, str]]] = None,
) -> Dict[str, Any]:
    """
    Computes Krippendorff's Alpha for Inter-annotator Agreement Analysis.

    Args:
        path (str): Path to the dataset (CSV, JSON, JSONL, etc.).
        df (pd.DataFrame): DataFrame containing annotations.
        text_col (str): Column containing the text.
        annotator_cols (list): List of annotator columns.
        weight_dict (dict): Dictionary mapping annotators to weights.
        metric (str): "nominal", "ordinal", "interval", or "ratio".
        ordinal_scale (list): Optional scale for ordinal distance calculation.

    Returns:
        dict: {
            "alpha": float,
            "observed_disagreement": float,
            "expected_disagreement": float,
            "per_category_scores": Dict[str, Dict[str, float]]
        }
    """
    # Preprocess data
    processed_data = preprocess_data(path=path, df=df, text_col=text_col, annotator_cols=annotator_cols, metric=metric)
    df_processed = processed_data.df
    column_mapping = processed_data.column_mapping
    nominal_mappings = processed_data.nominal_mappings
    ordinal_mappings = processed_data.ordinal_mappings

    # Compute reliability matrix
    reliability_matrix = compute_reliability_matrix(df_processed, column_mapping)

    # Compute Krippendorff's Alpha and get all details
    alpha_results = krippendorff_alpha(
        df=reliability_matrix,
        annotator_cols=column_mapping.annotator_cols,
        metric=metric,
        weight_dict=weight_dict,
        ordinal_scale=ordinal_scale,
        nominal_mappings=nominal_mappings,
        ordinal_mappings=ordinal_mappings,
    )

    return alpha_results
