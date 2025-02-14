from typing import Optional, Any, Dict

import pandas as pd

from krippendorff_alpha.metric import krippendorff_alpha
from krippendorff_alpha.preprocessing import preprocess_data
from krippendorff_alpha.reliability import compute_reliability_matrix
from krippendorff_alpha.schema import ColumnMapping, AnnotationSchema


def compute_alpha(
    df: pd.DataFrame,
    column_mapping: Optional[ColumnMapping] = None,
    annotation_schema: Optional[AnnotationSchema] = None,
    weight_dict: Optional[Dict[str, float]] = None,
    ordinal_scale: Optional[Dict[str, float]] = None,
) -> Any:
    """
    Computes Krippendorff's alpha for inter-annotator agreement.

    Parameters:
    - df (pd.DataFrame): The dataframe containing annotation data.
    - column_mapping (Optional[ColumnMapping]): A mapping specifying which columns correspond to annotators and annotations.
    - annotation_schema (Optional[AnnotationSchema]): The schema defining the annotation type (nominal, ordinal, etc.).
    - weight_dict (Optional[Dict[str, float]]): A dictionary specifying weights for individual annotators (if applicable).
    - ordinal_scale (Optional[Dict[str, float]]): A dictionary defining an ordinal scale if the data type is ordinal.

    Returns:
    - Any: A dictionary containing Krippendorff's alpha, observed and expected disagreement, and per-category scores.

    Raises:
    - ValueError: If df, column_mapping, or annotation_schema are not provided.
    """
    if df is None or column_mapping is None or annotation_schema is None:
        raise ValueError("df, column_mapping, and annotation_schema must be provided.")

    preprocessed_data, text_col = preprocess_data(df, column_mapping, annotation_schema)

    if preprocessed_data.nominal_mappings:
        preprocessed_data.nominal_mappings = {str(k): v for k, v in preprocessed_data.nominal_mappings.items()}

    if preprocessed_data.ordinal_mappings:
        preprocessed_data.ordinal_mappings = {str(k): v for k, v in preprocessed_data.ordinal_mappings.items()}

    # Extract the appropriate mapping based on the data type
    if annotation_schema.data_type == "nominal":
        mapping = preprocessed_data.nominal_mappings
    elif annotation_schema.data_type == "ordinal":
        mapping = preprocessed_data.ordinal_mappings
    else:
        mapping = None

    reliability_matrix = compute_reliability_matrix(preprocessed_data.df, column_mapping, text_col)

    results = krippendorff_alpha(
        reliability_matrix,
        data_type=annotation_schema.data_type,
        mapping=mapping,
        weight_dict=weight_dict,
        ordinal_scale=ordinal_scale,
    )

    return results
