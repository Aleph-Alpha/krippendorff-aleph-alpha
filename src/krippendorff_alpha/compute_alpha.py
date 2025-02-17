from typing import Optional, Any, Dict, List, Union
import pandas as pd
import json
from krippendorff_alpha.metric import krippendorff_alpha
from krippendorff_alpha.preprocessing import preprocess_data
from krippendorff_alpha.reliability import compute_reliability_matrix
from krippendorff_alpha.schema import (
    ColumnMapping,
    AnnotationSchema,
    AnnotationLevelEnum,
    MissingValueStrategyEnum,
    DataTypeEnum,
)


def compute_alpha(
    df: pd.DataFrame,
    data_type: str,
    column_mapping: Optional[Union[ColumnMapping, Dict[str, Any]]] = None,
    annotation_level: str = AnnotationLevelEnum.TEXT_LEVEL,
    weight_dict: Optional[Dict[str, float]] = None,
    ordinal_scale: Optional[List[Union[int, float, str]]] = None,
) -> Any:
    """
    Computes Krippendorff's alpha for inter-annotator agreement.

    Parameters:
    - df (pd.DataFrame): The dataframe containing annotation data.
    - data_type (str): The type of annotation data (e.g., "nominal" or "ordinal").
    - column_mapping (Optional[ColumnMapping]): Specifies which columns correspond to annotators and text.
      If None, it will be inferred automatically.
    - annotation_level (str, default="text_level"): The level of annotation (e.g., "text_level", "token_level").
    - weight_dict (Optional[Dict[str, float]]): A dictionary specifying weights for individual annotators (if applicable).
    - ordinal_scale: Optional[List[Union[int, float, str]]]: A list defining an ordinal scale if the data type is ordinal.

    Returns:
    - Any: A dictionary containing Krippendorff's alpha, observed and expected disagreement, and per-category scores.
    """

    if df is None:
        raise ValueError("A valid DataFrame (df) must be provided.")

    if column_mapping is None:
        inferred_text_col = next((col for col in df.columns if df[col].dtype == "object"), None)
        inferred_annotator_cols = [col for col in df.columns if col != inferred_text_col]
        if len(inferred_annotator_cols) < 3:
            raise ValueError("At least three annotator columns are required for reliability assessment.")

        column_mapping = ColumnMapping(text_col=inferred_text_col, annotator_cols=inferred_annotator_cols)

    elif isinstance(column_mapping, dict):
        column_mapping = ColumnMapping(**column_mapping)

    try:
        data_type_enum = DataTypeEnum(data_type.lower())
    except ValueError:
        raise ValueError(f"Invalid data_type '{data_type}'. Must be one of {[e.value for e in DataTypeEnum]}")

    # Create annotation schema with user-defined `data_type` and default `annotation_level`
    annotation_schema = AnnotationSchema(
        data_type=data_type_enum,
        annotation_level=annotation_level,
        missing_value_strategy=MissingValueStrategyEnum.IGNORE,  # Default to IGNORE
    )

    # Let `preprocess_data` handle column mapping inference
    preprocessed_data, text_col = preprocess_data(df, column_mapping, annotation_schema)

    # Convert mappings to string keys (for compatibility)
    if preprocessed_data.nominal_mappings:
        print(preprocessed_data.nominal_mappings)
        preprocessed_data.nominal_mappings = {str(k): v for k, v in preprocessed_data.nominal_mappings.items()}

    if preprocessed_data.ordinal_mappings:
        preprocessed_data.ordinal_mappings = {str(k): v for k, v in preprocessed_data.ordinal_mappings.items()}

    # Select appropriate mapping
    mapping = (
        preprocessed_data.nominal_mappings
        if preprocessed_data.annotation_schema.data_type == "nominal"
        else preprocessed_data.ordinal_mappings
        if preprocessed_data.annotation_schema.data_type == "ordinal"
        else None
    )

    # Compute reliability matrix
    reliability_matrix = compute_reliability_matrix(preprocessed_data.df, preprocessed_data.column_mapping, text_col)

    # Compute Krippendorff's alpha
    results = krippendorff_alpha(
        reliability_matrix,
        data_type=data_type_enum,
        mapping=mapping,
        weight_dict=weight_dict,
        ordinal_scale=ordinal_scale,
    )

    if results.get("per_category_scores") is None:
        del results["per_category_scores"]

    return json.dumps(results, indent=4)
