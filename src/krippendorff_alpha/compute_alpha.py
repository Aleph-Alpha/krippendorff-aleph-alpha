from typing import Any
import pandas as pd
import logging
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
from krippendorff_alpha.constants import MIN_ANNOTATORS_REQUIRED, load_custom_config, reset_config
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_alpha(
    df: pd.DataFrame,
    data_type: str,
    column_mapping: ColumnMapping | dict[str, Any] | None = None,
    annotation_level: str = AnnotationLevelEnum.TEXT_LEVEL,
    weight_dict: dict[str, float] | None = None,
    ordinal_scale: list[int | float | str] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
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
    - config_path (Optional[str | Path]): Path to a custom YAML configuration file. If None, uses default English config.

    Returns:
    - dict[str, Any]: A dictionary containing Krippendorff's alpha, observed and expected disagreement, and per-category scores.
    """

    if df is None:
        raise ValueError("A valid DataFrame (df) must be provided.")

    custom_config = None
    if config_path is not None:
        custom_config = load_custom_config(config_path)

    if column_mapping is None:
        inferred_text_col = next((col for col in df.columns if df[col].dtype == "object"), None)
        inferred_annotator_cols = [col for col in df.columns if col != inferred_text_col]
        if len(inferred_annotator_cols) < MIN_ANNOTATORS_REQUIRED:
            raise ValueError(
                f"At least {MIN_ANNOTATORS_REQUIRED} annotator columns are required for reliability assessment."
            )

        column_mapping = ColumnMapping(text_col=inferred_text_col, annotator_cols=inferred_annotator_cols)

    elif isinstance(column_mapping, dict):
        column_mapping = ColumnMapping(**column_mapping)

    try:
        data_type_enum = DataTypeEnum(data_type.lower())
    except ValueError:
        raise ValueError(f"Invalid data_type '{data_type}'. Must be one of {[e.value for e in DataTypeEnum]}")

    annotation_schema = AnnotationSchema(
        data_type=data_type_enum,
        annotation_level=annotation_level,
        missing_value_strategy=MissingValueStrategyEnum.IGNORE,
    )

    preprocessed_data, text_col = preprocess_data(df, column_mapping, annotation_schema, custom_config)

    if preprocessed_data.nominal_mappings:
        logger.debug(f"Nominal mappings: {preprocessed_data.nominal_mappings}")
        preprocessed_data.nominal_mappings = {str(k): v for k, v in preprocessed_data.nominal_mappings.items()}

    if preprocessed_data.ordinal_mappings:
        preprocessed_data.ordinal_mappings = {str(k): v for k, v in preprocessed_data.ordinal_mappings.items()}

    if preprocessed_data.annotation_schema.data_type == DataTypeEnum.NOMINAL:
        mapping = preprocessed_data.nominal_mappings
    elif preprocessed_data.annotation_schema.data_type == DataTypeEnum.ORDINAL:
        mapping = preprocessed_data.ordinal_mappings
    else:
        mapping = None

    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, text_col, custom_config
    )

    results = krippendorff_alpha(
        reliability_matrix,
        data_type=data_type_enum,
        mapping=mapping,
        weight_dict=weight_dict,
        ordinal_scale=ordinal_scale,
    )

    if results.get("per_category_scores") is None:
        results.pop("per_category_scores", None)

    if config_path is not None:
        reset_config()

    return results
