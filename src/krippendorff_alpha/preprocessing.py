import pandas as pd
import logging
from typing import Any

from krippendorff_alpha.constants import (
    get_text_column_aliases,
    get_word_column_aliases,
    get_ordinal_categories,
    get_annotator_regex,
)
from krippendorff_alpha.schema import (
    ColumnMapping,
    PreprocessedData,
    AnnotationSchema,
    DataTypeEnum,
    AnnotationLevelEnum,
    MissingValueStrategyEnum,
)

logger = logging.getLogger(__name__)


def detect_column(df: pd.DataFrame, column_aliases: set[str]) -> str | None:
    matches = [col for col in df.columns if col.lower().strip() in {name.lower() for name in column_aliases}]
    return matches[0] if matches else None


def detect_annotator_columns(df: pd.DataFrame, custom_config: dict[str, Any] | None = None) -> list[str]:
    annotator_regex = get_annotator_regex(custom_config)
    return [col for col in df.columns if annotator_regex.match(col)]


def create_global_mapping(
    df: pd.DataFrame,
    annotator_cols: list[str],
    data_type: DataTypeEnum | str,
    custom_config: dict[str, Any] | None = None,
) -> dict[Any, int]:
    """Creates a unified mapping across all annotator columns to ensure consistency."""
    unique_values = set()
    for col in annotator_cols:
        unique_values.update(df[col].dropna().unique())

    sorted_unique_values = list(sorted(unique_values, key=str))

    if isinstance(data_type, str):
        try:
            data_type_enum = DataTypeEnum(data_type.lower())
        except ValueError:
            data_type_enum = None
    else:
        data_type_enum = data_type

    if data_type_enum == DataTypeEnum.ORDINAL:
        normalized_labels = {label.lower(): label for label in sorted_unique_values}
        dataset_labels_lower = set(normalized_labels.keys())

        ordinal_categories = get_ordinal_categories(custom_config)
        for ordinal_scale in ordinal_categories:
            ordinal_scale_lower = [label.lower() for label in ordinal_scale]
            if dataset_labels_lower.issubset(set(ordinal_scale_lower)):
                return {label: ordinal_scale_lower.index(label.lower()) for label in sorted_unique_values}

    return {label: i for i, label in enumerate(sorted_unique_values)}


def preprocess_data(
    df: pd.DataFrame,
    column_mapping: ColumnMapping,
    annotation_schema: AnnotationSchema,
    custom_config: dict[str, Any] | None = None,
) -> tuple[PreprocessedData, str]:
    """
    Preprocesses annotation data by detecting relevant columns, mapping categorical labels to numeric values,
    and handling missing values based on the specified annotation schema.

    Args:
        df (pd.DataFrame): The input DataFrame containing annotation data.
        column_mapping (ColumnMapping): Object containing mappings for text and annotator columns.
        annotation_schema (AnnotationSchema): Object defining the annotation level, data type, and missing value strategy.

    Returns:
        Tuple[PreprocessedData, str]: A tuple containing the preprocessed data and detected text column name.
    """
    if df is None:
        raise ValueError("A DataFrame must be provided.")

    df = df.copy()

    column_mapping = (
        column_mapping.model_copy()
        if hasattr(column_mapping, "model_copy")
        else ColumnMapping(text_col=column_mapping.text_col, annotator_cols=column_mapping.annotator_cols)
    )

    annotator_cols = column_mapping.annotator_cols or detect_annotator_columns(df, custom_config)
    column_mapping.annotator_cols = annotator_cols
    text_col_aliases = (
        get_text_column_aliases(custom_config)
        if annotation_schema.annotation_level == AnnotationLevelEnum.TEXT_LEVEL
        else get_word_column_aliases(custom_config)
    )
    text_col = column_mapping.text_col or detect_column(df, text_col_aliases)

    if text_col is None:
        raise ValueError("Could not detect a valid text column. Please specify it in column_mapping.")

    global_mapping = create_global_mapping(df, annotator_cols, annotation_schema.data_type.value, custom_config)

    for col in annotator_cols:
        df[col] = df[col].map(global_mapping).fillna(-1).astype(int)

    ordinal_mappings = global_mapping if annotation_schema.data_type == DataTypeEnum.ORDINAL else {}
    nominal_mappings = global_mapping if annotation_schema.data_type == DataTypeEnum.NOMINAL else {}

    # Handle missing values
    if annotation_schema.missing_value_strategy == MissingValueStrategyEnum.DROP:
        df = df.dropna(subset=annotator_cols)
    elif annotation_schema.missing_value_strategy == MissingValueStrategyEnum.FILL:
        df[annotator_cols] = df[annotator_cols].fillna(-1)

    return PreprocessedData(
        df=df,
        column_mapping=column_mapping,
        annotation_schema=annotation_schema,
        ordinal_mappings=ordinal_mappings,
        nominal_mappings=nominal_mappings,
    ), text_col
