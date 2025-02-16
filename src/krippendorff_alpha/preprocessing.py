import pandas as pd
import logging
from typing import Optional, List, Tuple, Any

from krippendorff_alpha.constants import (
    TEXT_COLUMN_ALIASES,
    WORD_COLUMN_ALIASES,
    ORDINAL_CATEGORIES,
    ANNOTATOR_REGEX,
)
from krippendorff_alpha.schema import (
    ColumnMapping,
    PreprocessedData,
    AnnotationSchema,
    DataTypeEnum,
    AnnotationLevelEnum,
    MissingValueStrategyEnum,
)

logging.basicConfig(level=logging.INFO)


def detect_column(df: pd.DataFrame, column_aliases: set[str]) -> Optional[str]:
    matches = [col for col in df.columns if col.lower().strip() in {name.lower() for name in column_aliases}]
    return matches[0] if matches else None


def detect_annotator_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if ANNOTATOR_REGEX.match(col)]


def create_global_mapping(
    df: pd.DataFrame, annotator_cols: List[str], data_type: str
) -> dict[Any, int] | dict[Any, str]:
    """Creates a unified mapping across all annotator columns to ensure consistency."""
    unique_values = set()
    for col in annotator_cols:
        unique_values.update(df[col].dropna().unique())

    sorted_unique_values = list(sorted(unique_values, key=str))  # Default sorted order

    if data_type == DataTypeEnum.ORDINAL:
        # Normalize dataset labels (lowercase for comparison)
        normalized_labels = {label.lower(): label for label in sorted_unique_values}
        dataset_labels_lower = set(normalized_labels.keys())

        # Find the best matching predefined ordinal scale
        for ordinal_scale in ORDINAL_CATEGORIES:
            ordinal_scale_lower = [label.lower() for label in ordinal_scale]
            if dataset_labels_lower.issubset(set(ordinal_scale_lower)):
                return {label: ordinal_scale_lower.index(label.lower()) for label in sorted_unique_values}

    # Fallback: Default to sorted order if no match is found
    return {label: i for i, label in enumerate(sorted_unique_values)}


def preprocess_data(
    df: pd.DataFrame,
    column_mapping: ColumnMapping,
    annotation_schema: AnnotationSchema,
) -> Tuple[PreprocessedData, str]:
    if df is None:
        raise ValueError("A DataFrame must be provided.")

    annotator_cols = column_mapping.annotator_cols or detect_annotator_columns(df)
    column_mapping.annotator_cols = annotator_cols
    text_col = column_mapping.text_col or detect_column(
        df,
        TEXT_COLUMN_ALIASES
        if annotation_schema.annotation_level == AnnotationLevelEnum.TEXT_LEVEL
        else WORD_COLUMN_ALIASES,
    )

    if text_col is None:
        raise ValueError("Could not detect a valid text column. Please specify it in column_mapping.")

    # Generate a global mapping for all annotator columns
    global_mapping = create_global_mapping(df, annotator_cols, annotation_schema.data_type)

    # Apply the mapping to all annotator columns
    for col in annotator_cols:
        df[col] = df[col].map(global_mapping).fillna(-1).astype(int)

    # Store mappings in the correct place
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
