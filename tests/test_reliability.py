import logging

import pandas as pd

from src.krippendorff_alpha.schema import (
    ColumnMapping,
    AnnotationSchema,
)
from src.krippendorff_alpha.reliability import compute_reliability_matrix
from src.krippendorff_alpha.preprocessing import preprocess_data

logging.basicConfig(level=logging.INFO)


def test_compute_reliability_matrix(df_nominal: pd.DataFrame) -> None:
    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    column_mapping = ColumnMapping(text_col=None, annotator_cols=["annotator1", "annotator2", "annotator3"])

    preprocessed_data, detected_text_col = preprocess_data(df_nominal, column_mapping, annotation_schema)

    print("Preprocessed DataFrame:")
    print(preprocessed_data.df)

    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    print("Reliability Matrix:")
    print(reliability_matrix)

    assert reliability_matrix.shape == (3, 3)
    assert list(reliability_matrix.columns) == ["Hello world", "Goodbye world", "It is sunny"]
    assert list(reliability_matrix.index) == ["annotator1", "annotator2", "annotator3"]

    print("Test passed: compute_reliability_matrix works as expected!")


def test_compute_reliability_ordinal(df_ordinal: pd.DataFrame) -> None:
    annotation_schema = AnnotationSchema(
        data_type="ordinal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    column_mapping = ColumnMapping(text_col=None, annotator_cols=["annotator1", "annotator2", "annotator3"])

    # Preprocess data (auto-detects text column and annotator columns)
    preprocessed_data, detected_text_col = preprocess_data(df_ordinal, column_mapping, annotation_schema)

    print("Preprocessed DataFrame:")
    print(preprocessed_data.df)

    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    print("Reliability Matrix:")
    print(reliability_matrix)

    assert reliability_matrix.shape == (3, 3)
    assert list(reliability_matrix.columns) == ["it is very cold", "it is warm", "it is hot"]
    assert list(reliability_matrix.index) == ["annotator1", "annotator2", "annotator3"]

    print("Test passed: compute_reliability_matrix works as expected!")
