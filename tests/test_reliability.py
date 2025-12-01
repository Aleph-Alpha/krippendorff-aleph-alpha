import logging
import pandas as pd
import numpy as np
import pytest

from krippendorff_alpha.schema import ColumnMapping, AnnotationSchema
from krippendorff_alpha.reliability import compute_reliability_matrix
from krippendorff_alpha.preprocessing import preprocess_data

logging.basicConfig(level=logging.INFO)


def test_compute_reliability_matrix(df_nominal: pd.DataFrame) -> None:
    """Test reliability matrix computation for nominal data."""
    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    column_mapping = ColumnMapping(text_col=None, annotator_cols=["annotator1", "annotator2", "annotator3"])

    preprocessed_data, detected_text_col = preprocess_data(df_nominal, column_mapping, annotation_schema)

    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    assert reliability_matrix.shape == (3, 3)
    assert list(reliability_matrix.columns) == ["Hello world", "Goodbye world", "It is sunny"]
    assert list(reliability_matrix.index) == ["annotator1", "annotator2", "annotator3"]


def test_compute_reliability_ordinal(df_ordinal: pd.DataFrame) -> None:
    """Test reliability matrix computation for ordinal data."""
    annotation_schema = AnnotationSchema(
        data_type="ordinal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    column_mapping = ColumnMapping(text_col=None, annotator_cols=["annotator1", "annotator2", "annotator3"])

    preprocessed_data, detected_text_col = preprocess_data(df_ordinal, column_mapping, annotation_schema)

    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    assert reliability_matrix.shape == (3, 3)
    assert list(reliability_matrix.columns) == ["it is very cold", "it is warm", "it is hot"]
    assert list(reliability_matrix.index) == ["annotator1", "annotator2", "annotator3"]


def test_compute_reliability_matrix_with_missing_data() -> None:
    """Test reliability matrix computation with missing data."""
    df = pd.DataFrame(
        {
            "text": ["A", "B", "C"],
            "annotator1": ["positive", "negative", "positive"],
            "annotator2": ["positive", np.nan, "positive"],
            "annotator3": ["positive", "negative", np.nan],
        }
    )

    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])

    preprocessed_data, detected_text_col = preprocess_data(df, column_mapping, annotation_schema)

    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    assert reliability_matrix.shape == (3, 3)
    assert list(reliability_matrix.index) == ["annotator1", "annotator2", "annotator3"]


def test_compute_reliability_matrix_missing_columns() -> None:
    """Test reliability matrix computation with missing columns."""
    df = pd.DataFrame(
        {
            "text": ["A", "B", "C"],
            "annotator1": ["positive", "negative", "positive"],
            "annotator2": ["positive", "negative", "positive"],
            "annotator3": ["positive", "negative", "positive"],
        }
    )

    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator4"])

    with pytest.raises(KeyError, match="annotator4"):
        preprocess_data(df, column_mapping, annotation_schema)


def test_compute_reliability_matrix_auto_detect() -> None:
    """Test reliability matrix computation with auto-detected columns."""
    df = pd.DataFrame(
        {
            "text": ["A", "B", "C"],
            "annotator1": ["positive", "negative", "positive"],
            "annotator2": ["positive", "negative", "positive"],
            "annotator3": ["positive", "negative", "positive"],
        }
    )

    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    column_mapping = ColumnMapping()

    preprocessed_data, detected_text_col = preprocess_data(df, column_mapping, annotation_schema)

    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    assert reliability_matrix.shape[0] == 3
    assert reliability_matrix.shape[1] == 3
