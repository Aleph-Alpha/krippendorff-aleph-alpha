import pandas as pd
import pytest
import numpy as np
from krippendorff_alpha.preprocessing import preprocess_data, detect_column
from krippendorff_alpha.schema import ColumnMapping, AnnotationSchema, MissingValueStrategyEnum
from krippendorff_alpha.constants import WORD_COLUMN_ALIASES


def test_detect_column() -> None:
    """Test column detection."""
    df = pd.DataFrame(
        {
            "word": ["Sample1", "Sample2"],
            "Annotator1": ["A", "B"],
        }
    )

    assert detect_column(df, WORD_COLUMN_ALIASES) == "word"


def test_preprocess_data_nominal(df_nominal: pd.DataFrame) -> None:
    """Test preprocessing nominal data."""
    column_mapping = ColumnMapping(text_col=None, annotator_cols=["annotator1", "annotator2", "annotator3"])

    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    preprocessed_data, detected_text_col = preprocess_data(df_nominal, column_mapping, annotation_schema)

    assert detected_text_col == "text"
    assert preprocessed_data.df.shape == df_nominal.shape
    assert set(preprocessed_data.df.columns) == set(df_nominal.columns)
    assert len(preprocessed_data.nominal_mappings) > 0


def test_preprocess_word_nominal() -> None:
    """Test preprocessing word-level nominal data."""
    column_mapping = ColumnMapping()

    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="token_level", missing_value_strategy="ignore"
    )
    data = {
        "text": ["it is very cold", "it is warm", "it is hot"],
        "word": ["cold", "warm", "hot"],
        "annotator1": ["temp", "O", "temp"],
        "annotator2": ["temp", "temp", "O"],
        "annotator3": ["temp", "temp", "O"],
    }
    df = pd.DataFrame(data)

    preprocessed_data, detected_text_col = preprocess_data(df, column_mapping, annotation_schema)

    assert detected_text_col == "word"
    assert preprocessed_data.df.shape == df.shape
    assert set(preprocessed_data.df.columns) == set(df.columns)


def test_preprocess_data_ordinal(df_ordinal: pd.DataFrame) -> None:
    """Test preprocessing ordinal data."""
    column_mapping = ColumnMapping(text_col=None, annotator_cols=["annotator1", "annotator2", "annotator3"])

    annotation_schema = AnnotationSchema(
        data_type="ordinal",
        annotation_level="text_level",
        missing_value_strategy="ignore",
    )

    preprocessed_data, detected_text_col = preprocess_data(df_ordinal, column_mapping, annotation_schema)

    assert detected_text_col == "text"
    assert preprocessed_data.df.shape == df_ordinal.shape
    assert set(preprocessed_data.df.columns) == set(df_ordinal.columns)
    assert len(preprocessed_data.ordinal_mappings) > 0


def test_preprocess_data_with_missing_values_ignore() -> None:
    """Test preprocessing with missing value strategy 'ignore'."""
    df = pd.DataFrame({
        "text": ["A", "B", "C"],
        "annotator1": ["positive", "negative", "positive"],
        "annotator2": ["positive", np.nan, "positive"],
        "annotator3": ["positive", "negative", np.nan],
    })
    
    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="nominal",
        annotation_level="text_level",
        missing_value_strategy=MissingValueStrategyEnum.IGNORE,
    )
    
    preprocessed_data, _ = preprocess_data(df, column_mapping, annotation_schema)

    assert preprocessed_data.df.shape == df.shape


def test_preprocess_data_with_missing_values_drop() -> None:
    """Test preprocessing with missing value strategy 'drop'."""
    df = pd.DataFrame({
        "text": ["A", "B", "C"],
        "annotator1": ["positive", "negative", "positive"],
        "annotator2": ["positive", np.nan, "positive"],
        "annotator3": ["positive", "negative", np.nan],
    })
    
    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="nominal",
        annotation_level="text_level",
        missing_value_strategy=MissingValueStrategyEnum.DROP,
    )
    
    preprocessed_data, _ = preprocess_data(df, column_mapping, annotation_schema)

    assert preprocessed_data.df.shape[0] <= df.shape[0]


def test_preprocess_data_with_missing_values_fill() -> None:
    """Test preprocessing with missing value strategy 'fill'."""
    df = pd.DataFrame({
        "text": ["A", "B", "C"],
        "annotator1": ["positive", "negative", "positive"],
        "annotator2": ["positive", np.nan, "positive"],
        "annotator3": ["positive", "negative", np.nan],
    })
    
    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="nominal",
        annotation_level="text_level",
        missing_value_strategy=MissingValueStrategyEnum.FILL,
    )
    
    preprocessed_data, _ = preprocess_data(df, column_mapping, annotation_schema)

    assert preprocessed_data.df.shape == df.shape
    assert not preprocessed_data.df[column_mapping.annotator_cols].isna().any().any()


def test_preprocess_data_no_text_column() -> None:
    """Test preprocessing when text column cannot be detected."""
    df = pd.DataFrame({
        "col1": ["A", "B", "C"],
        "annotator1": ["positive", "negative", "positive"],
        "annotator2": ["positive", "negative", "positive"],
        "annotator3": ["positive", "negative", "positive"],
    })
    
    column_mapping = ColumnMapping(text_col=None, annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="nominal",
        annotation_level="text_level",
        missing_value_strategy="ignore",
    )
    
    with pytest.raises(ValueError, match="Could not detect a valid text column"):
        preprocess_data(df, column_mapping, annotation_schema)
