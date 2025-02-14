import pandas as pd
from src.krippendorff_alpha.preprocessing import detect_column, preprocess_data
from src.krippendorff_alpha.schema import ColumnMapping, AnnotationSchema
from src.krippendorff_alpha.constants import TEXT_COLUMN_ALIASES, WORD_COLUMN_ALIASES


def test_detect_column():
    df = pd.DataFrame(
        {
            "Text": ["Sample sentence 1", "Sample sentence 2"],
            "Annotator1": ["A", "B"],
        }
    )

    assert detect_column(df, TEXT_COLUMN_ALIASES) == "Text"
    assert detect_column(df, WORD_COLUMN_ALIASES) is None  # No match


def test_preprocess_data_nominal():
    df = pd.DataFrame(
        {
            "text": ["Hello world", "Goodbye world"],
            "annotator1": ["positive", "negative"],
            "annotator2": ["negative", "positive"],
            "annotator3": ["neutral", "negative"],
        }
    )

    column_mapping = ColumnMapping(text_col=None, annotator_cols=["annotator1", "annotator2", "annotator3"])

    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    preprocessed_data, detected_text_col = preprocess_data(df, column_mapping, annotation_schema)

    print("Detected text column:", detected_text_col)
    print("Preprocessed DataFrame:")
    print(preprocessed_data.df)
    print("Ordinal mappings:", preprocessed_data.ordinal_mappings)
    print("Nominal mappings:", preprocessed_data.nominal_mappings)

    assert detected_text_col == "text"
    assert preprocessed_data.df.shape == df.shape
    assert set(preprocessed_data.df.columns) == set(df.columns)


def test_preprocess_data_ordinal():
    df = pd.DataFrame(
        {
            "text": ["it is very cold", "it is warm"],
            "annotator1": ["low", "medium"],
            "annotator2": ["low", "high"],
            "annotator3": ["medium", "very high"],
        }
    )

    column_mapping = ColumnMapping(text_col=None, annotator_cols=["annotator1", "annotator2", "annotator3"])

    annotation_schema = AnnotationSchema(
        data_type="ordinal",  # 👈 User just enters a string now!
        annotation_level="text_level",
        missing_value_strategy="ignore",
    )

    preprocessed_data, detected_text_col = preprocess_data(df, column_mapping, annotation_schema)

    print("Detected text column:", detected_text_col)
    print("Preprocessed DataFrame:")
    print(preprocessed_data.df)
    print("Ordinal mappings:", preprocessed_data.ordinal_mappings)
    print("Nominal mappings:", preprocessed_data.nominal_mappings)

    assert detected_text_col == "text"
    assert preprocessed_data.df.shape == df.shape
    assert set(preprocessed_data.df.columns) == set(df.columns)
