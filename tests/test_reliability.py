import pandas as pd
import logging
from src.krippendorff_alpha.schema import (
    ColumnMapping,
    AnnotationSchema,
)
from src.krippendorff_alpha.reliability import compute_reliability_matrix
from src.krippendorff_alpha.preprocessing import preprocess_data

logging.basicConfig(level=logging.INFO)


def test_compute_reliability_matrix():
    # Create a sample DataFrame
    data = {
        "text": ["Hello world", "Goodbye world", "It is sunny"],
        "annotator1": ["positive", "negative", "positive"],
        "annotator2": ["negative", "positive", "negative"],
        "annotator3": ["positive", "negative", "positive"],
    }
    df = pd.DataFrame(data)

    # Define annotation schema

    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    # Preprocess data (auto-detects text column and annotator columns)
    preprocessed_data, detected_text_col = preprocess_data(df, ColumnMapping(), annotation_schema)

    # Print preprocessed DataFrame
    print("Preprocessed DataFrame:")
    print(preprocessed_data.df)

    # Compute reliability matrix without explicitly passing column_mapping
    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    # Print output reliability matrix
    print("Reliability Matrix:")
    print(reliability_matrix)

    # Assertions
    assert reliability_matrix.shape == (3, 3)  # Expecting 3 annotators x 3 texts
    assert list(reliability_matrix.columns) == ["Hello world", "Goodbye world", "It is sunny"]
    assert list(reliability_matrix.index) == ["annotator1", "annotator2", "annotator3"]

    print("Test passed: compute_reliability_matrix works as expected!")
