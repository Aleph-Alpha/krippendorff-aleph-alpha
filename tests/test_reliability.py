import pytest
import os
import pandas as pd

from src.krippendorff_alpha.constants import WORD_COLUMN_ALIASES, TEXT_COLUMN_ALIASES
from src.krippendorff_alpha.reliability import compute_reliability_matrix
from src.krippendorff_alpha.preprocessing import preprocess_data


@pytest.mark.parametrize(
    "filename, data_type",
    [
        ("interval_numeric_equalGaps_noAbsoluteZero.jsonl", "interval"),
        ("nominal_categorical_noOrder_sample.json", "nominal"),
        ("ordinal_orderedCategories_unequalGaps_sample.csv", "ordinal"),
        ("ratio_numeric_equalGaps_withAbsoluteZero.tsv", "ratio"),
    ],
)
def test_compute_reliability_matrix(example_data: str, filename: str, data_type: str) -> None:
    path: str = os.path.join(example_data, filename)

    print(f"\n--- Testing {filename} ({data_type}) ---")

    preprocessed = preprocess_data(path)
    df = preprocessed.df
    column_mapping = preprocessed.column_mapping

    print("\nOriginal Preprocessed DataFrame:")
    print(df.head())

    reliability_matrix = compute_reliability_matrix(df, column_mapping)

    print("\nTransformed Reliability Matrix:")
    print(reliability_matrix.head())

    assert isinstance(reliability_matrix, pd.DataFrame), "Output should be a DataFrame"
    assert not reliability_matrix.empty, "Reliability matrix should not be empty"
    assert set(reliability_matrix.index) == set(column_mapping.annotator_cols), "Rows should match annotators"
    expected_index_col = next((col for col in WORD_COLUMN_ALIASES if col in df.columns), None)
    if not expected_index_col:
        expected_index_col = next((col for col in TEXT_COLUMN_ALIASES if col in df.columns), None)

    assert expected_index_col is not None, "Neither word nor text column found in df."

    expected_columns = set(df[expected_index_col])

    assert set(reliability_matrix.columns) == expected_columns, (
        "Columns should match the expected word or text column values."
    )
