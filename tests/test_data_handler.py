import pytest
import pandas as pd
from krippendorff_alpha.data_handler import load_data, process_dataframe
from krippendorff_alpha.schema import DataType, DataSchema, AnnotationSchema
from typing import Dict, Any


@pytest.mark.parametrize(
    "file_type,data_type",
    [
        ("json", DataType.NOMINAL),
        ("jsonl", DataType.INTERVAL),
        ("csv", DataType.ORDINAL),
        ("tsv", DataType.RATIO),
    ],
)
def test_load_data(example_files: Dict[str, str], file_type: str, data_type: DataType) -> None:
    """Test loading data from different formats."""
    file_path: str = example_files[file_type]

    # Test output as DataSchema
    data: DataSchema = load_data(file_path, data_type, as_dataframe=False)
    assert isinstance(data, DataSchema), f"Expected DataSchema, got {type(data)}"
    assert len(data.annotations) > 0, "Annotations should not be empty"
    assert all(isinstance(a, AnnotationSchema) for a in data.annotations), "All annotations must be AnnotationSchema"

    # Test output as DataFrame
    df: pd.DataFrame = load_data(file_path, data_type, as_dataframe=True)
    assert not df.empty, "DataFrame should not be empty"
    assert "annotator_id" in df.columns, "Missing 'annotator_id' column"
    assert "label" in df.columns, "Missing 'label' column"


def test_flexible_column_names() -> None:
    """Test that different text/annotation column names are handled."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "Text": ["example 1", "example 2"],
            "Annotator_A": ["yes", "no"],
            "Annotator_B": ["no", "yes"],
            "Annotator_C": ["maybe", "maybe"],
        }
    )

    processed: pd.DataFrame = process_dataframe(df, DataType.NOMINAL, as_dataframe=True)
    assert not processed.empty, "Processed DataFrame should not be empty"
    assert "annotator_id" in processed.columns, "Missing 'annotator_id' column"
    assert "label" in processed.columns, "Missing 'label' column"


def test_categorical_to_numeric() -> None:
    """Test conversion of categorical labels to numeric indices."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "text": ["sample 1", "sample 2"],
            "annotator_1": ["happy", "sad"],
            "annotator_2": ["sad", "happy"],
            "annotator_3": ["sad", "happy"],
        }
    )
    processed: pd.DataFrame = process_dataframe(df, DataType.NOMINAL, as_dataframe=True)
    assert processed["label"].dtype in [int, float], "Labels should be numeric"


def test_handle_missing_values() -> None:
    """Ensure missing values (NaN) are handled properly."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "text": ["sample 1", "sample 2"],
            "annotator_1": ["happy", None],
            "annotator_2": [None, "sad"],
            "annotator_3": ["sad", "happy"],
        }
    )
    processed: pd.DataFrame = process_dataframe(df, DataType.NOMINAL, as_dataframe=True)
    assert processed["label"].isna().sum() > 0, "Missing values should be preserved"


def test_dynamic_annotators() -> None:
    """Test that new annotators can be added dynamically."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "text": ["sample 1", "sample 2"],
            "annotator_1": ["yes", "no"],
            "annotator_2": ["yes", "maybe"],
            "annotator_3": ["no", "yes"],
        }
    )
    processed1: pd.DataFrame = process_dataframe(df, DataType.NOMINAL, as_dataframe=True)
    df["annotator_4"] = ["maybe", "yes"]  # New annotator added
    processed2: pd.DataFrame = process_dataframe(df, DataType.NOMINAL, as_dataframe=True)
    assert len(processed2) > len(processed1), "New annotations should be added without overwriting existing ones"


def test_annotation_integrity() -> None:
    """Ensure existing annotations remain unchanged when adding new data."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "text": ["item 1", "item 2"],
            "annotator_1": ["happy", "sad"],
            "annotator_2": ["sad", "sad"],
            "annotator_3": ["sad", "happy"],
        }
    )
    processed1: pd.DataFrame = process_dataframe(df, DataType.NOMINAL, as_dataframe=True)
    original_labels = processed1[processed1["annotator_id"] == "annotator_1"]["label"].tolist()
    df["annotator_4"] = ["neutral", "angry"]  # Add new annotator
    processed2: pd.DataFrame = process_dataframe(df, DataType.NOMINAL, as_dataframe=True)
    new_labels = processed2[processed2["annotator_id"] == "annotator_1"]["label"].tolist()
    assert original_labels == new_labels, "Existing annotations must not change"


def test_process_efficiency(benchmark: Any) -> None:
    """Benchmark processing efficiency with large data."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "text": [f"sample {i}" for i in range(1000)],
            "annotator_1": ["yes" if i % 2 == 0 else "no" for i in range(1000)],
            "annotator_2": ["no" if i % 2 == 0 else "yes" for i in range(1000)],
            "annotator_3": ["no" if i % 2 == 0 else "yes" for i in range(1000)],
        }
    )
    benchmark(lambda: process_dataframe(df, DataType.NOMINAL, as_dataframe=True))
