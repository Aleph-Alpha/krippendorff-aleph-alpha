from typing import List, Tuple

import pandas as pd
import json
from pathlib import Path
from src.krippendorff_alpha.schema import DataTypeEnum, PreprocessedData
from src.krippendorff_alpha.constants import TEXT_COLUMN_ALIASES
from src.krippendorff_alpha.preprocessing import (
    detect_column,
    detect_annotator_columns,
    load_data,
    flatten_json,
    flatten_jsonl,
    infer_annotation_type,
    convert_ordinal_to_numeric,
    convert_nominal_to_numeric,
    preprocess_data,
)


def test_detect_column() -> None:
    df = pd.DataFrame({"text": ["sample text"], "annotator_1": [1]})
    print("\nDetect Column - DataFrame:\n", df)

    detected_col = detect_column(df, TEXT_COLUMN_ALIASES)
    assert detected_col == "text"

    detected_col = detect_column(df, {"nonexistent"})
    assert detected_col is None


def test_detect_annotator_columns() -> None:
    df = pd.DataFrame({"annotator_1": [1], "annotator_2": [2], "annotator_3": [3]})
    print("\nDetect Annotator Columns - DataFrame:\n", df)

    detected_cols = detect_annotator_columns(df)
    assert detected_cols == ["annotator_1", "annotator_2", "annotator_3"]


def test_load_data(example_data: Path) -> None:
    # Test loading a CSV file
    csv_path = Path(example_data) / "ordinal_orderedCategories_unequalGaps_sample.csv"
    df = load_data(str(csv_path))
    print("\nLoaded CSV DataFrame:\n", df.head())
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Test loading a JSON file
    json_path = Path(example_data) / "nominal_categorical_noOrder_sample.json"
    df = load_data(str(json_path))
    print("\nLoaded JSON DataFrame:\n", df.head())
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Test loading a JSONL file
    jsonl_path = Path(example_data) / "interval_numeric_equalGaps_noAbsoluteZero.jsonl"
    df = load_data(str(jsonl_path))
    print("\nLoaded JSONL DataFrame:\n", df.head())
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Test loading a TSV file
    tsv_path = Path(example_data) / "ratio_numeric_equalGaps_withAbsoluteZero.tsv"
    df = load_data(str(tsv_path))
    print("\nLoaded TSV DataFrame:\n", df.head())
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_flatten_json(example_data: Path) -> None:
    json_path = Path(example_data) / "nominal_categorical_noOrder_sample.json"
    with open(json_path, "r") as f:
        json_data = json.load(f)
    df = flatten_json(json_data)
    print("\nFlattened JSON DataFrame:\n", df.head())
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_flatten_jsonl(example_data: Path) -> None:
    jsonl_path = Path(example_data) / "interval_numeric_equalGaps_noAbsoluteZero.jsonl"
    with open(jsonl_path, "r") as f:
        jsonl_data = [json.loads(line) for line in f]
    df = flatten_jsonl(jsonl_data)
    print("\nFlattened JSONL DataFrame:\n", df.head())
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_infer_annotation_type(datasets: List[Tuple[str, str]], example_data: Path) -> None:
    for file_name, expected_type in datasets:
        file_path = Path(example_data) / file_name
        df = load_data(str(file_path))
        print(f"\nInfer Annotation Type - DataFrame ({file_name}):\n", df.head())

        annotator_cols = detect_annotator_columns(df)
        for col in annotator_cols:
            inferred_type = infer_annotation_type(df[col])
            print(f"Column: {col}, Inferred Type: {inferred_type}")
            assert inferred_type == DataTypeEnum[expected_type.upper()]


def test_convert_ordinal_to_numeric(example_data: Path) -> None:
    csv_path = Path(example_data) / "ordinal_orderedCategories_unequalGaps_sample.csv"
    df = load_data(str(csv_path))
    print("\nOriginal Ordinal DataFrame:\n", df.head())

    annotator_cols = detect_annotator_columns(df)
    for col in annotator_cols:
        converted = convert_ordinal_to_numeric(df[col])
        print(f"\nConverted Ordinal Column '{col}' to Numeric:\n", converted.head())
        assert all(isinstance(x, (int, float)) for x in converted)


def test_convert_nominal_to_numeric(example_data: Path) -> None:
    json_path = Path(example_data) / "nominal_categorical_noOrder_sample.json"
    df = load_data(str(json_path))
    print("\nOriginal Nominal DataFrame:\n", df.head())

    annotator_cols = detect_annotator_columns(df)
    for col in annotator_cols:
        converted = convert_nominal_to_numeric(df[col], col)
        print(f"\nConverted Nominal Column '{col}' to Numeric:\n", converted.head())
        assert all(isinstance(x, int) for x in converted)


def test_preprocess_data(datasets: List[Tuple[str, str]], example_data: Path) -> None:
    for file_name, _ in datasets:
        file_path = Path(example_data) / file_name
        preprocessed = preprocess_data(path=str(file_path))
        print(f"\nPreprocessed DataFrame ({file_name}):\n", preprocessed.df.head())
        assert isinstance(preprocessed, PreprocessedData)
        assert len(preprocessed.df) > 0
