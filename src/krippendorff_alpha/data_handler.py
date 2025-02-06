import pandas as pd
import json
from pathlib import Path
from typing import Union
from .schemas import DataSchema, AnnotationSchema, DataType


def load_data(
    file_path: Union[str, Path], data_type: DataType, as_dataframe: bool = False
) -> Union[DataSchema, pd.DataFrame]:
    """Load data from CSV, TSV, JSON, or JSONL and validate it."""
    file_path = Path(file_path)

    if file_path.suffix in {".csv", ".tsv"}:
        sep = "\t" if file_path.suffix == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)
    elif file_path.suffix == ".json":
        df = pd.DataFrame(json.load(file_path.open("r")))
    elif file_path.suffix == ".jsonl":
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError("Unsupported file format. Use CSV, TSV, JSON, or JSONL.")

    return process_dataframe(df, data_type, as_dataframe)


def process_dataframe(
    df: pd.DataFrame, data_type: DataType, as_dataframe: bool = False
) -> Union[DataSchema, pd.DataFrame]:
    """Process and validate a Pandas DataFrame."""

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.isnull().all().any():
        raise ValueError("Dataset contains only NaN values.")

    text_column = next((col for col in df.columns if col.lower() == "text"), None)

    annotator_columns = [col for col in df.columns if col not in {text_column}]

    if len(annotator_columns) < 3:
        raise ValueError("At least three annotators are required.")

    annotations = []
    for idx, row in df.iterrows():
        item_id = row[text_column] if text_column else idx
        for annotator in annotator_columns:
            annotations.append(AnnotationSchema(item_id=item_id, annotator_id=annotator, label=row[annotator]))

    data_schema = DataSchema(annotations=annotations, data_type=data_type)

    if as_dataframe:
        return pd.DataFrame([entry.dict() for entry in data_schema.annotations])
    return data_schema


def update_with_new_annotations(
    existing_data: DataSchema, new_df: pd.DataFrame, as_dataframe: bool = False
) -> Union[DataSchema, pd.DataFrame]:
    """Dynamically update existing annotations with new annotators."""

    new_annotator_columns = [col for col in new_df.columns if col not in {"text"}]

    annotations = existing_data.annotations.copy()

    for idx, row in new_df.iterrows():
        item_id = row.get("text", idx)
        for annotator in new_annotator_columns:
            annotations.append(AnnotationSchema(item_id=item_id, annotator_id=annotator, label=row[annotator]))

    updated_data = DataSchema(annotations=annotations, data_type=existing_data.data_type)

    if as_dataframe:
        return pd.DataFrame([entry.dict() for entry in updated_data.annotations])
    return updated_data
