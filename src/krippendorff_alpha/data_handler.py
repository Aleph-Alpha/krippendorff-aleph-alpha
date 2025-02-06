import pandas as pd
import json
from pathlib import Path
from typing import Union, List, Dict
from collections import defaultdict
from .schema import DataSchema, AnnotationSchema, DataType


def process_jsonl(file_path: Union[str, Path]) -> pd.DataFrame:
    """Process JSONL file into a DataFrame with annotators as columns."""
    data: List[Dict[str, Union[str, int, float, None]]] = []
    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"]
            row: Dict[str, Union[str, int, float, None]] = {"text": text}
            for annotation in entry["annotations"]:
                row[annotation["annotator"]] = annotation["label"]
            data.append(row)
    return pd.DataFrame(data)


def flatten_json(json_data: List[Dict[str, Union[str, List[Dict[str, Union[str, int, float, None]]]]]]) -> pd.DataFrame:
    """Flatten JSON format where annotations are nested."""
    data: List[Dict[str, Union[str, int, float, None]]] = []

    for entry in json_data:
        row: Dict[str, Union[str, int, float, None]] = defaultdict(lambda: None)

        # Ensure text is a string
        if isinstance(entry["text"], str):
            row["text"] = entry["text"]
        else:
            row["text"] = json.dumps(entry["text"])  # Convert list/dict to string if necessary

        # Ensure "annotations" is a list before iterating
        if isinstance(entry.get("annotations"), list):
            for annotation in entry["annotations"]:
                if isinstance(annotation, dict):  # Ensure it's a dictionary
                    for annotator, label in annotation.items():
                        if annotator.startswith("annotator_"):
                            row[annotator] = label

        data.append(dict(row))  # Convert defaultdict to dict before appending

    return pd.DataFrame(data)


def load_data(
    file_path: Union[str, Path], data_type: DataType, as_dataframe: bool = False
) -> Union[DataSchema, pd.DataFrame]:
    """Load data from a file (CSV, TSV, JSON, JSONL) and convert it into a DataSchema or DataFrame."""
    file_path = Path(file_path)

    if file_path.suffix in {".csv", ".tsv"}:
        sep = "\t" if file_path.suffix == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)
    elif file_path.suffix == ".json":
        with open(file_path, "r") as f:
            json_data = json.load(f)
        df = flatten_json(json_data)
    elif file_path.suffix == ".jsonl":
        df = process_jsonl(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV, TSV, JSON, or JSONL.")

    return process_dataframe(df, data_type, as_dataframe)


def process_dataframe(
    df: pd.DataFrame, data_type: DataType, as_dataframe: bool = False
) -> Union[DataSchema, pd.DataFrame]:
    """Process a DataFrame into DataSchema, ensuring the correct format and transformations."""
    if df.empty:
        raise ValueError("Dataset is empty.")

    df.columns = df.columns.str.lower().str.replace(" ", "_")
    annotator_columns: List[str] = [col for col in df.columns if "annotator" in col]
    if not annotator_columns:
        raise ValueError("At least one annotator column is required.")

    # Convert categorical labels to numerical indices for Nominal & Ordinal data types
    if data_type in {DataType.NOMINAL, DataType.ORDINAL}:
        unique_labels = sorted(
            {label for col in annotator_columns for label in df[col].dropna().unique() if isinstance(label, str)}
        )
        label_mapping: Dict[str, int] = {label: idx for idx, label in enumerate(unique_labels)}

        # Apply mapping while preserving missing values
        df[annotator_columns] = df[annotator_columns].map(lambda x: label_mapping.get(x, x) if pd.notna(x) else x)

    annotations: List[AnnotationSchema] = []
    for _, row in df.iterrows():
        item_id = row["text"]
        for annotator in annotator_columns:
            annotations.append(
                AnnotationSchema(
                    item_id=item_id,
                    annotator_id=annotator,
                    label=row[annotator],  # Keeps NaN/None as-is
                )
            )

    data_schema = DataSchema(annotations=annotations, data_type=data_type)
    return pd.DataFrame([entry.model_dump() for entry in data_schema.annotations]) if as_dataframe else data_schema
