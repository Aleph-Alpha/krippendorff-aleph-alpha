import pandas as pd
import json
from pathlib import Path
from typing import Union, List, Dict
from .schema import DataSchema, AnnotationSchema, DataType
from .schema import LabelMapping


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase and replace spaces with underscores."""
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def process_jsonl(file_path: Union[str, Path]) -> pd.DataFrame:
    """Process JSONL file into a DataFrame with annotators as columns."""
    data: List[Dict[str, Union[str, int, float, None]]] = []
    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"]
            row: Dict[str, Union[str, int, float, None]] = {"text": text}
            # Handle sentence-level annotations
            for annotation in entry["annotations"]:
                row[annotation["annotator"]] = annotation["label"]
            data.append(row)
    return pd.DataFrame(data)


def process_json(file_path: Union[str, Path]) -> pd.DataFrame:
    """Process JSON file into a DataFrame with annotators as columns."""
    data: List[Dict[str, Union[str, int, float, None]]] = []
    with open(file_path, "r") as f:
        json_data = json.load(f)
        for entry in json_data:
            text = entry["text"]
            row: Dict[str, Union[str, int, float, None]] = {"text": text}
            # Handle sentence-level annotations
            for annotation in entry["annotations"]:
                row[annotation["annotator"]] = annotation["label"]
            data.append(row)
    return pd.DataFrame(data)


def process_token_level_json(
    json_data: List[Dict[str, Union[str, List[Dict[str, Union[str, int, float, None]]]]]],
) -> pd.DataFrame:
    """Flatten token-level JSON format (e.g., NER) and transform into a DataFrame."""
    data: List[Dict[str, Union[str, int, float, None]]] = []

    for entry in json_data:
        text = entry.get("text")
        if not isinstance(text, str):
            continue

        annotations = entry.get("annotations", [])
        if isinstance(annotations, list):
            for annotation in annotations:
                word = annotation.get("word", "")
                for annotator, label in annotation.items():
                    if annotator.startswith("annotator_"):
                        # Check if the label is a valid type (str, int, float, or None)
                        if isinstance(label, (list, dict)):
                            label = None
                        elif isinstance(label, (str, int, float)):
                            pass
                        else:
                            label = None

                        if isinstance(label, (str, int, float, type(None))):
                            data.append(
                                {
                                    "text": text,
                                    "word": word,
                                    "annotator_id": annotator,
                                    "label": label if label not in {"nan", "none", ""} else None,
                                    # Handle missing labels
                                }
                            )
                        else:
                            # If label is still not valid, skip this annotation
                            continue

    return pd.DataFrame(data)


def process_tsv(file_path: Union[str, Path]) -> pd.DataFrame:
    """Process TSV file into a DataFrame with annotators as columns for token-level annotations."""
    df = pd.read_csv(file_path, sep="\t")

    df = normalize_column_names(df)

    text_columns = ["text", "sentence", "Text", "Sentence"]
    for col in text_columns:
        if col.lower() in df.columns:
            df = df.rename(columns={col.lower(): "text"})
            break

    annotator_columns = [col for col in df.columns if "annotator" in col]
    annotator_columns = [col for col in df.columns if any(keyword in col for keyword in ["annotator", "annotator_"])]

    word_columns = ["word", "token", "Word", "Token"]
    word_column = None
    for col in word_columns:
        if col.lower() in df.columns:
            word_column = col.lower()
            break

    data = []
    for _, row in df.iterrows():
        if word_column:
            for annotator in annotator_columns:
                data.append(
                    {
                        "text": row["text"],
                        "word": row[word_column],
                        "annotator_id": annotator,
                        "label": row[annotator] if row[annotator] not in {"nan", "none", ""} else None,
                    }
                )
        else:
            entry = {"text": row["text"]}
            for annotator in annotator_columns:
                entry[annotator] = row[annotator] if row[annotator] not in {"nan", "none", ""} else None
            data.append(entry)

    return pd.DataFrame(data)


def process_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """Process CSV file into a DataFrame with annotators as columns, supporting both sentence-level and token-level annotations."""
    df = pd.read_csv(file_path)

    df = normalize_column_names(df)

    text_columns = ["text", "sentence", "Text", "Sentence"]
    for col in text_columns:
        if col.lower() in df.columns:
            df = df.rename(columns={col.lower(): "text"})
            break

    annotator_columns = [col for col in df.columns if "annotator" in col]
    annotator_columns = [col for col in df.columns if any(keyword in col for keyword in ["annotator", "annotator_"])]

    word_columns = ["word", "token", "Word", "Token"]
    word_column = None
    for col in word_columns:
        if col.lower() in df.columns:
            word_column = col.lower()
            break

    data = []
    for _, row in df.iterrows():
        if word_column:
            for annotator in annotator_columns:
                data.append(
                    {
                        "text": row["text"],
                        "word": row[word_column],
                        "annotator_id": annotator,
                        "label": row[annotator] if row[annotator] not in {"nan", "none", ""} else None,
                    }
                )
        else:
            entry = {"text": row["text"]}
            for annotator in annotator_columns:
                entry[annotator] = row[annotator] if row[annotator] not in {"nan", "none", ""} else None
            data.append(entry)

    return pd.DataFrame(data)


def load_data(
    file_path: Union[str, Path], data_type: DataType, as_dataframe: bool = False
) -> Union[DataSchema, pd.DataFrame]:
    """Load data from a file (CSV, TSV, JSON, JSONL) and convert it into a DataSchema or DataFrame."""
    file_path = Path(file_path)

    if file_path.suffix in {".csv"}:
        df = process_csv(file_path)
    elif file_path.suffix in {".tsv"}:
        df = process_tsv(file_path)
    elif file_path.suffix == ".json":
        with open(file_path, "r") as f:
            json_data = json.load(f)
        # Check if it's token-level or sentence-level
        if isinstance(json_data[0]["annotations"][0], dict):  # Token-level
            df = process_token_level_json(json_data)
        else:  # Sentence-level
            df = process_json(file_path)
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

    df = normalize_column_names(df)

    # Ensure 'text' or 'Text' is available for item_id
    if "text" not in df.columns:
        raise ValueError("Missing 'text' column.")

    annotator_columns: List[str] = [col for col in df.columns if "annotator" in col]
    if not annotator_columns:
        raise ValueError("At least one annotator column is required.")

    label_mapping_dict: Dict[str, Union[int, str]] = {}

    if data_type in {DataType.NOMINAL, DataType.ORDINAL}:
        # Handle Nominal & Ordinal by mapping categorical labels to numeric indices
        unique_labels = sorted(
            {label for col in annotator_columns for label in df[col].dropna().unique() if isinstance(label, str)}
        )
        label_mapping_dict = {label: idx for idx, label in enumerate(unique_labels)}
        df[annotator_columns] = df[annotator_columns].infer_objects(copy=False)
        df[annotator_columns] = df[annotator_columns].apply(lambda x: x.astype(str).map(label_mapping_dict).fillna(x))

    elif data_type in {DataType.INTERVAL, DataType.RATIO}:
        pass

    label_mapping = LabelMapping(
        label_to_value=label_mapping_dict,
        value_to_label={v: k for k, v in label_mapping_dict.items()},  # Reverse the mapping
    )

    annotations: List[AnnotationSchema] = []
    for _, row in df.iterrows():
        item_id = row.get("text")
        if item_id is None:
            raise ValueError(f"Missing 'item_id' (text column) for row: {row}")

        for annotator in annotator_columns:
            # Get the value for this annotator (numerical value, e.g., 0, 2, 1, etc.)
            label = row[annotator]

            if pd.isna(label):
                label = None
            elif isinstance(label, str) and label.lower() in {"nan", "none", ""}:
                label = None

            annotations.append(
                AnnotationSchema(
                    item_id=item_id,
                    annotator_id=annotator,
                    label=label,
                )
            )

    data_schema = DataSchema(annotations=annotations, data_type=data_type, label_mapping=label_mapping)

    return pd.DataFrame([entry.model_dump() for entry in data_schema.annotations]) if as_dataframe else data_schema
