import pandas as pd
import json
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Set
from src.krippendorff_alpha.schema import ColumnMapping, PreprocessedData, AnnotationSchema, DataTypeEnum
from src.krippendorff_alpha.constants import (
    ORDINAL_CATEGORIES,
    TEXT_COLUMN_ALIASES,
    WORD_COLUMN_ALIASES,
    ANNOTATOR_REGEX,
)

logging.basicConfig(level=logging.INFO)


def detect_column(df: pd.DataFrame, column_aliases: Set[str]) -> Optional[str]:
    matches = [col for col in df.columns if col.lower().strip() in {name.lower() for name in column_aliases}]

    if len(matches) > 1:
        logging.warning(f"Multiple matching columns found {matches}. Using first detected: {matches[0]}")

    return matches[0] if matches else None


def detect_annotator_columns(df: pd.DataFrame) -> List[str]:
    matches = [col for col in df.columns if ANNOTATOR_REGEX.match(col)]

    if len(matches) < 3:
        logging.warning(f"Detected only {len(matches)} annotator columns: {matches}. At least 3 are recommended.")

    patterns = set(re.sub(r"\d+", "", col).lower() for col in matches)  # Normalize by removing numbers
    if len(patterns) > 1:
        logging.warning(f"Inconsistent annotator column formats detected: {patterns}. Ensure uniform naming.")

    return matches


def load_data(path: str) -> pd.DataFrame:
    file_ext = Path(path).suffix.lower()

    if file_ext in {".csv", ".tsv"}:
        sep = "\t" if file_ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    elif file_ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return flatten_json(data)
    elif file_ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return flatten_jsonl(data)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: CSV, TSV, JSON, JSONL.")


def flatten_json(data: List[Dict[str, str]]) -> pd.DataFrame:
    records = []

    if isinstance(data, dict):
        data = [data]

    for entry in data:
        text = entry.get("text", "")
        annotations: str | List[Dict[str, str]] = entry.get("annotations", [])

        if isinstance(annotations, str):
            annotations = []
        elif not isinstance(annotations, list):
            annotations = []
        elif isinstance(annotations, list) and not all(isinstance(a, dict) for a in annotations):
            annotations = []

        if annotations:
            for annotation in annotations:
                if isinstance(annotation, dict):
                    record = {"text": text}

                    if "word" in annotation or "token" in annotation:
                        token_col = "word" if "word" in annotation else "token"
                        record[token_col] = str(annotation.get(token_col, ""))  # Ensure it's always a string

                    for key, value in annotation.items():
                        if key not in {"word", "token"}:
                            record[key] = str(value)

                    records.append(record)
        else:
            records.append({"text": text})

    df = pd.DataFrame(records)

    for col in ["word", "token"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def flatten_jsonl(data: List[Dict[str, str]]) -> pd.DataFrame:
    records = []

    for entry in data:
        text = entry.get("text", "")
        annotations: str | List[Dict[str, str]] = entry.get("annotations", [])

        if isinstance(annotations, str):
            annotations = []
        elif not isinstance(annotations, list):
            annotations = []
        elif isinstance(annotations, list) and not all(isinstance(a, dict) for a in annotations):
            annotations = []

        if annotations and isinstance(annotations[0], dict):
            first_annotation = annotations[0]

            if "word" in first_annotation:
                for annotation in annotations:
                    if isinstance(annotation, dict):
                        record = {"text": text, "word": annotation.get("word", "")}

                        for key, value in annotation.items():
                            if key.startswith("annotator_"):
                                record[key] = value

                        records.append(record)

            elif "annotator" in first_annotation and "label" in first_annotation:
                record = {"text": text}

                for annotation in annotations:
                    if isinstance(annotation, dict):
                        annotator_name = annotation.get("annotator", "")
                        record[annotator_name] = annotation.get("label", "")

                records.append(record)

    df = pd.DataFrame(records)
    return df


def infer_annotation_type(column: pd.Series) -> DataTypeEnum:
    if column.dtype == "object":
        unique_values = list(column.dropna().unique())

        for ordinal_scale in ORDINAL_CATEGORIES:
            if set(unique_values).issubset(set(ordinal_scale)):
                return DataTypeEnum.ORDINAL

        return DataTypeEnum.NOMINAL

    elif column.dtype in ["int64", "float64"]:
        unique_values = column.dropna().unique()
        if len(unique_values) < 10 and all(isinstance(x, (int, float)) for x in unique_values):
            return DataTypeEnum.ORDINAL
        elif column.min() >= 0:
            return DataTypeEnum.RATIO
        else:
            return DataTypeEnum.INTERVAL

    return DataTypeEnum.NOMINAL


def convert_ordinal_to_numeric(column: pd.Series) -> pd.Series:
    unique_values = set(column.dropna().unique())

    for ordinal_scale in ORDINAL_CATEGORIES:
        if unique_values.issubset(set(ordinal_scale)):
            return column.map({label: i for i, label in enumerate(ordinal_scale)})

    mapping = {label: i for i, label in enumerate(sorted(unique_values))}
    return column.map(mapping)


nominal_mappings: Dict[str, Dict[str, int]] = {}


def convert_nominal_to_numeric(column: pd.Series, col_name: str) -> pd.Series:
    if col_name not in nominal_mappings:
        unique_values = sorted(set(column.dropna().unique()))
        nominal_mappings[col_name] = {label: i for i, label in enumerate(unique_values)}

    return column.map(nominal_mappings[col_name]).fillna(-1).astype(int)


def preprocess_data(
    path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    annotation_level: str = "sentence_level",
    text_col: Optional[str] = None,
    annotator_cols: Optional[List[str]] = None,
    missing_value_strategy: str = "ignore",
) -> PreprocessedData:
    if path:
        df = load_data(path)
    elif df is None:
        raise ValueError("Either a file path or a DataFrame must be provided.")

    if text_col is None:
        if annotation_level == "sentence_level":
            text_col = detect_column(df, TEXT_COLUMN_ALIASES)
        elif annotation_level == "token_level":
            text_col = detect_column(df, WORD_COLUMN_ALIASES)
        else:
            raise ValueError("annotation_level must be either 'sentence_level' or 'token_level'.")

        if text_col is None:
            raise ValueError(
                f"Could not infer the correct text column for {annotation_level}. Please specify it explicitly."
            )

    if annotator_cols is None:
        annotator_cols = detect_annotator_columns(df)
        if len(annotator_cols) < 3:
            raise ValueError("At least three annotator columns are required for reliability assessment.")

    annotation_types = {}
    for col in annotator_cols:
        annotation_type = infer_annotation_type(df[col])
        annotation_types[col] = str(annotation_type)

        if annotation_type == DataTypeEnum.ORDINAL:
            df[col] = convert_ordinal_to_numeric(df[col])
        elif annotation_type == DataTypeEnum.NOMINAL:
            df[col] = convert_nominal_to_numeric(df[col], col)

    if missing_value_strategy == "drop":
        df = df.dropna(subset=annotator_cols)
    elif missing_value_strategy == "fill":
        df[annotator_cols] = df[annotator_cols].fillna(-1)

    return PreprocessedData(
        df=df,
        column_mapping=ColumnMapping(text_col=text_col, annotator_cols=annotator_cols),
        annotation_schema=AnnotationSchema(
            annotation_level=annotation_level, data_type=annotation_types, missing_value_strategy=missing_value_strategy
        ),
    )
