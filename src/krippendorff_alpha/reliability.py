import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Literal, Optional, Dict
from src.krippendorff_alpha.schema import ColumnMapping
from src.krippendorff_alpha.constants import PICKLE_STORAGE_PATH, WORD_COLUMN_ALIASES, DAYS_TO_KEEP
from numpy.typing import NDArray

# Global mappings
ordinal_mappings: Dict[str, Dict[str, int]] = {}
nominal_mappings: Dict[str, Dict[str, int]] = {}


def clean_old_pickles() -> None:
    pickle_dir = os.path.dirname(PICKLE_STORAGE_PATH)
    now = datetime.now()

    if not os.path.exists(pickle_dir):
        return

    for filename in os.listdir(pickle_dir):
        if filename.endswith(".pkl"):
            file_path = os.path.join(pickle_dir, filename)
            modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))

            # Delete files older than DAYS_TO_KEEP
            if (now - modified_time).days > DAYS_TO_KEEP:
                os.remove(file_path)
                print(f"Deleted old pickle file: {file_path}")


def compute_reliability_matrix(df: pd.DataFrame, column_mapping: ColumnMapping) -> NDArray[np.float64]:
    annotator_cols = column_mapping.annotator_cols

    if not annotator_cols:
        raise ValueError("No annotator columns found in column mapping.")

    return np.asarray(df[annotator_cols].values, dtype=np.float64)


def save_mappings() -> None:
    mappings_path = PICKLE_STORAGE_PATH.replace(".pkl", "_mappings.pkl")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    versioned_mappings_path = mappings_path.replace(".pkl", f"_{timestamp}.pkl")

    with open(versioned_mappings_path, "wb") as f:
        pickle.dump({"ordinal": ordinal_mappings, "nominal": nominal_mappings}, f)

    with open(mappings_path, "wb") as f:
        pickle.dump({"ordinal": ordinal_mappings, "nominal": nominal_mappings}, f)

    print(f"Mappings saved to {mappings_path}")
    print(f"Versioned mappings saved as {versioned_mappings_path}")

    clean_old_pickles()


def load_mappings() -> None:
    global ordinal_mappings, nominal_mappings
    mappings_path = PICKLE_STORAGE_PATH.replace(".pkl", "_mappings.pkl")
    if os.path.exists(mappings_path):
        with open(mappings_path, "rb") as f:
            mappings = pickle.load(f)
            ordinal_mappings = mappings.get("ordinal", {})
            nominal_mappings = mappings.get("nominal", {})


def save_reliability_matrix(df: pd.DataFrame, versioning: bool = True) -> None:
    if df is None or df.empty:
        print("Warning: Attempted to save an empty or None DataFrame.")
        return

    try:
        os.makedirs(os.path.dirname(PICKLE_STORAGE_PATH), exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        versioned_path = PICKLE_STORAGE_PATH.replace(".pkl", f"_{timestamp}.pkl")

        with open(versioned_path, "wb") as f:
            pickle.dump(df, f)

        with open(PICKLE_STORAGE_PATH, "wb") as f:
            pickle.dump(df, f)

        save_mappings()
        print(f"Reliability matrix saved to {PICKLE_STORAGE_PATH}")
        print(f"Versioned copy saved as {versioned_path}")

        clean_old_pickles()
    except Exception as e:
        print(f"Error saving reliability matrix: {e}")


def load_reliability_matrix() -> Optional[pd.DataFrame]:
    if not os.path.exists(PICKLE_STORAGE_PATH):
        print("No previous reliability matrix found.")
        return None

    try:
        with open(PICKLE_STORAGE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading reliability matrix: {e}")
        return None


def detect_new_labels(new_data: pd.DataFrame) -> bool:
    global nominal_mappings, ordinal_mappings

    all_labels = set(new_data.iloc[:, 1:].dropna().values.flatten())

    known_labels = {label for mapping in nominal_mappings.values() for label in mapping}
    known_labels.update(label for mapping in ordinal_mappings.values() for label in mapping)

    return not all_labels.issubset(known_labels)


def update_reliability_matrix(
    current_df: pd.DataFrame,
    new_data: pd.DataFrame,
    column_mapping: ColumnMapping,
    update_mode: Literal["auto", "manual"] = "auto",
    update_type: Optional[Literal["new_doc", "new_annotator", "new_doc_annotator", "new_task"]] = None,
) -> pd.DataFrame:
    if update_mode == "manual" and update_type is None:
        raise ValueError("When update_mode is 'manual', update_type must be specified.")

    text_col = column_mapping.text_col
    word_col = next((col for col in current_df.columns if col in WORD_COLUMN_ALIASES), None)

    prev_df = load_reliability_matrix()
    load_mappings()

    if prev_df is None:
        prev_df = pd.DataFrame()

    if detect_new_labels(new_data):
        print("New labels detected! Treating this as a new dataset instead of an update.")
        return new_data.copy()

    if update_mode == "auto":
        if prev_df.empty:
            return new_data.copy()
        if set(new_data.columns) == set(prev_df.columns):
            update_type = "new_doc"
        elif set(new_data.columns) - set(prev_df.columns):
            update_type = "new_annotator"
        else:
            update_type = "new_doc_annotator"

    if update_type == "new_task":
        updated_df = new_data.copy()
    elif update_type == "new_doc":
        updated_df = pd.concat([current_df, new_data], ignore_index=True)
    elif update_type == "new_annotator":
        merge_cols = [text_col] if text_col in current_df.columns else []
        if word_col and word_col in current_df.columns:
            merge_cols.append(word_col)

        if not merge_cols:
            raise ValueError("No common columns found for merging new annotators.")

        updated_df = current_df.merge(new_data, on=merge_cols, how="left", suffixes=("", "_new"))
    elif update_type == "new_doc_annotator":
        all_columns = set(current_df.columns).union(set(new_data.columns))

        for col in all_columns:
            if col not in current_df.columns:
                current_df[col] = np.nan
            if col not in new_data.columns:
                new_data[col] = np.nan

        merge_cols = [text_col]
        if word_col:
            merge_cols.append(word_col)

        if all(col in new_data.columns and col in current_df.columns for col in merge_cols):
            new_data = new_data.set_index(merge_cols)
            current_df = current_df.set_index(merge_cols)

        updated_df = pd.concat([current_df, new_data], axis=0).reset_index()

    save_reliability_matrix(updated_df, versioning=True)
    return updated_df
