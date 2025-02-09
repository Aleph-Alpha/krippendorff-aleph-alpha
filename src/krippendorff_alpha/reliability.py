import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Literal, Optional
from src.krippendorff_alpha.schema import ColumnMapping
from src.krippendorff_alpha.constants import PICKLE_STORAGE_PATH, WORD_COLUMN_ALIASES
from numpy.typing import NDArray


def compute_reliability_matrix(df: pd.DataFrame, column_mapping: ColumnMapping) -> NDArray[np.float64]:
    annotator_cols = column_mapping.annotator_cols

    if not annotator_cols:
        raise ValueError("No annotator columns found in column mapping.")

    return np.asarray(df[annotator_cols].values, dtype=np.float64)


def save_reliability_matrix(df: pd.DataFrame, versioning: bool = True) -> None:
    if df is None or df.empty:
        print("Warning: Attempted to save an empty or None DataFrame.")
        return

    try:
        os.makedirs(os.path.dirname(PICKLE_STORAGE_PATH), exist_ok=True)

        if versioning:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            versioned_path = PICKLE_STORAGE_PATH.replace(".pkl", f"_{timestamp}.pkl")
            with open(versioned_path, "wb") as f:
                pickle.dump(df, f)

        # Save latest version (overwrite)
        with open(PICKLE_STORAGE_PATH, "wb") as f:
            pickle.dump(df, f)

        print(f"Reliability matrix saved to {PICKLE_STORAGE_PATH}")
        if versioning:
            print(f"Versioned copy saved as {versioned_path}")

    except Exception as e:
        print(f"Error saving reliability matrix: {e}")


def load_reliability_matrix() -> Optional[pd.DataFrame]:
    if not os.path.exists(PICKLE_STORAGE_PATH):
        print("No previous reliability matrix found.")
        return None

    try:
        with open(PICKLE_STORAGE_PATH, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
        print(f"Error loading reliability matrix: {e}")
        return None


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
    if prev_df is None:
        prev_df = pd.DataFrame()

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

        if current_df.duplicated(subset=merge_cols).any():
            raise ValueError("Duplicate text/word values detected. Ensure unique identifiers before merging.")

        updated_df = current_df.merge(new_data, on=merge_cols, how="left", suffixes=("", "_new"))

    elif update_type == "new_doc_annotator":
        all_columns = set(current_df.columns).union(set(new_data.columns))

        current_df = current_df.copy()
        new_data = new_data.copy()

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

        new_annotator_cols = set(updated_df.columns) - set(current_df.columns)
        valid_new_annotator = any(updated_df[col].notna().any() for col in new_annotator_cols)

        if not valid_new_annotator:
            raise ValueError("'new_doc_annotator' did not add valid annotations.")

    save_reliability_matrix(updated_df, versioning=True)
    return updated_df
