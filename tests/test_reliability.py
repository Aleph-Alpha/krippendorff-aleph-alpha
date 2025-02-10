from typing import List, Tuple, Any, Literal
from numpy.typing import NDArray
from pathlib import Path
import pandas as pd
import numpy as np
import os
from src.krippendorff_alpha.reliability import (
    compute_reliability_matrix,
    save_reliability_matrix,
    load_reliability_matrix,
    detect_new_labels,
    update_reliability_matrix,
)
from src.krippendorff_alpha.preprocessing import preprocess_data

PICKLE_STORAGE_PATH: str = ""

UpdateType = Literal["new_doc", "new_annotator", "new_doc_annotator", "new_task"]


def test_compute_reliability_matrix(datasets: List[Tuple[str, str]], example_data: str) -> None:
    for filename, data_type in datasets:
        path = os.path.join(example_data, filename)
        preprocessed = preprocess_data(path)
        matrix: NDArray[Any] = compute_reliability_matrix(preprocessed.df, preprocessed.column_mapping)

        print(f"\nComputed reliability matrix for {filename}:")
        print(matrix)
        print(f"Shape: {matrix.shape}")

        assert isinstance(matrix, np.ndarray)
        assert matrix.shape[0] > 0, "Reliability matrix should not be empty"
        assert matrix.shape[1] == len(preprocessed.column_mapping.annotator_cols), "Incorrect column count"


def test_save_and_load_reliability_matrix(datasets: List[Tuple[str, str]], example_data: str, tmp_path: Path) -> None:
    global PICKLE_STORAGE_PATH
    for filename, data_type in datasets:
        path = os.path.join(example_data, filename)
        preprocessed = preprocess_data(path)
        test_path = tmp_path / "test_reliability.pkl"

        PICKLE_STORAGE_PATH = str(test_path)
        save_reliability_matrix(preprocessed.df, versioning=False)

        loaded_df: pd.DataFrame = load_reliability_matrix()
        assert loaded_df is not None, "Loaded reliability matrix should not be None"
        pd.testing.assert_frame_equal(preprocessed.df, loaded_df, check_dtype=False)


def test_detect_new_labels(datasets: List[Tuple[str, str]], example_data: str) -> None:
    for filename, data_type in datasets:
        path = os.path.join(example_data, filename)
        preprocessed = preprocess_data(path)
        new_data: pd.DataFrame = preprocessed.df.copy()
        new_data.iloc[0, 1] = "New_Label_123"
        assert detect_new_labels(new_data), "New label detection failed"


def test_update_reliability_matrix(datasets: List[Tuple[str, str]], example_data: str) -> None:
    for filename, data_type in datasets:
        path = os.path.join(example_data, filename)
        preprocessed = preprocess_data(path)
        current_df: pd.DataFrame = preprocessed.df.copy()

        print(f"\nInitial DataFrame for {filename}:")
        print(current_df.head())

        prev_matrix = load_reliability_matrix()
        print(f"Previous stored matrix: {'Exists' if prev_matrix is not None else 'None'}")

        # Simulate new data (partial sample of original dataset)
        new_data: pd.DataFrame = current_df.sample(frac=0.2, replace=False).reset_index(drop=True)

        print("\nSimulated new data (to be added):")
        print(new_data.head())

        # Test different update types
        update_types: List[UpdateType] = ["new_doc", "new_annotator", "new_doc_annotator", "new_task"]
        for update_type in update_types:
            print(f"\nApplying update type: {update_type}")
            updated_df: pd.DataFrame = update_reliability_matrix(
                current_df, new_data, preprocessed.column_mapping, update_mode="manual", update_type=update_type
            )

            print(f"\nUpdated reliability matrix after '{update_type}':")
            print(updated_df.head())
