import os
import pytest
import pandas as pd
import numpy as np
from typing import Literal, List
from src.krippendorff_alpha.reliability import (
    compute_reliability_matrix,
    update_reliability_matrix,
    load_reliability_matrix,
)
from src.krippendorff_alpha.preprocessing import preprocess_data


# Adding the correct type annotations
@pytest.mark.parametrize(
    "dataset",
    [
        "interval_numeric_equalGaps_noAbsoluteZero.jsonl",
        "nominal_categorical_noOrder_sample.json",
        "ordinal_orderedCategories_unequalGaps_sample.csv",
        "ratio_numeric_equalGaps_withAbsoluteZero.tsv",
    ],
)
def test_compute_reliability_matrix(example_data: str, dataset: str) -> None:
    example_dir = example_data
    file_path = os.path.join(example_dir, dataset)
    print(f"\n🔍 Testing reliability matrix computation for: {dataset}")

    try:
        preprocessed_data = preprocess_data(path=file_path)
        assert preprocessed_data.df is not None and not preprocessed_data.df.empty, (
            f"❌ DataFrame is empty for {dataset}"
        )

        matrix = compute_reliability_matrix(preprocessed_data.df, preprocessed_data.column_mapping)

        assert isinstance(matrix, (pd.DataFrame, np.ndarray)), "❌ Output must be a DataFrame or NumPy array"
        assert matrix.shape[1] >= 3, f"❌ Reliability matrix must have at least 3 annotator columns for {dataset}"
        assert matrix.shape[0] > 0, f"❌ Reliability matrix must have rows for {dataset}"

        print(f"✅ Successfully computed reliability matrix for: {dataset}")
        print(f"📊 Matrix shape: {matrix.shape}")

    except Exception as e:
        pytest.fail(f"❌ Error computing reliability matrix for {dataset}: {e}")


@pytest.mark.parametrize(
    "dataset",
    [
        "interval_numeric_equalGaps_noAbsoluteZero.jsonl",
        "nominal_categorical_noOrder_sample.json",
        "ordinal_orderedCategories_unequalGaps_sample.csv",
        "ratio_numeric_equalGaps_withAbsoluteZero.tsv",
    ],
)
def test_update_reliability_matrix(example_data: str, dataset: str) -> None:
    example_dir = example_data
    file_path = os.path.join(example_dir, dataset)
    print(f"\n🔄 Testing reliability matrix update for: {dataset}")

    try:
        preprocessed_data = preprocess_data(path=file_path)
        current_df = preprocessed_data.df.copy()
        assert not current_df.empty, f"❌ Preprocessed DataFrame is empty for {dataset}"

        prev_matrix = load_reliability_matrix()
        print(f"📂 Previous stored matrix: {'Exists' if prev_matrix is not None else 'None'}")

        new_data = current_df.sample(frac=0.2, replace=False).reset_index(drop=True)  # Ensure unique selection

        num_new_annotators = 2
        for i in range(num_new_annotators):
            new_col_name = f"annotator_new_{i + 1}"
            if dataset.startswith("nominal") or dataset.startswith("ordinal"):
                existing_categories = list(
                    set(current_df[preprocessed_data.column_mapping.annotator_cols].values.flatten())
                )
                new_data[new_col_name] = np.random.choice(existing_categories, size=new_data.shape[0])
            else:
                new_data[new_col_name] = np.random.choice([0, 1, 2], size=new_data.shape[0])

        assert not new_data.empty, "❌ Simulated new data is empty"

        for col in preprocessed_data.column_mapping.annotator_cols:
            if col in new_data.columns and col in current_df.columns:
                new_data[col] = new_data[col].astype(current_df[col].dtype)

        update_types: List[Literal["new_doc", "new_annotator", "new_doc_annotator", "new_task"]] = [
            "new_doc",
            "new_annotator",
            "new_doc_annotator",
            "new_task",
        ]
        for update_type in update_types:
            print(f"🔄 Testing update type: {update_type}")

            updated_df = update_reliability_matrix(
                current_df,
                new_data,
                preprocessed_data.column_mapping,
                update_mode="manual",
                update_type=update_type,
            )

            assert not updated_df.empty, f"❌ Updated DataFrame is empty for {dataset}, update type: {update_type}"
            assert set(current_df.columns) <= set(updated_df.columns), f"❌ Column mismatch after update: {update_type}"

            if update_type in ["new_annotator", "new_doc_annotator"]:
                assert updated_df.shape[1] >= current_df.shape[1], "❌ Missing annotator columns after update"
            if update_type in ["new_doc", "new_doc_annotator"]:
                assert updated_df.shape[0] >= current_df.shape[0], "❌ Missing data rows after update"

            print(f"✅ Successfully updated reliability matrix for: {dataset} (update type: {update_type})")

    except Exception as e:
        pytest.fail(f"❌ Error updating reliability matrix for {dataset}: {e}")
