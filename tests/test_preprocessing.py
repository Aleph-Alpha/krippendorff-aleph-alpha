import os
import pytest
from src.krippendorff_alpha.preprocessing import preprocess_data


@pytest.mark.parametrize(
    "dataset",
    [
        "interval_numeric_equalGaps_noAbsoluteZero.jsonl",
        "nominal_categorical_noOrder_sample.json",
        "ordinal_orderedCategories_unequalGaps_sample.csv",
        "ratio_numeric_equalGaps_withAbsoluteZero.tsv",
    ],
)
def test_preprocessing(example_data: str, dataset: str) -> None:
    example_dir = example_data  # Get the directory from the fixture
    file_path = os.path.join(example_dir, dataset)
    print(f"\nTesting preprocessing for: {file_path}")

    try:
        preprocessed_data = preprocess_data(path=file_path)
        assert preprocessed_data.df is not None, "❌ DataFrame is empty!"
        print(f"✅ Successfully processed: {dataset}")
        print(f"📌 Preprocessed DataFrame (first 5 rows):\n{preprocessed_data.df.head()}")
        print(f"📝 Column Mapping: {preprocessed_data.column_mapping}")
        print(f"📊 Annotation Schema: {preprocessed_data.annotation_schema}\n")
    except Exception as e:
        assert False, f"❌ Error processing {dataset}: {e}"
