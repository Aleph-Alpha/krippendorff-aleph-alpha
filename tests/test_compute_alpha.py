import os
import pandas as pd
from typing import List, Tuple, Dict, Any
from src.krippendorff_alpha.compute_alpha import compute_alpha
from src.krippendorff_alpha.schema import DataTypeEnum


def test_compute_alpha_with_path(example_data: str, datasets: List[Tuple[str, str]]) -> None:
    for dataset_file, expected_metric in datasets:
        dataset_path = os.path.join(example_data, dataset_file)

        print(f"\nTesting dataset: {dataset_file} (Metric: {expected_metric})")
        result: Dict[str, Any] = compute_alpha(path=dataset_path, metric=expected_metric)

        print(f"Computed Alpha for {dataset_file}: {result['alpha']}")
        print(f"Observed Disagreement: {result['observed_disagreement']}")
        print(f"Expected Disagreement: {result['expected_disagreement']}")
        print("Per-Category Scores:")
        for category, scores in result.get("per_category_scores", {}).items():
            print(
                f"  {category}: Observed={scores['observed_disagreement']}, Expected={scores['expected_disagreement']}"
            )

        assert "alpha" in result, f"Missing 'alpha' key in result for {dataset_file}"
        assert isinstance(result["alpha"], float), f"Alpha is not a float for {dataset_file}"
        assert "observed_disagreement" in result
        assert "expected_disagreement" in result
        assert "per_category_scores" in result


def test_compute_alpha_with_dataframe() -> None:
    data = {
        "text": ["Sentence 1", "Sentence 2", "Sentence 3"],
        "annotator_1": ["dark", "light", "neutral"],
        "annotator_2": ["dark", "light", "light"],
        "annotator_3": ["dark", "dark", "neutral"],
    }
    df: pd.DataFrame = pd.DataFrame(data)

    print("\nTesting with manually created DataFrame:")
    print(df)

    result: Dict[str, Any] = compute_alpha(
        df=df,
        text_col="text",
        annotator_cols=["annotator_1", "annotator_2", "annotator_3"],
        metric=DataTypeEnum.ORDINAL,  # Use the Enum instead of a string
    )

    print(f"Computed Alpha for DataFrame: {result['alpha']}")
    print(f"Observed Disagreement: {result['observed_disagreement']}")
    print(f"Expected Disagreement: {result['expected_disagreement']}")
    print("Per-Category Scores:")
    for category, scores in result.get("per_category_scores", {}).items():
        print(f"  {category}: Observed={scores['observed_disagreement']}, Expected={scores['expected_disagreement']}")

    assert "alpha" in result
    assert isinstance(result["alpha"], float)
    assert "observed_disagreement" in result
    assert "expected_disagreement" in result
    assert "per_category_scores" in result
