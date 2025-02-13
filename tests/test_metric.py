from typing import List, Tuple
import pandas as pd

from src.krippendorff_alpha.metric import krippendorff_alpha, parse_annotator_name
from src.krippendorff_alpha.preprocessing import preprocess_data
from src.krippendorff_alpha.reliability import compute_reliability_matrix
from src.krippendorff_alpha.schema import DataTypeEnum


def test_krippendorff_alpha(datasets: List[Tuple[str, DataTypeEnum]], example_data: str) -> None:
    for filename, metric in datasets:
        if isinstance(metric, str):
            print(f"DEBUG: Converting '{metric}' to DataTypeEnum")  # Debugging
            metric = DataTypeEnum[metric.upper()]  # Convert string to Enum

        print(f"\nProcessing dataset: {filename} with metric: {metric.value}")
        file_path = f"{example_data}/{filename}"
        print(f"\nProcessing dataset: {filename} with metric: {metric.value}")

        preprocessed = preprocess_data(file_path)
        df: pd.DataFrame = preprocessed.df
        column_mapping = preprocessed.column_mapping

        print("\nPreprocessed DataFrame:")
        print(df.head())

        reliability_matrix: pd.DataFrame = compute_reliability_matrix(df, column_mapping)

        print("\nTransformed Reliability Matrix:")
        print(reliability_matrix.head())

        result = krippendorff_alpha(
            df=reliability_matrix,
            annotator_cols=list(reliability_matrix.columns),
            metric=metric,
            nominal_mappings=preprocessed.nominal_mappings,
            ordinal_mappings=preprocessed.ordinal_mappings,
        )

        print("\nKrippendorff’s Alpha Results:")
        print(f"Alpha: {result['alpha']:.4f}")
        print(f"Observed Disagreement: {result['observed_disagreement']:.4f}")
        print(f"Expected Disagreement: {result['expected_disagreement']:.4f}")

        print("\nPer-Category Scores:")
        for category, scores in result["per_category_scores"].items():
            print(f"  {category}:")
            print(f"    Observed Disagreement: {scores['observed_disagreement']:.4f}")
            print(f"    Expected Disagreement: {scores['expected_disagreement']:.4f}")

        # Validate results
        assert "alpha" in result, "Missing alpha value in result"
        assert isinstance(result["alpha"], float), "Alpha value must be a float"
        assert -1.0 <= result["alpha"] <= 1.0, "Alpha should be in the range [-1, 1]"
        assert "per_category_scores" in result, "Missing per-category scores"

        for category, scores in result["per_category_scores"].items():
            assert "observed_disagreement" in scores, f"Missing observed disagreement for category {category}"
            assert "expected_disagreement" in scores, f"Missing expected disagreement for category {category}"


def test_krippendorff_alpha_with_weights(datasets: List[Tuple[str, DataTypeEnum]], example_data: str) -> None:
    for filename, metric in datasets:
        file_path = f"{example_data}/{filename}"

        preprocessed = preprocess_data(file_path)
        df: pd.DataFrame = preprocessed.df
        column_mapping = preprocessed.column_mapping

        reliability_matrix: pd.DataFrame = compute_reliability_matrix(df, column_mapping)

        # Define weights: Annotator 1 has more weight
        annotators: List[str] = list(reliability_matrix.index)  # Annotators are now in index
        weight_dict = {parse_annotator_name("annotator_1"): 1.5}  # Higher weight for the first annotator
        for annotator in annotators[1:]:
            weight_dict[parse_annotator_name(annotator)] = 1.0  # Default weight

        # Compute Krippendorff’s alpha with weights
        result = krippendorff_alpha(
            df=reliability_matrix,
            annotator_cols=annotators,
            metric=metric,
            weight_dict=weight_dict,
            nominal_mappings=preprocessed.nominal_mappings,
            ordinal_mappings=preprocessed.ordinal_mappings,
        )

        print("\nKrippendorff’s Alpha Results:")
        print(f"Alpha: {result['alpha']:.4f}")
        print(f"Observed Disagreement: {result['observed_disagreement']:.4f}")
        print(f"Expected Disagreement: {result['expected_disagreement']:.4f}")

        print("\nPer-Category Scores:")
        for category, scores in result["per_category_scores"].items():
            print(f"  {category}:")
            print(f"    Observed Disagreement: {scores['observed_disagreement']:.4f}")
            print(f"    Expected Disagreement: {scores['expected_disagreement']:.4f}")

        # Validate results
        assert "alpha" in result, "Missing alpha value in result"
        assert isinstance(result["alpha"], float), "Alpha value must be a float"
        assert -1.0 <= result["alpha"] <= 1.0, "Alpha should be in the range [-1, 1]"
        assert "per_category_scores" in result, "Missing per-category scores"

        for category, scores in result["per_category_scores"].items():
            assert "observed_disagreement" in scores, f"Missing observed disagreement for category {category}"
            assert "expected_disagreement" in scores, f"Missing expected disagreement for category {category}"
