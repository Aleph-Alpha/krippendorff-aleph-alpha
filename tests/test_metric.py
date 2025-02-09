import os
from typing import List, Tuple

from src.krippendorff_alpha.preprocessing import preprocess_data
from src.krippendorff_alpha.reliability import compute_reliability_matrix
from src.krippendorff_alpha.metric import krippendorff_alpha


def test_krippendorff_alpha_with_real_data(example_data: str, datasets: List[Tuple[str, str]]) -> None:
    for filename, metric in datasets:
        file_path = os.path.join(example_data, filename)

        # Preprocess data
        preprocessed = preprocess_data(file_path)
        data, column_mapping = preprocessed.df, preprocessed.column_mapping

        reliability_matrix = compute_reliability_matrix(data, column_mapping)

        alpha_results = krippendorff_alpha(reliability_matrix, metric=metric)

        print(f"\n📊 Dataset: {filename} | Metric: {metric}")
        print(f"🔢 Krippendorff's Alpha: {alpha_results['alpha']:.4f}")
        print(f"🔍 Observed Disagreement: {alpha_results['observed_disagreement']:.4f}")
        print(f"🎯 Expected Disagreement: {alpha_results['expected_disagreement']:.4f}")

        if metric in ["nominal", "ordinal"]:
            print("📌 Per-category Scores:")
            for category, scores in alpha_results["per_category_scores"].items():
                print(
                    f"  - Category {category}: Obs: {scores['observed_disagreement']:.4f}, "
                    f"Exp: {scores['expected_disagreement']:.4f}"
                )

        assert 0 <= alpha_results["alpha"] <= 1, (
            f"❌ Krippendorff Alpha out of range for {filename}: {alpha_results['alpha']}"
        )
        assert alpha_results["observed_disagreement"] >= 0, f"❌ Observed disagreement invalid for {filename}"
        assert alpha_results["expected_disagreement"] >= 0, f"❌ Expected disagreement invalid for {filename}"

        if metric in ["nominal", "ordinal"]:
            for category, scores in alpha_results["per_category_scores"].items():
                assert scores["observed_disagreement"] >= 0, f"❌ Invalid observed disagreement for category {category}"
                assert scores["expected_disagreement"] >= 0, f"❌ Invalid expected disagreement for category {category}"
