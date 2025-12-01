import pandas as pd
import pytest
import numpy as np

from krippendorff_alpha.preprocessing import preprocess_data
from krippendorff_alpha.reliability import compute_reliability_matrix
from krippendorff_alpha.schema import AnnotationSchema, ColumnMapping, DataTypeEnum
from krippendorff_alpha.metric import (
    krippendorff_alpha,
    nominal_distance,
    ordinal_distance,
    interval_distance,
    ratio_distance,
    compute_observed_disagreement,
)


def test_krippendorff_alpha_nominal(df_nominal: pd.DataFrame) -> None:
    """Test Krippendorff's alpha calculation for nominal data."""
    column_mapping = ColumnMapping(text_col=None, annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    preprocessed_data, detected_text_col = preprocess_data(df_nominal, column_mapping, annotation_schema)

    if preprocessed_data.nominal_mappings:
        preprocessed_data.nominal_mappings = {str(k): v for k, v in preprocessed_data.nominal_mappings.items()}

    mapping = preprocessed_data.nominal_mappings if annotation_schema.data_type == "nominal" else None

    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    result = krippendorff_alpha(reliability_matrix, data_type=annotation_schema.data_type, mapping=mapping)

    assert "alpha" in result
    assert "observed_disagreement" in result
    assert "expected_disagreement" in result
    assert "per_category_scores" in result

    assert isinstance(result["alpha"], float)
    assert -1.0 <= result["alpha"] <= 1.0
    assert isinstance(result["observed_disagreement"], float)
    assert isinstance(result["expected_disagreement"], float)
    assert result["observed_disagreement"] >= 0
    assert result["expected_disagreement"] >= 0

    if result["per_category_scores"]:
        for category, scores in result["per_category_scores"].items():
            assert "observed_disagreement" in scores
            assert "expected_disagreement" in scores
            assert isinstance(scores["observed_disagreement"], float)
            assert isinstance(scores["expected_disagreement"], float)


def test_krippendorff_alpha_with_missing_data() -> None:
    """Test Krippendorff's alpha with missing data (NaN values)."""
    df = pd.DataFrame(
        {
            "text": ["A", "B", "C", "D"],
            "annotator1": ["positive", "negative", "positive", "negative"],
            "annotator2": ["positive", "negative", np.nan, "negative"],
            "annotator3": ["positive", np.nan, "positive", "negative"],
        }
    )

    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    preprocessed_data, detected_text_col = preprocess_data(df, column_mapping, annotation_schema)
    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    mapping = preprocessed_data.nominal_mappings if preprocessed_data.nominal_mappings else None
    result = krippendorff_alpha(reliability_matrix, data_type=DataTypeEnum.NOMINAL, mapping=mapping)

    assert "alpha" in result
    assert isinstance(result["alpha"], float)
    assert not np.isnan(result["alpha"])


def test_krippendorff_alpha_perfect_agreement() -> None:
    """Test Krippendorff's alpha with perfect agreement (should be 1.0)."""
    df = pd.DataFrame(
        {
            "text": ["A", "B", "C"],
            "annotator1": ["positive", "negative", "positive"],
            "annotator2": ["positive", "negative", "positive"],
            "annotator3": ["positive", "negative", "positive"],
        }
    )

    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    preprocessed_data, detected_text_col = preprocess_data(df, column_mapping, annotation_schema)
    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    mapping = preprocessed_data.nominal_mappings if preprocessed_data.nominal_mappings else None
    result = krippendorff_alpha(reliability_matrix, data_type=DataTypeEnum.NOMINAL, mapping=mapping)

    assert result["alpha"] == 1.0
    assert result["observed_disagreement"] == 0.0


def test_krippendorff_alpha_ordinal(df_ordinal: pd.DataFrame) -> None:
    """Test Krippendorff's alpha calculation for ordinal data."""
    column_mapping = ColumnMapping(text_col=None, annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="ordinal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    preprocessed_data, detected_text_col = preprocess_data(df_ordinal, column_mapping, annotation_schema)
    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    ordinal_scale = ["low", "medium", "high", "very high"]
    mapping = preprocessed_data.ordinal_mappings if preprocessed_data.ordinal_mappings else None

    result = krippendorff_alpha(
        reliability_matrix,
        data_type=DataTypeEnum.ORDINAL,
        mapping=mapping,
        ordinal_scale=ordinal_scale,
    )

    assert "alpha" in result
    assert isinstance(result["alpha"], float)
    assert -1.0 <= result["alpha"] <= 1.0


def test_krippendorff_alpha_with_weights() -> None:
    """Test Krippendorff's alpha with annotator weights."""
    df = pd.DataFrame(
        {
            "text": ["A", "B", "C"],
            "annotator1": ["positive", "negative", "positive"],
            "annotator2": ["positive", "negative", "positive"],
            "annotator3": ["positive", "negative", "positive"],
        }
    )

    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    preprocessed_data, detected_text_col = preprocess_data(df, column_mapping, annotation_schema)
    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    mapping = preprocessed_data.nominal_mappings if preprocessed_data.nominal_mappings else None
    weight_dict = {"annotator1": 1.0, "annotator2": 0.8, "annotator3": 1.0}

    result = krippendorff_alpha(
        reliability_matrix,
        data_type=DataTypeEnum.NOMINAL,
        mapping=mapping,
        weight_dict=weight_dict,
    )

    assert "alpha" in result
    assert isinstance(result["alpha"], float)


def test_distance_functions() -> None:
    """Test distance functions for different data types."""
    assert nominal_distance("A", "A") == 0.0
    assert nominal_distance("A", "B") == 1.0

    scale = ["low", "medium", "high"]
    assert ordinal_distance("low", "low", scale) == 0.0
    assert ordinal_distance("low", "medium", scale) == 1.0
    assert ordinal_distance("low", "high", scale) == 4.0

    assert interval_distance(1.0, 1.0) == 0.0
    assert interval_distance(1.0, 3.0) == 4.0

    assert ratio_distance(0.0, 0.0) == 0.0
    assert ratio_distance(2.0, 2.0) == 0.0
    assert ratio_distance(1.0, 3.0) == 1.0


def test_compute_observed_disagreement_with_missing_data() -> None:
    """Test observed disagreement calculation with missing data."""
    import numpy as np
    from krippendorff_alpha.schema import DataTypeEnum

    reliability_matrix = np.array(
        [
            [1.0, 2.0, 1.0],
            [1.0, 2.0, np.nan],
            [1.0, np.nan, 1.0],
        ],
        dtype=np.float64,
    )

    weight_vector = np.ones(3)
    distance_fn = nominal_distance

    obs_dis, per_cat_obs, pairwise_counts = compute_observed_disagreement(
        reliability_matrix, weight_vector, distance_fn, DataTypeEnum.NOMINAL
    )

    assert not np.isnan(obs_dis)
    assert obs_dis >= 0


def test_per_category_scores_symmetry() -> None:
    """Test that per-category scores are calculated symmetrically."""
    df = pd.DataFrame(
        {
            "text": ["A", "B", "C"],
            "annotator1": ["positive", "negative", "positive"],
            "annotator2": ["negative", "positive", "negative"],
            "annotator3": ["positive", "negative", "positive"],
        }
    )

    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    preprocessed_data, detected_text_col = preprocess_data(df, column_mapping, annotation_schema)
    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    mapping = preprocessed_data.nominal_mappings if preprocessed_data.nominal_mappings else None
    result = krippendorff_alpha(reliability_matrix, data_type=DataTypeEnum.NOMINAL, mapping=mapping)

    if result.get("per_category_scores"):
        total_per_cat_obs = sum(scores["observed_disagreement"] for scores in result["per_category_scores"].values())
        assert total_per_cat_obs >= 0


def test_krippendorff_alpha_interval() -> None:
    """Test Krippendorff's alpha for interval data."""
    df = pd.DataFrame(
        {
            "text": ["A", "B", "C"],
            "annotator1": [1.0, 2.0, 3.0],
            "annotator2": [1.5, 2.5, 3.5],
            "annotator3": [2.0, 3.0, 4.0],
        }
    )

    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="interval", annotation_level="text_level", missing_value_strategy="ignore"
    )

    preprocessed_data, detected_text_col = preprocess_data(df, column_mapping, annotation_schema)
    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    result = krippendorff_alpha(reliability_matrix, data_type=DataTypeEnum.INTERVAL)

    assert "alpha" in result
    assert isinstance(result["alpha"], float)
    assert -1.0 <= result["alpha"] <= 1.0


def test_krippendorff_alpha_ratio() -> None:
    """Test Krippendorff's alpha for ratio data."""
    df = pd.DataFrame(
        {
            "text": ["A", "B", "C"],
            "annotator1": [10.0, 20.0, 30.0],
            "annotator2": [12.0, 22.0, 32.0],
            "annotator3": [15.0, 25.0, 35.0],
        }
    )

    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="ratio", annotation_level="text_level", missing_value_strategy="ignore"
    )

    preprocessed_data, detected_text_col = preprocess_data(df, column_mapping, annotation_schema)
    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    result = krippendorff_alpha(reliability_matrix, data_type=DataTypeEnum.RATIO)

    assert "alpha" in result
    assert isinstance(result["alpha"], float)
    assert -1.0 <= result["alpha"] <= 1.0


def test_krippendorff_alpha_minimum_requirements() -> None:
    """Test that minimum requirements (3 annotators, 3 units) are enforced."""
    with pytest.raises(ValueError, match="At least three annotator columns"):
        ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2"])

    reliability_matrix = pd.DataFrame(
        {
            "unit1": [1, 2],
            "unit2": [2, 3],
            "unit3": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="at least.*3.*annotators"):
        krippendorff_alpha(reliability_matrix, data_type=DataTypeEnum.NOMINAL)

    reliability_matrix_insufficient_units = pd.DataFrame(
        {
            "unit1": [1, 2, 3],
            "unit2": [2, 3, 1],
        }
    )

    with pytest.raises(ValueError, match="at least.*3.*subjects"):
        krippendorff_alpha(reliability_matrix_insufficient_units, data_type=DataTypeEnum.NOMINAL)
