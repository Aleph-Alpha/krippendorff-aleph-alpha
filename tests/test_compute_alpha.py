import pandas as pd
import pytest
import numpy as np
import tempfile
import yaml
from pathlib import Path
from krippendorff_alpha.schema import ColumnMapping
from krippendorff_alpha.compute_alpha import compute_alpha


def test_compute_alpha_nominal(df_nominal: pd.DataFrame) -> None:
    """Test compute_alpha with nominal data."""
    column_mapping = ColumnMapping(annotator_cols=["annotator1", "annotator2", "annotator3"], text_col="text")

    results = compute_alpha(df_nominal, data_type="nominal", column_mapping=column_mapping)

    assert "alpha" in results
    assert isinstance(results["alpha"], float)
    assert -1.0 <= results["alpha"] <= 1.0
    assert "observed_disagreement" in results
    assert "expected_disagreement" in results
    assert "per_category_scores" in results


def test_compute_alpha_ordinal(df_ordinal: pd.DataFrame) -> None:
    """Test compute_alpha with ordinal data."""
    column_mapping = ColumnMapping(annotator_cols=["annotator1", "annotator2", "annotator3"], text_col="text")
    ordinal_scale = ["low", "medium", "high", "very high"]

    results = compute_alpha(df_ordinal, data_type="ordinal", column_mapping=column_mapping, ordinal_scale=ordinal_scale)

    assert "alpha" in results
    assert isinstance(results["alpha"], float)
    assert -1.0 <= results["alpha"] <= 1.0


def test_compute_alpha_default_annotation_level(df_nominal: pd.DataFrame) -> None:
    """Test compute_alpha with default annotation level."""
    results = compute_alpha(df_nominal, data_type="nominal", column_mapping=ColumnMapping())

    assert "alpha" in results
    assert isinstance(results["alpha"], float)


def test_compute_alpha_invalid_input() -> None:
    """Test compute_alpha with invalid input."""
    with pytest.raises(ValueError):
        compute_alpha(None, data_type="nominal")


def test_compute_alpha_with_missing_data() -> None:
    """Test compute_alpha with missing data (NaN values)."""
    df = pd.DataFrame({
        "text": ["A", "B", "C", "D"],
        "annotator1": ["positive", "negative", "positive", "negative"],
        "annotator2": ["positive", "negative", np.nan, "negative"],
        "annotator3": ["positive", np.nan, "positive", "negative"],
    })
    
    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    results = compute_alpha(df, data_type="nominal", column_mapping=column_mapping)

    assert "alpha" in results
    assert isinstance(results["alpha"], float)
    assert not np.isnan(results["alpha"])


def test_compute_alpha_with_weights() -> None:
    """Test compute_alpha with annotator weights."""
    df = pd.DataFrame({
        "text": ["A", "B", "C"],
        "annotator1": ["positive", "negative", "positive"],
        "annotator2": ["positive", "negative", "positive"],
        "annotator3": ["positive", "negative", "positive"],
    })
    
    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    weight_dict = {"annotator1": 1.0, "annotator2": 0.8, "annotator3": 1.0}

    results = compute_alpha(df, data_type="nominal", column_mapping=column_mapping, weight_dict=weight_dict)

    assert "alpha" in results
    assert isinstance(results["alpha"], float)


def test_compute_alpha_interval() -> None:
    """Test compute_alpha with interval data."""
    df = pd.DataFrame({
        "text": ["A", "B", "C"],
        "annotator1": [1.0, 2.0, 3.0],
        "annotator2": [1.5, 2.5, 3.5],
        "annotator3": [2.0, 3.0, 4.0],
    })
    
    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    results = compute_alpha(df, data_type="interval", column_mapping=column_mapping)

    assert "alpha" in results
    assert isinstance(results["alpha"], float)
    assert "per_category_scores" not in results or results["per_category_scores"] is None


def test_compute_alpha_ratio() -> None:
    """Test compute_alpha with ratio data."""
    df = pd.DataFrame({
        "text": ["A", "B", "C"],
        "annotator1": [10.0, 20.0, 30.0],
        "annotator2": [12.0, 22.0, 32.0],
        "annotator3": [15.0, 25.0, 35.0],
    })
    
    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    results = compute_alpha(df, data_type="ratio", column_mapping=column_mapping)

    assert "alpha" in results
    assert isinstance(results["alpha"], float)
    assert "per_category_scores" not in results or results["per_category_scores"] is None


def test_compute_alpha_invalid_data_type() -> None:
    """Test compute_alpha with invalid data type."""
    df = pd.DataFrame({
        "text": ["A", "B", "C"],
        "annotator1": [1, 2, 3],
        "annotator2": [1, 2, 3],
        "annotator3": [1, 2, 3],
    })
    
    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    
    with pytest.raises(ValueError, match="Invalid data_type"):
        compute_alpha(df, data_type="invalid_type", column_mapping=column_mapping)


def test_compute_alpha_per_category_scores() -> None:
    """Test that per-category scores are included for nominal/ordinal data."""
    df = pd.DataFrame({
        "text": ["A", "B", "C"],
        "annotator1": ["positive", "negative", "positive"],
        "annotator2": ["negative", "positive", "negative"],
        "annotator3": ["positive", "negative", "positive"],
    })
    
    column_mapping = ColumnMapping(text_col="text", annotator_cols=["annotator1", "annotator2", "annotator3"])
    results = compute_alpha(df, data_type="nominal", column_mapping=column_mapping)
    
    assert "per_category_scores" in results
    if results["per_category_scores"]:
        for category, scores in results["per_category_scores"].items():
            assert "observed_disagreement" in scores
            assert "expected_disagreement" in scores


def test_compute_alpha_custom_config() -> None:
    """Test compute_alpha with custom configuration file."""
    df = pd.DataFrame({
        "texto": ["A", "B", "C"],
        "annotator1": ["Low", "Medium", "High"],
        "annotator2": ["Low", "Medium", "High"],
        "annotator3": ["Low", "Medium", "High"],
    })
    
    custom_config = {
        "ordinal_categories": {
            "test_scale": [["Low", "Medium", "High"]]
        },
        "text_column_aliases": ["texto", "text"],
        "word_column_aliases": ["word", "token"],
        "annotator_regex": "^(annotator|annotation|label|rater|coder)[_\\s]*[a-zA-Z0-9]+$"
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(custom_config, f)
        config_path = Path(f.name)
    
    try:
        column_mapping = ColumnMapping(text_col="texto", annotator_cols=["annotator1", "annotator2", "annotator3"])
        results = compute_alpha(df, data_type="ordinal", column_mapping=column_mapping, config_path=config_path)
        
        assert "alpha" in results
        assert isinstance(results["alpha"], float)
        assert -1.0 <= results["alpha"] <= 1.0
    finally:
        config_path.unlink()
