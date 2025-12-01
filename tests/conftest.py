import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def df_nominal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["Hello world", "Goodbye world", "It is sunny"],
            "annotator1": ["positive", "negative", "positive"],
            "annotator2": ["negative", "positive", "negative"],
            "annotator3": ["positive", "negative", "positive"],
        }
    )


@pytest.fixture
def df_ordinal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["it is very cold", "it is warm", "it is hot"],
            "annotator1": ["low", "medium", "high"],
            "annotator2": ["low", "high", "very high"],
            "annotator3": ["medium", "very high", "very high"],
        }
    )


@pytest.fixture
def df_nominal_with_missing() -> pd.DataFrame:
    """Fixture with missing data (NaN values)."""
    return pd.DataFrame(
        {
            "text": ["A", "B", "C", "D"],
            "annotator1": ["positive", "negative", "positive", "negative"],
            "annotator2": ["positive", "negative", np.nan, "negative"],
            "annotator3": ["positive", np.nan, "positive", "negative"],
        }
    )


@pytest.fixture
def df_perfect_agreement() -> pd.DataFrame:
    """Fixture with perfect agreement (all annotators agree)."""
    return pd.DataFrame(
        {
            "text": ["A", "B", "C"],
            "annotator1": ["positive", "negative", "positive"],
            "annotator2": ["positive", "negative", "positive"],
            "annotator3": ["positive", "negative", "positive"],
        }
    )


@pytest.fixture
def df_interval() -> pd.DataFrame:
    """Fixture with interval data."""
    return pd.DataFrame(
        {
            "text": ["A", "B", "C"],
            "annotator1": [1.0, 2.0, 3.0],
            "annotator2": [1.5, 2.5, 3.5],
            "annotator3": [2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def df_ratio() -> pd.DataFrame:
    """Fixture with ratio data."""
    return pd.DataFrame(
        {
            "text": ["A", "B", "C"],
            "annotator1": [10.0, 20.0, 30.0],
            "annotator2": [12.0, 22.0, 32.0],
            "annotator3": [15.0, 25.0, 35.0],
        }
    )
