import pytest
import pandas as pd


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
