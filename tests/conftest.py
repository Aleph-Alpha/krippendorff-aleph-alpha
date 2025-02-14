import pytest
import pandas as pd

@pytest.fixture
def df_nominal():
    return pd.DataFrame(
        {
            "text": ["Hello world", "Goodbye world", "It is sunny"],
            "annotator1": ["positive", "negative", "positive"],
            "annotator2": ["negative", "positive", "negative"],
            "annotator3": ["positive", "negative", "positive"],
        }
    )

@pytest.fixture
def df_ordinal():
    return pd.DataFrame(
        {
            "text": ["it is very cold", "it is warm"],
            "annotator1": ["low", "medium"],
            "annotator2": ["low", "high"],
            "annotator3": ["medium", "very high"],
        }
    )
