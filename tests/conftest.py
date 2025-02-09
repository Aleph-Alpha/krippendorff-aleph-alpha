from typing import List, Tuple

import pytest
import os


@pytest.fixture
def example_data() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.join(base_dir, "../examples")
    return example_dir


@pytest.fixture
def datasets() -> List[Tuple[str, str]]:
    return [
        ("interval_numeric_equalGaps_noAbsoluteZero.jsonl", "interval"),
        ("nominal_categorical_noOrder_sample.json", "nominal"),
        ("ordinal_orderedCategories_unequalGaps_sample.csv", "ordinal"),
        ("ratio_numeric_equalGaps_withAbsoluteZero.tsv", "ratio"),
    ]
