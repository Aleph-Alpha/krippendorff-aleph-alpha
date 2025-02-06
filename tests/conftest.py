import pytest
from pathlib import Path
from typing import Dict


@pytest.fixture
def example_files() -> Dict[str, Path]:
    """Provide paths to example files."""
    root_dir = Path(__file__).resolve().parent.parent  # Move up to the project root
    files = {
        "json": root_dir / "examples/nominal_categorical_noOrder_sample.json",
        "jsonl": root_dir / "examples/interval_numeric_equalGaps_noAbsoluteZero.jsonl",
        "csv": root_dir / "examples/ordinal_orderedCategories_unequalGaps_sample.csv",
        "tsv": root_dir / "examples/ratio_numeric_equalGaps_withAbsoluteZero.tsv",
    }

    for key, file in files.items():
        if not file.exists():
            pytest.fail(f"Test file for {key} is missing: {file}")

    return files
