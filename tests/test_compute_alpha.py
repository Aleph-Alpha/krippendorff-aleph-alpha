import pandas as pd
import pytest
from krippendorff_alpha.schema import ColumnMapping, AnnotationSchema
from krippendorff_alpha.compute_alpha import compute_alpha


def test_compute_alpha_nominal(df_nominal: pd.DataFrame) -> None:
    print("Testing compute_alpha with nominal data")
    print("Input DataFrame:")
    print(df_nominal)

    column_mapping = ColumnMapping(annotator_cols=["annotator1", "annotator2", "annotator3"], text_col="text")
    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    print("Column Mapping:", column_mapping)
    print("Annotation Schema:", annotation_schema)

    results = compute_alpha(df_nominal, column_mapping, annotation_schema)

    print("Results:", results)
    assert "alpha" in results
    assert isinstance(results["alpha"], float)


def test_compute_alpha_ordinal(df_ordinal: pd.DataFrame) -> None:
    print("Testing compute_alpha with ordinal data")
    print("Input DataFrame:")
    print(df_ordinal)

    column_mapping = ColumnMapping(annotator_cols=["annotator1", "annotator2", "annotator3"], text_col="text")
    annotation_schema = AnnotationSchema(
        data_type="ordinal", annotation_level="text_level", missing_value_strategy="ignore"
    )
    ordinal_scale = {"low": 1.0, "medium": 2.0, "high": 3.0, "very high": 4.0}  # Define ordinal scale

    print("Column Mapping:", column_mapping)
    print("Annotation Schema:", annotation_schema)
    print("Ordinal Scale:", ordinal_scale)

    results = compute_alpha(df_ordinal, column_mapping, annotation_schema, ordinal_scale=ordinal_scale)

    print("Results:", results)
    assert "alpha" in results
    assert isinstance(results["alpha"], float)


def test_compute_alpha_invalid_input() -> None:
    print("Testing compute_alpha with invalid input")
    with pytest.raises(ValueError):
        compute_alpha(None, None, None)
