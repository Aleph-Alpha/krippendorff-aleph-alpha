import pandas as pd

from src.krippendorff_alpha.preprocessing import preprocess_data
from src.krippendorff_alpha.reliability import compute_reliability_matrix
from src.krippendorff_alpha.schema import AnnotationSchema, ColumnMapping
from src.krippendorff_alpha.metric import krippendorff_alpha


def test_krippendorff_alpha(df_nominal: pd.DataFrame) -> None:
    column_mapping = ColumnMapping(text_col=None, annotator_cols=["annotator1", "annotator2", "annotator3"])
    annotation_schema = AnnotationSchema(
        data_type="nominal", annotation_level="text_level", missing_value_strategy="ignore"
    )

    preprocessed_data, detected_text_col = preprocess_data(df_nominal, column_mapping, annotation_schema)

    if preprocessed_data.nominal_mappings:
        preprocessed_data.nominal_mappings = {str(k): v for k, v in preprocessed_data.nominal_mappings.items()}

    # Extract the appropriate mapping based on the data type
    if annotation_schema.data_type == "nominal":
        mapping = preprocessed_data.nominal_mappings
    elif annotation_schema.data_type == "ordinal":
        mapping = preprocessed_data.ordinal_mappings
    else:
        mapping = None

    # Compute the reliability matrix
    reliability_matrix = compute_reliability_matrix(
        preprocessed_data.df, preprocessed_data.column_mapping, detected_text_col
    )

    print("Nominal mappings:", preprocessed_data.nominal_mappings)
    print("Nominal mapping keys:", list(preprocessed_data.nominal_mappings.keys()))
    print("Nominal mapping values:", list(preprocessed_data.nominal_mappings.values()))

    # Call Krippendorff's alpha, passing the automatically selected mapping
    result = krippendorff_alpha(reliability_matrix, data_type=annotation_schema.data_type, mapping=mapping)

    # Print the result to see the alpha value and other metrics
    print("Krippendorff's Alpha Calculation Result:")
    print(result)
