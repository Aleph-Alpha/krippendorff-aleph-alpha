import pandas as pd
import logging
from krippendorff_alpha.schema import ColumnMapping
from krippendorff_alpha.preprocessing import detect_annotator_columns, detect_column
from krippendorff_alpha.constants import TEXT_COLUMN_ALIASES

logging.basicConfig(level=logging.INFO)


def compute_reliability_matrix(
    df: pd.DataFrame, column_mapping: ColumnMapping | None = None, text_col: str | None = None
) -> pd.DataFrame:
    """
    Computes the reliability matrix for the given DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_mapping (Optional[ColumnMapping]): Column mapping (optional, inferred if not provided).
    - text_col (Optional[str]): The detected text column from preprocessing (optional).

    Returns:
    - pd.DataFrame: The transposed reliability matrix with annotator columns as rows and text indices as columns.
    """

    logging.info("Starting computation of reliability matrix.")

    # Detect columns if column_mapping is not provided
    annotator_cols = column_mapping.annotator_cols if column_mapping else detect_annotator_columns(df)
    text_col = text_col or (column_mapping.text_col if column_mapping else detect_column(df, TEXT_COLUMN_ALIASES))

    logging.info(f"Detected annotator columns: {annotator_cols}")
    logging.info(f"Detected text column: {text_col}")

    if not annotator_cols or text_col not in df.columns:
        logging.error("Missing annotator columns or a valid text column in the data.")
        raise ValueError("Missing annotator columns or a valid text column in the data.")

    # Convert DataFrame to reliability matrix format
    annotator_matrix = df[annotator_cols].to_numpy()
    text_index = df[text_col].to_numpy()

    reliability_matrix = pd.DataFrame(annotator_matrix, columns=annotator_cols, index=text_index)

    logging.info(f"Reliability matrix computed with shape {reliability_matrix.shape}.")

    return reliability_matrix.T
