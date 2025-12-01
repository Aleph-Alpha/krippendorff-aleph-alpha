import pandas as pd
import logging
from typing import Any
from krippendorff_alpha.schema import ColumnMapping
from krippendorff_alpha.preprocessing import detect_annotator_columns, detect_column
from krippendorff_alpha.constants import get_text_column_aliases

logger = logging.getLogger(__name__)


def compute_reliability_matrix(
    df: pd.DataFrame,
    column_mapping: ColumnMapping | None = None,
    text_col: str | None = None,
    custom_config: dict[str, Any] | None = None,
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

    logger.info("Starting computation of reliability matrix.")

    annotator_cols = column_mapping.annotator_cols if column_mapping else detect_annotator_columns(df, custom_config)
    text_col_aliases = get_text_column_aliases(custom_config)
    text_col = text_col or (column_mapping.text_col if column_mapping else detect_column(df, text_col_aliases))

    logger.info(f"Detected annotator columns: {annotator_cols}")
    logger.info(f"Detected text column: {text_col}")

    if not annotator_cols or text_col not in df.columns:
        logger.error("Missing annotator columns or a valid text column in the data.")
        raise ValueError("Missing annotator columns or a valid text column in the data.")

    annotator_matrix = df[annotator_cols].to_numpy()
    text_index = df[text_col].to_numpy()

    reliability_matrix = pd.DataFrame(annotator_matrix, columns=annotator_cols, index=text_index)

    logger.info(f"Reliability matrix computed with shape {reliability_matrix.shape}.")

    return reliability_matrix.T
