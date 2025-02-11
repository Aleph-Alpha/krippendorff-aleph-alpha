import pandas as pd
import numpy as np
import logging
from src.krippendorff_alpha.schema import ColumnMapping
from src.krippendorff_alpha.constants import WORD_COLUMN_ALIASES, TEXT_COLUMN_ALIASES

logging.basicConfig(level=logging.INFO)


def compute_reliability_matrix(df: pd.DataFrame, column_mapping: ColumnMapping) -> pd.DataFrame:
    logging.info("Starting computation of reliability matrix.")
    annotator_cols = column_mapping.annotator_cols

    logging.info(f"Annotator columns identified: {annotator_cols}")

    # Check if any word-based column exists in the dataset
    word_col = next((col for col in WORD_COLUMN_ALIASES if col in df.columns), None)
    text_col = next((col for col in TEXT_COLUMN_ALIASES if col in df.columns), None)

    logging.info(f"Identified word column: {word_col}, text column: {text_col}")

    index_col = word_col if word_col and word_col in df.columns else text_col

    if not annotator_cols or index_col not in df.columns:
        logging.error("Missing annotator columns or a valid word/text column in the data.")
        raise ValueError("Missing annotator columns or a valid word/text column in the data.")

    logging.info(f"Using '{index_col}' as the index column.")

    annotator_matrix = df[annotator_cols].to_numpy(dtype=np.float32)
    text_index = df[index_col].to_numpy()
    reliability_matrix = pd.DataFrame(annotator_matrix, columns=annotator_cols, index=text_index)

    logging.info(f"Reliability matrix computed with shape {reliability_matrix.shape}.")

    return reliability_matrix.T
