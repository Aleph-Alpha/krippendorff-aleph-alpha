import pandas as pd
from typing import Optional, List, Dict, Union, Literal
from src.krippendorff_alpha.schema import ColumnMapping
from src.krippendorff_alpha.reliability import (
    load_reliability_matrix,
    compute_reliability_matrix,
    update_reliability_matrix,
)
from src.krippendorff_alpha.preprocessing import preprocess_data
from src.krippendorff_alpha.metric import krippendorff_alpha


def compute_alpha(
    path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    update_mode: Literal["auto", "manual"] = "auto",  # Literal for 'auto' or 'manual'
    update_type: Optional[Literal["new_doc", "new_annotator", "new_doc_annotator", "new_task"]] = None,
    text_col: Optional[str] = None,
    annotator_cols: Optional[List[str]] = None,
    weight_dict: Optional[Dict[str, float]] = None,
    metric: str = "nominal",
    ordinal_scale: Optional[List[Union[int, float, str]]] = None,
) -> Dict[str, float]:
    """
    Computes Krippendorff's Alpha for reliability analysis.

    Args:
        path (str): Path to the dataset (CSV, JSON, JSONL, etc.).
        df (pd.DataFrame): DataFrame containing annotations.
        update_mode (str): "auto" (default) or "manual" for updating the reliability matrix.
        update_type (str): Type of update ("new_doc", "new_annotator", "new_doc_annotator", "new_task").
        text_col (str): Column containing the text.
        annotator_cols (list): List of annotator columns.
        weight_dict (dict): Dictionary mapping annotators to weights.
        metric (str): "nominal", "ordinal", "interval", or "ratio".
        ordinal_scale (list): Optional scale for ordinal distance calculation.

    Returns:
        dict: {"alpha": float} with Krippendorff's Alpha value.
    """
    # Load previous reliability matrix (if exists)
    prev_matrix = load_reliability_matrix()

    new_data = preprocess_data(
        path=path,
        df=df,
        annotation_level="sentence_level",
        text_col=text_col,
        annotator_cols=annotator_cols,
    )

    annotator_cols = annotator_cols if annotator_cols is not None else []

    updated_matrix = update_reliability_matrix(
        current_df=prev_matrix if prev_matrix is not None else pd.DataFrame(),
        new_data=new_data,
        column_mapping=ColumnMapping(text_col=text_col, annotator_cols=annotator_cols),
        update_mode=update_mode,
        update_type=update_type,
    )

    reliability_matrix = compute_reliability_matrix(
        updated_matrix, ColumnMapping(text_col=text_col, annotator_cols=annotator_cols)
    )

    alpha_value = krippendorff_alpha(
        reliability_matrix=reliability_matrix.to_numpy()
        if isinstance(reliability_matrix, pd.DataFrame)
        else reliability_matrix,
        metric=metric,
        weight_dict=weight_dict,
        ordinal_scale=ordinal_scale,
    )

    return {"alpha": float(alpha_value["alpha"])}
