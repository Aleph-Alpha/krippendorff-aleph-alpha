from pydantic import BaseModel, Field, validator
from typing import List, Union, Literal, Dict, Set
import pandas as pd
import numpy as np

DataType = Literal["nominal", "ordinal", "interval", "ratio"]


class AnnotationSchema(BaseModel):
    item_id: Union[int, str] = Field(..., description="Unique identifier for the annotated item")
    annotator_id: Union[int, str] = Field(..., description="Unique identifier for the annotator")
    label: Union[int, float, str, None] = Field(..., description="Annotation label (can be categorical or numeric)")

    @validator("label", pre=True, always=True)
    def convert_missing_values(cls, v: Union[int, float, str, None]) -> Union[int, float, str, np.float64]:
        """Convert missing values to np.nan for consistency."""
        if v is None or str(v).lower() in {"", "nan", "none"}:
            return np.float64(np.nan)
        return v


class DataSchema(BaseModel):
    annotations: List[AnnotationSchema] = Field(..., description="List of annotation entries")
    data_type: DataType = Field(..., description="Type of data (nominal, ordinal, interval, ratio)")

    @validator("annotations", pre=True, always=True)
    def check_at_least_two_annotators(cls, v: List[AnnotationSchema]) -> List[AnnotationSchema]:
        """Ensure each item has at least two annotators."""
        item_rater_count: Dict[Union[int, str], Set[Union[int, str]]] = {}
        for entry in v:
            item_rater_count.setdefault(entry.item_id, set()).add(entry.annotator_id)
        if any(len(raters) < 2 for raters in item_rater_count.values()):
            raise ValueError("Each item must have ratings from at least two annotators.")
        return v


def load_data_from_dataframe(df: pd.DataFrame, data_type: DataType) -> DataSchema:
    """Convert a Pandas DataFrame into the DataSchema."""
    if df.isnull().all().any():
        raise ValueError("Dataset contains only NaN values.")

    annotations: List[AnnotationSchema] = [
        AnnotationSchema(item_id=row["item_id"], annotator_id=row["annotator_id"], label=row["label"])
        for _, row in df.iterrows()
    ]
    return DataSchema(annotations=annotations, data_type=data_type)
