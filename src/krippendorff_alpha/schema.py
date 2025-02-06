from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Union, Dict, Set, Any
import pandas as pd
from enum import Enum


class DataType(Enum):
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"
    RATIO = "ratio"


class AnnotationSchema(BaseModel):
    item_id: Union[int, str] = Field(..., description="Unique identifier for the annotated item")
    annotator_id: Union[int, str] = Field(..., description="Unique identifier for the annotator")
    label: Union[int, float, str, None] = Field(..., description="Annotation label (can be categorical or numeric)")

    @staticmethod
    @field_validator("label", mode="before")
    def convert_missing_values(cls: type["AnnotationSchema"], v: Any) -> Union[int, float, str, None]:
        """Convert missing values to NaN for consistency."""
        if v is None or str(v).lower() in {"", "nan", "none"}:
            return float("nan")

        # Ensure `v` is of the correct type before returning
        if isinstance(v, (int, float, str)):
            return v

        raise TypeError(f"Invalid label type: {type(v)}. Expected int, float, str, or None.")


class DataSchema(BaseModel):
    annotations: List[AnnotationSchema]
    data_type: DataType

    @staticmethod
    @model_validator(mode="before")
    def check_at_least_three_annotators(cls: type["DataSchema"], values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure each item has at least three annotators."""
        if isinstance(values, dict) and "annotations" in values:
            item_rater_count: Dict[Union[int, str], Set[Union[int, str]]] = {}
            for entry in values["annotations"]:
                item_rater_count.setdefault(entry.item_id, set()).add(entry.annotator_id)
            if any(len(raters) < 3 for raters in item_rater_count.values()):
                raise ValueError("Each item must have ratings from at least three annotators.")
        return values


def load_data_from_dataframe(df: pd.DataFrame, data_type: DataType) -> DataSchema:
    """Convert a Pandas DataFrame into the DataSchema."""
    if df.isnull().all().any():
        raise ValueError("Dataset contains only NaN values.")

    annotations: List[AnnotationSchema] = [
        AnnotationSchema(item_id=row["item_id"], annotator_id=row["annotator_id"], label=row["label"])
        for _, row in df.iterrows()
    ]
    return DataSchema(annotations=annotations, data_type=data_type)
