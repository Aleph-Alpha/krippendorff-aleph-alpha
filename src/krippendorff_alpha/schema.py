from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Union, Dict, Any, Set
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
    label: Union[int, str, None] = Field(..., description="Annotation label (can be categorical or numeric)")

    @staticmethod
    @field_validator("label", mode="before")
    def convert_missing_values(cls: type["AnnotationSchema"], v: Union[int, str, None]) -> Union[int, str, None]:
        """Convert missing values to None for consistency."""
        if v is None or (isinstance(v, str) and str(v).lower() in {"", "nan", "none"}):
            return None  # Return None instead of NaN
        return v


class LabelMapping(BaseModel):
    label_to_value: Dict[Union[str, int], Union[int, float]]
    value_to_label: Dict[Union[int, float], Union[str, int]]


class DataSchema(BaseModel):
    annotations: List[AnnotationSchema]
    data_type: DataType
    label_mapping: LabelMapping

    @staticmethod
    @model_validator(mode="before")
    def check_at_least_three_annotators(cls: type["DataSchema"], values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure each item has at least three annotators."""
        if isinstance(values, dict) and "annotations" in values:
            item_rater_count: Dict[Union[int, str], Set[Union[int, str]]] = {}  # Specify the type of the set
            for entry in values["annotations"]:
                item_rater_count.setdefault(entry.item_id, set()).add(entry.annotator_id)
            if any(len(raters) < 3 for raters in item_rater_count.values()):
                raise ValueError("Each item must have ratings from at least three annotators.")
        return values


def load_data_from_dataframe(df: pd.DataFrame, data_type: DataType, label_mapping: LabelMapping) -> DataSchema:
    """Convert a Pandas DataFrame into the DataSchema."""
    if df.isnull().all().any():
        raise ValueError("Dataset contains only NaN values.")

    annotations: List[AnnotationSchema] = []
    for _, row in df.iterrows():
        label = row["label"]

        if data_type in [DataType.ORDINAL, DataType.INTERVAL, DataType.RATIO]:
            if isinstance(label, str):
                label = float(label)
        else:
            if isinstance(label, (int, float)):
                label = str(label)

        annotations.append(AnnotationSchema(item_id=row["item_id"], annotator_id=row["annotator_id"], label=label))

    return DataSchema(annotations=annotations, data_type=data_type, label_mapping=label_mapping)
