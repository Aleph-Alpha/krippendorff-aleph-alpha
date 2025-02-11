from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any
import pandas as pd
import enum


class ColumnMapping(BaseModel):
    text_col: Optional[str] = Field(None, description="Column containing the text being annotated.")
    annotator_cols: List[str] = Field(..., description="List of annotator columns.")

    @field_validator("annotator_cols", mode="before")
    @classmethod
    def validate_annotator_cols(cls, v: List[str]) -> List[str]:
        if len(v) < 3:
            raise ValueError("At least three annotator columns are required for reliability assessment.")
        return v


class DataTypeEnum(str, enum.Enum):
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"
    RATIO = "ratio"


class AnnotationSchema(BaseModel):
    data_type: Dict[str, str] = Field(
        ..., description="Type of annotation per annotator column: nominal, ordinal, interval, or ratio."
    )
    missing_value_strategy: str = Field("ignore", description="Strategy to handle missing values (ignore, drop, fill).")
    annotation_level: str = Field(..., description="Annotation level: sentence_level or token_level.")


class PreprocessedData(BaseModel):
    df: pd.DataFrame = Field(..., description="Preprocessed Pandas DataFrame ready for analysis.")
    column_mapping: ColumnMapping
    annotation_schema: AnnotationSchema
    ordinal_mappings: Dict[str, Any]
    nominal_mappings: Dict[str, Any]
    model_config = ConfigDict(arbitrary_types_allowed=True)
