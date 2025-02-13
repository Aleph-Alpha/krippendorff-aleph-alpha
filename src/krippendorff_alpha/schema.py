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


class MissingValueStrategyEnum(str, enum.Enum):
    IGNORE = "ignore"
    DROP = "drop"
    FILL = "fill"


class AnnotationLevelEnum(str, enum.Enum):
    SENTENCE_LEVEL = "sentence_level"
    TOKEN_LEVEL = "token_level" # nosec


class AnnotationSchema(BaseModel):
    data_type: Dict[str, DataTypeEnum] = Field(
        ..., description="Type of annotation per annotator column: nominal, ordinal, interval, or ratio."
    )
    missing_value_strategy: MissingValueStrategyEnum = Field(
        MissingValueStrategyEnum.IGNORE, description="Strategy to handle missing values (ignore, drop, fill)."
    )
    annotation_level: AnnotationLevelEnum = Field(..., description="Annotation level: sentence_level or token_level.")


class PreprocessedData(BaseModel):
    df: pd.DataFrame = Field(..., description="Preprocessed Pandas DataFrame ready for analysis.")
    column_mapping: ColumnMapping
    annotation_schema: AnnotationSchema
    ordinal_mappings: Dict[str, Any]
    nominal_mappings: Dict[str, Any]
    model_config = ConfigDict(arbitrary_types_allowed=True)
