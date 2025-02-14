from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import enum


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
    TEXT_LEVEL = "text_level"
    TOKEN_LEVEL = "token_level"  # nosec


class ColumnMapping(BaseModel):
    text_col: Optional[str] = Field(None, description="Column containing the text being annotated.")
    annotator_cols: Optional[List[str]] = Field(
        None, description="List of annotator columns. If None, it will be inferred."
    )

    @field_validator("annotator_cols", mode="before")
    @classmethod
    def validate_annotator_cols(cls, v):
        if v is None:
            return None  # Allow None for later inference

        if not isinstance(v, list):
            raise ValueError("Annotator columns must be provided as a list of strings.")

        if len(v) < 3:
            raise ValueError("At least three annotator columns are required for reliability assessment.")

        return v


class AnnotationSchema(BaseModel):
    data_type: Union[str, DataTypeEnum] = Field(
        ..., description="Type of annotation: nominal, ordinal, interval, or ratio."
    )
    missing_value_strategy: Union[str, MissingValueStrategyEnum] = Field(
        MissingValueStrategyEnum.IGNORE, description="Strategy to handle missing values (ignore, drop, fill)."
    )
    annotation_level: Union[str, AnnotationLevelEnum] = Field(
        ..., description="Annotation level: text_level or token_level."
    )

    @field_validator("data_type", mode="before")
    @classmethod
    def validate_data_type(cls, v):
        if isinstance(v, DataTypeEnum):
            return v  # Already an enum, return as is
        try:
            return DataTypeEnum(v.lower())  # Convert string to enum
        except ValueError:
            raise ValueError(f"Invalid data_type: {v}. Must be one of {list(DataTypeEnum)}.")

    @field_validator("missing_value_strategy", mode="before")
    @classmethod
    def validate_missing_value_strategy(cls, v):
        if isinstance(v, MissingValueStrategyEnum):
            return v
        try:
            return MissingValueStrategyEnum(v.lower())
        except ValueError:
            raise ValueError(f"Invalid missing_value_strategy: {v}. Must be one of {list(MissingValueStrategyEnum)}.")

    @field_validator("annotation_level", mode="before")
    @classmethod
    def validate_annotation_level(cls, v):
        if isinstance(v, AnnotationLevelEnum):
            return v
        try:
            return AnnotationLevelEnum(v.lower())
        except ValueError:
            raise ValueError(f"Invalid annotation_level: {v}. Must be one of {list(AnnotationLevelEnum)}.")

    def get_data_type_mapping(self, annotator_cols: List[str]) -> Dict[str, DataTypeEnum]:
        """Generate a dictionary mapping each annotator column to the specified data type."""
        return {col: self.data_type for col in annotator_cols}


class InputData(BaseModel):
    df: pd.DataFrame = Field(..., description="DataFrame containing text and annotator annotations.")
    column_mapping: ColumnMapping
    annotation_schema: AnnotationSchema

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("df", mode="before")
    @classmethod
    def validate_dataframe_format(cls, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")
        return value

    def to_dataframe(self) -> pd.DataFrame:
        """Return the structured Pandas DataFrame."""
        return self.df


class PreprocessedData(BaseModel):
    df: pd.DataFrame = Field(..., description="Preprocessed Pandas DataFrame ready for analysis.")
    column_mapping: ColumnMapping
    annotation_schema: AnnotationSchema
    ordinal_mappings: Dict[str, Any]
    nominal_mappings: Dict[str, Any]
    model_config = ConfigDict(arbitrary_types_allowed=True)
