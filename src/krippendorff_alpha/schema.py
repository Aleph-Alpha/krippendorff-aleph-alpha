from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Any
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
    text_col: str | None = Field(None, description="Column containing the text being annotated.")
    annotator_cols: list[str] | None = Field(
        None, description="List of annotator columns. If None, it will be inferred."
    )

    @field_validator("annotator_cols", mode="before")
    @classmethod
    def validate_annotator_cols(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("Annotator columns must be provided as a list of strings.")

        if len(v) < 3:
            raise ValueError("At least three annotator columns are required for reliability assessment.")

        return v


class AnnotationSchema(BaseModel):
    data_type: str | DataTypeEnum = Field(..., description="Type of annotation: nominal, ordinal, interval, or ratio.")
    missing_value_strategy: str | MissingValueStrategyEnum = Field(
        MissingValueStrategyEnum.IGNORE, description="Strategy to handle missing values (ignore, drop, fill)."
    )
    annotation_level: str | AnnotationLevelEnum = Field(..., description="Annotation level: text_level or token_level.")

    @classmethod
    def _validate_enum_field(
        cls, v: str | enum.Enum, enum_class: type[enum.Enum], field_name: str
    ) -> enum.Enum:
        """
        Generic validator for enum fields that handles both string and enum inputs.
        
        Args:
            v: The value to validate (can be string or enum instance)
            enum_class: The enum class to validate against
            field_name: Name of the field being validated (for error messages)
            
        Returns:
            The validated enum instance
            
        Raises:
            ValueError: If the value cannot be converted to a valid enum value
        """
        if isinstance(v, enum_class):
            return v
        try:
            return enum_class(v.lower())
        except ValueError:
            valid_values = [e.value for e in enum_class]
            raise ValueError(
                f"Invalid {field_name}: {v}. Must be one of {valid_values}."
            )

    @field_validator("data_type", mode="before")
    @classmethod
    def validate_data_type(cls, v: str | DataTypeEnum) -> DataTypeEnum:
        return cls._validate_enum_field(v, DataTypeEnum, "data_type")

    @field_validator("missing_value_strategy", mode="before")
    @classmethod
    def validate_missing_value_strategy(cls, v: str | MissingValueStrategyEnum) -> MissingValueStrategyEnum:
        return cls._validate_enum_field(v, MissingValueStrategyEnum, "missing_value_strategy")

    @field_validator("annotation_level", mode="before")
    @classmethod
    def validate_annotation_level(cls, v: str | AnnotationLevelEnum) -> AnnotationLevelEnum:
        return cls._validate_enum_field(v, AnnotationLevelEnum, "annotation_level")

    def get_data_type_mapping(self, annotator_cols: list[str]) -> dict[str, DataTypeEnum]:
        return {col: DataTypeEnum(self.data_type) for col in annotator_cols}


class PreprocessedData(BaseModel):
    df: pd.DataFrame = Field(..., description="Preprocessed Pandas DataFrame ready for analysis.")
    column_mapping: ColumnMapping
    annotation_schema: AnnotationSchema
    ordinal_mappings: dict[str, Any]
    nominal_mappings: dict[str, Any]
    model_config = ConfigDict(arbitrary_types_allowed=True)
