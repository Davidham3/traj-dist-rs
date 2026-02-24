from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DistanceType(str, Enum):
    """Distance type enum"""

    EUCLIDEAN = "euclidean"
    SPHERICAL = "spherical"


class ImplementationType(str, Enum):
    """Implementation type enum"""

    CYTHON = "cydist"
    PYTHON = "pydist"


class Metainfo(BaseModel):
    """
    Data model for individual test case metadata.
    Corresponds to each line in metainfo/*.jsonl files.
    """

    algorithm: str = Field(..., description="Algorithm name")
    type_d: DistanceType = Field(..., description="Distance type")
    implemented_by: ImplementationType = Field(
        ..., description="Implementation source (Cython or Python)"
    )
    sample_file: str = Field(..., description="Associated Parquet sample file path")

    # Optional hyperparameters
    eps: Optional[float] = Field(None, description="eps parameter for LCSS and EDR")
    g: Optional[List[float]] = Field(
        None, description="g parameter for ERP (a list of two floats)"
    )
    precision: Optional[int] = Field(None, description="precision parameter for SOWD")
    converted: Optional[bool] = Field(None, description="converted parameter for SOWD")

    model_config = ConfigDict(
        use_enum_values=True  # Use enum values during serialization (e.g., "euclidean" instead of DistanceType.EUCLIDEAN)
    )
