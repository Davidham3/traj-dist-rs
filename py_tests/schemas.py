from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DistanceType(str, Enum):
    """距离类型枚举"""

    EUCLIDEAN = "euclidean"
    SPHERICAL = "spherical"


class ImplementationType(str, Enum):
    """实现类型枚举"""

    CYTHON = "cydist"
    PYTHON = "pydist"


class Metainfo(BaseModel):
    """
    单个测试用例元数据的数据模型。
    对应 metainfo/*.jsonl 文件中的每一行。
    """

    algorithm: str = Field(..., description="算法名称")
    type_d: DistanceType = Field(..., description="距离类型")
    implemented_by: ImplementationType = Field(
        ..., description="实现来源 (Cython 或 Python)"
    )
    sample_file: str = Field(..., description="关联的 Parquet 样本文件路径")

    # 可选的超参数
    eps: Optional[float] = Field(None, description="LCSS 和 EDR 的 eps 参数")
    g: Optional[List[float]] = Field(
        None, description="ERP 的 g 参数 (一个包含两个浮点数的列表)"
    )
    precision: Optional[int] = Field(None, description="SOWD 的 precision 参数")
    converted: Optional[bool] = Field(None, description="SOWD 的 converted 参数")

    model_config = ConfigDict(
        use_enum_values=True  # 在序列化时使用枚举的值 (例如 "euclidean" 而不是 DistanceType.EUCLIDEAN)
    )
