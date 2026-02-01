"""
统一测试框架基类

提供轨迹距离算法测试的通用方法和工具
"""

import pytest
import math
from pathlib import Path
from typing import Optional, Union, List, Dict, TypedDict
from abc import ABC, abstractmethod
from pydantic import ValidationError
from schemas import Metainfo
import traj_dist_rs._lib as traj_dist_rs


class AlgorithmConfig(TypedDict):
    """算法配置类型定义"""

    name: str  # 算法名称
    function_name: str  # Python 绑定的函数名
    distance_types: List[str]  # 支持的距离类型 ["euclidean", "spherical"]
    has_hyperparameters: bool  # 是否有超参数
    hyperparameter_name: Optional[str]  # 超参数名称
    hyperparameter_values: Optional[
        Dict[str, List[Union[float, List[float]]]]
    ]  # 超参数值配置


def get_algorithm_config_from_metainfo(
    algorithm: str, all_metainfo: Optional[Dict[str, List[Metainfo]]] = None
) -> AlgorithmConfig:
    """
    从元数据动态获取算法配置信息

    Args:
        algorithm: 算法名称
        all_metainfo: 所有元数据（Pydantic 模型对象列表）

    Returns:
        算法配置字典
    """
    if all_metainfo is None:
        all_metainfo = load_all_metainfo_from_data_dir()

    if algorithm not in all_metainfo:
        raise KeyError(f"算法 {algorithm} 的元数据不存在")

    metainfo_list = all_metainfo[algorithm]

    # 获取算法的基本信息
    first_metainfo = metainfo_list[0] if metainfo_list else None
    if not first_metainfo:
        raise ValueError(f"算法 {algorithm} 的元数据列表为空")

    # 找出该算法支持的距离类型 (使用 Pydantic 模型属性)
    distance_types = list({m.type_d for m in metainfo_list})

    # 检查是否包含超参数字段 (使用 Pydantic 模型属性)
    has_hyperparameters = any(
        m.eps is not None or m.g is not None or m.precision is not None
        for m in metainfo_list
    )

    # 确定超参数名称 (使用 Pydantic 模型属性)
    hyperparameter_name = None
    if any(m.eps is not None for m in metainfo_list):
        hyperparameter_name = "eps"
    elif any(m.g is not None for m in metainfo_list):
        hyperparameter_name = "g"
    elif any(m.precision is not None for m in metainfo_list):
        hyperparameter_name = "precision"

    # 生成算法名称（首字母大写）
    name = algorithm.replace("_", " ").title()

    config = {
        "name": name,
        "function_name": algorithm,
        "distance_types": distance_types,
        "has_hyperparameters": has_hyperparameters,
        "hyperparameter_name": hyperparameter_name,
        "hyperparameter_values": None,  # 从元数据获取值，而不是硬编码
    }

    return config


def get_algorithm_config(algorithm: str) -> AlgorithmConfig:
    """获取指定算法的配置（从元数据动态加载）"""
    return get_algorithm_config_from_metainfo(algorithm)


def load_all_metainfo_from_data_dir(
    data_dir: Optional[Path] = None,
) -> Dict[str, List[Metainfo]]:
    """
    从 data 目录动态加载所有元数据，并使用 Pydantic 模型进行解析

    Args:
        data_dir: 数据目录（默认为 py_tests/data）

    Returns:
        字典，键为算法名称，值为 Metainfo 对象列表
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    metainfo_dir = data_dir / "metainfo"
    all_metainfo = {}

    for metainfo_file in metainfo_dir.glob("*.jsonl"):
        algorithm = metainfo_file.stem
        metainfo_list = []
        with open(metainfo_file, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        # 使用 Pydantic 模型解析和验证 JSON 行
                        meta_obj = Metainfo.model_validate_json(line)
                        metainfo_list.append(meta_obj)
                    except ValidationError as e:
                        print(
                            f"警告: 解析文件 {metainfo_file.name} 第 {i+1} 行失败: {e}"
                        )
        all_metainfo[algorithm] = metainfo_list

    return all_metainfo


def load_metainfo(
    algorithm: str, data_dir: Optional[Path] = None
) -> List[Metainfo]:
    """
    加载指定算法的元数据

    Args:
        algorithm: 算法名称
        data_dir: 数据目录（默认为 py_tests/data）

    Returns:
        Metainfo 对象列表
    """
    all_metainfo = load_all_metainfo_from_data_dir(data_dir)
    if algorithm not in all_metainfo:
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        metainfo_file = data_dir / "metainfo" / f"{algorithm}.jsonl"
        raise FileNotFoundError(f"元数据文件不存在: {metainfo_file}")
    return all_metainfo[algorithm]


def get_sample_path(metainfo: Metainfo, data_dir: Optional[Path] = None) -> Path:
    """
    从元数据对象获取样本文件路径

    Args:
        metainfo: Metainfo 对象
        data_dir: 数据目录（默认为 py_tests/data）

    Returns:
        样本文件路径
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    # 使用属性访问代替字典键访问
    return data_dir / metainfo.sample_file


class BaseDistanceTest(ABC):
    """
    距离算法测试基类

    提供通用的测试方法，子类只需指定算法名称即可自动生成测试用例
    """

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """算法名称，子类必须实现"""
        pass

    @property
    def config(self):
        """获取算法配置"""
        return get_algorithm_config(self.algorithm_name)

    @property
    def function_name(self) -> str:
        """获取 Python 绑定的函数名"""
        return self.config["function_name"]

    def _get_hyperparameter_value_from_metainfo(
        self, all_metainfo: Dict[str, List[Metainfo]], distance_type: str
    ) -> Optional[Union[float, List[float], int]]:
        """
        从元数据中获取超参数值 (已更新以使用 Pydantic 模型)
        """
        algorithm_metainfo = all_metainfo.get(self.algorithm_name, [])
        return HyperparameterUtil.get_hyperparameter_value(
            self.algorithm_name, distance_type, algorithm_metainfo
        )

    def _get_distance_function(self):
        """获取距离计算函数"""
        try:
            import traj_dist_rs._lib as traj_dist_rs

            return getattr(traj_dist_rs, self.function_name)
        except ImportError:
            pytest.skip("traj_dist_rs 模块未安装，请先运行 'maturin develop'")
        except AttributeError:
            pytest.skip(f"函数 {self.function_name} 未实现")

    def _call_distance_function(
        self,
        traj1: List[List[float]],
        traj2: List[List[float]],
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
    ) -> float:
        """
        调用距离计算函数

        Args:
            traj1: 轨迹1
            traj2: 轨迹2
            distance_type: 距离类型
            hyperparameter_value: 超参数值（可选）

        Returns:
            计算得到的距离值
        """
        func = self._get_distance_function()
        config = self.config

        # 特殊处理 DTW 算法，它有 use_full_matrix 参数但没有超参数
        if self.algorithm_name == "dtw":
            result = func(traj1, traj2, distance_type, use_full_matrix=False)
            # DTW 现在返回 DpResult 对象
            if hasattr(result, 'distance'):
                return result.distance
            return result

        # 特殊处理 LCSS、EDR 算法，它们有 use_full_matrix 参数和 eps 超参数
        if self.algorithm_name in ["lcss", "edr"]:
            param_name = config["hyperparameter_name"]
            if hyperparameter_value is None:
                hyperparameter_value = 1e-6
            result = func(traj1, traj2, distance_type, eps=hyperparameter_value, use_full_matrix=False)
            # 这些算法现在返回 DpResult 对象
            if hasattr(result, 'distance'):
                return result.distance
            return result

        # 特殊处理 ERP 算法，它有 use_full_matrix 参数和 g 超参数
        if self.algorithm_name == "erp":
            param_name = config["hyperparameter_name"]
            if hyperparameter_value is None:
                hyperparameter_value = [0.0, 0.0]
            # 获取实际的函数名（可能是 erp_compat_traj_dist 或 erp_standard）
            func_name = self.function_name
            if func_name == "erp_compat_traj_dist":
                result = traj_dist_rs.erp_compat_traj_dist(traj1, traj2, distance_type, g=hyperparameter_value, use_full_matrix=False)
            elif func_name == "erp_standard":
                result = traj_dist_rs.erp_standard(traj1, traj2, distance_type, g=hyperparameter_value, use_full_matrix=False)
            else:
                result = func(traj1, traj2, distance_type, g=hyperparameter_value, use_full_matrix=False)
            # ERP 现在返回 DpResult 对象
            if hasattr(result, 'distance'):
                return result.distance
            return result

        # 特殊处理 Discret Frechet 算法，它有 use_full_matrix 参数但没有超参数
        if self.algorithm_name == "discret_frechet":
            result = func(traj1, traj2, distance_type, use_full_matrix=False)
            # Discret Frechet 现在返回 DpResult 对象
            if hasattr(result, 'distance'):
                return result.distance
            return result

        if not config["has_hyperparameters"]:
            result = func(traj1, traj2, distance_type)
            # 检查是否返回 DpResult 对象
            if hasattr(result, 'distance'):
                return result.distance
            return result
        else:
            # 如果没有提供超参数值，使用默认值
            if hyperparameter_value is None:
                param_name = config["hyperparameter_name"]
                # 从配置中获取默认值，如果配置中没有则使用默认值
                if param_name == "eps":
                    # 对于 eps 参数，使用一个非常小的值，这样对于大多数情况都可以匹配
                    hyperparameter_value = 1e-6
                elif param_name == "g":
                    # 对于 g 参数（ERP），使用 0,0，这样惩罚将等同于欧几里得距离
                    hyperparameter_value = [0.0, 0.0]
                elif param_name == "precision":
                    # 对于 precision 参数（SOWD），使用默认值 5
                    hyperparameter_value = 5

            param_name = config["hyperparameter_name"]
            if param_name == "g" and isinstance(hyperparameter_value, list):
                # ERP 的 g 参数是列表
                result = func(traj1, traj2, distance_type, g=hyperparameter_value)
            elif param_name == "precision":
                # SOWD 的 precision 参数
                result = func(traj1, traj2, distance_type, precision=hyperparameter_value)
            else:
                # LCSS 和 EDR 的 eps 参数
                result = func(traj1, traj2, distance_type, eps=hyperparameter_value)

            # 检查是否返回 DpResult 对象
            if hasattr(result, 'distance'):
                return result.distance
            return result

    def _get_expected_distance(self, test_data, hyperparameter_value=None):
        """
        从测试数据中获取期望距离

        Args:
            test_data: 测试数据 DataFrame
            hyperparameter_value: 超参数值（用于验证）

        Returns:
            期望距离列表
        """
        if test_data is None:
            pytest.skip(f"{self.algorithm_name} 测试数据不存在")

        distances = []
        for row in test_data.iter_rows(named=True):
            distances.append((row["traj1"], row["traj2"], row["distance"]))
        return distances

    def _test_accuracy(
        self,
        test_data,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        tolerance: float = 1e-8,
    ):
        """
        测试准确率的通用方法

        Args:
            test_data: 测试数据
            distance_type: 距离类型
            hyperparameter_value: 超参数值（可选）
            tolerance: 容差
        """
        test_cases = self._get_expected_distance(test_data, hyperparameter_value)

        for traj1, traj2, expected_distance in test_cases:
            actual_distance = self._call_distance_function(
                traj1, traj2, distance_type, hyperparameter_value
            )

            # 处理 DpResult 返回类型
            if hasattr(actual_distance, 'distance'):
                actual_distance = actual_distance.distance

            error = abs(actual_distance - expected_distance)
            assert error < tolerance, (
                f"{self.algorithm_name.upper()} 距离计算误差过大: {error}\n"
                f"距离类型: {distance_type}\n"
                f"超参数: {hyperparameter_value}\n"
                f"轨迹1: {traj1}\n"
                f"轨迹2: {traj2}\n"
                f"期望距离: {expected_distance}\n"
                f"实际距离: {actual_distance}"
            )

    def _check_identical_trajectories(
        self,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        tolerance: float = 1e-8,
        **kwargs,
    ):
        """
        检查相同轨迹的距离应该接近 0

        Args:
            distance_type: 距离类型
            hyperparameter_value: 超参数值（可选）
            tolerance: 容差
            **kwargs: 其他超参数（如 g, eps, precision）
        """
        # 优先使用 kwargs 中的超参数
        if kwargs:
            hyperparameter_value = kwargs.get(
                "eps", kwargs.get("g", kwargs.get("precision", hyperparameter_value))
            )

        if distance_type == "spherical":
            traj = [[-122.39548, 37.77668], [-122.39539, 37.77644]]
        else:
            traj = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]

        distance = self._call_distance_function(
            traj, traj, distance_type, hyperparameter_value
        )

        # 处理 DpResult 返回类型
        if hasattr(distance, 'distance'):
            distance = distance.distance

        # 对于 LCSS 和 EDR，距离在 [0, 1] 范围内，相同轨迹应该接近 0
        if self.algorithm_name in ["lcss", "edr"]:
            assert (
                distance < tolerance
            ), f"{self.algorithm_name.upper()} 相同轨迹的距离应该接近 0，实际: {distance}"
        else:
            assert (
                distance < tolerance
            ), f"{self.algorithm_name.upper()} 相同轨迹的距离应该接近 0，实际: {distance}"

    def _check_simple_case(
        self,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        **kwargs,
    ):
        """
        检查简单轨迹对

        Args:
            distance_type: 距离类型
            hyperparameter_value: 超参数值（可选）
            **kwargs: 其他超参数（如 g, eps, precision）
        """
        # 优先使用 kwargs 中的超参数
        if kwargs:
            hyperparameter_value = kwargs.get(
                "eps", kwargs.get("g", kwargs.get("precision", hyperparameter_value))
            )

        if distance_type == "spherical":
            traj1 = [[-122.39548, 37.77668], [-122.39539, 37.77644]]
            traj2 = [[-122.39548, 37.77668], [-122.39537, 37.77663]]
        else:
            traj1 = [[0.0, 0.0], [1.0, 1.0]]
            traj2 = [[0.0, 1.0], [1.0, 0.0]]

        distance = self._call_distance_function(
            traj1, traj2, distance_type, hyperparameter_value
        )

        # 处理 DpResult 返回类型
        if hasattr(distance, 'distance'):
            distance = distance.distance

        # 距离应该大于 0
        assert (
            distance > 0
        ), f"{self.algorithm_name.upper()} 不同轨迹的距离应该大于 0，实际: {distance}"

    def _check_empty_trajectory(
        self,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        **kwargs,
    ):
        """
        检查空轨迹的情况

        Args:
            distance_type: 距离类型
            hyperparameter_value: 超参数值（可选）
            **kwargs: 其他超参数（如 g, eps, precision）
        """
        # 优先使用 kwargs 中的超参数
        if kwargs:
            hyperparameter_value = kwargs.get(
                "eps", kwargs.get("g", kwargs.get("precision", hyperparameter_value))
            )

        if distance_type == "spherical":
            traj1 = [[-122.39548, 37.77668], [-122.39539, 37.77644]]
        else:
            traj1 = [[0.0, 0.0], [1.0, 1.0]]

        traj2 = []

        distance = self._call_distance_function(
            traj1, traj2, distance_type, hyperparameter_value
        )

        # 处理 DpResult 返回类型
        if hasattr(distance, 'distance'):
            distance = distance.distance

        # 对于 LCSS 和 EDR，空轨迹应该返回 1.0
        if self.algorithm_name in ["lcss", "edr"]:
            assert (
                distance == 1.0
            ), f"{self.algorithm_name.upper()} 空轨迹应该返回 1.0，实际: {distance}"
        else:
            # 其他算法应该返回最大值
            assert (
                distance == float("inf") or distance > 1e100
            ), f"{self.algorithm_name.upper()} 空轨迹应该返回最大值，实际: {distance}"

    def _check_single_point(
        self,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        **kwargs,
    ):
        """
        检查单点轨迹

        注意：某些算法（如 SSPD）单点轨迹没有段，会返回 inf

        Args:
            distance_type: 距离类型
            hyperparameter_value: 超参数值（可选）
            **kwargs: 其他超参数（如 g, eps, precision）
        """
        # 优先使用 kwargs 中的超参数
        if kwargs:
            hyperparameter_value = kwargs.get(
                "eps", kwargs.get("g", kwargs.get("precision", hyperparameter_value))
            )

        if distance_type == "spherical":
            traj1 = [[-122.39548, 37.77668]]
            traj2 = [[-122.39539, 37.77644]]
        else:
            traj1 = [[0.0, 0.0]]
            traj2 = [[1.0, 1.0]]

        distance = self._call_distance_function(
            traj1, traj2, distance_type, hyperparameter_value
        )

        # 处理 DpResult 返回类型
        if hasattr(distance, 'distance'):
            distance = distance.distance

        # SSPD 单点轨迹没有段，返回 inf
        if self.algorithm_name == "sspd":
            assert distance == float("inf"), (
                f"{self.algorithm_name.upper()} 单点轨迹的距离应该是 inf\n"
                f"实际距离: {distance}"
            )
        # LCSS 和 EDR 单点轨迹不匹配，返回 1.0
        elif self.algorithm_name in ["lcss", "edr"]:
            if (
                hyperparameter_value is None
                or isinstance(hyperparameter_value, (int, float))
                and hyperparameter_value < 1.0
            ):
                assert (
                    distance == 1.0
                ), f"{self.algorithm_name.upper()} 单点不匹配轨迹的距离应该是 1.0，实际: {distance}"
            else:
                # 大 eps 时应该匹配
                assert (
                    distance < 1e-8
                ), f"{self.algorithm_name.upper()} 大 eps 时单点匹配轨迹的距离应该接近 0，实际: {distance}"
        # DTW 和 Hausdorff 单点轨迹的距离应该是两点之间的距离
        elif self.algorithm_name in ["dtw", "hausdorff"]:
            if distance_type == "spherical":
                # 球面距离，使用 Haversine 公式
                # 简单验证距离 > 0
                assert distance > 0, (
                    f"{self.algorithm_name.upper()} 单点轨迹的距离应该大于 0\n"
                    f"实际距离: {distance}"
                )
            else:
                expected_distance = math.sqrt((1.0 - 0.0) ** 2 + (1.0 - 0.0) ** 2)
                assert abs(distance - expected_distance) < 1e-6, (
                    f"{self.algorithm_name.upper()} 单点轨迹的距离应该是两点之间的欧几里得距离\n"
                    f"期望距离: {expected_distance}\n"
                    f"实际距离: {distance}"
                )

    def _check_invalid_distance_type(self):
        """检查无效的距离类型应该抛出异常"""
        traj1 = [[0.0, 0.0], [1.0, 1.0]]
        traj2 = [[0.0, 1.0], [1.0, 0.0]]

        with pytest.raises(ValueError, match="Invalid distance type"):
            self._call_distance_function(traj1, traj2, "invalid")

    def _check_valid_distance_types(self):
        """检查有效的距离类型"""
        traj1 = [[0.0, 0.0], [1.0, 1.0]]
        traj2 = [[0.0, 1.0], [1.0, 0.0]]

        for distance_type in self.config["distance_types"]:
            distance = self._call_distance_function(traj1, traj2, distance_type)
            # 处理 DpResult 返回类型
            if hasattr(distance, 'distance'):
                distance = distance.distance
            assert (
                distance > 0
            ), f"{self.algorithm_name.upper()} {distance_type} 距离应该大于 0，实际: {distance}"

    def _check_invalid_trajectory_format(self):
        """检查无效的轨迹格式应该抛出异常"""
        # 测试非列表输入
        with pytest.raises(ValueError):
            self._call_distance_function("not a list", [[0.0, 1.0]], "euclidean")

        # 测试点坐标不是 2 个元素
        with pytest.raises(ValueError):
            self._call_distance_function([[0.0, 1.0, 2.0]], [[0.0, 1.0]], "euclidean")


class DistanceTestWithHyperparameters(BaseDistanceTest):
    """
    带超参数的距离算法测试基类

    继承自 BaseDistanceTest，添加超参数相关的测试方法
    """

    def _check_hyperparameter_effect(
        self,
        distance_type: str,
        small_value: Union[float, int],
        large_value: Union[float, int],
    ):
        """
        检查超参数的影响

        Args:
            distance_type: 距离类型
            small_value: 小超参数值
            large_value: 大超参数值
        """
        if distance_type == "spherical":
            traj1 = [[-122.39548, 37.77668], [-122.39539, 37.77644]]
            traj2 = [[-122.39548, 37.77668], [-122.39537, 37.77663]]
        else:
            traj1 = [[0.0, 0.0], [1.0, 1.0]]
            traj2 = [[0.0, 0.1], [1.0, 1.1]]

        param_name = self.config["hyperparameter_name"]

        if param_name == "eps":
            # 对于 LCSS 和 EDR，大 eps 产生更小的距离
            distance_small = self._call_distance_function(
                traj1, traj2, distance_type, small_value
            )
            distance_large = self._call_distance_function(
                traj1, traj2, distance_type, large_value
            )

            # 处理 DpResult 返回类型
            if hasattr(distance_small, 'distance'):
                distance_small = distance_small.distance
            if hasattr(distance_large, 'distance'):
                distance_large = distance_large.distance

            assert distance_large <= distance_small, (
                f"{self.algorithm_name.upper()} 大 eps 应该产生更小的距离\n"
                f"小 eps 距离: {distance_small}\n"
                f"大 eps 距离: {distance_large}"
            )
        elif param_name == "g":
            # 对于 ERP，g 参数的影响
            distance_small = self._call_distance_function(
                traj1, traj2, distance_type, small_value
            )
            distance_large = self._call_distance_function(
                traj1, traj2, distance_type, large_value
            )

            # 处理 DpResult 返回类型
            if hasattr(distance_small, 'distance'):
                distance_small = distance_small.distance
            if hasattr(distance_large, 'distance'):
                distance_large = distance_large.distance

            # ERP 的距离受 g 参数影响，但不一定是单调的
            # 这里只验证距离是合理的
            assert distance_small >= 0 and distance_large >= 0, (
                f"{self.algorithm_name.upper()} 距离应该非负\n"
                f"小 g 距离: {distance_small}\n"
                f"大 g 距离: {distance_large}"
            )
        elif param_name == "precision":
            # 对于 SOWD，precision 越高，精度越高，距离可能更准确
            distance_small = self._call_distance_function(
                traj1, traj2, distance_type, small_value
            )
            distance_large = self._call_distance_function(
                traj1, traj2, distance_type, large_value
            )

            # SOWD 的距离受 precision 参数影响
            assert distance_small >= 0 and distance_large >= 0, (
                f"{self.algorithm_name.upper()} 距离应该非负\n"
                f"小 precision 距离: {distance_small}\n"
                f"大 precision 距离: {distance_large}"
            )

    def _check_negative_hyperparameter(
        self, distance_type: str, negative_value: Union[float, int]
    ):
        """
        检查负的超参数值

        Args:
            distance_type: 距离类型
            negative_value: 负超参数值
        """
        param_name = self.config["hyperparameter_name"]

        if param_name == "eps":
            # LCSS 和 EDR 的 eps < 0 应该不匹配任何点（因为距离总是 >= 0）
            traj1 = [[0.0, 0.0], [1.0, 1.0]]
            traj2 = [[0.0, 0.0], [1.0, 1.0]]

            distance = self._call_distance_function(
                traj1, traj2, distance_type, negative_value
            )
            assert (
                distance == 1.0
            ), f"{self.algorithm_name.upper()} 负 eps 应该不匹配任何点，距离应该是 1.0，实际: {distance}"


def load_test_data_by_metainfo(metainfo: Metainfo, data_dir):
    """
    根据元数据对象加载测试数据

    Args:
        metainfo: Metainfo 对象
        data_dir: 数据目录

    Returns:
        测试数据 DataFrame
    """
    # 此函数现在接收一个 Metainfo 对象
    sample_path = get_sample_path(metainfo, data_dir)

    try:
        import pyarrow.parquet as pq
        import polars as pl

        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取测试数据 {sample_path}: {e}")


def get_hyperparameter_value_from_metainfo(metainfo: Metainfo):
    """
    从元数据对象中提取超参数值

    Args:
        metainfo: Metainfo 对象

    Returns:
        超参数值
    """
    # 使用属性访问
    if metainfo.eps is not None:
        return metainfo.eps
    elif metainfo.g is not None:
        return metainfo.g
    elif metainfo.precision is not None:
        return metainfo.precision
    return None


class HyperparameterUtil:
    """
    超参数工具类，用于从元数据中获取超参数值（已更新）
    """

    @staticmethod
    def get_hyperparameter_value(
        algorithm_name: str, distance_type: str, metainfo_list: List[Metainfo]
    ) -> Optional[Union[float, List[float], int]]:
        """
        从元数据列表中获取算法在特定距离类型下的超参数值

        Args:
            algorithm_name: 算法名称
            distance_type: 距离类型
            metainfo_list: Metainfo 对象列表

        Returns:
            超参数值，如果算法没有超参数则返回 None
        """
        config = get_algorithm_config(algorithm_name)
        if not config["has_hyperparameters"]:
            return None

        # 查找匹配算法名称和距离类型的元数据
        for metainfo in metainfo_list:
            if (
                metainfo.algorithm == algorithm_name
                and metainfo.type_d.value == distance_type
            ):
                return get_hyperparameter_value_from_metainfo(metainfo)

        # 如果没有找到完全匹配，返回第一个匹配算法名称的元数据中的超参数值
        for metainfo in metainfo_list:
            if metainfo.algorithm == algorithm_name:
                hyperparam_value = get_hyperparameter_value_from_metainfo(metainfo)
                if hyperparam_value is not None:
                    return hyperparam_value

        # 如果还是没有找到，返回配置中的默认值
        param_name = config["hyperparameter_name"]
        if param_name == "eps":
            return 1e-6
        elif param_name == "g":
            return [0.0, 0.0]
        elif param_name == "precision":
            return 5
        return None
