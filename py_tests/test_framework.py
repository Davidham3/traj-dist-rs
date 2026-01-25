"""
统一测试框架基类

提供轨迹距离算法测试的通用方法和工具
"""

import pytest
import math
import json
from pathlib import Path
from typing import Optional, Union, List, Callable, Dict, Any, TypedDict
from abc import ABC, abstractmethod


class AlgorithmConfig(TypedDict):
    """算法配置类型定义"""
    name: str  # 算法名称
    function_name: str  # Python 绑定的函数名
    distance_types: List[str]  # 支持的距离类型 ["euclidean", "spherical"]
    has_hyperparameters: bool  # 是否有超参数
    hyperparameter_name: Optional[str]  # 超参数名称
    hyperparameter_values: Optional[Dict[str, List[Union[float, List[float]]]]]  # 超参数值配置


def get_algorithm_config_from_metainfo(algorithm: str, all_metainfo: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> AlgorithmConfig:
    """
    从元数据动态获取算法配置信息

    Args:
        algorithm: 算法名称
        all_metainfo: 所有元数据（如果为 None，则自动加载）

    Returns:
        算法配置字典
    """
    if all_metainfo is None:
        # 动态加载所有元数据
        all_metainfo = load_all_metainfo_from_data_dir()
    
    if algorithm not in all_metainfo:
        raise KeyError(f"算法 {algorithm} 的元数据不存在")
    
    metainfo_list = all_metainfo[algorithm]
    
    # 获取算法的基本信息
    first_metainfo = metainfo_list[0] if metainfo_list else {}
    
    # 找出该算法支持的距离类型
    distance_types = list(set([m["type_d"] for m in metainfo_list]))
    
    # 检查是否包含超参数字段来判断是否有超参数
    has_hyperparameters = any(["eps" in m or "g" in m or "precision" in m for m in metainfo_list])
    
    # 确定超参数名称
    hyperparameter_name = None
    if any("eps" in m for m in metainfo_list):
        hyperparameter_name = "eps"
    elif any("g" in m for m in metainfo_list):
        hyperparameter_name = "g"
    elif any("precision" in m for m in metainfo_list):
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
    """
    获取指定算法的配置（从元数据动态加载）

    Args:
        algorithm: 算法名称

    Returns:
        算法配置字典

    Raises:
        KeyError: 如果算法不存在
    """
    return get_algorithm_config_from_metainfo(algorithm)


def load_all_metainfo_from_data_dir(data_dir: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    从 data 目录动态加载所有元数据

    Args:
        data_dir: 数据目录（默认为 py_tests/data）

    Returns:
        字典，键为算法名称，值为元数据列表
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    metainfo_dir = data_dir / "metainfo"
    all_metainfo = {}

    for metainfo_file in metainfo_dir.glob("*.jsonl"):
        algorithm = metainfo_file.stem
        metainfo_list = []
        with open(metainfo_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    metainfo_list.append(json.loads(line))
        all_metainfo[algorithm] = metainfo_list

    return all_metainfo


def load_metainfo(algorithm: str, data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    加载指定算法的元数据

    Args:
        algorithm: 算法名称
        data_dir: 数据目录（默认为 py_tests/data）

    Returns:
        元数据列表（每个元素是一个字典）
    """
    # 使用通用加载函数
    all_metainfo = load_all_metainfo_from_data_dir(data_dir)
    if algorithm not in all_metainfo:
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        metainfo_file = data_dir / "metainfo" / f"{algorithm}.jsonl"
        raise FileNotFoundError(f"元数据文件不存在: {metainfo_file}")
    return all_metainfo[algorithm]


def get_sample_path(metainfo: Dict[str, Any], data_dir: Optional[Path] = None) -> Path:
    """
    从元数据获取样本文件路径

    Args:
        metainfo: 元数据字典
        data_dir: 数据目录（默认为 py_tests/data）

    Returns:
        样本文件路径
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    sample_file = metainfo["sample_file"]
    # sample_file 是相对于 metainfo_dir 的路径
    # 例如: ../samples/edr_euclidean_eps_0.01.parquet
    samples_dir = data_dir / "samples"
    return samples_dir / Path(sample_file).name


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

    def _get_hyperparameter_value_from_metainfo(self, all_metainfo: Dict[str, List[Dict[str, Any]]], distance_type: str) -> Optional[Union[float, List[float], int]]:
        """
        从元数据中获取超参数值
        
        Args:
            all_metainfo: 所有元数据字典
            distance_type: 距离类型
            
        Returns:
            超参数值
        """
        algorithm_metainfo = all_metainfo.get(self.algorithm_name, [])
        return HyperparameterUtil.get_hyperparameter_value(self.algorithm_name, distance_type, algorithm_metainfo)

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
        hyperparameter_value: Optional[Union[float, List[float], int]] = None
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

        if not config["has_hyperparameters"]:
            return func(traj1, traj2, distance_type)
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
                return func(traj1, traj2, distance_type, g=hyperparameter_value)
            elif param_name == "precision":
                # SOWD 的 precision 参数
                return func(traj1, traj2, distance_type, precision=hyperparameter_value)
            else:
                # LCSS 和 EDR 的 eps 参数
                return func(traj1, traj2, distance_type, eps=hyperparameter_value)

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
            distances.append((row["traj_1"], row["traj_2"], row["distance"]))
        return distances

    def _test_accuracy(
        self,
        test_data,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        tolerance: float = 1e-8
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
        **kwargs
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
            hyperparameter_value = kwargs.get('eps', kwargs.get('g', kwargs.get('precision', hyperparameter_value)))

        if distance_type == "spherical":
            traj = [[-122.39548, 37.77668], [-122.39539, 37.77644]]
        else:
            traj = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]

        distance = self._call_distance_function(traj, traj, distance_type, hyperparameter_value)

        # 对于 LCSS 和 EDR，距离在 [0, 1] 范围内，相同轨迹应该接近 0
        if self.algorithm_name in ["lcss", "edr"]:
            assert distance < tolerance, (
                f"{self.algorithm_name.upper()} 相同轨迹的距离应该接近 0，实际: {distance}"
            )
        else:
            assert distance < tolerance, (
                f"{self.algorithm_name.upper()} 相同轨迹的距离应该接近 0，实际: {distance}"
            )

    def _check_simple_case(
        self,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        **kwargs
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
            hyperparameter_value = kwargs.get('eps', kwargs.get('g', kwargs.get('precision', hyperparameter_value)))

        if distance_type == "spherical":
            traj1 = [[-122.39548, 37.77668], [-122.39539, 37.77644]]
            traj2 = [[-122.39548, 37.77668], [-122.39537, 37.77663]]
        else:
            traj1 = [[0.0, 0.0], [1.0, 1.0]]
            traj2 = [[0.0, 1.0], [1.0, 0.0]]

        distance = self._call_distance_function(traj1, traj2, distance_type, hyperparameter_value)

        # 距离应该大于 0
        assert distance > 0, (
            f"{self.algorithm_name.upper()} 不同轨迹的距离应该大于 0，实际: {distance}"
        )

    def _check_empty_trajectory(
        self,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        **kwargs
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
            hyperparameter_value = kwargs.get('eps', kwargs.get('g', kwargs.get('precision', hyperparameter_value)))

        if distance_type == "spherical":
            traj1 = [[-122.39548, 37.77668], [-122.39539, 37.77644]]
        else:
            traj1 = [[0.0, 0.0], [1.0, 1.0]]

        traj2 = []

        distance = self._call_distance_function(traj1, traj2, distance_type, hyperparameter_value)

        # 对于 LCSS 和 EDR，空轨迹应该返回 1.0
        if self.algorithm_name in ["lcss", "edr"]:
            assert distance == 1.0, (
                f"{self.algorithm_name.upper()} 空轨迹应该返回 1.0，实际: {distance}"
            )
        else:
            # 其他算法应该返回最大值
            assert distance == float("inf") or distance > 1e100, (
                f"{self.algorithm_name.upper()} 空轨迹应该返回最大值，实际: {distance}"
            )

    def _check_single_point(
        self,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        **kwargs
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
            hyperparameter_value = kwargs.get('eps', kwargs.get('g', kwargs.get('precision', hyperparameter_value)))

        if distance_type == "spherical":
            traj1 = [[-122.39548, 37.77668]]
            traj2 = [[-122.39539, 37.77644]]
        else:
            traj1 = [[0.0, 0.0]]
            traj2 = [[1.0, 1.0]]

        distance = self._call_distance_function(traj1, traj2, distance_type, hyperparameter_value)

        # SSPD 单点轨迹没有段，返回 inf
        if self.algorithm_name == "sspd":
            assert distance == float("inf"), (
                f"{self.algorithm_name.upper()} 单点轨迹的距离应该是 inf\n"
                f"实际距离: {distance}"
            )
        # LCSS 和 EDR 单点轨迹不匹配，返回 1.0
        elif self.algorithm_name in ["lcss", "edr"]:
            if hyperparameter_value is None or isinstance(hyperparameter_value, (int, float)) and hyperparameter_value < 1.0:
                assert distance == 1.0, (
                    f"{self.algorithm_name.upper()} 单点不匹配轨迹的距离应该是 1.0，实际: {distance}"
                )
            else:
                # 大 eps 时应该匹配
                assert distance < 1e-8, (
                    f"{self.algorithm_name.upper()} 大 eps 时单点匹配轨迹的距离应该接近 0，实际: {distance}"
                )
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
            assert distance > 0, (
                f"{self.algorithm_name.upper()} {distance_type} 距离应该大于 0，实际: {distance}"
            )

    def _check_invalid_trajectory_format(self):
        """检查无效的轨迹格式应该抛出异常"""
        # 测试非列表输入
        with pytest.raises(TypeError):
            self._call_distance_function("not a list", [[0.0, 1.0]], "euclidean")

        # 测试点坐标不是 2 个元素
        with pytest.raises(ValueError, match="should have exactly 2 elements"):
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
        large_value: Union[float, int]
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
            distance_small = self._call_distance_function(traj1, traj2, distance_type, small_value)
            distance_large = self._call_distance_function(traj1, traj2, distance_type, large_value)

            assert distance_large <= distance_small, (
                f"{self.algorithm_name.upper()} 大 eps 应该产生更小的距离\n"
                f"小 eps 距离: {distance_small}\n"
                f"大 eps 距离: {distance_large}"
            )
        elif param_name == "g":
            # 对于 ERP，g 参数的影响
            distance_small = self._call_distance_function(traj1, traj2, distance_type, small_value)
            distance_large = self._call_distance_function(traj1, traj2, distance_type, large_value)

            # ERP 的距离受 g 参数影响，但不一定是单调的
            # 这里只验证距离是合理的
            assert distance_small >= 0 and distance_large >= 0, (
                f"{self.algorithm_name.upper()} 距离应该非负\n"
                f"小 g 距离: {distance_small}\n"
                f"大 g 距离: {distance_large}"
            )
        elif param_name == "precision":
            # 对于 SOWD，precision 越高，精度越高，距离可能更准确
            distance_small = self._call_distance_function(traj1, traj2, distance_type, small_value)
            distance_large = self._call_distance_function(traj1, traj2, distance_type, large_value)

            # SOWD 的距离受 precision 参数影响
            assert distance_small >= 0 and distance_large >= 0, (
                f"{self.algorithm_name.upper()} 距离应该非负\n"
                f"小 precision 距离: {distance_small}\n"
                f"大 precision 距离: {distance_large}"
            )

    def _check_negative_hyperparameter(
        self,
        distance_type: str,
        negative_value: Union[float, int]
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

            distance = self._call_distance_function(traj1, traj2, distance_type, negative_value)
            assert distance == 1.0, (
                f"{self.algorithm_name.upper()} 负 eps 应该不匹配任何点，距离应该是 1.0，实际: {distance}"
            )


def load_test_data_by_metainfo(metainfo: Dict[str, Any], data_dir):
    """
    根据元数据加载测试数据

    Args:
        metainfo: 元数据字典
        data_dir: 数据目录

    Returns:
        测试数据 DataFrame
    """
    sample_path = get_sample_path(metainfo, data_dir)

    try:
        import pyarrow.parquet as pq
        import polars as pl
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取测试数据 {sample_path}: {e}")


def get_hyperparameter_value_from_metainfo(metainfo: Dict[str, Any]):
    """
    从元数据中提取超参数值

    Args:
        metainfo: 元数据字典

    Returns:
        超参数值
    """
    if "eps" in metainfo:
        return metainfo["eps"]
    elif "g" in metainfo:
        return metainfo["g"]
    elif "precision" in metainfo:
        return metainfo["precision"]
    return None


class HyperparameterUtil:
    """
    超参数工具类，用于从元数据中获取超参数值
    """
    
    @staticmethod
    def get_hyperparameter_value(algorithm_name: str, distance_type: str, metainfo_list: List[Dict[str, Any]]) -> Optional[Union[float, List[float], int]]:
        """
        从元数据列表中获取算法在特定距离类型下的超参数值
        
        Args:
            algorithm_name: 算法名称
            distance_type: 距离类型
            metainfo_list: 元数据列表
            
        Returns:
            超参数值，如果算法没有超参数则返回 None
        """
        config = get_algorithm_config(algorithm_name)
        if not config["has_hyperparameters"]:
            return None
            
        # 查找匹配算法名称和距离类型的元数据
        for metainfo in metainfo_list:
            if metainfo.get("algorithm") == algorithm_name and metainfo.get("type_d") == distance_type:
                return get_hyperparameter_value_from_metainfo(metainfo)
                
        # 如果没有找到完全匹配，返回第一个匹配算法名称的元数据中的超参数值
        for metainfo in metainfo_list:
            if metainfo.get("algorithm") == algorithm_name:
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