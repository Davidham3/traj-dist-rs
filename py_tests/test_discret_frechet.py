"""
Discret Frechet 算法测试用例

测试 Rust 实现的 Discret Frechet 算法与原始 traj-dist 实现的准确率对比
"""

import pytest
from test_framework import BaseDistanceTest, load_test_data_by_metainfo


class TestDiscretFrechetEuclidean(BaseDistanceTest):
    """测试 Discret Frechet 欧几里得距离"""

    @property
    def algorithm_name(self) -> str:
        return "discret_frechet"

    def test_discret_frechet_euclidean_accuracy(self, all_metainfo, data_dir):
        """
        测试 Discret Frechet 欧几里得距离的准确率

        验证 Rust 实现的结果与原始 traj-dist 实现的误差在 1e-8 以内
        """
        discret_frechet_metainfo = all_metainfo.get("discret_frechet", [])
        euclidean_metainfo = [m for m in discret_frechet_metainfo if m["type_d"] == "euclidean"]

        if not euclidean_metainfo:
            pytest.skip("Discret Frechet 欧几里得距离测试数据不存在")

        test_data = load_test_data_by_metainfo(euclidean_metainfo[0], data_dir)
        self._test_accuracy(test_data, "euclidean")

    def test_discret_frechet_euclidean_identical_trajectories(self):
        """测试相同轨迹的距离应该接近 0"""
        distance_type = "euclidean"
        self._check_identical_trajectories(distance_type)

    def test_discret_frechet_euclidean_simple_case(self):
        """测试简单轨迹对"""
        distance_type = "euclidean"
        self._check_simple_case(distance_type)

    def test_discret_frechet_euclidean_empty_trajectory(self):
        """测试空轨迹的情况"""
        distance_type = "euclidean"
        self._check_empty_trajectory(distance_type)

    def test_discret_frechet_euclidean_single_point(self):
        """测试单点轨迹"""
        distance_type = "euclidean"
        self._check_single_point(distance_type)


class TestDiscretFrechetParameterValidation(BaseDistanceTest):
    """测试 Discret Frechet 参数验证"""

    @property
    def algorithm_name(self) -> str:
        return "discret_frechet"

    def test_discret_frechet_invalid_distance_type(self):
        """测试无效的距离类型应该抛出异常"""
        self._check_invalid_distance_type()

    def test_discret_frechet_valid_distance_types(self):
        """测试有效的距离类型"""
        self._check_valid_distance_types()

    def test_discret_frechet_invalid_trajectory_format(self):
        """测试无效的轨迹格式应该抛出异常"""
        self._check_invalid_trajectory_format()