"""
LCSS (Longest Common Subsequence) 算法测试用例

测试 Rust 实现的 LCSS 算法与原始 traj-dist 实现的准确率对比
"""

import pytest
from test_framework import DistanceTestWithHyperparameters, load_test_data_by_metainfo, get_hyperparameter_value_from_metainfo


class TestLCSSEuclidean(DistanceTestWithHyperparameters):
    """测试 LCSS 欧几里得距离"""

    @property
    def algorithm_name(self) -> str:
        return "lcss"

    def test_lcss_euclidean_accuracy(self, all_metainfo, data_dir):
        """
        测试 LCSS 欧几里得距离的准确率

        验证 Rust 实现的结果与原始 traj-dist 实现的误差在 1e-8 以内
        """
        lcss_metainfo = all_metainfo.get("lcss", [])
        euclidean_metainfo = [m for m in lcss_metainfo if m["type_d"] == "euclidean"]

        if not euclidean_metainfo:
            pytest.skip("LCSS 欧几里得距离测试数据不存在")

        # 测试所有 eps 值
        for metainfo in euclidean_metainfo:
            eps = get_hyperparameter_value_from_metainfo(metainfo)
            test_data = load_test_data_by_metainfo(metainfo, data_dir)
            self._test_accuracy(test_data, "euclidean", eps)

    def test_lcss_euclidean_identical_trajectories(self):
        """测试相同轨迹的距离应该接近 0"""
        self._check_identical_trajectories("euclidean", eps=1e-6)

    def test_lcss_euclidean_simple_case(self):
        """测试简单轨迹对"""
        self._check_simple_case("euclidean", eps=0.5)

    def test_lcss_euclidean_empty_trajectory(self):
        """测试空轨迹的情况"""
        self._check_empty_trajectory("euclidean", eps=1.0)

    def test_lcss_euclidean_eps_effect(self):
        """测试 eps 参数的影响"""
        self._check_hyperparameter_effect("euclidean", 0.05, 1000.0)

    def test_lcss_euclidean_single_point(self):
        """测试单点轨迹"""
        self._check_single_point("euclidean", eps=0.5)


class TestLCSSSpherical(DistanceTestWithHyperparameters):
    """测试 LCSS 球面距离"""

    @property
    def algorithm_name(self) -> str:
        return "lcss"

    def test_lcss_spherical_accuracy(self, all_metainfo, data_dir):
        """
        测试 LCSS 球面距离的准确率

        验证 Rust 实现的结果与原始 traj-dist 实现的误差在 1e-8 以内
        """
        lcss_metainfo = all_metainfo.get("lcss", [])
        spherical_metainfo = [m for m in lcss_metainfo if m["type_d"] == "spherical"]

        if not spherical_metainfo:
            pytest.skip("LCSS 球面距离测试数据不存在")

        # 测试所有 eps 值
        for metainfo in spherical_metainfo:
            eps = get_hyperparameter_value_from_metainfo(metainfo)
            test_data = load_test_data_by_metainfo(metainfo, data_dir)
            self._test_accuracy(test_data, "spherical", eps)

    def test_lcss_spherical_identical_trajectories(self):
        """测试相同轨迹的距离应该接近 0"""
        self._check_identical_trajectories("spherical", eps=1e-6)

    def test_lcss_spherical_simple_case(self):
        """测试简单轨迹对"""
        self._check_simple_case("spherical", eps=1.0)

    def test_lcss_spherical_eps_effect(self):
        """测试 eps 参数的影响"""
        self._check_hyperparameter_effect("spherical", 10.0, 100000.0)


class TestLCSSParameterValidation(DistanceTestWithHyperparameters):
    """测试 LCSS 参数验证"""

    @property
    def algorithm_name(self) -> str:
        return "lcss"

    def test_lcss_invalid_distance_type(self):
        """测试无效的距离类型应该抛出异常"""
        self._check_invalid_distance_type()

    def test_lcss_valid_distance_types(self):
        """测试有效的距离类型"""
        self._check_valid_distance_types()

    def test_lcss_invalid_trajectory_format(self):
        """测试无效的轨迹格式应该抛出异常"""
        self._check_invalid_trajectory_format()

    def test_lcss_eps_parameter(self):
        """测试 eps 参数的影响"""
        traj1 = [[0.0, 0.0], [1.0, 1.0]]
        traj2 = [[0.0, 0.0], [1.0, 1.0]]

        # 小 eps 应该匹配完全相同的点（距离为 0）
        distance_small = self._call_distance_function(traj1, traj2, "euclidean", 1e-6)
        assert distance_small < 1e-8

        # eps < 0 应该也正常工作（不会匹配任何点，因为距离总是 >= 0）
        distance_negative = self._call_distance_function(traj1, traj2, "euclidean", -1.0)
        assert distance_negative == 1.0

        # 非常大的 eps 应该匹配所有点
        distance_large = self._call_distance_function(traj1, traj2, "euclidean", 1e10)
        assert distance_large < 1e-8