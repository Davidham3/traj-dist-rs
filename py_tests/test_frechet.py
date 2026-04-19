"""
Tests for Frechet Distance (continuous) algorithm
"""

import math

import pytest
import traj_dist_rs
from test_framework import BaseDistanceTest, load_test_data_by_metainfo


class TestFrechetVsCythonReference(BaseDistanceTest):
    """Test that Rust Frechet implementation matches Cython reference"""

    @property
    def algorithm_name(self) -> str:
        return "frechet"

    def test_frechet_accuracy(self, all_metainfo, data_dir):
        """Test Frechet accuracy against Cython reference implementation (1225 trajectory pairs)"""
        frechet_metainfo = all_metainfo.get("frechet", [])
        euclidean_metainfo = [m for m in frechet_metainfo if m.type_d == "euclidean"]

        if not euclidean_metainfo:
            pytest.skip("Frechet Euclidean distance test data not found")

        test_data = load_test_data_by_metainfo(euclidean_metainfo[0], data_dir)
        self._test_accuracy(test_data, "euclidean", tolerance=1.5e-8)

    def test_frechet_identical(self):
        """Test Frechet with identical trajectories"""
        traj = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
        dist = traj_dist_rs.frechet(traj, traj)
        assert (
            dist < 1.5e-8
        ), f"Identical trajectories should have distance ~0, got {dist}"

    def test_frechet_simple(self):
        """Test Frechet with simple known case"""
        traj1 = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        traj2 = [[0.0, 1.0], [3.0, 1.0]]
        dist = traj_dist_rs.frechet(traj1, traj2)
        # Parallel lines at distance 1.0
        assert abs(dist - 1.0) < 1.5e-8, f"Expected ~1.0, got {dist}"

    def test_frechet_empty(self):
        """Test Frechet with empty trajectories"""
        empty = []
        traj = [[0.0, 0.0], [1.0, 1.0]]
        dist = traj_dist_rs.frechet(empty, traj)
        assert dist == float("inf") or dist > 1e300

    def test_frechet_single_point(self):
        """Test Frechet with single point trajectories"""
        single = [[0.0, 0.0]]
        traj = [[0.0, 0.0], [1.0, 1.0]]
        dist = traj_dist_rs.frechet(single, traj)
        assert dist == float("inf") or dist > 1e300


class TestFrechetProperties(BaseDistanceTest):
    """Test mathematical properties of Frechet distance"""

    @property
    def algorithm_name(self) -> str:
        return "frechet"

    def test_frechet_symmetry(self):
        """Test Frechet symmetry: d(A,B) == d(B,A)"""
        traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
        traj2 = [[0.0, 0.5], [1.0, 1.5], [2.0, 0.5]]

        dist12 = traj_dist_rs.frechet(traj1, traj2)
        dist21 = traj_dist_rs.frechet(traj2, traj1)

        assert (
            abs(dist12 - dist21) < 1.5e-8
        ), f"Frechet should be symmetric: {dist12} vs {dist21}"

    def test_frechet_leq_discret_frechet(self):
        """Test that continuous Frechet <= discrete Frechet"""
        traj1 = [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]]
        traj2 = [[0.5, 0.5], [2.0, 1.5], [3.5, 2.5]]

        cont_dist = traj_dist_rs.frechet(traj1, traj2)
        disc_result = traj_dist_rs.discret_frechet(traj1, traj2, "euclidean")
        disc_dist = disc_result.distance

        assert (
            cont_dist <= disc_dist + 1.5e-8
        ), f"Continuous Frechet ({cont_dist}) should be <= Discrete Frechet ({disc_dist})"

    def test_frechet_triangle_inequality_approx(self):
        """Test approximate triangle inequality: d(A,C) <= d(A,B) + d(B,C)"""
        traj_a = [[0.0, 0.0], [1.0, 0.0]]
        traj_b = [[0.0, 1.0], [1.0, 1.0]]
        traj_c = [[0.0, 2.0], [1.0, 2.0]]

        dist_ab = traj_dist_rs.frechet(traj_a, traj_b)
        dist_bc = traj_dist_rs.frechet(traj_b, traj_c)
        dist_ac = traj_dist_rs.frechet(traj_a, traj_c)

        assert (
            dist_ac <= dist_ab + dist_bc + 1e-10
        ), f"Triangle inequality violated: d(A,C)={dist_ac} > d(A,B)+d(B,C)={dist_ab + dist_bc}"


class TestFrechetBoundaryCases(BaseDistanceTest):
    """Test Frechet boundary cases"""

    @property
    def algorithm_name(self) -> str:
        return "frechet"

    def test_frechet_zero_coordinates(self):
        """Test Frechet with zero coordinates"""
        traj1 = [[0.0, 0.0], [0.0, 0.0]]
        traj2 = [[1.0, 1.0], [1.0, 1.0]]

        dist = traj_dist_rs.frechet(traj1, traj2)
        assert dist > 0
        assert math.isfinite(dist)

    def test_frechet_negative_coordinates(self):
        """Test Frechet with negative coordinates"""
        traj1 = [[-1.0, -1.0], [-2.0, -2.0]]
        traj2 = [[-3.0, -3.0], [-4.0, -4.0]]

        dist = traj_dist_rs.frechet(traj1, traj2)
        assert dist > 0
        assert math.isfinite(dist)

    def test_frechet_large_coordinates(self):
        """Test Frechet with large coordinates"""
        traj1 = [[1e6, 1e6], [1e6 + 1.0, 1e6 + 1.0]]
        traj2 = [[1e6 + 10.0, 1e6 + 10.0], [1e6 + 11.0, 1e6 + 11.0]]

        dist = traj_dist_rs.frechet(traj1, traj2)
        assert dist > 0
        assert math.isfinite(dist)

    def test_frechet_different_lengths(self):
        """Test Frechet with different length trajectories"""
        traj1 = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        traj2 = [[0.0, 1.0], [3.0, 1.0]]

        dist = traj_dist_rs.frechet(traj1, traj2)
        assert dist > 0
        assert math.isfinite(dist)


class TestFrechetBatchComputation(BaseDistanceTest):
    """Test Frechet with batch computation (Metric API)"""

    @property
    def algorithm_name(self) -> str:
        return "frechet"

    def test_frechet_metric_factory(self):
        """Test Metric.frechet() factory method"""
        metric = traj_dist_rs.Metric.frechet()
        assert metric is not None

    def test_frechet_pdist(self):
        """Test Frechet with pdist"""
        import numpy as np

        trajectories = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.5, 0.5], [1.5, 1.5]]),
        ]
        metric = traj_dist_rs.Metric.frechet()
        distances = traj_dist_rs.pdist(trajectories, metric=metric)
        assert len(distances) == 3  # 3 * 2 / 2
        for dist in distances:
            assert dist >= 0
            assert math.isfinite(dist)

    def test_frechet_cdist(self):
        """Test Frechet with cdist"""
        import numpy as np

        traj_a = [np.array([[0.0, 0.0], [1.0, 1.0]])]
        traj_b = [np.array([[0.0, 1.0], [1.0, 0.0]])]
        metric = traj_dist_rs.Metric.frechet()
        matrix = traj_dist_rs.cdist(traj_a, traj_b, metric=metric)
        assert len(matrix) == 1
        dist_value = matrix.flat[0]
        assert dist_value >= 0
        assert math.isfinite(dist_value)
