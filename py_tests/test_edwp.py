"""
Tests for EDwP (Edit Distance with Projections) algorithm
"""

import math

import traj_dist_rs
from test_framework import BaseDistanceTest


class TestEDwPMatrixConsistency(BaseDistanceTest):
    """Test that EDwP produces consistent results with and without matrix"""

    @property
    def algorithm_name(self) -> str:
        return "edwp"

    def test_edwp_matrix_consistency(self):
        """Test that EDwP produces same distance with and without matrix"""
        traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

        # Without matrix
        result_no_matrix = traj_dist_rs.edwp(traj1, traj2, False)
        assert result_no_matrix.matrix is None

        # With matrix
        result_with_matrix = traj_dist_rs.edwp(traj1, traj2, True)
        assert result_with_matrix.matrix is not None
        assert len(result_with_matrix.matrix) == len(traj1) * len(traj2)

        # Distances should be the same
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-6


class TestEDwPVsPythonReference(BaseDistanceTest):
    """Test that Rust EDwP implementation matches Python reference"""

    @property
    def algorithm_name(self) -> str:
        return "edwp"

    def test_edwp_accuracy(self, edwp_test_data):
        """Test EDwP accuracy against Python reference implementation"""
        self._test_accuracy(edwp_test_data, "euclidean", tolerance=1.5e-8)

    def test_edwp_identical(self):
        """Test EDwP with identical trajectories"""
        self._check_identical_trajectories("euclidean", tolerance=1e-6)

    def test_edwp_empty(self):
        """Test EDwP with empty trajectories"""
        self._check_empty_trajectory("euclidean")

    def test_edwp_single_point(self):
        """Test EDwP with single point trajectories"""
        self._check_single_point("euclidean")

    def test_edwp_symmetry(self):
        """Test EDwP symmetry property"""
        traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

        dist1 = traj_dist_rs.edwp(traj1, traj2, False)
        dist2 = traj_dist_rs.edwp(traj2, traj1, False)

        # EDwP should be symmetric
        assert abs(dist1.distance - dist2.distance) < 1e-6


class TestEDwPBoundaryCases(BaseDistanceTest):
    """Test EDwP boundary cases"""

    @property
    def algorithm_name(self) -> str:
        return "edwp"

    def test_edwp_zero_coordinates(self):
        """Test EDwP with zero coordinates"""
        traj1 = [[0.0, 0.0], [0.0, 0.0]]
        traj2 = [[1.0, 1.0], [1.0, 1.0]]

        dist = traj_dist_rs.edwp(traj1, traj2, False)
        assert dist.distance > 0
        assert math.isfinite(dist.distance)

    def test_edwp_negative_coordinates(self):
        """Test EDwP with negative coordinates"""
        traj1 = [[-1.0, -1.0], [-2.0, -2.0]]
        traj2 = [[-3.0, -3.0], [-4.0, -4.0]]

        dist = traj_dist_rs.edwp(traj1, traj2, False)
        assert dist.distance > 0
        assert math.isfinite(dist.distance)

    def test_edwp_large_coordinates(self):
        """Test EDwP with large coordinates"""
        traj1 = [[1e6, 1e6], [1e6 + 1.0, 1e6 + 1.0]]
        traj2 = [[1e6 + 10.0, 1e6 + 10.0], [1e6 + 11.0, 1e6 + 11.0]]

        dist = traj_dist_rs.edwp(traj1, traj2, False)
        assert dist.distance > 0
        assert math.isfinite(dist.distance)

    def test_edwp_small_coordinates(self):
        """Test EDwP with small coordinates"""
        traj1 = [[1e-6, 1e-6], [2e-6, 2e-6]]
        traj2 = [[3e-6, 3e-6], [4e-6, 4e-6]]

        dist = traj_dist_rs.edwp(traj1, traj2, False)
        assert dist.distance > 0
        assert math.isfinite(dist.distance)
