"""Test batch computation functions pdist and cdist"""

import numpy as np
import pytest
import traj_dist_rs


class TestMetricFactory:
    """Test Metric factory methods"""

    def test_metric_sspd(self):
        """Test Metric.sspd factory method"""
        metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
        assert metric is not None

    def test_metric_dtw(self):
        """Test Metric.dtw factory method"""
        metric = traj_dist_rs.Metric.dtw(type_d="euclidean")
        assert metric is not None

    def test_metric_hausdorff(self):
        """Test Metric.hausdorff factory method"""
        metric = traj_dist_rs.Metric.hausdorff(type_d="euclidean")
        assert metric is not None

    def test_metric_discret_frechet(self):
        """Test Metric.discret_frechet factory method"""
        metric = traj_dist_rs.Metric.discret_frechet(type_d="euclidean")
        assert metric is not None

    def test_metric_lcss(self):
        """Test Metric.lcss factory method"""
        metric = traj_dist_rs.Metric.lcss(eps=5.0, type_d="euclidean")
        assert metric is not None

    def test_metric_edr(self):
        """Test Metric.edr factory method"""
        metric = traj_dist_rs.Metric.edr(eps=5.0, type_d="euclidean")
        assert metric is not None

    def test_metric_erp(self):
        """Test Metric.erp factory method"""
        metric = traj_dist_rs.Metric.erp(g=[0.0, 1.0], type_d="euclidean")
        assert metric is not None

    def test_metric_erp_invalid_gap(self):
        """Test Metric.erp with invalid gap parameter"""
        with pytest.raises(ValueError, match="must have at least 2 elements"):
            traj_dist_rs.Metric.erp(g=[0.0], type_d="euclidean")

    def test_metric_invalid_distance_type(self):
        """Test Metric factory method with invalid distance type"""
        with pytest.raises(ValueError, match="Invalid distance type"):
            traj_dist_rs.Metric.sspd(type_d="invalid")

    def test_metric_direct_construction(self):
        """Test that direct Metric construction is forbidden"""
        with pytest.raises(ValueError, match="cannot be created directly"):
            traj_dist_rs.Metric()


class TestPdist:
    """Test pdist function"""

    def test_pdist_simple_case(self):
        """Test pdist with simple case"""
        trajectories = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.5, 0.5], [1.5, 1.5]]),
        ]

        metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
        distances = traj_dist_rs.pdist(trajectories, metric=metric)

        # Expected size: n * (n - 1) / 2 = 3 * 2 / 2 = 3
        assert len(distances) == 3

        # Verify all distances are positive and finite
        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0)

    def test_pdist_with_numpy_arrays_zero_copy(self):
        """Test pdist with numpy arrays (zero-copy)"""
        trajectories = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.5, 0.5], [1.5, 1.5]]),
        ]

        metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
        distances_np = traj_dist_rs.pdist(trajectories, metric=metric)

        # Verify with list input (will be copied)
        trajectories_list = [t.tolist() for t in trajectories]
        distances_list = traj_dist_rs.pdist(trajectories_list, metric=metric)

        # Results should be the same
        np.testing.assert_array_almost_equal(distances_np, distances_list)

    def test_pdist_all_algorithms(self):
        """Test pdist with all supported algorithms"""
        trajectories = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]

        algorithms = [
            (traj_dist_rs.Metric.sspd(type_d="euclidean")),
            (traj_dist_rs.Metric.dtw(type_d="euclidean")),
            (traj_dist_rs.Metric.hausdorff(type_d="euclidean")),
            (traj_dist_rs.Metric.discret_frechet(type_d="euclidean")),
        ]

        for metric in algorithms:
            distances = traj_dist_rs.pdist(trajectories, metric=metric)
            assert len(distances) == 1
            assert np.all(np.isfinite(distances))
            assert np.all(distances >= 0)

    def test_pdist_with_lcss(self):
        """Test pdist with LCSS algorithm"""
        trajectories = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]

        metric = traj_dist_rs.Metric.lcss(eps=0.5, type_d="euclidean")
        distances = traj_dist_rs.pdist(trajectories, metric=metric)
        assert len(distances) == 1
        assert np.all(np.isfinite(distances))
        assert 0 <= distances[0] <= 1  # LCSS should be between 0 and 1

    def test_pdist_with_edr(self):
        """Test pdist with EDR algorithm"""
        trajectories = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]

        metric = traj_dist_rs.Metric.edr(eps=0.5, type_d="euclidean")
        distances = traj_dist_rs.pdist(trajectories, metric=metric)
        assert len(distances) == 1
        assert np.all(np.isfinite(distances))
        assert 0 <= distances[0] <= 1  # EDR should be between 0 and 1

    def test_pdist_with_erp(self):
        """Test pdist with ERP algorithm"""
        trajectories = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]

        metric = traj_dist_rs.Metric.erp(g=[0.0, 0.0], type_d="euclidean")
        distances = traj_dist_rs.pdist(trajectories, metric=metric)
        assert len(distances) == 1
        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0)

    def test_pdist_spherical(self):
        """Test pdist with spherical distance type"""
        trajectories = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]

        metric = traj_dist_rs.Metric.sspd(type_d="spherical")
        distances = traj_dist_rs.pdist(trajectories, metric=metric)
        assert len(distances) == 1
        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0)

    def test_pdist_parallel_vs_sequential(self):
        """Test that parallel and sequential produce the same results"""
        trajectories = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.5, 0.5], [1.5, 1.5]]),
        ]

        metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
        distances_parallel = traj_dist_rs.pdist(
            trajectories, metric=metric, parallel=True
        )
        distances_sequential = traj_dist_rs.pdist(
            trajectories, metric=metric, parallel=False
        )

        np.testing.assert_array_almost_equal(distances_parallel, distances_sequential)

    def test_pdist_invalid_input(self):
        """Test pdist with invalid input"""
        trajectories = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
        ]

        metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
        with pytest.raises(ValueError, match="at least 2 trajectories"):
            traj_dist_rs.pdist(trajectories, metric=metric)


class TestCdist:
    """Test cdist function"""

    def test_cdist_simple_case(self):
        """Test cdist with simple case"""
        trajectories_a = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]
        trajectories_b = [
            np.array([[0.5, 0.5], [1.5, 1.5]]),
            np.array([[0.5, 1.5], [1.5, 0.5]]),
        ]

        metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
        distances = traj_dist_rs.cdist(trajectories_a, trajectories_b, metric=metric)

        # Expected shape: (n_a, n_b) = (2, 2)
        assert distances.shape == (2, 2)

        # Verify all distances are positive and finite
        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0)

    def test_cdist_with_numpy_arrays_zero_copy(self):
        """Test cdist with numpy arrays (zero-copy)"""
        trajectories_a = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]
        trajectories_b = [
            np.array([[0.5, 0.5], [1.5, 1.5]]),
            np.array([[0.5, 1.5], [1.5, 0.5]]),
        ]

        metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
        distances_np = traj_dist_rs.cdist(trajectories_a, trajectories_b, metric=metric)

        # Verify with list input (will be copied)
        trajectories_a_list = [t.tolist() for t in trajectories_a]
        trajectories_b_list = [t.tolist() for t in trajectories_b]
        distances_list = traj_dist_rs.cdist(
            trajectories_a_list, trajectories_b_list, metric=metric
        )

        # Results should be the same
        np.testing.assert_array_almost_equal(distances_np, distances_list)

    def test_cdist_all_algorithms(self):
        """Test cdist with all supported algorithms"""
        trajectories_a = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
        ]
        trajectories_b = [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]

        algorithms = [
            (traj_dist_rs.Metric.sspd(type_d="euclidean")),
            (traj_dist_rs.Metric.dtw(type_d="euclidean")),
            (traj_dist_rs.Metric.hausdorff(type_d="euclidean")),
            (traj_dist_rs.Metric.discret_frechet(type_d="euclidean")),
        ]

        for metric in algorithms:
            distances = traj_dist_rs.cdist(
                trajectories_a, trajectories_b, metric=metric
            )
            assert distances.shape == (1, 1)
            assert np.all(np.isfinite(distances))
            assert np.all(distances >= 0)

    def test_cdist_with_lcss(self):
        """Test cdist with LCSS algorithm"""
        trajectories_a = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
        ]
        trajectories_b = [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]

        metric = traj_dist_rs.Metric.lcss(eps=0.5, type_d="euclidean")
        distances = traj_dist_rs.cdist(trajectories_a, trajectories_b, metric=metric)
        assert distances.shape == (1, 1)
        assert np.all(np.isfinite(distances))
        assert 0 <= distances[0, 0] <= 1  # LCSS should be between 0 and 1

    def test_cdist_with_edr(self):
        """Test cdist with EDR algorithm"""
        trajectories_a = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
        ]
        trajectories_b = [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]

        metric = traj_dist_rs.Metric.edr(eps=0.5, type_d="euclidean")
        distances = traj_dist_rs.cdist(trajectories_a, trajectories_b, metric=metric)
        assert distances.shape == (1, 1)
        assert np.all(np.isfinite(distances))
        assert 0 <= distances[0, 0] <= 1  # EDR should be between 0 and 1

    def test_cdist_with_erp(self):
        """Test cdist with ERP algorithm"""
        trajectories_a = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
        ]
        trajectories_b = [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]

        metric = traj_dist_rs.Metric.erp(g=[0.0, 0.0], type_d="euclidean")
        distances = traj_dist_rs.cdist(trajectories_a, trajectories_b, metric=metric)
        assert distances.shape == (1, 1)
        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0)

    def test_cdist_spherical(self):
        """Test cdist with spherical distance type"""
        trajectories_a = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
        ]
        trajectories_b = [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]

        metric = traj_dist_rs.Metric.sspd(type_d="spherical")
        distances = traj_dist_rs.cdist(trajectories_a, trajectories_b, metric=metric)
        assert distances.shape == (1, 1)
        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0)

    def test_cdist_parallel_vs_sequential(self):
        """Test that parallel and sequential produce the same results"""
        trajectories_a = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]
        trajectories_b = [
            np.array([[0.5, 0.5], [1.5, 1.5]]),
            np.array([[0.5, 1.5], [1.5, 0.5]]),
        ]

        metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
        distances_parallel = traj_dist_rs.cdist(
            trajectories_a, trajectories_b, metric=metric, parallel=True
        )
        distances_sequential = traj_dist_rs.cdist(
            trajectories_a, trajectories_b, metric=metric, parallel=False
        )

        np.testing.assert_array_almost_equal(distances_parallel, distances_sequential)

    def test_cdist_invalid_input(self):
        """Test cdist with invalid input"""
        trajectories_a = []
        trajectories_b = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
        ]

        metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
        with pytest.raises(ValueError, match="at least 1 trajectory"):
            traj_dist_rs.cdist(trajectories_a, trajectories_b, metric=metric)


class TestBatchPerformance:
    """Test batch computation performance"""

    def test_pdist_larger_dataset(self):
        """Test pdist with larger dataset"""
        # Create 10 trajectories
        trajectories = [
            np.array([[i * 0.1, i * 0.1], [(i + 1) * 0.1, (i + 1) * 0.1]])
            for i in range(10)
        ]

        metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
        distances = traj_dist_rs.pdist(trajectories, metric=metric)

        # Expected size: 10 * 9 / 2 = 45
        assert len(distances) == 45
        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0)

    def test_cdist_larger_dataset(self):
        """Test cdist with larger dataset"""
        # Create 5 trajectories in each collection
        trajectories_a = [
            np.array([[i * 0.1, i * 0.1], [(i + 1) * 0.1, (i + 1) * 0.1]])
            for i in range(5)
        ]
        trajectories_b = [
            np.array([[i * 0.2, i * 0.2], [(i + 1) * 0.2, (i + 1) * 0.2]])
            for i in range(5)
        ]

        metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
        distances = traj_dist_rs.cdist(trajectories_a, trajectories_b, metric=metric)

        # Expected shape: (5, 5)
        assert distances.shape == (5, 5)
        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0)
