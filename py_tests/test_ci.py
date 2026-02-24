"""
CI-specific tests

These tests do not depend on external test data and are suitable for GitHub Actions CI environment.
These tests mainly verify that basic functionality works correctly, not accuracy.
"""

import numpy as np
import pytest
import traj_dist_rs


def test_sspd_basic():
    """Test SSPD basic functionality"""
    traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

    dist = traj_dist_rs.sspd(traj1, traj2, "euclidean")

    # Only verify that return value is a valid number
    assert isinstance(dist, float)
    assert not np.isnan(dist)
    assert dist >= 0


def test_sspd_spherical():
    """Test SSPD spherical distance"""
    traj1 = [[-122.4, 37.7], [-122.3, 37.8]]
    traj2 = [[-122.4, 37.8], [-122.3, 37.7]]

    dist = traj_dist_rs.sspd(traj1, traj2, "spherical")

    assert isinstance(dist, float)
    assert not np.isnan(dist)
    assert dist >= 0


def test_dtw_basic():
    """Test DTW basic functionality"""
    traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

    result = traj_dist_rs.dtw(traj1, traj2, "euclidean", use_full_matrix=False)

    # Check that result has distance and matrix attributes
    assert hasattr(result, "distance")
    assert hasattr(result, "matrix")
    assert isinstance(result.distance, float)
    assert not np.isnan(result.distance)
    assert result.distance >= 0
    assert result.matrix is None


def test_dtw_with_matrix():
    """Test DTW return matrix"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]

    result = traj_dist_rs.dtw(traj1, traj2, "euclidean", use_full_matrix=True)

    assert hasattr(result, "distance")
    assert hasattr(result, "matrix")
    assert isinstance(result.distance, float)
    assert result.matrix is not None
    assert isinstance(result.matrix, np.ndarray)


def test_hausdorff_basic():
    """Test Hausdorff basic functionality"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]

    dist = traj_dist_rs.hausdorff(traj1, traj2, "euclidean")

    assert isinstance(dist, float)
    assert not np.isnan(dist)
    assert dist >= 0


def test_lcss_basic():
    """Test LCSS basic functionality"""
    traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

    result = traj_dist_rs.lcss(
        traj1, traj2, "euclidean", eps=0.5, use_full_matrix=False
    )

    assert hasattr(result, "distance")
    assert hasattr(result, "matrix")
    assert isinstance(result.distance, float)
    assert 0 <= result.distance <= 1


def test_edr_basic():
    """Test EDR basic functionality"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]

    result = traj_dist_rs.edr(traj1, traj2, "euclidean", eps=0.5, use_full_matrix=False)

    assert hasattr(result, "distance")
    assert hasattr(result, "matrix")
    assert isinstance(result.distance, float)
    assert 0 <= result.distance <= 1


def test_erp_standard_basic():
    """Test ERP standard basic functionality"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]

    result = traj_dist_rs.erp_standard(
        traj1, traj2, "euclidean", g=[0.0, 0.0], use_full_matrix=False
    )

    assert hasattr(result, "distance")
    assert hasattr(result, "matrix")
    assert isinstance(result.distance, float)
    assert result.distance >= 0


def test_erp_compat_basic():
    """Test ERP compat_traj_dist basic functionality"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]

    result = traj_dist_rs.erp_compat_traj_dist(
        traj1, traj2, "euclidean", g=[0.0, 0.0], use_full_matrix=False
    )

    assert hasattr(result, "distance")
    assert hasattr(result, "matrix")
    assert isinstance(result.distance, float)
    assert result.distance >= 0


def test_discret_frechet_basic():
    """Test Discret Frechet basic functionality"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]

    result = traj_dist_rs.discret_frechet(
        traj1, traj2, "euclidean", use_full_matrix=False
    )

    assert hasattr(result, "distance")
    assert hasattr(result, "matrix")
    assert isinstance(result.distance, float)
    assert result.distance >= 0


def test_same_trajectory_zero_distance():
    """Test that distance for identical trajectories should be close to 0"""
    traj = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]

    # SSPD
    dist = traj_dist_rs.sspd(traj, traj, "euclidean")
    assert dist < 0.01  # Allow small floating point error

    # Hausdorff
    dist = traj_dist_rs.hausdorff(traj, traj, "euclidean")
    assert dist < 0.01


def test_numpy_array_input():
    """Test numpy array input"""
    traj1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    traj2 = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])

    dist = traj_dist_rs.sspd(traj1, traj2, "euclidean")

    assert isinstance(dist, float)
    assert not np.isnan(dist)
    assert dist >= 0


def test_empty_trajectory_handling():
    """Test empty trajectory handling"""
    # Single point trajectory (SSPD should return inf)
    traj1 = [[0.0, 0.0]]
    traj2 = [[0.0, 1.0]]

    dist = traj_dist_rs.sspd(traj1, traj2, "euclidean")
    assert dist == float("inf")


def test_invalid_distance_type():
    """Test invalid distance type"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]

    # Should raise exception or return error
    with pytest.raises(Exception):
        traj_dist_rs.sspd(traj1, traj2, "invalid_distance_type")


def test_readme_examples():
    """Test example code from README"""
    traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

    # SSPD
    distance = traj_dist_rs.sspd(traj1, traj2, dist_type="euclidean")
    assert isinstance(distance, float)
    assert not np.isnan(distance)

    # DTW
    result = traj_dist_rs.dtw(
        traj1, traj2, dist_type="euclidean", use_full_matrix=False
    )
    assert isinstance(result.distance, float)

    # Hausdorff
    distance = traj_dist_rs.hausdorff(traj1, traj2, dist_type="spherical")
    assert isinstance(distance, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
