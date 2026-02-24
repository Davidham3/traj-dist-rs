"""
Precomputed distance matrix interface test cases

Tests distance calculation functionality using precomputed distance matrices
"""

import numpy as np
import pytest
import traj_dist_rs


class TestDTWWithMatrix:
    """Test DTW with precomputed distance matrix"""

    def test_dtw_with_matrix_basic(self):
        """Test basic precomputed distance matrix functionality"""
        t1 = [[0.0, 0.0], [1.0, 1.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0]]

        # Calculate using standard interface
        result_standard = traj_dist_rs.dtw(t1, t2, "euclidean")

        # Manually precompute distance matrix
        dist_matrix = np.zeros((2, 2))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # Calculate using precomputed matrix
        result_matrix = traj_dist_rs.dtw_with_matrix(dist_matrix)

        # Results from both methods should be consistent
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10

    def test_dtw_with_matrix_and_full_matrix(self):
        """Test precomputed matrix with full DP matrix return"""
        dist_matrix = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        # Don't return full matrix
        result_no_matrix = traj_dist_rs.dtw_with_matrix(
            dist_matrix, use_full_matrix=False
        )
        assert result_no_matrix.matrix is None

        # Return full matrix
        result_with_matrix = traj_dist_rs.dtw_with_matrix(
            dist_matrix, use_full_matrix=True
        )
        assert result_with_matrix.matrix is not None

        # Distance values should be consistent
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-10

    def test_dtw_with_matrix_consistency(self):
        """Test consistency between precomputed matrix and standard interface"""
        t1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]]

        # Use standard interface
        result_standard = traj_dist_rs.dtw(t1, t2, "euclidean")

        # Precompute distance matrix
        dist_matrix = np.zeros((3, 3))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # Use precomputed matrix
        result_matrix = traj_dist_rs.dtw_with_matrix(dist_matrix)

        # Results from both methods should be consistent
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10


class TestLCSSWithMatrix:
    """Test LCSS with precomputed distance matrix"""

    def test_lcss_with_matrix_basic(self):
        """Test basic precomputed distance matrix functionality"""
        t1 = [[0.0, 0.0], [1.0, 1.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0]]

        # Calculate using standard interface
        result_standard = traj_dist_rs.lcss(t1, t2, "euclidean", eps=0.5)

        # Manually precompute distance matrix
        dist_matrix = np.zeros((2, 2))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # Calculate using precomputed matrix
        result_matrix = traj_dist_rs.lcss_with_matrix(dist_matrix, eps=0.5)

        # Results from both methods should be consistent
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10

    def test_lcss_with_matrix_and_full_matrix(self):
        """Test precomputed matrix with full DP matrix return"""
        dist_matrix = np.array(
            [
                [0.1, 0.1],
                [0.1, 0.1],
            ]
        )

        eps = 0.5

        # Don't return full matrix
        result_no_matrix = traj_dist_rs.lcss_with_matrix(
            dist_matrix, eps, use_full_matrix=False
        )
        assert result_no_matrix.matrix is None

        # Return full matrix
        result_with_matrix = traj_dist_rs.lcss_with_matrix(
            dist_matrix, eps, use_full_matrix=True
        )
        assert result_with_matrix.matrix is not None

        # Distance values should be consistent
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-10


class TestEDRWithMatrix:
    """Test EDR with precomputed distance matrix"""

    def test_edr_with_matrix_basic(self):
        """Test basic precomputed distance matrix functionality"""
        t1 = [[0.0, 0.0], [1.0, 1.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0]]

        # Calculate using standard interface
        result_standard = traj_dist_rs.edr(t1, t2, "euclidean", eps=0.5)

        # Manually precompute distance matrix
        dist_matrix = np.zeros((2, 2))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # Calculate using precomputed matrix
        result_matrix = traj_dist_rs.edr_with_matrix(dist_matrix, eps=0.5)

        # Results from both methods should be consistent
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10

    def test_edr_with_matrix_and_full_matrix(self):
        """Test precomputed matrix with full DP matrix return"""
        dist_matrix = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        eps = 0.5

        # Don't return full matrix
        result_no_matrix = traj_dist_rs.edr_with_matrix(
            dist_matrix, eps, use_full_matrix=False
        )
        assert result_no_matrix.matrix is None

        # Return full matrix
        result_with_matrix = traj_dist_rs.edr_with_matrix(
            dist_matrix, eps, use_full_matrix=True
        )
        assert result_with_matrix.matrix is not None

        # Distance values should be consistent
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-10


class TestDiscretFrechetWithMatrix:
    """Test Discret Frechet with precomputed distance matrix"""

    def test_discret_frechet_with_matrix_basic(self):
        """Test basic precomputed distance matrix functionality"""
        t1 = [[0.0, 0.0], [1.0, 1.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0]]

        # Calculate using standard interface
        result_standard = traj_dist_rs.discret_frechet(t1, t2, "euclidean")

        # Manually precompute distance matrix
        dist_matrix = np.zeros((2, 2))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # Calculate using precomputed matrix
        result_matrix = traj_dist_rs.discret_frechet_with_matrix(dist_matrix)

        # Results from both methods should be consistent
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10

    def test_discret_frechet_with_matrix_and_full_matrix(self):
        """Test precomputed matrix with full DP matrix return"""
        dist_matrix = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        # Don't return full matrix
        result_no_matrix = traj_dist_rs.discret_frechet_with_matrix(
            dist_matrix, use_full_matrix=False
        )
        assert result_no_matrix.matrix is None

        # Return full matrix
        result_with_matrix = traj_dist_rs.discret_frechet_with_matrix(
            dist_matrix, use_full_matrix=True
        )
        assert result_with_matrix.matrix is not None

        # Distance values should be consistent
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-10


class TestERPStandardWithMatrix:
    """Test ERP standard with precomputed distance matrix"""

    def test_erp_standard_with_matrix_basic(self):
        """Test basic precomputed distance matrix functionality"""
        t1 = [[0.0, 0.0], [1.0, 1.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0]]
        g = [0.0, 0.0]

        # Calculate using standard interface
        result_standard = traj_dist_rs.erp_standard(t1, t2, "euclidean", g)

        # Manually precompute distance matrix
        dist_matrix = np.zeros((2, 2))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # Precompute gap distances
        seq0_gap_dists = np.zeros(2)
        for i in range(len(t1)):
            dx = t1[i][0] - g[0]
            dy = t1[i][1] - g[1]
            seq0_gap_dists[i] = np.sqrt(dx * dx + dy * dy)

        seq1_gap_dists = np.zeros(2)
        for j in range(len(t2)):
            dx = t2[j][0] - g[0]
            dy = t2[j][1] - g[1]
            seq1_gap_dists[j] = np.sqrt(dx * dx + dy * dy)

        # Calculate using precomputed matrix
        result_matrix = traj_dist_rs.erp_standard_with_matrix(
            dist_matrix, seq0_gap_dists, seq1_gap_dists
        )

        # Results from both methods should be consistent
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10

    def test_erp_standard_with_matrix_and_full_matrix(self):
        """Test precomputed matrix with full DP matrix return"""
        dist_matrix = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )
        seq0_gap_dists = np.array([1.0, 1.0])
        seq1_gap_dists = np.array([1.0, 1.0])

        # Don't return full matrix
        result_no_matrix = traj_dist_rs.erp_standard_with_matrix(
            dist_matrix, seq0_gap_dists, seq1_gap_dists, use_full_matrix=False
        )
        assert result_no_matrix.matrix is None

        # Return full matrix
        result_with_matrix = traj_dist_rs.erp_standard_with_matrix(
            dist_matrix, seq0_gap_dists, seq1_gap_dists, use_full_matrix=True
        )
        assert result_with_matrix.matrix is not None

        # Distance values should be consistent
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-10


class TestERPCompatWithMatrix:
    """Test ERP compat with precomputed distance matrix"""

    def test_erp_compat_with_matrix_basic(self):
        """Test basic precomputed distance matrix functionality"""
        t1 = [[0.0, 0.0], [1.0, 1.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0]]
        g = [0.0, 0.0]

        # Calculate using standard interface
        result_standard = traj_dist_rs.erp_compat_traj_dist(t1, t2, "euclidean", g)

        # Manually precompute distance matrix
        dist_matrix = np.zeros((2, 2))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # Precompute gap distances
        seq0_gap_dists = np.zeros(2)
        for i in range(len(t1)):
            dx = t1[i][0] - g[0]
            dy = t1[i][1] - g[1]
            seq0_gap_dists[i] = np.sqrt(dx * dx + dy * dy)

        seq1_gap_dists = np.zeros(2)
        for j in range(len(t2)):
            dx = t2[j][0] - g[0]
            dy = t2[j][1] - g[1]
            seq1_gap_dists[j] = np.sqrt(dx * dx + dy * dy)

        # Calculate using precomputed matrix
        result_matrix = traj_dist_rs.erp_compat_traj_dist_with_matrix(
            dist_matrix, seq0_gap_dists, seq1_gap_dists
        )

        # Results from both methods should be consistent
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10

    def test_erp_compat_with_matrix_and_full_matrix(self):
        """Test precomputed matrix with full DP matrix return"""
        dist_matrix = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )
        seq0_gap_dists = np.array([1.0, 1.0])
        seq1_gap_dists = np.array([1.0, 1.0])

        # Don't return full matrix
        result_no_matrix = traj_dist_rs.erp_compat_traj_dist_with_matrix(
            dist_matrix, seq0_gap_dists, seq1_gap_dists, use_full_matrix=False
        )
        assert result_no_matrix.matrix is None

        # Return full matrix
        result_with_matrix = traj_dist_rs.erp_compat_traj_dist_with_matrix(
            dist_matrix, seq0_gap_dists, seq1_gap_dists, use_full_matrix=True
        )
        assert result_with_matrix.matrix is not None

        # Distance values should be consistent
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-10


class TestMatrixValidation:
    """Test input validation"""

    def test_invalid_matrix_shape(self):
        """Test invalid matrix shape"""
        # 1D array should fail
        with pytest.raises(ValueError, match="must be a 2D numpy array"):
            traj_dist_rs.dtw_with_matrix(np.array([1.0, 2.0, 3.0]))

        # 3D array should fail
        with pytest.raises(ValueError, match="must be a 2D numpy array"):
            traj_dist_rs.dtw_with_matrix(np.array([[[1.0]]]))

    def test_invalid_matrix_dtype(self):
        """Test invalid matrix data type"""
        # Integer array should fail
        with pytest.raises(
            ValueError, match="must be a 2D numpy array of float64 values"
        ):
            traj_dist_rs.dtw_with_matrix(np.array([[1, 2], [3, 4]]))


class TestEmptyMatrix:
    """Test empty matrix cases"""

    def test_dtw_empty_matrix(self):
        """Test empty matrix"""
        dist_matrix = np.zeros((0, 2))
        result = traj_dist_rs.dtw_with_matrix(dist_matrix)
        # Empty trajectory should return f64::MAX
        assert result.distance == 1.7976931348623157e308

    def test_dtw_matrix_with_empty_row(self):
        """Test matrix with empty row"""
        dist_matrix = np.zeros((2, 0))
        result = traj_dist_rs.dtw_with_matrix(dist_matrix)
        # Empty trajectory should return f64::MAX
        assert result.distance == 1.7976931348623157e308
