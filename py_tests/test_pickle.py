"""
Pickle serialization test cases for DpResult

Tests that DpResult objects can be pickled and unpickled correctly,
which is required for joblib parallel processing.
"""

import pickle

import numpy as np
import pytest
import traj_dist_rs


class TestDpResultPickle:
    """Test pickle serialization of DpResult"""

    def test_dtw_result_pickle_without_matrix(self):
        """Test pickling DTW result without matrix"""
        traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

        result = traj_dist_rs.dtw(traj1, traj2, "euclidean", use_full_matrix=False)

        # Pickle and unpickle
        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)

        # Verify distance is preserved
        assert unpickled.distance == pytest.approx(result.distance, rel=1e-8)

        # Verify matrix is None
        assert unpickled.matrix is None

    def test_dtw_result_pickle_with_matrix(self):
        """Test pickling DTW result with matrix"""
        traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

        result = traj_dist_rs.dtw(traj1, traj2, "euclidean", use_full_matrix=True)

        # Pickle and unpickle
        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)

        # Verify distance is preserved
        assert unpickled.distance == pytest.approx(result.distance, rel=1e-8)

        # Verify matrix is preserved
        assert unpickled.matrix is not None
        np.testing.assert_array_almost_equal(unpickled.matrix, result.matrix, decimal=8)

    def test_lcss_result_pickle(self):
        """Test pickling LCSS result"""
        traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

        result = traj_dist_rs.lcss(
            traj1, traj2, "euclidean", eps=0.5, use_full_matrix=True
        )

        # Pickle and unpickle
        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)

        # Verify distance is preserved
        assert unpickled.distance == pytest.approx(result.distance, rel=1e-8)

        # Verify matrix is preserved
        assert unpickled.matrix is not None
        np.testing.assert_array_almost_equal(unpickled.matrix, result.matrix, decimal=8)

    def test_edr_result_pickle(self):
        """Test pickling EDR result"""
        traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

        result = traj_dist_rs.edr(
            traj1, traj2, "euclidean", eps=0.5, use_full_matrix=True
        )

        # Pickle and unpickle
        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)

        # Verify distance is preserved
        assert unpickled.distance == pytest.approx(result.distance, rel=1e-8)

        # Verify matrix is preserved
        assert unpickled.matrix is not None
        np.testing.assert_array_almost_equal(unpickled.matrix, result.matrix, decimal=8)

    def test_erp_result_pickle(self):
        """Test pickling ERP result"""
        traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

        result = traj_dist_rs.erp_standard(
            traj1, traj2, "euclidean", g=[0.0, 0.0], use_full_matrix=True
        )

        # Pickle and unpickle
        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)

        # Verify distance is preserved
        assert unpickled.distance == pytest.approx(result.distance, rel=1e-8)

        # Verify matrix is preserved
        assert unpickled.matrix is not None
        np.testing.assert_array_almost_equal(unpickled.matrix, result.matrix, decimal=8)

    def test_discret_frechet_result_pickle(self):
        """Test pickling Discret Frechet result"""
        traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

        result = traj_dist_rs.discret_frechet(
            traj1, traj2, "euclidean", use_full_matrix=True
        )

        # Pickle and unpickle
        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)

        # Verify distance is preserved
        assert unpickled.distance == pytest.approx(result.distance, rel=1e-8)

        # Verify matrix is preserved
        assert unpickled.matrix is not None
        np.testing.assert_array_almost_equal(unpickled.matrix, result.matrix, decimal=8)

    def test_pickle_protocol_versions(self):
        """Test pickle serialization with different protocol versions"""
        traj1 = [[0.0, 0.0], [1.0, 1.0]]
        traj2 = [[0.1, 0.1], [1.1, 1.1]]

        result = traj_dist_rs.dtw(traj1, traj2, "euclidean", use_full_matrix=True)

        # Test different pickle protocols
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            pickled = pickle.dumps(result, protocol=protocol)
            unpickled = pickle.loads(pickled)

            assert unpickled.distance == pytest.approx(result.distance, rel=1e-8)
            assert unpickled.matrix is not None
            np.testing.assert_array_almost_equal(
                unpickled.matrix, result.matrix, decimal=8
            )
