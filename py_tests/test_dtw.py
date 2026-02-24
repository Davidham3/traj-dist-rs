"""
DTW (Dynamic Time Warping) algorithm test cases

Tests the accuracy comparison between Rust implementation and original traj-dist implementation
"""

import pytest
from test_framework import BaseDistanceTest, load_test_data_by_metainfo


class TestDTWEuclidean(BaseDistanceTest):
    """Test DTW Euclidean distance"""

    @property
    def algorithm_name(self) -> str:
        return "dtw"

    def test_dtw_euclidean_accuracy(self, all_metainfo, data_dir):
        """
        Test DTW Euclidean distance accuracy

        Verify that Rust implementation results match original traj-dist implementation within 1e-8 tolerance
        """
        dtw_metainfo = all_metainfo.get("dtw", [])
        euclidean_metainfo = [m for m in dtw_metainfo if m.type_d == "euclidean"]

        if not euclidean_metainfo:
            pytest.skip("DTW Euclidean distance test data not found")

        test_data = load_test_data_by_metainfo(euclidean_metainfo[0], data_dir)
        self._test_accuracy(test_data, "euclidean")

    def test_dtw_euclidean_identical_trajectories(self):
        """Test that distance for identical trajectories should be close to 0"""
        self._check_identical_trajectories("euclidean")

    def test_dtw_euclidean_simple_case(self):
        """Test simple trajectory pair"""
        self._check_simple_case("euclidean")

    def test_dtw_euclidean_empty_trajectory(self):
        """Test empty trajectory case"""
        self._check_empty_trajectory("euclidean")

    def test_dtw_euclidean_single_point(self):
        """Test single point trajectory"""
        self._check_single_point("euclidean")


class TestDTWSpherical(BaseDistanceTest):
    """Test DTW Spherical distance"""

    @property
    def algorithm_name(self) -> str:
        return "dtw"

    def test_dtw_spherical_accuracy(self, all_metainfo, data_dir):
        """
        Test DTW Spherical distance accuracy

        Verify that Rust implementation results match original traj-dist implementation within 1e-8 tolerance
        """
        dtw_metainfo = all_metainfo.get("dtw", [])
        spherical_metainfo = [m for m in dtw_metainfo if m.type_d == "spherical"]

        if not spherical_metainfo:
            pytest.skip("DTW Spherical distance test data not found")

        test_data = load_test_data_by_metainfo(spherical_metainfo[0], data_dir)
        self._test_accuracy(test_data, "spherical")

    def test_dtw_spherical_identical_trajectories(self):
        """Test that distance for identical trajectories should be close to 0"""
        self._check_identical_trajectories("spherical")

    def test_dtw_spherical_simple_case(self):
        """Test simple trajectory pair"""
        self._check_simple_case("spherical")


class TestDTWParameterValidation(BaseDistanceTest):
    """Test DTW parameter validation"""

    @property
    def algorithm_name(self) -> str:
        return "dtw"

    def test_dtw_invalid_distance_type(self):
        """Test that invalid distance type should raise exception"""
        self._check_invalid_distance_type()

    def test_dtw_valid_distance_types(self):
        """Test valid distance types"""
        self._check_valid_distance_types()

    def test_dtw_invalid_trajectory_format(self):
        """Test that invalid trajectory format should raise exception"""
        self._check_invalid_trajectory_format()
