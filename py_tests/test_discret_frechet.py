"""
Discret Frechet algorithm test cases

Tests the accuracy comparison between Rust implementation and original traj-dist implementation
"""

import pytest
from test_framework import BaseDistanceTest, load_test_data_by_metainfo


class TestDiscretFrechetEuclidean(BaseDistanceTest):
    """Test Discret Frechet Euclidean distance"""

    @property
    def algorithm_name(self) -> str:
        return "discret_frechet"

    def test_discret_frechet_euclidean_accuracy(self, all_metainfo, data_dir):
        """
        Test Discret Frechet Euclidean distance accuracy

        Verify that Rust implementation results match original traj-dist implementation within 1e-8 tolerance
        """
        discret_frechet_metainfo = all_metainfo.get("discret_frechet", [])
        euclidean_metainfo = [
            m for m in discret_frechet_metainfo if m.type_d == "euclidean"
        ]

        if not euclidean_metainfo:
            pytest.skip("Discret Frechet Euclidean distance test data not found")

        test_data = load_test_data_by_metainfo(euclidean_metainfo[0], data_dir)
        self._test_accuracy(test_data, "euclidean")

    def test_discret_frechet_euclidean_identical_trajectories(self):
        """Test that distance for identical trajectories should be close to 0"""
        distance_type = "euclidean"
        self._check_identical_trajectories(distance_type)

    def test_discret_frechet_euclidean_simple_case(self):
        """Test simple trajectory pair"""
        distance_type = "euclidean"
        self._check_simple_case(distance_type)

    def test_discret_frechet_euclidean_empty_trajectory(self):
        """Test empty trajectory case"""
        distance_type = "euclidean"
        self._check_empty_trajectory(distance_type)

    def test_discret_frechet_euclidean_single_point(self):
        """Test single point trajectory"""
        distance_type = "euclidean"
        self._check_single_point(distance_type)


class TestDiscretFrechetParameterValidation(BaseDistanceTest):
    """Test Discret Frechet parameter validation"""

    @property
    def algorithm_name(self) -> str:
        return "discret_frechet"

    def test_discret_frechet_invalid_distance_type(self):
        """Test that invalid distance type should raise exception"""
        self._check_invalid_distance_type()

    def test_discret_frechet_valid_distance_types(self):
        """Test valid distance types"""
        self._check_valid_distance_types()

    def test_discret_frechet_invalid_trajectory_format(self):
        """Test that invalid trajectory format should raise exception"""
        self._check_invalid_trajectory_format()
