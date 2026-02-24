"""
EDR (Edit Distance on Real sequence) algorithm test cases

Tests the accuracy comparison between Rust implementation and original traj-dist implementation
"""

import pytest
from test_framework import (DistanceTestWithHyperparameters,
                            get_hyperparameter_value_from_metainfo,
                            load_test_data_by_metainfo)


class TestEDREuclidean(DistanceTestWithHyperparameters):
    """Test EDR Euclidean distance"""

    @property
    def algorithm_name(self) -> str:
        return "edr"

    def test_edr_euclidean_accuracy(self, all_metainfo, data_dir):
        """
        Test EDR Euclidean distance accuracy

        Verify that Rust implementation results match original traj-dist implementation within 1e-8 tolerance
        """
        edr_metainfo = all_metainfo.get("edr", [])
        euclidean_metainfo = [m for m in edr_metainfo if m.type_d == "euclidean"]

        if not euclidean_metainfo:
            pytest.skip("EDR Euclidean distance test data not found")

        # Test all eps values
        for metainfo in euclidean_metainfo:
            eps = get_hyperparameter_value_from_metainfo(metainfo)
            test_data = load_test_data_by_metainfo(metainfo, data_dir)
            self._test_accuracy(test_data, "euclidean", eps)

    def test_edr_euclidean_identical_trajectories(self):
        """Test that distance for identical trajectories should be close to 0"""
        self._check_identical_trajectories("euclidean", eps=1e-6)

    def test_edr_euclidean_simple_case(self):
        """Test simple trajectory pair"""
        self._check_simple_case("euclidean", eps=0.5)

    def test_edr_euclidean_empty_trajectory(self):
        """Test empty trajectory case"""
        self._check_empty_trajectory("euclidean", eps=1.0)

    def test_edr_euclidean_eps_effect(self):
        """Test eps parameter effect"""
        self._check_hyperparameter_effect("euclidean", 0.05, 1000.0)

    def test_edr_euclidean_single_point(self):
        """Test single point trajectory"""
        self._check_single_point("euclidean", eps=0.5)


class TestEDRSpherical(DistanceTestWithHyperparameters):
    """Test EDR Spherical distance"""

    @property
    def algorithm_name(self) -> str:
        return "edr"

    def test_edr_spherical_accuracy(self, all_metainfo, data_dir):
        """
        Test EDR Spherical distance accuracy

        Verify that Rust implementation results match original traj-dist implementation within 1e-8 tolerance
        """
        edr_metainfo = all_metainfo.get("edr", [])
        spherical_metainfo = [m for m in edr_metainfo if m.type_d == "spherical"]

        if not spherical_metainfo:
            pytest.skip("EDR Spherical distance test data not found")

        # Test all eps values
        for metainfo in spherical_metainfo:
            eps = get_hyperparameter_value_from_metainfo(metainfo)
            test_data = load_test_data_by_metainfo(metainfo, data_dir)
            self._test_accuracy(test_data, "spherical", eps)

    def test_edr_spherical_identical_trajectories(self):
        """Test that distance for identical trajectories should be close to 0"""
        self._check_identical_trajectories("spherical", eps=1e-6)

    def test_edr_spherical_simple_case(self):
        """Test simple trajectory pair"""
        # Use smaller eps value to ensure trajectories don't completely match
        self._check_simple_case("spherical", eps=10.0)

    def test_edr_spherical_eps_effect(self):
        """Test eps parameter effect"""
        self._check_hyperparameter_effect("spherical", 10.0, 100000.0)


class TestEDRParameterValidation(DistanceTestWithHyperparameters):
    """Test EDR parameter validation"""

    @property
    def algorithm_name(self) -> str:
        return "edr"

    def test_edr_invalid_distance_type(self):
        """Test that invalid distance type should raise exception"""
        self._check_invalid_distance_type()

    def test_edr_valid_distance_types(self):
        """Test valid distance types"""
        self._check_valid_distance_types()

    def test_edr_invalid_trajectory_format(self):
        """Test that invalid trajectory format should raise exception"""
        self._check_invalid_trajectory_format()

    def test_edr_eps_parameter(self):
        """Test eps parameter effect"""
        traj1 = [[0.0, 0.0], [1.0, 1.0]]
        traj2 = [[0.0, 0.0], [1.0, 1.0]]

        # Small eps should match identical points (distance is 0)
        distance_small = self._call_distance_function(traj1, traj2, "euclidean", 1e-6)
        assert distance_small < 1e-8

        # eps < 0 should also work (won't match any points since distance is always >= 0)
        distance_negative = self._call_distance_function(
            traj1, traj2, "euclidean", -1.0
        )
        assert distance_negative == 1.0

        # Very large eps should match all points
        distance_large = self._call_distance_function(traj1, traj2, "euclidean", 1e10)
        assert distance_large < 1e-8
