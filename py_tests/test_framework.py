"""
Unified test framework base class

Provides common methods and utilities for trajectory distance algorithm testing
"""

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Union

import pytest
import traj_dist_rs._lib as traj_dist_rs
from pydantic import ValidationError
from schemas import Metainfo


class AlgorithmConfig(TypedDict):
    """Algorithm configuration type definition"""

    name: str  # Algorithm name
    function_name: str  # Python binding function name
    distance_types: List[str]  # Supported distance types ["euclidean", "spherical"]
    has_hyperparameters: bool  # Whether it has hyperparameters
    hyperparameter_name: Optional[str]  # Hyperparameter name
    hyperparameter_values: Optional[
        Dict[str, List[Union[float, List[float]]]]
    ]  # Hyperparameter value configuration


def get_algorithm_config_from_metainfo(
    algorithm: str, all_metainfo: Optional[Dict[str, List[Metainfo]]] = None
) -> AlgorithmConfig:
    """
    Dynamically get algorithm configuration information from metadata

    Args:
        algorithm: Algorithm name
        all_metainfo: All metadata (list of Pydantic model objects)

    Returns:
        Algorithm configuration dictionary
    """
    if all_metainfo is None:
        all_metainfo = load_all_metainfo_from_data_dir()

    if algorithm not in all_metainfo:
        raise KeyError(f"Metadata for algorithm {algorithm} not found")

    metainfo_list = all_metainfo[algorithm]

    # Get basic algorithm information
    first_metainfo = metainfo_list[0] if metainfo_list else None
    if not first_metainfo:
        raise ValueError(f"Metadata list for algorithm {algorithm} is empty")

    # Find supported distance types for this algorithm (using Pydantic model attributes)
    distance_types = list({m.type_d for m in metainfo_list})

    # Check if it contains hyperparameter fields (using Pydantic model attributes)
    has_hyperparameters = any(
        m.eps is not None or m.g is not None or m.precision is not None
        for m in metainfo_list
    )

    # Determine hyperparameter name (using Pydantic model attributes)
    hyperparameter_name = None
    if any(m.eps is not None for m in metainfo_list):
        hyperparameter_name = "eps"
    elif any(m.g is not None for m in metainfo_list):
        hyperparameter_name = "g"
    elif any(m.precision is not None for m in metainfo_list):
        hyperparameter_name = "precision"

    # Generate algorithm name (title case)
    name = algorithm.replace("_", " ").title()

    config = {
        "name": name,
        "function_name": algorithm,
        "distance_types": distance_types,
        "has_hyperparameters": has_hyperparameters,
        "hyperparameter_name": hyperparameter_name,
        "hyperparameter_values": None,  # Get values from metadata, not hardcoded
    }

    return config


def get_algorithm_config(algorithm: str) -> AlgorithmConfig:
    """Get configuration for specified algorithm (dynamically loaded from metadata)"""
    return get_algorithm_config_from_metainfo(algorithm)


def load_all_metainfo_from_data_dir(
    data_dir: Optional[Path] = None,
) -> Dict[str, List[Metainfo]]:
    """
    Dynamically load all metadata from data directory and parse using Pydantic models

    Args:
        data_dir: Data directory (defaults to py_tests/data)

    Returns:
        Dictionary with algorithm names as keys and lists of Metainfo objects as values
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    metainfo_dir = data_dir / "metainfo"
    all_metainfo = {}

    for metainfo_file in metainfo_dir.glob("*.jsonl"):
        algorithm = metainfo_file.stem
        metainfo_list = []
        with open(metainfo_file, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        # Parse and validate JSON line using Pydantic model
                        meta_obj = Metainfo.model_validate_json(line)
                        metainfo_list.append(meta_obj)
                    except ValidationError as e:
                        print(
                            f"Warning: Failed to parse line {i+1} in file {metainfo_file.name}: {e}"
                        )
        all_metainfo[algorithm] = metainfo_list

    return all_metainfo


def load_metainfo(algorithm: str, data_dir: Optional[Path] = None) -> List[Metainfo]:
    """
    Load metadata for specified algorithm

    Args:
        algorithm: Algorithm name
        data_dir: Data directory (defaults to py_tests/data)

    Returns:
        List of Metainfo objects
    """
    all_metainfo = load_all_metainfo_from_data_dir(data_dir)
    if algorithm not in all_metainfo:
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        metainfo_file = data_dir / "metainfo" / f"{algorithm}.jsonl"
        raise FileNotFoundError(f"Metadata file not found: {metainfo_file}")
    return all_metainfo[algorithm]


def get_sample_path(metainfo: Metainfo, data_dir: Optional[Path] = None) -> Path:
    """
    Get sample file path from metadata object

    Args:
        metainfo: Metainfo object
        data_dir: Data directory (defaults to py_tests/data)

    Returns:
        Sample file path
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    # Use attribute access instead of dictionary key access
    return data_dir / metainfo.sample_file


class BaseDistanceTest(ABC):
    """
    Distance algorithm test base class

    Provides common test methods, subclasses only need to specify algorithm name to automatically generate test cases
    """

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Algorithm name, must be implemented by subclass"""

    @property
    def config(self):
        """Get algorithm configuration"""
        return get_algorithm_config(self.algorithm_name)

    @property
    def function_name(self) -> str:
        """Get Python binding function name"""
        return self.config["function_name"]

    def _get_hyperparameter_value_from_metainfo(
        self, all_metainfo: Dict[str, List[Metainfo]], distance_type: str
    ) -> Optional[Union[float, List[float], int]]:
        """
        Get hyperparameter value from metadata (updated to use Pydantic model)
        """
        algorithm_metainfo = all_metainfo.get(self.algorithm_name, [])
        return HyperparameterUtil.get_hyperparameter_value(
            self.algorithm_name, distance_type, algorithm_metainfo
        )

    def _get_distance_function(self):
        """Get distance calculation function"""
        try:
            import traj_dist_rs._lib as traj_dist_rs

            return getattr(traj_dist_rs, self.function_name)
        except ImportError:
            pytest.skip(
                "traj_dist_rs module not installed, please run 'maturin develop' first"
            )
        except AttributeError:
            pytest.skip(f"Function {self.function_name} not implemented")

    def _call_distance_function(
        self,
        traj1: List[List[float]],
        traj2: List[List[float]],
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
    ) -> float:
        """
        Call distance calculation function

        Args:
            traj1: Trajectory 1
            traj2: Trajectory 2
            distance_type: Distance type
            hyperparameter_value: Hyperparameter value (optional)

        Returns:
            Calculated distance value
        """
        func = self._get_distance_function()
        config = self.config

        # Special handling for DTW algorithm, it has use_full_matrix parameter but no hyperparameters
        if self.algorithm_name == "dtw":
            result = func(traj1, traj2, distance_type, use_full_matrix=False)
            # DTW now returns DpResult object
            if hasattr(result, "distance"):
                return result.distance
            return result

        # Special handling for LCSS, EDR algorithms, they have use_full_matrix parameter and eps hyperparameter
        if self.algorithm_name in ["lcss", "edr"]:
            param_name = config["hyperparameter_name"]
            if hyperparameter_value is None:
                hyperparameter_value = 1e-6
            result = func(
                traj1,
                traj2,
                distance_type,
                eps=hyperparameter_value,
                use_full_matrix=False,
            )
            # These algorithms now return DpResult object
            if hasattr(result, "distance"):
                return result.distance
            return result

        # Special handling for ERP algorithm, it has use_full_matrix parameter and g hyperparameter
        if self.algorithm_name == "erp":
            param_name = config["hyperparameter_name"]
            if hyperparameter_value is None:
                hyperparameter_value = [0.0, 0.0]
            # Get actual function name (might be erp_compat_traj_dist or erp_standard)
            func_name = self.function_name
            if func_name == "erp_compat_traj_dist":
                result = traj_dist_rs.erp_compat_traj_dist(
                    traj1,
                    traj2,
                    distance_type,
                    g=hyperparameter_value,
                    use_full_matrix=False,
                )
            elif func_name == "erp_standard":
                result = traj_dist_rs.erp_standard(
                    traj1,
                    traj2,
                    distance_type,
                    g=hyperparameter_value,
                    use_full_matrix=False,
                )
            else:
                result = func(
                    traj1,
                    traj2,
                    distance_type,
                    g=hyperparameter_value,
                    use_full_matrix=False,
                )
            # ERP now returns DpResult object
            if hasattr(result, "distance"):
                return result.distance
            return result

        # Special handling for Discret Frechet algorithm, it has use_full_matrix parameter but no hyperparameters
        if self.algorithm_name == "discret_frechet":
            result = func(traj1, traj2, distance_type, use_full_matrix=False)
            # Discret Frechet now returns DpResult object
            if hasattr(result, "distance"):
                return result.distance
            return result

        if not config["has_hyperparameters"]:
            result = func(traj1, traj2, distance_type)
            # Check if DpResult object is returned
            if hasattr(result, "distance"):
                return result.distance
            return result
        else:
            # If no hyperparameter value provided, use default value
            if hyperparameter_value is None:
                param_name = config["hyperparameter_name"]
                # Get default value from config, if not in config use default value
                if param_name == "eps":
                    # For eps parameter, use a very small value so it matches for most cases
                    hyperparameter_value = 1e-6
                elif param_name == "g":
                    # For g parameter (ERP), use 0,0, so penalty equals Euclidean distance
                    hyperparameter_value = [0.0, 0.0]
                elif param_name == "precision":
                    # For precision parameter (SOWD), use default value 5
                    hyperparameter_value = 5

            param_name = config["hyperparameter_name"]
            if param_name == "g" and isinstance(hyperparameter_value, list):
                # ERP's g parameter is a list
                result = func(traj1, traj2, distance_type, g=hyperparameter_value)
            elif param_name == "precision":
                # SOWD's precision parameter
                result = func(
                    traj1, traj2, distance_type, precision=hyperparameter_value
                )
            else:
                # LCSS and EDR's eps parameter
                result = func(traj1, traj2, distance_type, eps=hyperparameter_value)

            # Check if DpResult object is returned
            if hasattr(result, "distance"):
                return result.distance
            return result

    def _get_expected_distance(self, test_data, hyperparameter_value=None):
        """
        Get expected distance from test data

        Args:
            test_data: Test data DataFrame
            hyperparameter_value: Hyperparameter value (for validation)

        Returns:
            List of expected distances
        """
        if test_data is None:
            pytest.skip(f"{self.algorithm_name} test data not found")

        distances = []
        for row in test_data.iter_rows(named=True):
            distances.append((row["traj1"], row["traj2"], row["distance"]))
        return distances

    def _test_accuracy(
        self,
        test_data,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        tolerance: float = 1e-8,
    ):
        """
        Common method for testing accuracy

        Args:
            test_data: Test data
            distance_type: Distance type
            hyperparameter_value: Hyperparameter value (optional)
            tolerance: Tolerance
        """
        test_cases = self._get_expected_distance(test_data, hyperparameter_value)

        for traj1, traj2, expected_distance in test_cases:
            actual_distance = self._call_distance_function(
                traj1, traj2, distance_type, hyperparameter_value
            )

            # Handle DpResult return type
            if hasattr(actual_distance, "distance"):
                actual_distance = actual_distance.distance

            error = abs(actual_distance - expected_distance)
            assert error < tolerance, (
                f"{self.algorithm_name.upper()} distance calculation error too large: {error}\n"
                f"Distance type: {distance_type}\n"
                f"Hyperparameter: {hyperparameter_value}\n"
                f"Trajectory 1: {traj1}\n"
                f"Trajectory 2: {traj2}\n"
                f"Expected distance: {expected_distance}\n"
                f"Actual distance: {actual_distance}"
            )

    def _check_identical_trajectories(
        self,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        tolerance: float = 1e-8,
        **kwargs,
    ):
        """
        Check that distance between identical trajectories should be close to 0

        Args:
            distance_type: Distance type
            hyperparameter_value: Hyperparameter value (optional)
            tolerance: Tolerance
            **kwargs: Other hyperparameters (e.g., g, eps, precision)
        """
        # Prefer hyperparameters from kwargs
        if kwargs:
            hyperparameter_value = kwargs.get(
                "eps", kwargs.get("g", kwargs.get("precision", hyperparameter_value))
            )

        if distance_type == "spherical":
            traj = [[-122.39548, 37.77668], [-122.39539, 37.77644]]
        else:
            traj = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]

        distance = self._call_distance_function(
            traj, traj, distance_type, hyperparameter_value
        )

        # Handle DpResult return type
        if hasattr(distance, "distance"):
            distance = distance.distance

        # For LCSS and EDR, distance is in [0, 1] range, identical trajectories should be close to 0
        if self.algorithm_name in ["lcss", "edr"]:
            assert (
                distance < tolerance
            ), f"{self.algorithm_name.upper()} distance for identical trajectories should be close to 0, actual: {distance}"
        else:
            assert (
                distance < tolerance
            ), f"{self.algorithm_name.upper()} distance for identical trajectories should be close to 0, actual: {distance}"

    def _check_simple_case(
        self,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        **kwargs,
    ):
        """
        Check simple trajectory pair

        Args:
            distance_type: Distance type
            hyperparameter_value: Hyperparameter value (optional)
            **kwargs: Other hyperparameters (e.g., g, eps, precision)
        """
        # Prefer hyperparameters from kwargs
        if kwargs:
            hyperparameter_value = kwargs.get(
                "eps", kwargs.get("g", kwargs.get("precision", hyperparameter_value))
            )

        if distance_type == "spherical":
            traj1 = [[-122.39548, 37.77668], [-122.39539, 37.77644]]
            traj2 = [[-122.39548, 37.77668], [-122.39537, 37.77663]]
        else:
            traj1 = [[0.0, 0.0], [1.0, 1.0]]
            traj2 = [[0.0, 1.0], [1.0, 0.0]]

        distance = self._call_distance_function(
            traj1, traj2, distance_type, hyperparameter_value
        )

        # Handle DpResult return type
        if hasattr(distance, "distance"):
            distance = distance.distance

        # Distance should be greater than 0
        assert (
            distance > 0
        ), f"{self.algorithm_name.upper()} distance for different trajectories should be greater than 0, actual: {distance}"

    def _check_empty_trajectory(
        self,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        **kwargs,
    ):
        """
        Check empty trajectory case

        Args:
            distance_type: Distance type
            hyperparameter_value: Hyperparameter value (optional)
            **kwargs: Other hyperparameters (e.g., g, eps, precision)
        """
        # Prefer hyperparameters from kwargs
        if kwargs:
            hyperparameter_value = kwargs.get(
                "eps", kwargs.get("g", kwargs.get("precision", hyperparameter_value))
            )

        if distance_type == "spherical":
            traj1 = [[-122.39548, 37.77668], [-122.39539, 37.77644]]
        else:
            traj1 = [[0.0, 0.0], [1.0, 1.0]]

        traj2 = []

        distance = self._call_distance_function(
            traj1, traj2, distance_type, hyperparameter_value
        )

        # Handle DpResult return type
        if hasattr(distance, "distance"):
            distance = distance.distance

        # For LCSS and EDR, empty trajectory should return 1.0
        if self.algorithm_name in ["lcss", "edr"]:
            assert (
                distance == 1.0
            ), f"{self.algorithm_name.upper()} empty trajectory should return 1.0, actual: {distance}"
        else:
            # Other algorithms should return maximum value
            assert (
                distance == float("inf") or distance > 1e100
            ), f"{self.algorithm_name.upper()} empty trajectory should return maximum value, actual: {distance}"

    def _check_single_point(
        self,
        distance_type: str,
        hyperparameter_value: Optional[Union[float, List[float], int]] = None,
        **kwargs,
    ):
        """
        Check single point trajectory

        Note: Some algorithms (e.g., SSPD) return inf for single point trajectories as they have no segments

        Args:
            distance_type: Distance type
            hyperparameter_value: Hyperparameter value (optional)
            **kwargs: Other hyperparameters (e.g., g, eps, precision)
        """
        # Prefer hyperparameters from kwargs
        if kwargs:
            hyperparameter_value = kwargs.get(
                "eps", kwargs.get("g", kwargs.get("precision", hyperparameter_value))
            )

        if distance_type == "spherical":
            traj1 = [[-122.39548, 37.77668]]
            traj2 = [[-122.39539, 37.77644]]
        else:
            traj1 = [[0.0, 0.0]]
            traj2 = [[1.0, 1.0]]

        distance = self._call_distance_function(
            traj1, traj2, distance_type, hyperparameter_value
        )

        # Handle DpResult return type
        if hasattr(distance, "distance"):
            distance = distance.distance

        # SSPD single point trajectory has no segments, returns inf
        if self.algorithm_name == "sspd":
            assert distance == float("inf"), (
                f"{self.algorithm_name.upper()} single point trajectory distance should be inf\n"
                f"Actual distance: {distance}"
            )
        # LCSS and EDR single point trajectories don't match, return 1.0
        elif self.algorithm_name in ["lcss", "edr"]:
            if (
                hyperparameter_value is None
                or isinstance(hyperparameter_value, (int, float))
                and hyperparameter_value < 1.0
            ):
                assert (
                    distance == 1.0
                ), f"{self.algorithm_name.upper()} single point mismatched trajectory distance should be 1.0, actual: {distance}"
            else:
                # Should match with large eps
                assert (
                    distance < 1e-8
                ), f"{self.algorithm_name.upper()} single point matched trajectory distance with large eps should be close to 0, actual: {distance}"
        # DTW and Hausdorff single point trajectory distance should be distance between two points
        elif self.algorithm_name in ["dtw", "hausdorff"]:
            if distance_type == "spherical":
                # Spherical distance, uses Haversine formula
                # Simply verify distance > 0
                assert distance > 0, (
                    f"{self.algorithm_name.upper()} single point trajectory distance should be greater than 0\n"
                    f"Actual distance: {distance}"
                )
            else:
                expected_distance = math.sqrt((1.0 - 0.0) ** 2 + (1.0 - 0.0) ** 2)
                assert abs(distance - expected_distance) < 1e-6, (
                    f"{self.algorithm_name.upper()} single point trajectory distance should be Euclidean distance between two points\n"
                    f"Expected distance: {expected_distance}\n"
                    f"Actual distance: {distance}"
                )

    def _check_invalid_distance_type(self):
        """Check that invalid distance type should raise exception"""
        traj1 = [[0.0, 0.0], [1.0, 1.0]]
        traj2 = [[0.0, 1.0], [1.0, 0.0]]

        with pytest.raises(ValueError, match="Invalid distance type"):
            self._call_distance_function(traj1, traj2, "invalid")

    def _check_valid_distance_types(self):
        """Check valid distance types"""
        traj1 = [[0.0, 0.0], [1.0, 1.0]]
        traj2 = [[0.0, 1.0], [1.0, 0.0]]

        for distance_type in self.config["distance_types"]:
            distance = self._call_distance_function(traj1, traj2, distance_type)
            # Handle DpResult return type
            if hasattr(distance, "distance"):
                distance = distance.distance
            assert (
                distance > 0
            ), f"{self.algorithm_name.upper()} {distance_type} distance should be greater than 0, actual: {distance}"

    def _check_invalid_trajectory_format(self):
        """Check that invalid trajectory format should raise exception"""
        # Test non-list input
        with pytest.raises(ValueError):
            self._call_distance_function("not a list", [[0.0, 1.0]], "euclidean")

        # Test point coordinates not having 2 elements
        with pytest.raises(ValueError):
            self._call_distance_function([[0.0, 1.0, 2.0]], [[0.0, 1.0]], "euclidean")


class DistanceTestWithHyperparameters(BaseDistanceTest):
    """
    Distance algorithm test base class with hyperparameters

    Inherits from BaseDistanceTest, adds hyperparameter-related test methods
    """

    def _check_hyperparameter_effect(
        self,
        distance_type: str,
        small_value: Union[float, int],
        large_value: Union[float, int],
    ):
        """
        Check hyperparameter effect

        Args:
            distance_type: Distance type
            small_value: Small hyperparameter value
            large_value: Large hyperparameter value
        """
        if distance_type == "spherical":
            traj1 = [[-122.39548, 37.77668], [-122.39539, 37.77644]]
            traj2 = [[-122.39548, 37.77668], [-122.39537, 37.77663]]
        else:
            traj1 = [[0.0, 0.0], [1.0, 1.0]]
            traj2 = [[0.0, 0.1], [1.0, 1.1]]

        param_name = self.config["hyperparameter_name"]

        if param_name == "eps":
            # For LCSS and EDR, large eps produces smaller distance
            distance_small = self._call_distance_function(
                traj1, traj2, distance_type, small_value
            )
            distance_large = self._call_distance_function(
                traj1, traj2, distance_type, large_value
            )

            # Handle DpResult return type
            if hasattr(distance_small, "distance"):
                distance_small = distance_small.distance
            if hasattr(distance_large, "distance"):
                distance_large = distance_large.distance

            assert distance_large <= distance_small, (
                f"{self.algorithm_name.upper()} large eps should produce smaller distance\n"
                f"Small eps distance: {distance_small}\n"
                f"Large eps distance: {distance_large}"
            )
        elif param_name == "g":
            # For ERP, effect of g parameter
            distance_small = self._call_distance_function(
                traj1, traj2, distance_type, small_value
            )
            distance_large = self._call_distance_function(
                traj1, traj2, distance_type, large_value
            )

            # Handle DpResult return type
            if hasattr(distance_small, "distance"):
                distance_small = distance_small.distance
            if hasattr(distance_large, "distance"):
                distance_large = distance_large.distance

            # ERP distance is affected by g parameter but not necessarily monotonic
            # Here only verify distance is reasonable
            assert distance_small >= 0 and distance_large >= 0, (
                f"{self.algorithm_name.upper()} distance should be non-negative\n"
                f"Small g distance: {distance_small}\n"
                f"Large g distance: {distance_large}"
            )
        elif param_name == "precision":
            # For SOWD, higher precision means higher accuracy, distance may be more accurate
            distance_small = self._call_distance_function(
                traj1, traj2, distance_type, small_value
            )
            distance_large = self._call_distance_function(
                traj1, traj2, distance_type, large_value
            )

            # SOWD distance is affected by precision parameter
            assert distance_small >= 0 and distance_large >= 0, (
                f"{self.algorithm_name.upper()} distance should be non-negative\n"
                f"Small precision distance: {distance_small}\n"
                f"Large precision distance: {distance_large}"
            )

    def _check_negative_hyperparameter(
        self, distance_type: str, negative_value: Union[float, int]
    ):
        """
        Check negative hyperparameter value

        Args:
            distance_type: Distance type
            negative_value: Negative hyperparameter value
        """
        param_name = self.config["hyperparameter_name"]

        if param_name == "eps":
            # For LCSS and EDR, eps < 0 should not match any points (since distance is always >= 0)
            traj1 = [[0.0, 0.0], [1.0, 1.0]]
            traj2 = [[0.0, 0.0], [1.0, 1.0]]

            distance = self._call_distance_function(
                traj1, traj2, distance_type, negative_value
            )
            assert (
                distance == 1.0
            ), f"{self.algorithm_name.upper()} negative eps should not match any points, distance should be 1.0, actual: {distance}"


def load_test_data_by_metainfo(metainfo: Metainfo, data_dir):
    """
    Load test data based on metadata object

    Args:
        metainfo: Metainfo object
        data_dir: Data directory

    Returns:
        Test data DataFrame
    """
    # This function now receives a Metainfo object
    sample_path = get_sample_path(metainfo, data_dir)

    try:
        import polars as pl
        import pyarrow.parquet as pq

        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"Failed to read test data {sample_path}: {e}")


def get_hyperparameter_value_from_metainfo(metainfo: Metainfo):
    """
    Extract hyperparameter value from metadata object

    Args:
        metainfo: Metainfo object

    Returns:
        Hyperparameter value
    """
    # Use attribute access
    if metainfo.eps is not None:
        return metainfo.eps
    elif metainfo.g is not None:
        return metainfo.g
    elif metainfo.precision is not None:
        return metainfo.precision
    return None


class HyperparameterUtil:
    """
    Hyperparameter utility class for getting hyperparameter values from metadata (updated)
    """

    @staticmethod
    def get_hyperparameter_value(
        algorithm_name: str, distance_type: str, metainfo_list: List[Metainfo]
    ) -> Optional[Union[float, List[float], int]]:
        """
        Get hyperparameter value for algorithm with specific distance type from metadata list

        Args:
            algorithm_name: Algorithm name
            distance_type: Distance type
            metainfo_list: List of Metainfo objects

        Returns:
            Hyperparameter value, returns None if algorithm has no hyperparameters
        """
        config = get_algorithm_config(algorithm_name)
        if not config["has_hyperparameters"]:
            return None

        # Find metadata matching algorithm name and distance type
        for metainfo in metainfo_list:
            if (
                metainfo.algorithm == algorithm_name
                and metainfo.type_d.value == distance_type
            ):
                return get_hyperparameter_value_from_metainfo(metainfo)

        # If no exact match found, return hyperparameter value from first metadata matching algorithm name
        for metainfo in metainfo_list:
            if metainfo.algorithm == algorithm_name:
                hyperparam_value = get_hyperparameter_value_from_metainfo(metainfo)
                if hyperparam_value is not None:
                    return hyperparam_value

        # If still not found, return default value from configuration
        param_name = config["hyperparameter_name"]
        if param_name == "eps":
            return 1e-6
        elif param_name == "g":
            return [0.0, 0.0]
        elif param_name == "precision":
            return 5
        return None
