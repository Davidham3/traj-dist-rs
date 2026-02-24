#!/usr/bin/env python3
"""
traj-dist-rs batch computation (pdist/cdist) performance benchmark script

Tests the performance of pdist and cdist functions in traj-dist-rs (Rust implementation).
Uses randomly generated trajectories to test performance at different scales:
- pdist: [10, 100, 1000] trajectories
- cdist: [10x10, 100x100, 1000x1000] trajectory pairs
Tests both parallel and sequential execution.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import traj_dist_rs


def parse_args():
    parser = argparse.ArgumentParser(
        description="traj-dist-rs batch computation performance benchmark script"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="algorithms_config.json",
        help="Algorithm configuration file path",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warmup runs, default 3",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of test runs, default 5",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for trajectory generation, default 42",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load algorithm configuration"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def generate_random_trajectory(num_points: int, seed: int) -> np.ndarray:
    """Generate a random trajectory with specified number of points"""
    rng = np.random.RandomState(seed)
    # Generate random 2D points in range [0, 100]
    trajectory = rng.uniform(0, 100, size=(num_points, 2))
    return trajectory


def generate_random_trajectories(
    num_trajectories: int, trajectory_length: int, seed: int
) -> List[np.ndarray]:
    """Generate a list of random trajectories with specified length"""
    trajectories = []
    for i in range(num_trajectories):
        traj = generate_random_trajectory(trajectory_length, seed + i)
        trajectories.append(traj)
    return trajectories


def create_metric(
    algorithm_name: str, algorithm_config: Dict[str, Any], distance_type: str
):
    """Create a Metric object for the specified algorithm"""
    has_hyperparameters = algorithm_config["has_hyperparameters"]

    if has_hyperparameters:
        hyperparam = algorithm_config["hyperparameter"]
        hyperparam["name"]
        hyperparam["type"]
        hyperparameter_value = hyperparam["values"][0]

        if algorithm_name == "lcss":
            return traj_dist_rs.Metric.lcss(
                eps=hyperparameter_value, type_d=distance_type
            )
        elif algorithm_name == "edr":
            return traj_dist_rs.Metric.edr(
                eps=hyperparameter_value, type_d=distance_type
            )
        elif algorithm_name == "erp":
            g = list(hyperparameter_value)
            return traj_dist_rs.Metric.erp(g=g, type_d=distance_type)
        else:
            raise ValueError(
                f"Unknown algorithm with hyperparameters: {algorithm_name}"
            )
    else:
        # Algorithms without hyperparameters
        if algorithm_name == "sspd":
            return traj_dist_rs.Metric.sspd(type_d=distance_type)
        elif algorithm_name == "dtw":
            return traj_dist_rs.Metric.dtw(type_d=distance_type)
        elif algorithm_name == "hausdorff":
            return traj_dist_rs.Metric.hausdorff(type_d=distance_type)
        elif algorithm_name == "discret_frechet":
            return traj_dist_rs.Metric.discret_frechet(type_d=distance_type)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")


def benchmark_pdist(
    algorithm_config: Dict[str, Any],
    distance_type: str,
    trajectories: List[np.ndarray],
    trajectory_length: int,
    warmup_runs: int,
    num_runs: int,
    parallel: bool,
) -> List[Dict[str, Any]]:
    """
    Benchmark pdist function with specified trajectory length

    Returns:
        List of test results
    """
    algorithm_name = algorithm_config["name"]

    # Check if algorithm supports the distance type
    if distance_type not in algorithm_config["distance_types"]:
        return []

    # Use all provided trajectories (fixed at 5)
    if len(trajectories) < 2:
        return []

    # Create metric
    try:
        metric = create_metric(algorithm_name, algorithm_config, distance_type)
    except ValueError as e:
        print(f"Warning: {e}, skipping")
        return []

    results = []
    num_trajectories = len(trajectories)
    expected_distances = num_trajectories * (num_trajectories - 1) // 2

    parallel_str = "parallel" if parallel else "sequential"
    print(
        f"\nTesting pdist: {algorithm_name} ({distance_type}) - {num_trajectories} trajectories, length={trajectory_length} ({parallel_str})"
    )

    # Warmup
    for _ in range(warmup_runs):
        _ = traj_dist_rs.pdist(trajectories, metric=metric, parallel=parallel)

    # Measure time
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = traj_dist_rs.pdist(trajectories, metric=metric, parallel=parallel)
        end = time.perf_counter()
        times.append(end - start)

        # Verify result shape
        if len(result) != expected_distances:
            print(
                f"  Warning: Expected {expected_distances} distances, got {len(result)}"
            )

    # Store results
    result_dict = {
        "algorithm": algorithm_name,
        "distance_type": distance_type,
        "implementation": "rust",
        "function": "pdist",
        "parallel": parallel,
        "num_trajectories": num_trajectories,
        "trajectory_length": trajectory_length,
        "num_distances": expected_distances,
    }

    if algorithm_config["has_hyperparameters"]:
        hyperparam = algorithm_config["hyperparameter"]
        hyperparam_name = hyperparam["name"]
        hyperparam_type = hyperparam["type"]
        hyperparameter_value = hyperparam["values"][0]
        if hyperparam_type == "list":
            result_dict[f"hyperparam_{hyperparam_name}"] = json.dumps(
                hyperparameter_value
            )
        else:
            result_dict[f"hyperparam_{hyperparam_name}"] = str(hyperparameter_value)

    result_dict["times"] = times

    results.append(result_dict)

    return results


def benchmark_cdist(
    algorithm_config: Dict[str, Any],
    distance_type: str,
    trajectories_a: List[np.ndarray],
    trajectories_b: List[np.ndarray],
    trajectory_length: int,
    warmup_runs: int,
    num_runs: int,
    parallel: bool,
) -> List[Dict[str, Any]]:
    """
    Benchmark cdist function with specified trajectory length

    Returns:
        List of test results
    """
    algorithm_name = algorithm_config["name"]

    # Check if algorithm supports the distance type
    if distance_type not in algorithm_config["distance_types"]:
        return []

    if len(trajectories_a) == 0 or len(trajectories_b) == 0:
        return []

    # Create metric
    try:
        metric = create_metric(algorithm_name, algorithm_config, distance_type)
    except ValueError as e:
        print(f"Warning: {e}, skipping")
        return []

    results = []
    num_trajectories_a = len(trajectories_a)
    num_trajectories_b = len(trajectories_b)
    expected_distances = num_trajectories_a * num_trajectories_b

    parallel_str = "parallel" if parallel else "sequential"
    print(
        f"\nTesting cdist: {algorithm_name} ({distance_type}) - {num_trajectories_a}x{num_trajectories_b}, length={trajectory_length} ({parallel_str})"
    )

    # Warmup
    for _ in range(warmup_runs):
        _ = traj_dist_rs.cdist(
            trajectories_a, trajectories_b, metric=metric, parallel=parallel
        )

    # Measure time
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = traj_dist_rs.cdist(
            trajectories_a, trajectories_b, metric=metric, parallel=parallel
        )
        end = time.perf_counter()
        times.append(end - start)

        # Verify result shape
        if result.shape != (num_trajectories_a, num_trajectories_b):
            print(
                f"  Warning: Expected shape ({num_trajectories_a}, {num_trajectories_b}), got {result.shape}"
            )

    # Store results
    result_dict = {
        "algorithm": algorithm_name,
        "distance_type": distance_type,
        "implementation": "rust",
        "function": "cdist",
        "parallel": parallel,
        "num_trajectories_a": num_trajectories_a,
        "num_trajectories_b": num_trajectories_b,
        "trajectory_length": trajectory_length,
        "num_distances": expected_distances,
    }

    if algorithm_config["has_hyperparameters"]:
        hyperparam = algorithm_config["hyperparameter"]
        hyperparam_name = hyperparam["name"]
        hyperparam_type = hyperparam["type"]
        hyperparameter_value = hyperparam["values"][0]
        if hyperparam_type == "list":
            result_dict[f"hyperparam_{hyperparam_name}"] = json.dumps(
                hyperparameter_value
            )
        else:
            result_dict[f"hyperparam_{hyperparam_name}"] = str(hyperparameter_value)

    result_dict["times"] = times

    results.append(result_dict)

    return results


def save_results(results: List[Dict[str, Any]], output_path: Path):
    """Save results as parquet file"""
    # Prepare data
    data = {}

    # Extract all possible column names
    columns = set()
    for result in results:
        columns.update(result.keys())
        columns.discard("times")

    # Add times column
    columns.add("times")

    # Organize data by column
    for col in columns:
        if col == "times":
            data[col] = [r.get(col, []) for r in results]
        else:
            data[col] = [r.get(col, None) for r in results]

    # Define schema
    schema_fields = []
    for col in sorted(columns):
        if col == "times":
            schema_fields.append((col, pa.list_(pa.float64())))
        elif col.startswith("hyperparam_"):
            schema_fields.append((col, pa.string()))
        elif col in ["algorithm", "distance_type", "implementation", "function"]:
            schema_fields.append((col, pa.string()))
        elif col == "parallel":
            schema_fields.append((col, pa.bool_()))
        elif col in [
            "num_trajectories",
            "num_trajectories_a",
            "num_trajectories_b",
            "num_distances",
            "trajectory_length",
        ]:
            schema_fields.append((col, pa.int64()))
        else:
            schema_fields.append((col, pa.float64()))

    schema = pa.schema(schema_fields)

    # Create table
    table = pa.table(data, schema=schema)

    # Save
    pq.write_table(table, output_path)
    print(f"Saved {len(results)} results to: {output_path}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config_path = Path(args.config_file)
    if not config_path.exists():
        config_path = Path(__file__).parent / args.config_file

    config = load_config(config_path)

    # Number of trajectories (fixed at 5)
    num_trajectories = 5

    # Test scales for trajectory length [10, 100, 1000]
    trajectory_lengths = [10, 100, 1000]

    # Run all algorithm tests
    all_results = []

    # Only test DTW algorithm
    algorithm_name = "dtw"

    for algorithm_config in config["algorithms"]:
        if algorithm_config["name"] != algorithm_name:
            continue

        for distance_type in algorithm_config["distance_types"]:
            # Test pdist at different trajectory lengths (both parallel and sequential)
            for trajectory_length in trajectory_lengths:
                # Generate trajectories with specified length
                np.random.seed(args.seed)
                trajectories = generate_random_trajectories(
                    num_trajectories, trajectory_length, args.seed
                )

                for parallel in [False, True]:
                    results = benchmark_pdist(
                        algorithm_config,
                        distance_type,
                        trajectories,
                        trajectory_length,
                        args.warmup_runs,
                        args.num_runs,
                        parallel,
                    )
                    all_results.extend(results)

            # Test cdist at different trajectory lengths (both parallel and sequential)
            for trajectory_length in trajectory_lengths:
                # Generate trajectories with specified length
                np.random.seed(args.seed)
                trajectories_a = generate_random_trajectories(
                    num_trajectories, trajectory_length, args.seed
                )
                trajectories_b = generate_random_trajectories(
                    num_trajectories, trajectory_length, args.seed + num_trajectories
                )

                for parallel in [False, True]:
                    results = benchmark_cdist(
                        algorithm_config,
                        distance_type,
                        trajectories_a,
                        trajectories_b,
                        trajectory_length,
                        args.warmup_runs,
                        args.num_runs,
                        parallel,
                    )
                    all_results.extend(results)

    # Save results
    output_file = "traj_dist_rs_rust_batch_benchmark.parquet"
    output_path = output_dir / output_file
    save_results(all_results, output_path)

    print(f"\nTesting complete! Tested {len(all_results)} test cases")


if __name__ == "__main__":
    main()
