#!/usr/bin/env python3
"""
traj-dist performance benchmark script

Supports both Cython and Python implementations, supports command-line arguments to specify warmup runs and test runs.
Saves results as parquet files with algorithm, distance type, hyperparameters, and other metadata.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description="traj-dist performance benchmark script"
    )
    parser.add_argument(
        "--baseline-file",
        type=str,
        default="output/baseline_trajectories.parquet",
        help="Baseline trajectory file path",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="algorithms_config.json",
        help="Algorithm configuration file path",
    )
    parser.add_argument(
        "--implementation",
        type=str,
        required=True,
        choices=["cython", "python"],
        help="Implementation type: cython or python",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup runs, default 5",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of test runs, default 10",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load algorithm configuration"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_baseline_trajectories(baseline_path: Path) -> pl.DataFrame:
    """Load baseline trajectory data"""
    print(f"Loading baseline trajectory data: {baseline_path}")
    df = pl.read_parquet(baseline_path)
    print(f"Loaded {len(df)} trajectory pairs")
    return df


def get_traj_dist_functions(implementation: str):
    """Get traj-dist distance functions"""
    if implementation == "cython":
        import traj_dist.distance as tdist

        functions = {
            "sspd": {
                "euclidean": tdist.c_e_sspd,
                "spherical": tdist.c_g_sspd,
            },
            "dtw": {
                "euclidean": tdist.c_e_dtw,
                "spherical": tdist.c_g_dtw,
            },
            "hausdorff": {
                "euclidean": tdist.c_e_hausdorff,
                "spherical": tdist.c_g_hausdorff,
            },
            "discret_frechet": {
                "euclidean": tdist.c_discret_frechet,
            },
            "lcss": {
                "euclidean": tdist.c_e_lcss,
                "spherical": tdist.c_g_lcss,
            },
            "edr": {
                "euclidean": tdist.c_e_edr,
                "spherical": tdist.c_g_edr,
            },
            "erp": {
                "euclidean": tdist.c_e_erp,
                "spherical": tdist.c_g_erp,
            },
        }
    else:  # python
        import traj_dist.pydist.discret_frechet as py_discret_frechet
        import traj_dist.pydist.dtw as py_dtw
        import traj_dist.pydist.edr as py_edr
        import traj_dist.pydist.erp as py_erp
        import traj_dist.pydist.hausdorff as py_hausdorff
        import traj_dist.pydist.lcss as py_lcss
        import traj_dist.pydist.sspd as py_sspd

        functions = {
            "sspd": {
                "euclidean": py_sspd.e_sspd,
                "spherical": py_sspd.s_sspd,
            },
            "dtw": {
                "euclidean": py_dtw.e_dtw,
                "spherical": py_dtw.s_dtw,
            },
            "hausdorff": {
                "euclidean": py_hausdorff.e_hausdorff,
                "spherical": py_hausdorff.s_hausdorff,
            },
            "discret_frechet": {
                "euclidean": py_discret_frechet.discret_frechet,
            },
            "lcss": {
                "euclidean": py_lcss.e_lcss,
                "spherical": py_lcss.s_lcss,
            },
            "edr": {
                "euclidean": py_edr.e_edr,
                "spherical": py_edr.s_edr,
            },
            "erp": {
                "euclidean": py_erp.e_erp,
                "spherical": py_erp.s_erp,
            },
        }
    return functions


def benchmark_algorithm(
    algorithm_config: Dict[str, Any],
    distance_type: str,
    baseline_df: pl.DataFrame,
    functions: Dict[str, Dict[str, Any]],
    warmup_runs: int,
    num_runs: int,
    implementation: str,
) -> List[Dict[str, Any]]:
    """
    Benchmark a single algorithm with specific configuration

    Returns:
        List of test results, each containing algorithm name, distance type, hyperparameters, time list, etc.
    """
    algorithm_name = algorithm_config["name"]
    has_hyperparameters = algorithm_config["has_hyperparameters"]

    # Get distance function
    if algorithm_name not in functions:
        print(
            f"Warning: {algorithm_name} is not available in {implementation} implementation, skipping"
        )
        return []

    if distance_type not in functions[algorithm_name]:
        print(
            f"Warning: {algorithm_name} does not support {distance_type} distance, skipping"
        )
        return []

    dist_func = functions[algorithm_name][distance_type]

    # Build call parameters
    call_params = {}
    hyperparameter_value = None

    if has_hyperparameters:
        hyperparam = algorithm_config["hyperparameter"]
        hyperparam_name = hyperparam["name"]
        hyperparam_type = hyperparam["type"]
        hyperparameter_value = hyperparam["values"][
            0
        ]  # Only take the first hyperparameter value

        if hyperparam_type == "list":
            call_params[hyperparam_name] = np.array(
                hyperparameter_value, dtype=np.float64
            )
        else:
            call_params[hyperparam_name] = hyperparameter_value

    results = []

    print(f"\nTesting: {algorithm_name} ({distance_type}) - {implementation}")
    if hyperparameter_value is not None:
        print(f"  Hyperparameter: {hyperparam_name} = {hyperparameter_value}")

    for row_idx in range(len(baseline_df)):
        row = baseline_df[row_idx]
        traj1 = row["traj1"].item().to_numpy()
        traj2 = row["traj2"].item().to_numpy()

        # Warmup
        for _ in range(warmup_runs):
            dist_func(traj1, traj2, **call_params)

        # Measure time
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            dist_func(traj1, traj2, **call_params)
            end = time.perf_counter()
            times.append(end - start)

        # Store results
        result_dict = {
            "algorithm": algorithm_name,
            "distance_type": distance_type,
            "implementation": implementation,
            "traj_index": row_idx,
        }

        if has_hyperparameters:
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
            schema_fields.append(
                (col, pa.string())
            )  # Hyperparameters stored as strings
        elif col == "algorithm":
            schema_fields.append((col, pa.string()))
        elif col == "distance_type":
            schema_fields.append((col, pa.string()))
        elif col == "implementation":
            schema_fields.append((col, pa.string()))
        elif col == "traj_index":
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

    # Load baseline trajectories
    baseline_path = Path(args.baseline_file)
    if not baseline_path.exists():
        baseline_path = output_dir / args.baseline_file

    baseline_df = load_baseline_trajectories(baseline_path)

    # Get distance functions
    functions = get_traj_dist_functions(args.implementation)

    # Run all algorithm tests
    all_results = []

    for algorithm_config in config["algorithms"]:
        algorithm_config["name"]

        for distance_type in algorithm_config["distance_types"]:
            results = benchmark_algorithm(
                algorithm_config,
                distance_type,
                baseline_df,
                functions,
                args.warmup_runs,
                args.num_runs,
                args.implementation,
            )
            all_results.extend(results)

    # Save results
    output_file = f"traj_dist_{args.implementation}_benchmark.parquet"
    output_path = output_dir / output_file
    save_results(all_results, output_path)

    print(f"\nTesting complete! Tested {len(all_results)} test cases")


if __name__ == "__main__":
    main()
