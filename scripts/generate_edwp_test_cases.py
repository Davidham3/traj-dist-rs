#!/usr/bin/env python3
"""
Generate test cases for EDwP algorithm (Python reference implementation)

Uses the EDwP implementation from scripts/benchmark/extra_algos/edwp.py
as the reference implementation for correctness validation.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Import EDwP reference implementation
sys.path.append(str(Path(__file__).parent / "benchmark" / "extra_algos"))
from edwp import edwp

sys.path.append(str(Path(__file__).parent.parent))
from py_tests.schemas import ImplementationType, Metainfo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate test cases for EDwP algorithm (Python reference implementation)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="py_tests/data", help="Output directory"
    )
    parser.add_argument(
        "--traj-data",
        type=str,
        default="../traj-dist/data/benchmark_trajectories.pkl",
        help="Trajectory data file",
    )
    parser.add_argument(
        "--num-traj", type=int, default=50, help="Number of trajectories to use"
    )
    return parser.parse_args()


def build_sample_filename():
    """
    Build sample file name for EDwP

    Format: edwp_euclidean.parquet
    """
    return "edwp_euclidean.parquet"


def build_metainfo_filename():
    """
    Build metadata file name

    Format: edwp.jsonl
    """
    return "edwp.jsonl"


def generate_test_cases(args):
    """
    Generate EDwP test cases using Python reference implementation
    """
    # Load trajectory data
    traj_data_path = Path(args.traj_data)
    if not traj_data_path.exists():
        # Try relative path
        traj_data_path = (
            Path(__file__).parent.parent.parent
            / "traj-dist"
            / "data"
            / "benchmark_trajectories.pkl"
        )

    print(f"Loading trajectory data from: {traj_data_path}")
    with open(traj_data_path, "rb") as f:
        traj_list = pickle.load(f, encoding="latin1")

    # Use specified number of trajectories
    traj_list = traj_list[: args.num_traj]
    print(f"Using {len(traj_list)} trajectories")

    # Build output directory
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "cython_samples"
    metainfo_dir = output_dir / "metainfo"
    samples_dir.mkdir(parents=True, exist_ok=True)
    metainfo_dir.mkdir(parents=True, exist_ok=True)

    # Build sample file name and path
    sample_filename = build_sample_filename()
    sample_path = samples_dir / sample_filename

    print("Generating test cases for: edwp (using Python reference implementation)")
    print(f"Sample file: {sample_path}")

    # Generate test cases
    results = []
    num_measurements = 10  # Number of measurements

    for i, traj1_orig in enumerate(traj_list):
        for j, traj2_orig in enumerate(traj_list):
            if i >= j:  # Avoid duplicate calculations
                continue

            traj1 = traj1_orig
            traj2 = traj2_orig

            # Warmup - run a few times to warm up the cache
            for _ in range(5):
                edwp(traj1, traj2)

            # Measure computation time - multiple measurements for statistics
            times = []
            for _ in range(num_measurements):
                start = time.perf_counter()
                result = edwp(traj1, traj2)
                end = time.perf_counter()
                times.append(end - start)

            # Calculate statistical metrics
            times_array = np.array(times)
            time_stats = {
                "mean": float(np.mean(times_array)),
                "std": float(np.std(times_array)),
                "median": float(np.median(times_array)),
                "min": float(np.min(times_array)),
                "max": float(np.max(times_array)),
            }

            results.append(
                (
                    traj1.tolist(),
                    traj2.tolist(),
                    result,
                    times,  # Store all measurement times
                    time_stats,  # Store statistical information
                )
            )

    traj1, traj2, distance, times_list, time_stats_list = zip(*results)

    traj_type = pa.large_list(pa.list_(pa.float64(), 2))
    time_list_type = pa.list_(pa.float64())

    # Convert statistics dictionary to struct type
    time_stats_fields = [
        ("mean", pa.float64()),
        ("std", pa.float64()),
        ("median", pa.float64()),
        ("min", pa.float64()),
        ("max", pa.float64()),
    ]
    time_stats_type = pa.struct(time_stats_fields)

    table = pa.table(
        {
            "traj1": traj1,
            "traj2": traj2,
            "distance": distance,
            "times": times_list,
            "time_stats": time_stats_list,
        },
        schema=pa.schema(
            [
                ("traj1", traj_type),
                ("traj2", traj_type),
                ("distance", pa.float64()),
                ("times", time_list_type),
                ("time_stats", time_stats_type),
            ]
        ),
    )

    pq.write_table(table, sample_path)
    print(f"Generated {len(results)} test cases")
    print(f"Saved to: {sample_path}")

    # Build metadata
    impl_type = ImplementationType.PYTHON  # Using Python reference implementation
    try:
        metainfo = Metainfo(
            algorithm="edwp",
            type_d="euclidean",
            implemented_by=impl_type,
            eps=None,
            g=None,
            precision=None,
            converted=None,
            # Path relative to metainfo_dir's parent directory
            sample_file=f"cython_samples/{sample_filename}",
        )
    except Exception as e:
        print(f"Error creating Pydantic model: {e}")
        return

    # Build metadata file name and path
    metainfo_filename = build_metainfo_filename()
    metainfo_path = metainfo_dir / metainfo_filename

    # Serialize using model's .json() method and append to metadata file
    with open(metainfo_path, "a") as f:
        # metainfo.json() will generate a compact JSON string
        f.write(metainfo.model_dump_json() + "\n")

    print(f"Metadata appended to: {metainfo_path}")


def main():
    args = parse_args()

    # Generate test cases for EDwP
    print("==========================================")
    print("Generating test cases for EDwP (Python reference implementation)")
    print("==========================================")
    generate_test_cases(args)


if __name__ == "__main__":
    main()
