#!/usr/bin/env python3
"""
Generate test cases for a single algorithm (Cython implementation only)

Use argparse to specify algorithm name and hyperparameter values, generate Parquet files and metadata JSONL files.
Parquet file names include algorithm name, distance type, and hyperparameter values.
Files contain 5 columns: 2 columns of trajectory data, 1 column of distance values, 1 column of measurement time lists, 1 column of statistics.

Note: Correctness validation only uses Cython implementation as "correct answer", Python implementation performance testing is handled by benchmark scripts.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
# Import Cython implementation
import traj_dist.distance as tdist

sys.path.append(str(Path(__file__).parent.parent))
from py_tests.schemas import ImplementationType, Metainfo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate test cases for a single algorithm (Cython implementation only)"
    )
    parser.add_argument("--algorithm", type=str, required=True, help="Algorithm name")
    parser.add_argument(
        "--type_d",
        type=str,
        default="euclidean",
        choices=["euclidean", "spherical"],
        help="Distance type",
    )
    parser.add_argument(
        "--eps", type=float, default=None, help="LCSS and EDR algorithm eps parameter"
    )
    parser.add_argument(
        "--g",
        type=float,
        nargs=2,
        default=None,
        help="ERP algorithm g parameter (two values)",
    )
    parser.add_argument(
        "--precision", type=int, default=None, help="SOWD algorithm precision parameter"
    )
    parser.add_argument(
        "--converted",
        type=bool,
        default=None,
        help="SOWD algorithm converted parameter",
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


def build_sample_filename(algorithm, type_d, eps=None, g=None, precision=None):
    """
    Build sample file name, including algorithm name, distance type, and hyperparameters

    Format: {algorithm}_{type_d}_{param}_{value}.parquet
    Example: edr_euclidean_eps_0.01.parquet
    """
    parts = [algorithm, type_d]
    if eps is not None:
        parts.append(f"eps_{eps}")
    if g is not None:
        # For g parameter, use specific format
        parts.append(f"g_{g[0]}_{g[1]}")
    if precision is not None:
        parts.append(f"precision_{precision}")
    return "_".join(parts) + ".parquet"


def build_metainfo_filename(algorithm):
    """
    Build metadata file name

    Format: {algorithm}.jsonl
    Example: edr.jsonl
    """
    return f"{algorithm}.jsonl"


def get_distance_function(args):
    """
    Return the corresponding distance function based on Cython implementation algorithm and distance type
    """
    if args.algorithm == "sspd":
        return tdist.c_e_sspd if args.type_d == "euclidean" else tdist.c_g_sspd
    elif args.algorithm == "dtw":
        return tdist.c_e_dtw if args.type_d == "euclidean" else tdist.c_g_dtw
    elif args.algorithm == "hausdorff":
        return (
            tdist.c_e_hausdorff if args.type_d == "euclidean" else tdist.c_g_hausdorff
        )
    elif args.algorithm == "discret_frechet":
        return (
            tdist.c_discret_frechet
        )  # discret_frechet only supports Euclidean distance
    elif args.algorithm == "lcss":
        return tdist.c_e_lcss if args.type_d == "euclidean" else tdist.c_g_lcss
    elif args.algorithm == "edr":
        return tdist.c_e_edr if args.type_d == "euclidean" else tdist.c_g_edr
    elif args.algorithm == "erp":
        return tdist.c_e_erp if args.type_d == "euclidean" else tdist.c_g_erp
    elif args.algorithm == "sowd_grid":
        return tdist.c_sowd_grid  # SOWD only supports spherical/geographical
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")


def generate_test_cases(args):
    """
    Generate Cython implementation test cases
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

    # Build parameter dictionary
    params = {"type_d": args.type_d}
    if args.eps is not None:
        params["eps"] = args.eps
    if args.g is not None:
        params["g"] = np.array(args.g, dtype=np.float64)
    if args.precision is not None:
        params["precision"] = args.precision
    if args.converted is not None:
        params["converted"] = args.converted

    # Build output directory
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "cython_samples"
    metainfo_dir = output_dir / "metainfo"
    samples_dir.mkdir(parents=True, exist_ok=True)
    metainfo_dir.mkdir(parents=True, exist_ok=True)

    # Build sample file name and path
    sample_filename = build_sample_filename(
        args.algorithm, args.type_d, args.eps, args.g, args.precision
    )
    sample_path = samples_dir / sample_filename

    print(f"Generating test cases for: {args.algorithm} (using Cython implementation)")
    print(f"Parameters: {params}")
    print(f"Sample file: {sample_path}")

    # Get the corresponding distance function
    dist_func = get_distance_function(args)

    # Generate test cases
    results = []
    num_measurements = (
        10  # Number of measurements: 10 is usually sufficient for reliable statistics
    )

    for i, traj1_orig in enumerate(traj_list):
        for j, traj2_orig in enumerate(traj_list):
            if i >= j:  # Avoid duplicate calculations
                continue

            traj1 = traj1_orig
            traj2 = traj2_orig

            # SOWD algorithm requires special handling: convert to cell format
            call_params = {}
            if args.algorithm == "sowd_grid":
                from traj_dist.pydist.linecell import trajectory_set_grid

                precision = params.get("precision", 7)
                cells_list, _, _, _, _ = trajectory_set_grid([traj1, traj2], precision)
                traj1 = np.array([[c[0], c[1]] for c in cells_list[0]], dtype=np.int64)
                traj2 = np.array([[c[0], c[1]] for c in cells_list[1]], dtype=np.int64)
                # sowd_grid function does not accept extra parameters

            # Add hyperparameters
            if args.eps is not None:
                call_params["eps"] = args.eps
            if args.g is not None:
                call_params["g"] = np.array(args.g, dtype=np.float64)
            # SOWD algorithm's precision parameter is already used when converting to cell format, no need to pass to sowd_grid function

            # Warmup - run a few times to warm up the cache
            for _ in range(5):
                dist_func(traj1, traj2, **call_params)

            # Measure computation time - multiple measurements for statistics
            times = []
            for _ in range(num_measurements):
                start = time.perf_counter()
                result = dist_func(traj1, traj2, **call_params)
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
    impl_type = ImplementationType.CYTHON
    try:
        metainfo = Metainfo(
            algorithm=args.algorithm,
            type_d=args.type_d,
            implemented_by=impl_type,
            eps=args.eps,
            g=args.g,  # Pydantic will automatically handle list
            precision=args.precision,
            converted=args.converted,
            # Path relative to metainfo_dir's parent directory
            sample_file=f"cython_samples/{sample_filename}",
        )
    except Exception as e:
        print(f"Error creating Pydantic model: {e}")
        return

    # Build metadata file name and path
    metainfo_filename = build_metainfo_filename(args.algorithm)
    metainfo_path = metainfo_dir / metainfo_filename

    # Serialize using model's .json() method and append to metadata file
    with open(metainfo_path, "a") as f:
        # metainfo.json() will generate a compact JSON string
        f.write(metainfo.model_dump_json() + "\n")

    print(f"Metadata appended to: {metainfo_path}")


def main():
    args = parse_args()

    # Generate test cases for Cython implementation
    print("==========================================")
    print("Generating test cases for Cython implementation (correctness validation)")
    print("==========================================")
    generate_test_cases(args)


if __name__ == "__main__":
    main()
