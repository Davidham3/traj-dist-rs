#!/usr/bin/env python3
"""
Baseline trajectory generation script

Randomly sample k trajectory pairs from traj-dist/data/benchmark_trajectories.pkl,
store in parquet format, using pa.large_list(pa.list_(pa.float64(), 2)) type.
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baseline trajectory data from pkl file"
    )
    parser.add_argument(
        "--pkl-file",
        type=str,
        default="../traj-dist/data/benchmark_trajectories.pkl",
        help="Trajectory data pkl file path",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Number of trajectory pairs to sample (number of trajectories ≈ sqrt(2*k))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to ensure reproducible results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="baseline_trajectories.parquet",
        help="Output file name",
    )
    return parser.parse_args()


def load_trajectories(pkl_path: Path) -> list:
    """Load trajectory data"""
    print(f"Loading trajectory data: {pkl_path}")
    with open(pkl_path, "rb") as f:
        traj_list = pickle.load(f, encoding="latin1")
    print(f"Loaded {len(traj_list)} trajectories")
    return traj_list


def generate_trajectory_pairs(traj_list: list, num_pairs: int, seed: int) -> list:
    """
    Generate trajectory pairs

    Args:
        traj_list: List of trajectories
        num_pairs: Number of trajectory pairs to generate
        seed: Random seed

    Returns:
        List of trajectory pairs [(traj1, traj2), ...]
    """
    random.seed(seed)
    np.random.seed(seed)

    # Calculate required number of trajectories
    # For n trajectories, we can generate n*(n-1)/2 unique trajectory pairs
    # So if we need k trajectory pairs, we need approximately sqrt(2*k) trajectories
    num_traj = int(np.sqrt(2 * num_pairs)) + 1

    # Ensure we don't exceed available trajectory count
    num_traj = min(num_traj, len(traj_list))

    print(
        f"Randomly selecting {num_traj} trajectories from {len(traj_list)} trajectories"
    )
    selected_indices = random.sample(range(len(traj_list)), num_traj)
    selected_trajectories = [traj_list[i] for i in selected_indices]

    # Generate all possible trajectory pairs (i < j to avoid duplicates)
    pairs = []
    for i in range(num_traj):
        for j in range(i + 1, num_traj):
            pairs.append((selected_trajectories[i], selected_trajectories[j]))

    # If generated pairs exceed required number, randomly sample
    if len(pairs) > num_pairs:
        pairs = random.sample(pairs, num_pairs)

    print(f"Generated {len(pairs)} trajectory pairs")
    return pairs


def save_to_parquet(pairs: list, output_path: Path):
    """Save trajectory pairs as parquet file"""
    traj1_list = []
    traj2_list = []

    for traj1, traj2 in pairs:
        traj1_list.append(traj1.tolist() if isinstance(traj1, np.ndarray) else traj1)
        traj2_list.append(traj2.tolist() if isinstance(traj2, np.ndarray) else traj2)

    # Define schema
    traj_type = pa.large_list(pa.list_(pa.float64(), 2))

    schema = pa.schema(
        [
            ("traj1", traj_type),
            ("traj2", traj_type),
        ]
    )

    # Create table
    table = pa.table(
        {
            "traj1": traj1_list,
            "traj2": traj2_list,
        },
        schema=schema,
    )

    # Save
    pq.write_table(table, output_path)
    print(f"Saved {len(pairs)} trajectory pairs to: {output_path}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original pkl_file parameter (for metadata)
    original_pkl_file = args.pkl_file

    # Load trajectory data
    pkl_path = Path(args.pkl_file)
    # If relative path, resolve relative to script directory
    if not pkl_path.is_absolute():
        pkl_path = (Path(__file__).parent / pkl_path).resolve()

    if not pkl_path.exists():
        # Try default path
        pkl_path = (
            Path(__file__).parent.parent.parent.parent
            / "traj-dist"
            / "data"
            / "benchmark_trajectories.pkl"
        ).resolve()
        original_pkl_file = "../traj-dist/data/benchmark_trajectories.pkl"

    traj_list = load_trajectories(pkl_path)

    # Generate trajectory pairs
    pairs = generate_trajectory_pairs(traj_list, args.k, args.seed)

    # Save as parquet
    output_path = output_dir / args.output_file
    save_to_parquet(pairs, output_path)

    # Save metadata (using relative path)
    metadata = {
        "num_pairs": len(pairs),
        "seed": args.seed,
        "pkl_file": original_pkl_file,
    }
    metadata_path = output_dir / f"{args.output_file}.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
