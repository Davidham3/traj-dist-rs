#!/usr/bin/env python3
"""
Result summary analysis script

Reads three types of parquet files (traj_dist_cython_benchmark.parquet,
traj_dist_python_benchmark.parquet, traj_dist_rs_rust_benchmark.parquet),
summarizes timing details, and provides data analysis results.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl


def parse_args():
    parser = argparse.ArgumentParser(description="Result summary analysis script")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="benchmark_analysis_report.md",
        help="Output report file name",
    )
    parser.add_argument(
        "--use-median",
        action="store_true",
        default=True,
        help="Use median for statistics (default)",
    )
    parser.add_argument(
        "--use-mean",
        action="store_true",
        default=False,
        help="Use mean for statistics",
    )
    return parser.parse_args()


def load_benchmark_results(output_dir: Path) -> Dict[str, pl.DataFrame]:
    """Load all benchmark results"""
    results = {}

    # Try to load results from each implementation
    files = {
        "cython": "traj_dist_cython_benchmark.parquet",
        "python": "traj_dist_python_benchmark.parquet",
        "rust": "traj_dist_rs_rust_benchmark.parquet",
    }

    for impl, filename in files.items():
        filepath = output_dir / filename
        if filepath.exists():
            print(f"Loading {impl} results: {filepath}")
            results[impl] = pl.read_parquet(filepath)
        else:
            print(f"Warning: {impl} result file not found: {filepath}")

    return results


def calculate_statistics(
    times_list: List[float], use_median: bool = True
) -> Dict[str, float]:
    """Calculate statistical metrics"""
    times_array = np.array(times_list)

    if use_median:
        central_value = np.median(times_array)
    else:
        central_value = np.mean(times_array)

    return {
        "mean": float(np.mean(times_array)),
        "median": float(np.median(times_array)),
        "std": float(np.std(times_array)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
        "central": float(central_value),  # Central value for comparison
        "cv": (
            float(np.std(times_array) / np.mean(times_array))
            if np.mean(times_array) > 0
            else 0.0
        ),
    }


def group_and_analyze(
    df: pl.DataFrame, use_median: bool = True
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    Group by algorithm, distance type, and hyperparameters, then analyze

    Returns:
        {(algorithm, distance_type, hyperparam_str): {"times": [...], "stats": {...}}}
    """
    grouped = {}

    # Get hyperparameter columns
    hyperparam_cols = [col for col in df.columns if col.startswith("hyperparam_")]

    if not hyperparam_cols:
        # No hyperparameters
        for row in df.iter_rows(named=True):
            key = (row["algorithm"], row["distance_type"], "")
            if key not in grouped:
                grouped[key] = {"times": []}
            grouped[key]["times"].extend(row["times"])
    else:
        # With hyperparameters
        for row in df.iter_rows(named=True):
            # Build hyperparameter string (use semicolon separator to avoid markdown table conflicts)
            hyperparam_str = "; ".join(
                f"{col.replace('hyperparam_', '')}={row[col]}"
                for col in hyperparam_cols
            )
            key = (row["algorithm"], row["distance_type"], hyperparam_str)
            if key not in grouped:
                grouped[key] = {"times": []}
            grouped[key]["times"].extend(row["times"])

    # Calculate statistics
    for key in grouped:
        grouped[key]["stats"] = calculate_statistics(grouped[key]["times"], use_median)

    return grouped


def compare_implementations(
    results: Dict[str, pl.DataFrame],
    use_median: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compare performance across different implementations

    Returns:
        List of comparison results
    """
    # Get all unique (algorithm, distance_type, hyperparam) combinations
    all_keys = set()

    for impl, df in results.items():
        grouped = group_and_analyze(df, use_median)
        all_keys.update(grouped.keys())

    # Comparison results
    comparison_results = []

    for key in sorted(all_keys):
        algorithm, distance_type, hyperparam_str = key

        result = {
            "algorithm": algorithm,
            "distance_type": distance_type,
            "hyperparam": hyperparam_str if hyperparam_str else "N/A",
            "implementations": {},
        }

        # Get statistics for each implementation
        for impl, df in results.items():
            grouped = group_and_analyze(df, use_median)
            if key in grouped:
                stats = grouped[key]["stats"]
                result["implementations"][impl] = stats

        # Calculate performance improvement
        impl_stats = result["implementations"]

        if "rust" in impl_stats and "cython" in impl_stats:
            rust_time = impl_stats["rust"]["central"]
            cython_time = impl_stats["cython"]["central"]
            if rust_time > 0:
                result["rust_vs_cython"] = cython_time / rust_time

        if "rust" in impl_stats and "python" in impl_stats:
            rust_time = impl_stats["rust"]["central"]
            python_time = impl_stats["python"]["central"]
            if rust_time > 0:
                result["rust_vs_python"] = python_time / rust_time

        if "cython" in impl_stats and "python" in impl_stats:
            cython_time = impl_stats["cython"]["central"]
            python_time = impl_stats["python"]["central"]
            if cython_time > 0:
                result["cython_vs_python"] = python_time / cython_time

        comparison_results.append(result)

    return comparison_results


def analyze_batch_computation(output_dir: Path, use_median: bool = True) -> str:
    """
    Analyze batch computation (pdist/cdist) performance

    Returns:
        Markdown report section for batch computation
    """
    lines = []
    lines.append("## Batch Computation Performance")
    lines.append("")
    lines.append(
        "Performance comparison for batch distance computation (pdist and cdist):"
    )
    lines.append("")

    # Load batch computation results
    batch_files = {
        "cython": "traj_dist_cython_batch_benchmark.parquet",
        "rust": "traj_dist_rs_rust_batch_benchmark.parquet",
    }

    batch_results = {}
    for impl, filename in batch_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            print(f"Loading {impl} batch results: {filepath}")
            batch_results[impl] = pl.read_parquet(filepath)
        else:
            print(f"Warning: {impl} batch result file not found: {filepath}")

    if not batch_results:
        return ""

    # Add test configuration section
    lines.append("### Test Configuration")
    lines.append("")

    # Get test configuration from first result
    first_impl = list(batch_results.keys())[0]
    first_df = batch_results[first_impl]
    if first_df.height > 0:
        first_row = first_df.row(0, named=True)
        lines.append(f"- **Algorithm**: {first_row['algorithm']}")

        # Get number of trajectories (should be fixed at 5)
        num_traj = first_row["num_trajectories"]
        if num_traj is not None:
            lines.append(f"- **Number of trajectories**: {num_traj} (fixed)")

            # Calculate number of distances
            if first_row["function"] == "pdist":
                num_distances = num_traj * (num_traj - 1) // 2
                lines.append(
                    f"- **pdist computation**: {num_distances} distances ({num_traj}×{num_traj-1}/2)"
                )
            else:
                num_distances = num_traj * num_traj
                lines.append(
                    f"- **cdist computation**: {num_distances} distances ({num_traj}×{num_traj})"
                )

        # Get trajectory lengths
        traj_lengths = sorted(first_df["trajectory_length"].unique().to_list())
        lines.append(
            f"- **Trajectory lengths tested**: {', '.join(map(str, traj_lengths))} points"
        )

        # Get distance types
        distance_types = sorted(first_df["distance_type"].unique().to_list())
        lines.append(f"- **Distance types**: {', '.join(distance_types)}")

    lines.append("")

    # Separate pdist and cdist results
    pdist_results = {}
    cdist_results = {}

    for impl, df in batch_results.items():
        pdist_results[impl] = df.filter(pl.col("function") == "pdist")
        cdist_results[impl] = df.filter(pl.col("function") == "cdist")

    # pdist analysis
    if (
        pdist_results["cython"].height > 0
        or "rust" in pdist_results
        and pdist_results["rust"].height > 0
    ):
        lines.append("### pdist Performance")
        lines.append("")
        lines.append(
            "Performance for pairwise distance computation (compressed distance matrix) with varying trajectory lengths:"
        )
        lines.append("")
        lines.append(
            "| Distance Type | Traj Length | Distances | Cython (ms) | Rust Seq (ms) | Rust Par (ms) | Speedup (Seq) | Speedup (Par) | Parallel Efficiency |"
        )
        lines.append(
            "|---------------|-------------|-----------|-------------|--------------|--------------|--------------|--------------|---------------------|"
        )

        # Get unique trajectory lengths and distance types
        traj_lengths = set()
        distance_types = set()
        for impl, df in pdist_results.items():
            if df.height > 0:
                traj_lengths.update(df["trajectory_length"].unique().to_list())
                distance_types.update(df["distance_type"].unique().to_list())

        for distance_type in sorted(distance_types):
            for traj_length in sorted(traj_lengths):
                row_data = {
                    "cython": None,
                    "rust_seq": None,
                    "rust_par": None,
                    "num_distances": None,
                }

                # Get Cython results
                if pdist_results["cython"].height > 0:
                    cython_rows = pdist_results["cython"].filter(
                        (pl.col("distance_type") == distance_type)
                        & (pl.col("trajectory_length") == traj_length)
                    )
                    if cython_rows.height > 0:
                        row = cython_rows.row(0, named=True)
                        times = row["times"]
                        if times:
                            row_data["cython"] = (
                                np.median(times) if use_median else np.mean(times)
                            )
                            row_data["num_distances"] = row["num_distances"]

                # Get Rust sequential results
                if "rust" in pdist_results:
                    rust_rows = pdist_results["rust"].filter(
                        (pl.col("distance_type") == distance_type)
                        & (pl.col("trajectory_length") == traj_length)
                        & (not pl.col("parallel"))
                    )
                    if rust_rows.height > 0:
                        row = rust_rows.row(0, named=True)
                        times = row["times"]
                        if times:
                            row_data["rust_seq"] = (
                                np.median(times) if use_median else np.mean(times)
                            )

                # Get Rust parallel results
                if "rust" in pdist_results:
                    rust_rows = pdist_results["rust"].filter(
                        (pl.col("distance_type") == distance_type)
                        & (pl.col("trajectory_length") == traj_length)
                        & (pl.col("parallel"))
                    )
                    if rust_rows.height > 0:
                        row = rust_rows.row(0, named=True)
                        times = row["times"]
                        if times:
                            row_data["rust_par"] = (
                                np.median(times) if use_median else np.mean(times)
                            )

                # Calculate speedups
                num_distances = (
                    row_data["num_distances"]
                    if row_data["num_distances"] is not None
                    else "N/A"
                )
                cython_str = (
                    f"{row_data['cython']*1000:.4f}"
                    if row_data["cython"] is not None
                    else "N/A"
                )
                rust_seq_str = (
                    f"{row_data['rust_seq']*1000:.4f}"
                    if row_data["rust_seq"] is not None
                    else "N/A"
                )
                rust_par_str = (
                    f"{row_data['rust_par']*1000:.4f}"
                    if row_data["rust_par"] is not None
                    else "N/A"
                )

                speedup_seq = (
                    row_data["cython"] / row_data["rust_seq"]
                    if row_data["cython"] and row_data["rust_seq"]
                    else None
                )
                speedup_par = (
                    row_data["cython"] / row_data["rust_par"]
                    if row_data["cython"] and row_data["rust_par"]
                    else None
                )
                parallel_eff = (
                    row_data["rust_seq"] / row_data["rust_par"]
                    if row_data["rust_seq"] and row_data["rust_par"]
                    else None
                )

                speedup_seq_str = f"{speedup_seq:.2f}x" if speedup_seq else "N/A"
                speedup_par_str = f"{speedup_par:.2f}x" if speedup_par else "N/A"
                parallel_eff_str = f"{parallel_eff:.3f}x" if parallel_eff else "N/A"

                lines.append(
                    f"| {distance_type} | {traj_length} | {num_distances} | {cython_str} | {rust_seq_str} | {rust_par_str} | "
                    f"{speedup_seq_str} | {speedup_par_str} | {parallel_eff_str} |"
                )

        lines.append("")

    # cdist analysis
    if (
        cdist_results["cython"].height > 0
        or "rust" in cdist_results
        and cdist_results["rust"].height > 0
    ):
        lines.append("### cdist Performance")
        lines.append("")
        lines.append(
            "Performance for distance computation between two trajectory collections with varying trajectory lengths:"
        )
        lines.append("")
        lines.append(
            "| Distance Type | Traj Length | Distances | Cython (ms) | Rust Seq (ms) | Rust Par (ms) | Speedup (Seq) | Speedup (Par) | Parallel Efficiency |"
        )
        lines.append(
            "|---------------|-------------|-----------|-------------|--------------|--------------|--------------|--------------|---------------------|"
        )

        # Get unique trajectory lengths and distance types
        traj_lengths = set()
        distance_types = set()
        for impl, df in cdist_results.items():
            if df.height > 0:
                traj_lengths.update(df["trajectory_length"].unique().to_list())
                distance_types.update(df["distance_type"].unique().to_list())

        for distance_type in sorted(distance_types):
            for traj_length in sorted(traj_lengths):
                row_data = {
                    "cython": None,
                    "rust_seq": None,
                    "rust_par": None,
                    "num_distances": None,
                }

                # Get Cython results
                if cdist_results["cython"].height > 0:
                    cython_rows = cdist_results["cython"].filter(
                        (pl.col("distance_type") == distance_type)
                        & (pl.col("trajectory_length") == traj_length)
                    )
                    if cython_rows.height > 0:
                        row = cython_rows.row(0, named=True)
                        times = row["times"]
                        if times:
                            row_data["cython"] = (
                                np.median(times) if use_median else np.mean(times)
                            )
                            row_data["num_distances"] = row["num_distances"]

                # Get Rust sequential results
                if "rust" in cdist_results:
                    rust_rows = cdist_results["rust"].filter(
                        (pl.col("distance_type") == distance_type)
                        & (pl.col("trajectory_length") == traj_length)
                        & (not pl.col("parallel"))
                    )
                    if rust_rows.height > 0:
                        row = rust_rows.row(0, named=True)
                        times = row["times"]
                        if times:
                            row_data["rust_seq"] = (
                                np.median(times) if use_median else np.mean(times)
                            )

                # Get Rust parallel results
                if "rust" in cdist_results:
                    rust_rows = cdist_results["rust"].filter(
                        (pl.col("distance_type") == distance_type)
                        & (pl.col("trajectory_length") == traj_length)
                        & (pl.col("parallel"))
                    )
                    if rust_rows.height > 0:
                        row = rust_rows.row(0, named=True)
                        times = row["times"]
                        if times:
                            row_data["rust_par"] = (
                                np.median(times) if use_median else np.mean(times)
                            )

                # Calculate speedups
                num_distances = (
                    row_data["num_distances"]
                    if row_data["num_distances"] is not None
                    else "N/A"
                )
                cython_str = (
                    f"{row_data['cython']*1000:.4f}"
                    if row_data["cython"] is not None
                    else "N/A"
                )
                rust_seq_str = (
                    f"{row_data['rust_seq']*1000:.4f}"
                    if row_data["rust_seq"] is not None
                    else "N/A"
                )
                rust_par_str = (
                    f"{row_data['rust_par']*1000:.4f}"
                    if row_data["rust_par"] is not None
                    else "N/A"
                )

                speedup_seq = (
                    row_data["cython"] / row_data["rust_seq"]
                    if row_data["cython"] and row_data["rust_seq"]
                    else None
                )
                speedup_par = (
                    row_data["cython"] / row_data["rust_par"]
                    if row_data["cython"] and row_data["rust_par"]
                    else None
                )
                parallel_eff = (
                    row_data["rust_seq"] / row_data["rust_par"]
                    if row_data["rust_seq"] and row_data["rust_par"]
                    else None
                )

                speedup_seq_str = f"{speedup_seq:.2f}x" if speedup_seq else "N/A"
                speedup_par_str = f"{speedup_par:.2f}x" if speedup_par else "N/A"
                parallel_eff_str = f"{parallel_eff:.3f}x" if parallel_eff else "N/A"

                lines.append(
                    f"| {distance_type} | {traj_length} | {num_distances} | {cython_str} | {rust_seq_str} | {rust_par_str} | "
                    f"{speedup_seq_str} | {speedup_par_str} | {parallel_eff_str} |"
                )

        lines.append("")

    # Batch computation summary
    lines.append("### Batch Computation Summary")
    lines.append("")

    # Calculate overall speedups by distance type
    for distance_type in ["euclidean", "spherical"]:
        lines.append(f"**{distance_type.capitalize()} Distance**:")

        # Collect pdist speedups
        pdist_speedups_seq = []
        pdist_speedups_par = []

        if "cython" in pdist_results and "rust" in pdist_results:
            for traj_length in sorted(traj_lengths):
                cython_time = None
                rust_seq_time = None
                rust_par_time = None

                cython_rows = pdist_results["cython"].filter(
                    (pl.col("distance_type") == distance_type)
                    & (pl.col("trajectory_length") == traj_length)
                )
                if cython_rows.height > 0:
                    row = cython_rows.row(0, named=True)
                    times = row["times"]
                    if times:
                        cython_time = np.median(times) if use_median else np.mean(times)

                rust_rows = pdist_results["rust"].filter(
                    (pl.col("distance_type") == distance_type)
                    & (pl.col("trajectory_length") == traj_length)
                    & (not pl.col("parallel"))
                )
                if rust_rows.height > 0:
                    row = rust_rows.row(0, named=True)
                    times = row["times"]
                    if times:
                        rust_seq_time = (
                            np.median(times) if use_median else np.mean(times)
                        )

                rust_rows = pdist_results["rust"].filter(
                    (pl.col("distance_type") == distance_type)
                    & (pl.col("trajectory_length") == traj_length)
                    & (pl.col("parallel"))
                )
                if rust_rows.height > 0:
                    row = rust_rows.row(0, named=True)
                    times = row["times"]
                    if times:
                        rust_par_time = (
                            np.median(times) if use_median else np.mean(times)
                        )

                if cython_time and rust_seq_time and rust_seq_time > 0:
                    pdist_speedups_seq.append(cython_time / rust_seq_time)
                if cython_time and rust_par_time and rust_par_time > 0:
                    pdist_speedups_par.append(cython_time / rust_par_time)

        if pdist_speedups_seq:
            avg_seq = np.mean(pdist_speedups_seq)
            lines.append(
                f"- **pdist** - Rust (sequential) vs Cython: Average {avg_seq:.2f}x speedup (range: {np.min(pdist_speedups_seq):.2f}x - {np.max(pdist_speedups_seq):.2f}x)"
            )
        if pdist_speedups_par:
            avg_par = np.mean(pdist_speedups_par)
            lines.append(
                f"- **pdist** - Rust (parallel) vs Cython: Average {avg_par:.2f}x speedup (range: {np.min(pdist_speedups_par):.2f}x - {np.max(pdist_speedups_par):.2f}x)"
            )

        # Collect cdist speedups
        cdist_speedups_seq = []
        cdist_speedups_par = []

        if "cython" in cdist_results and "rust" in cdist_results:
            for traj_length in sorted(traj_lengths):
                cython_time = None
                rust_seq_time = None
                rust_par_time = None

                cython_rows = cdist_results["cython"].filter(
                    (pl.col("distance_type") == distance_type)
                    & (pl.col("trajectory_length") == traj_length)
                )
                if cython_rows.height > 0:
                    row = cython_rows.row(0, named=True)
                    times = row["times"]
                    if times:
                        cython_time = np.median(times) if use_median else np.mean(times)

                rust_rows = cdist_results["rust"].filter(
                    (pl.col("distance_type") == distance_type)
                    & (pl.col("trajectory_length") == traj_length)
                    & (not pl.col("parallel"))
                )
                if rust_rows.height > 0:
                    row = rust_rows.row(0, named=True)
                    times = row["times"]
                    if times:
                        rust_seq_time = (
                            np.median(times) if use_median else np.mean(times)
                        )

                rust_rows = cdist_results["rust"].filter(
                    (pl.col("distance_type") == distance_type)
                    & (pl.col("trajectory_length") == traj_length)
                    & (pl.col("parallel"))
                )
                if rust_rows.height > 0:
                    row = rust_rows.row(0, named=True)
                    times = row["times"]
                    if times:
                        rust_par_time = (
                            np.median(times) if use_median else np.mean(times)
                        )

                if cython_time and rust_seq_time and rust_seq_time > 0:
                    cdist_speedups_seq.append(cython_time / rust_seq_time)
                if cython_time and rust_par_time and rust_par_time > 0:
                    cdist_speedups_par.append(cython_time / rust_par_time)

        if cdist_speedups_seq:
            avg_seq = np.mean(cdist_speedups_seq)
            lines.append(
                f"- **cdist** - Rust (sequential) vs Cython: Average {avg_seq:.2f}x speedup (range: {np.min(cdist_speedups_seq):.2f}x - {np.max(cdist_speedups_seq):.2f}x)"
            )
        if cdist_speedups_par:
            avg_par = np.mean(cdist_speedups_par)
            lines.append(
                f"- **cdist** - Rust (parallel) vs Cython: Average {avg_par:.2f}x speedup (range: {np.min(cdist_speedups_par):.2f}x - {np.max(cdist_speedups_par):.2f}x)"
            )

        lines.append("")

    lines.append(
        "**Note**: Parallel efficiency measures how much faster the parallel implementation is compared to the sequential implementation. For small datasets, parallel overhead may outweigh benefits."
    )

    return "\n".join(lines)


def generate_markdown_report(
    comparison_results: List[Dict[str, Any]],
    output_dir: Path,
    use_median: bool = True,
) -> str:
    """Generate Markdown format report"""
    central_metric = "Median" if use_median else "Mean"

    lines = []
    lines.append("# Performance Benchmark Report")
    lines.append("")
    lines.append(f"**Statistical Metric**: {central_metric}")
    lines.append("")

    # Summary table
    lines.append("## Summary Table")
    lines.append("")
    # Summary table column order: Algorithm, Distance Type, Hyperparameters, Rust, Cython, Python, Rust/Cython, Rust/Python
    lines.append(
        "| Algorithm | Distance Type | Hyperparameters | Rust | Cython | Python | Rust/Cython | Rust/Python |"
    )
    lines.append(
        "|-----------|---------------|-----------------|------|--------|--------|-------------|-------------|"
    )

    for result in comparison_results:
        impl_stats = result["implementations"]

        rust_str = "N/A"
        cython_str = "N/A"
        python_str = "N/A"
        rust_c_str = "N/A"
        rust_p_str = "N/A"

        if "rust" in impl_stats:
            stats = impl_stats["rust"]
            rust_str = f"{stats['central']*1000:.4f}ms"
        if "cython" in impl_stats:
            stats = impl_stats["cython"]
            cython_str = f"{stats['central']*1000:.4f}ms"
        if "python" in impl_stats:
            stats = impl_stats["python"]
            python_str = f"{stats['central']*1000:.4f}ms"

        if "rust_vs_cython" in result:
            rust_c_str = f"{result['rust_vs_cython']:.2f}x"
        if "rust_vs_python" in result:
            rust_p_str = f"{result['rust_vs_python']:.2f}x"

        lines.append(
            f"| {result['algorithm']} | {result['distance_type']} | {result['hyperparam']} | "
            f"{rust_str} | {cython_str} | {python_str} | {rust_c_str} | {rust_p_str} |"
        )

    lines.append("")

    # Detailed statistics table
    lines.append("## Detailed Statistics")
    lines.append("")

    for result in comparison_results:
        algorithm = result["algorithm"]
        distance_type = result["distance_type"]
        hyperparam = result["hyperparam"]

        lines.append(f"### {algorithm.upper()} ({distance_type})")
        if hyperparam != "N/A":
            lines.append(f"Hyperparameters: {hyperparam}")
        lines.append("")

        impl_stats = result["implementations"]

        # Create time statistics table - order: Rust, Cython, Python
        lines.append("#### Time Statistics")
        lines.append("")
        lines.append(
            "| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |"
        )
        lines.append(
            "|----------------|-------------|-----------|--------------|----------|----------|--------|"
        )

        for impl in ["rust", "cython", "python"]:
            if impl in impl_stats:
                stats = impl_stats[impl]
                lines.append(
                    f"| {impl.upper()} | {stats['median']*1000:.4f} | {stats['mean']*1000:.4f} | "
                    f"{stats['std']*1000:.4f} | {stats['min']*1000:.4f} | {stats['max']*1000:.4f} | {stats['cv']*100:.2f} |"
                )

        lines.append("")

        # Performance improvement - focus on Rust vs Cython, remove Cython vs Python
        lines.append("#### Performance Improvement")
        lines.append("")

        if "rust_vs_cython" in result:
            lines.append(f"- **Rust vs Cython**: {result['rust_vs_cython']:.2f}x")
        if "rust_vs_python" in result:
            lines.append(f"- Rust vs Python: {result['rust_vs_python']:.2f}x")

        lines.append("")

    # Analysis by algorithm
    lines.append("## Analysis by Algorithm")
    lines.append("")
    lines.append(
        "Performance comparison across different implementations for each algorithm:"
    )
    lines.append("")

    # Group by algorithm
    algorithm_stats = {}
    for result in comparison_results:
        algorithm = result["algorithm"]
        if algorithm not in algorithm_stats:
            algorithm_stats[algorithm] = []
        algorithm_stats[algorithm].append(result)

    for algorithm in sorted(algorithm_stats.keys()):
        lines.append(f"### {algorithm.upper()}")
        lines.append("")

        rust_vs_cython_speedups = []
        rust_vs_python_speedups = []

        for result in algorithm_stats[algorithm]:
            if "rust_vs_cython" in result:
                rust_vs_cython_speedups.append(result["rust_vs_cython"])
            if "rust_vs_python" in result:
                rust_vs_python_speedups.append(result["rust_vs_python"])

        if rust_vs_cython_speedups:
            avg = np.mean(rust_vs_cython_speedups)
            lines.append(
                f"- **Rust vs Cython**: Average improvement {avg:.2f}x (range: {np.min(rust_vs_cython_speedups):.2f}x - {np.max(rust_vs_cython_speedups):.2f}x)"
            )
        if rust_vs_python_speedups:
            avg = np.mean(rust_vs_python_speedups)
            lines.append(
                f"- Rust vs Python: Average improvement {avg:.2f}x (range: {np.min(rust_vs_python_speedups):.2f}x - {np.max(rust_vs_python_speedups):.2f}x)"
            )

        lines.append("")

    # Analysis by distance type
    lines.append("## Analysis by Distance Type")
    lines.append("")
    lines.append("Performance comparison across different distance types:")
    lines.append("")

    # Group by distance type
    distance_type_stats = {"euclidean": [], "spherical": []}

    for result in comparison_results:
        distance_type = result["distance_type"]
        if distance_type in distance_type_stats:
            distance_type_stats[distance_type].append(result)

    for distance_type in ["euclidean", "spherical"]:
        lines.append(f"### {distance_type.upper()} Distance")
        lines.append("")

        rust_vs_cython_speedups = []
        rust_vs_python_speedups = []

        for result in distance_type_stats[distance_type]:
            if "rust_vs_cython" in result:
                rust_vs_cython_speedups.append(result["rust_vs_cython"])
            if "rust_vs_python" in result:
                rust_vs_python_speedups.append(result["rust_vs_python"])

        if rust_vs_cython_speedups:
            avg = np.mean(rust_vs_cython_speedups)
            lines.append(
                f"- **Rust vs Cython**: Average improvement {avg:.2f}x (range: {np.min(rust_vs_cython_speedups):.2f}x - {np.max(rust_vs_cython_speedups):.2f}x)"
            )
        if rust_vs_python_speedups:
            avg = np.mean(rust_vs_python_speedups)
            lines.append(
                f"- Rust vs Python: Average improvement {avg:.2f}x (range: {np.min(rust_vs_python_speedups):.2f}x - {np.max(rust_vs_python_speedups):.2f}x)"
            )

        # Find the algorithm with the best performance improvement for this distance type (based on Rust vs Cython)
        if rust_vs_cython_speedups:
            max_speedup = np.max(rust_vs_cython_speedups)
            best_algo = None
            for result in distance_type_stats[distance_type]:
                if (
                    "rust_vs_cython" in result
                    and result["rust_vs_cython"] == max_speedup
                ):
                    best_algo = result["algorithm"]
                    break
            lines.append(
                f"- **Best Performance Improvement Algorithm**: {best_algo} ({max_speedup:.2f}x)"
            )

        lines.append("")

    # Overall statistics
    lines.append("## Overall Statistics")
    lines.append("")

    # Calculate overall average time for each implementation
    overall_stats = {
        "python": {"times": []},
        "cython": {"times": []},
        "rust": {"times": []},
    }

    for result in comparison_results:
        for impl, stats in result["implementations"].items():
            if impl in overall_stats:
                overall_stats[impl]["times"].append(stats["central"])

    # Output in order: Rust, Cython, Python
    for impl in ["rust", "cython", "python"]:
        if overall_stats[impl]["times"]:
            avg_time = np.mean(overall_stats[impl]["times"])
            lines.append(
                f"- {impl.upper()} overall average time: {avg_time*1000:.4f} ms"
            )

    lines.append("")

    # Calculate overall performance improvement: Rust vs Cython, Rust vs Python
    if "rust" in overall_stats and "cython" in overall_stats:
        if overall_stats["rust"]["times"] and overall_stats["cython"]["times"]:
            rust_avg = np.mean(overall_stats["rust"]["times"])
            cython_avg = np.mean(overall_stats["cython"]["times"])
            if rust_avg > 0:
                lines.append(
                    f"- Rust vs Cython overall average improvement: {cython_avg/rust_avg:.2f}x"
                )

    if "rust" in overall_stats and "python" in overall_stats:
        if overall_stats["rust"]["times"] and overall_stats["python"]["times"]:
            rust_avg = np.mean(overall_stats["rust"]["times"])
            python_avg = np.mean(overall_stats["python"]["times"])
            if rust_avg > 0:
                lines.append(
                    f"- Rust vs Python overall average improvement: {python_avg/rust_avg:.2f}x"
                )

    # Batch computation analysis
    batch_report = analyze_batch_computation(output_dir, use_median)
    if batch_report:
        lines.append("\n" + batch_report)

    return "\n".join(lines)


def main():
    args = parse_args()

    # Determine whether to use median or mean
    use_median = args.use_median and not args.use_mean

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all benchmark results
    results = load_benchmark_results(output_dir)

    if not results:
        raise ValueError("Error: No benchmark result files found")

    # Compare implementations
    comparison_results = compare_implementations(results, use_median)

    # Generate report
    report = generate_markdown_report(comparison_results, output_dir, use_median)

    # Save report
    output_path = output_dir / args.output_file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("Performance Comparison Summary")
    print("=" * 100)
    print(
        f"\n{'Algorithm':<15} {'Distance Type':<12} {'Hyperparameters':<20} {'Python(ms)':<15} {'Cython(ms)':<15} {'Rust(ms)':<15} {'Rust/C':<10} {'Rust/P':<10} {'C/P':<10}"
    )
    print("-" * 140)

    for result in comparison_results:
        impl_stats = result["implementations"]

        python_str = "N/A"
        cython_str = "N/A"
        rust_str = "N/A"
        rust_c_str = "N/A"
        rust_p_str = "N/A"
        c_p_str = "N/A"

        if "python" in impl_stats:
            stats = impl_stats["python"]
            python_str = f"{stats['central']*1000:.4f}"
        if "cython" in impl_stats:
            stats = impl_stats["cython"]
            cython_str = f"{stats['central']*1000:.4f}"
        if "rust" in impl_stats:
            stats = impl_stats["rust"]
            rust_str = f"{stats['central']*1000:.4f}"

        if "rust_vs_cython" in result:
            rust_c_str = f"{result['rust_vs_cython']:.2f}x"
        if "rust_vs_python" in result:
            rust_p_str = f"{result['rust_vs_python']:.2f}x"
        if "cython_vs_python" in result:
            c_p_str = f"{result['cython_vs_python']:.2f}x"

        hyperparam_str = result["hyperparam"] if result["hyperparam"] != "N/A" else ""

        print(
            f"{result['algorithm']:<15} {result['distance_type']:<12} {hyperparam_str:<20} "
            f"{python_str:<15} {cython_str:<15} {rust_str:<15} {rust_c_str:<10} {rust_p_str:<10} {c_p_str:<10}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
