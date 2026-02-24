import numpy as np
import polars as pl

# 读取结果
df_cython = pl.read_parquet("benchmark_output/traj_dist_cython_batch_benchmark.parquet")
df_rust = pl.read_parquet("benchmark_output/traj_dist_rs_rust_batch_benchmark.parquet")


# 计算中位数
def compute_median_time(times):
    return np.median(times)


# 处理 Cython 数据
cython_results = []
for row in df_cython.iter_rows(named=True):
    median_time = compute_median_time(row["times"])
    cython_results.append(
        {
            "algorithm": row["algorithm"],
            "distance_type": row["distance_type"],
            "function": row["function"],
            "trajectory_length": row["trajectory_length"],
            "num_distances": row["num_distances"],
            "median_time_ms": median_time * 1000,  # 转换为毫秒
        }
    )

# 处理 Rust 数据
rust_results = []
for row in df_rust.iter_rows(named=True):
    median_time = compute_median_time(row["times"])
    rust_results.append(
        {
            "algorithm": row["algorithm"],
            "distance_type": row["distance_type"],
            "function": row["function"],
            "trajectory_length": row["trajectory_length"],
            "num_distances": row["num_distances"],
            "parallel": row["parallel"],
            "median_time_ms": median_time * 1000,  # 转换为毫秒
        }
    )


# 生成 Batch Computation Performance 部分
def generate_batch_performance_section():
    output = []
    output.append("## Batch Computation Performance")
    output.append("")
    output.append(
        "Performance comparison for batch distance computation (pdist and cdist):"
    )
    output.append("")

    # Test Configuration
    output.append("### Test Configuration")
    output.append("")
    output.append("- **Algorithm**: DTW (Dynamic Time Warping)")
    output.append("- **Number of trajectories**: 5 (fixed)")
    output.append("- **pdist computation**: 10 distances (5×4/2)")
    output.append("- **cdist computation**: 25 distances (5×5)")
    output.append("- **Trajectory lengths tested**: 10, 100, 1000 points")
    output.append("- **Distance types**: Euclidean and Spherical")
    output.append("")

    # pdist Performance
    output.append("### pdist Performance")
    output.append("")
    output.append(
        "Performance for pairwise distance computation (compressed distance matrix) with varying trajectory lengths:"
    )
    output.append("")

    pdist_table = []
    pdist_table.append(
        "| Distance Type | Traj Length | Distances | Cython (ms) | Rust Seq (ms) | Rust Par (ms) | Speedup (Seq) | Speedup (Par) | Parallel Efficiency |"
    )
    pdist_table.append(
        "|---------------|-------------|-----------|-------------|--------------|--------------|--------------|--------------|---------------------|"
    )

    for distance_type in ["euclidean", "spherical"]:
        for trajectory_length in [10, 100, 1000]:
            # Find Cython result
            cython_result = None
            for r in cython_results:
                if (
                    r["function"] == "pdist"
                    and r["distance_type"] == distance_type
                    and r["trajectory_length"] == trajectory_length
                ):
                    cython_result = r
                    break

            # Find Rust sequential result
            rust_seq_result = None
            for r in rust_results:
                if (
                    r["function"] == "pdist"
                    and r["distance_type"] == distance_type
                    and r["trajectory_length"] == trajectory_length
                    and not r["parallel"]
                ):
                    rust_seq_result = r
                    break

            # Find Rust parallel result
            rust_par_result = None
            for r in rust_results:
                if (
                    r["function"] == "pdist"
                    and r["distance_type"] == distance_type
                    and r["trajectory_length"] == trajectory_length
                    and r["parallel"]
                ):
                    rust_par_result = r
                    break

            if cython_result and rust_seq_result and rust_par_result:
                num_distances = cython_result["num_distances"]
                cython_time = cython_result["median_time_ms"]
                rust_seq_time = rust_seq_result["median_time_ms"]
                rust_par_time = rust_par_result["median_time_ms"]

                speedup_seq = cython_time / rust_seq_time
                speedup_par = cython_time / rust_par_time
                parallel_efficiency = rust_seq_time / rust_par_time

                pdist_table.append(
                    f"| {distance_type.capitalize()} | {trajectory_length} | {num_distances} | {cython_time:.4f} | {rust_seq_time:.4f} | {rust_par_time:.4f} | {speedup_seq:.2f}x | {speedup_par:.2f}x | {parallel_efficiency:.3f}x |"
                )

    output.extend(pdist_table)
    output.append("")

    # cdist Performance
    output.append("### cdist Performance")
    output.append("")
    output.append(
        "Performance for distance computation between two trajectory collections with varying trajectory lengths:"
    )
    output.append("")

    cdist_table = []
    cdist_table.append(
        "| Distance Type | Traj Length | Distances | Cython (ms) | Rust Seq (ms) | Rust Par (ms) | Speedup (Seq) | Speedup (Par) | Parallel Efficiency |"
    )
    cdist_table.append(
        "|---------------|-------------|-----------|-------------|--------------|--------------|--------------|--------------|---------------------|"
    )

    for distance_type in ["euclidean", "spherical"]:
        for trajectory_length in [10, 100, 1000]:
            # Find Cython result
            cython_result = None
            for r in cython_results:
                if (
                    r["function"] == "cdist"
                    and r["distance_type"] == distance_type
                    and r["trajectory_length"] == trajectory_length
                ):
                    cython_result = r
                    break

            # Find Rust sequential result
            rust_seq_result = None
            for r in rust_results:
                if (
                    r["function"] == "cdist"
                    and r["distance_type"] == distance_type
                    and r["trajectory_length"] == trajectory_length
                    and not r["parallel"]
                ):
                    rust_seq_result = r
                    break

            # Find Rust parallel result
            rust_par_result = None
            for r in rust_results:
                if (
                    r["function"] == "cdist"
                    and r["distance_type"] == distance_type
                    and r["trajectory_length"] == trajectory_length
                    and r["parallel"]
                ):
                    rust_par_result = r
                    break

            if cython_result and rust_seq_result and rust_par_result:
                num_distances = cython_result["num_distances"]
                cython_time = cython_result["median_time_ms"]
                rust_seq_time = rust_seq_result["median_time_ms"]
                rust_par_time = rust_par_result["median_time_ms"]

                speedup_seq = cython_time / rust_seq_time
                speedup_par = cython_time / rust_par_time
                parallel_efficiency = rust_seq_time / rust_par_time

                cdist_table.append(
                    f"| {distance_type.capitalize()} | {trajectory_length} | {num_distances} | {cython_time:.4f} | {rust_seq_time:.4f} | {rust_par_time:.4f} | {speedup_seq:.2f}x | {speedup_par:.2f}x | {parallel_efficiency:.3f}x |"
                )

    output.extend(cdist_table)
    output.append("")

    # Batch Computation Summary
    output.append("### Batch Computation Summary")
    output.append("")

    # Calculate summary statistics
    euclidean_pdist_seq_speedups = []
    euclidean_pdist_par_speedups = []
    euclidean_cdist_seq_speedups = []
    euclidean_cdist_par_speedups = []

    spherical_pdist_seq_speedups = []
    spherical_pdist_par_speedups = []
    spherical_cdist_seq_speedups = []
    spherical_cdist_par_speedups = []

    for distance_type in ["euclidean", "spherical"]:
        for trajectory_length in [10, 100, 1000]:
            # pdist
            cython_result = None
            rust_seq_result = None
            rust_par_result = None

            for r in cython_results:
                if (
                    r["function"] == "pdist"
                    and r["distance_type"] == distance_type
                    and r["trajectory_length"] == trajectory_length
                ):
                    cython_result = r
                    break

            for r in rust_results:
                if (
                    r["function"] == "pdist"
                    and r["distance_type"] == distance_type
                    and r["trajectory_length"] == trajectory_length
                ):
                    if not r["parallel"]:
                        rust_seq_result = r
                    else:
                        rust_par_result = r

            if cython_result and rust_seq_result and rust_par_result:
                cython_time = cython_result["median_time_ms"]
                rust_seq_time = rust_seq_result["median_time_ms"]
                rust_par_time = rust_par_result["median_time_ms"]

                speedup_seq = cython_time / rust_seq_time
                speedup_par = cython_time / rust_par_time

                if distance_type == "euclidean":
                    euclidean_pdist_seq_speedups.append(speedup_seq)
                    euclidean_pdist_par_speedups.append(speedup_par)
                else:
                    spherical_pdist_seq_speedups.append(speedup_seq)
                    spherical_pdist_par_speedups.append(speedup_par)

            # cdist
            cython_result = None
            rust_seq_result = None
            rust_par_result = None

            for r in cython_results:
                if (
                    r["function"] == "cdist"
                    and r["distance_type"] == distance_type
                    and r["trajectory_length"] == trajectory_length
                ):
                    cython_result = r
                    break

            for r in rust_results:
                if (
                    r["function"] == "cdist"
                    and r["distance_type"] == distance_type
                    and r["trajectory_length"] == trajectory_length
                ):
                    if not r["parallel"]:
                        rust_seq_result = r
                    else:
                        rust_par_result = r

            if cython_result and rust_seq_result and rust_par_result:
                cython_time = cython_result["median_time_ms"]
                rust_seq_time = rust_seq_result["median_time_ms"]
                rust_par_time = rust_par_result["median_time_ms"]

                speedup_seq = cython_time / rust_seq_time
                speedup_par = cython_time / rust_par_time

                if distance_type == "euclidean":
                    euclidean_cdist_seq_speedups.append(speedup_seq)
                    euclidean_cdist_par_speedups.append(speedup_par)
                else:
                    spherical_cdist_seq_speedups.append(speedup_seq)
                    spherical_cdist_par_speedups.append(speedup_par)

    # Euclidean summary
    output.append("**Euclidean Distance**:")
    output.append(
        f"- **pdist** - Rust (sequential) vs Cython: Average {np.mean(euclidean_pdist_seq_speedups):.2f}x speedup (range: {min(euclidean_pdist_seq_speedups):.2f}x - {max(euclidean_pdist_seq_speedups):.2f}x)"
    )
    output.append(
        f"- **pdist** - Rust (parallel) vs Cython: Average {np.mean(euclidean_pdist_par_speedups):.2f}x speedup (range: {min(euclidean_pdist_par_speedups):.2f}x - {max(euclidean_pdist_par_speedups):.2f}x)"
    )
    output.append(
        f"- **cdist** - Rust (sequential) vs Cython: Average {np.mean(euclidean_cdist_seq_speedups):.2f}x speedup (range: {min(euclidean_cdist_seq_speedups):.2f}x - {max(euclidean_cdist_seq_speedups):.2f}x)"
    )
    output.append(
        f"- **cdist** - Rust (parallel) vs Cython: Average {np.mean(euclidean_cdist_par_speedups):.2f}x speedup (range: {min(euclidean_cdist_par_speedups):.2f}x - {max(euclidean_cdist_par_speedups):.2f}x)"
    )
    output.append("")

    # Spherical summary
    output.append("**Spherical Distance**:")
    output.append(
        f"- **pdist** - Rust (sequential) vs Cython: Average {np.mean(spherical_pdist_seq_speedups):.2f}x speedup (range: {min(spherical_pdist_seq_speedups):.2f}x - {max(spherical_pdist_seq_speedups):.2f}x)"
    )
    output.append(
        f"- **pdist** - Rust (parallel) vs Cython: Average {np.mean(spherical_pdist_par_speedups):.2f}x speedup (range: {min(spherical_pdist_par_speedups):.2f}x - {max(spherical_pdist_par_speedups):.2f}x)"
    )
    output.append(
        f"- **cdist** - Rust (sequential) vs Cython: Average {np.mean(spherical_cdist_seq_speedups):.2f}x speedup (range: {min(spherical_cdist_seq_speedups):.2f}x - {max(spherical_cdist_seq_speedups):.2f}x)"
    )
    output.append(
        f"- **cdist** - Rust (parallel) vs Cython: Average {np.mean(spherical_cdist_par_speedups):.2f}x speedup (range: {min(spherical_cdist_par_speedups):.2f}x - {max(spherical_cdist_par_speedups):.2f}x)"
    )
    output.append("")

    output.append(
        "**Note**: Parallel efficiency measures how much faster the parallel implementation is compared to the sequential implementation. For small datasets, parallel overhead may outweigh benefits."
    )

    return "\n".join(output)
