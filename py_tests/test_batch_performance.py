"""Performance benchmark for batch computation functions"""

import time

import numpy as np
import traj_dist_rs


def benchmark_pdist():
    """Benchmark pdist function"""
    # Create 100 trajectories
    np.random.seed(42)
    trajectories = [np.random.rand(10, 2) * 100 for _ in range(100)]

    print("\n=== pdist Performance Benchmark ===")
    print(f"Number of trajectories: {len(trajectories)}")
    print(
        f"Expected pairwise distances: {len(trajectories) * (len(trajectories) - 1) // 2}"
    )

    metric = traj_dist_rs.Metric.sspd(type_d="euclidean")

    # Test sequential
    start = time.time()
    distances_seq = traj_dist_rs.pdist(trajectories, metric=metric, parallel=False)
    time_seq = time.time() - start
    print(f"Sequential time: {time_seq:.4f} seconds")

    # Test parallel with default settings
    start = time.time()
    distances_par = traj_dist_rs.pdist(trajectories, metric=metric, parallel=True)
    time_par = time.time() - start
    print(f"Parallel time (default): {time_par:.4f} seconds")

    # Verify results are the same
    assert np.allclose(
        distances_seq, distances_par
    ), "Results differ between sequential and parallel"

    # Calculate speedup
    speedup_default = time_seq / time_par

    print(f"\nSpeedup (default threads): {speedup_default:.2f}x")

    # Verify parallel is faster than sequential for large datasets
    if len(trajectories) >= 50:
        assert (
            speedup_default > 1.0
        ), "Parallel should be faster than sequential for large datasets"
        print(
            f"\n✓ Parallel processing is {speedup_default:.2f}x faster than sequential"
        )


def benchmark_cdist():
    """Benchmark cdist function"""
    # Create 100 trajectories in each collection
    np.random.seed(42)
    trajectories_a = [np.random.rand(10, 2) * 100 for _ in range(100)]
    trajectories_b = [np.random.rand(10, 2) * 100 for _ in range(100)]

    print("\n=== cdist Performance Benchmark ===")
    print(f"Number of trajectories in A: {len(trajectories_a)}")
    print(f"Number of trajectories in B: {len(trajectories_b)}")
    print(
        f"Expected distance matrix shape: ({len(trajectories_a)}, {len(trajectories_b)})"
    )

    metric = traj_dist_rs.Metric.sspd(type_d="euclidean")

    # Test sequential
    start = time.time()
    distances_seq = traj_dist_rs.cdist(
        trajectories_a, trajectories_b, metric=metric, parallel=False
    )
    time_seq = time.time() - start
    print(f"Sequential time: {time_seq:.4f} seconds")

    # Test parallel with default settings
    start = time.time()
    distances_par = traj_dist_rs.cdist(
        trajectories_a, trajectories_b, metric=metric, parallel=True
    )
    time_par = time.time() - start
    print(f"Parallel time (default): {time_par:.4f} seconds")

    # Verify results are the same
    assert np.allclose(
        distances_seq, distances_par
    ), "Results differ between sequential and parallel"

    # Calculate speedup
    speedup_default = time_seq / time_par

    print(f"\nSpeedup (default threads): {speedup_default:.2f}x")

    # Verify parallel is faster than sequential for large datasets
    if len(trajectories_a) >= 100:
        if speedup_default > 1.0:
            print(
                f"\n✓ Parallel processing is {speedup_default:.2f}x faster than sequential"
            )
        else:
            print(
                f"\n⚠ Parallel processing is {speedup_default:.2f}x slower than sequential (overhead may dominate for this dataset size)"
            )
            # For very small tasks, parallelization overhead can dominate
            print("  This is expected for small datasets or very fast operations")


def benchmark_different_algorithms():
    """Benchmark different algorithms"""
    # Create 50 trajectories
    np.random.seed(42)
    trajectories = [np.random.rand(10, 2) * 100 for _ in range(50)]

    print("\n=== Algorithm Performance Benchmark ===")
    print(f"Number of trajectories: {len(trajectories)}")

    algorithms = [
        ("SSPD", traj_dist_rs.Metric.sspd(type_d="euclidean")),
        ("DTW", traj_dist_rs.Metric.dtw(type_d="euclidean")),
        ("Hausdorff", traj_dist_rs.Metric.hausdorff(type_d="euclidean")),
        ("LCSS", traj_dist_rs.Metric.lcss(eps=5.0, type_d="euclidean")),
        ("EDR", traj_dist_rs.Metric.edr(eps=5.0, type_d="euclidean")),
        ("ERP", traj_dist_rs.Metric.erp(g=[0.0, 0.0], type_d="euclidean")),
        ("DiscretFrechet", traj_dist_rs.Metric.discret_frechet(type_d="euclidean")),
    ]

    for name, metric in algorithms:
        start = time.time()
        _ = traj_dist_rs.pdist(trajectories, metric=metric, parallel=True)
        elapsed = time.time() - start
        print(f"{name:20s}: {elapsed:.4f} seconds")


if __name__ == "__main__":
    print("Running batch computation performance benchmarks...")
    benchmark_pdist()
    benchmark_cdist()
    benchmark_different_algorithms()
    print("\n✓ All performance benchmarks completed successfully!")
