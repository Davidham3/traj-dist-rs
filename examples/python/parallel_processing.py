"""
Parallel Processing Examples

This example demonstrates how to use traj-dist-rs with joblib for
parallel processing of trajectory distance calculations.
"""

import time

import numpy as np
import traj_dist_rs
from joblib import Parallel, delayed

print("=" * 60)
print("Parallel Processing Examples")
print("=" * 60)

# Create sample trajectories
num_trajectories = 50
trajectory_length = 1000

print(
    f"\nCreating {num_trajectories} trajectories with {trajectory_length} points each..."
)
trajectories = [
    np.random.rand(trajectory_length, 2) * 10 for _ in range(num_trajectories)
]

# Example 1: Sequential vs Parallel Comparison
print("\n1. Sequential vs Parallel Performance Comparison")
print("-" * 60)


def compute_dtw_sequential(trajs):
    """Compute DTW distances sequentially."""
    results = []
    for i, traj1 in enumerate(trajs):
        for j, traj2 in enumerate(trajs):
            if i < j:  # Avoid duplicates and self-comparison
                result = traj_dist_rs.dtw(
                    traj1, traj2, "euclidean", use_full_matrix=False
                )
                results.append(result.distance)
    return results


def compute_dtw_parallel(trajs):
    """Compute DTW distances in parallel using joblib."""

    def compute_pair(i, j):
        result = traj_dist_rs.dtw(
            trajs[i], trajs[j], "euclidean", use_full_matrix=False
        )
        return result.distance

    # Create all pairs
    pairs = [(i, j) for i in range(len(trajs)) for j in range(i + 1, len(trajs))]

    # Compute in parallel
    results = Parallel(n_jobs=-1)(delayed(compute_pair)(i, j) for i, j in pairs)
    return results


# Measure sequential execution time
print("  Running sequential computation...")
start_time = time.time()
sequential_results = compute_dtw_sequential(trajectories[:10])  # Use 10 for demo
sequential_time = time.time() - start_time
print(f"  Sequential time: {sequential_time:.3f}s")

# Measure parallel execution time
print("  Running parallel computation...")
start_time = time.time()
parallel_results = compute_dtw_parallel(trajectories[:10])  # Use 10 for demo
parallel_time = time.time() - start_time
print(f"  Parallel time: {parallel_time:.3f}s")
print(f"  Speedup: {sequential_time / parallel_time:.2f}x")

# Verify results are identical
print(f"  Results match: {np.allclose(sequential_results, parallel_results)}")

# Example 2: Parallel computation with different algorithms
print("\n2. Parallel Processing with Different Algorithms")
print("-" * 60)


def compute_distance_pairwise(algo_name, trajs, dist_type="euclidean", **kwargs):
    """Compute pairwise distances for a specific algorithm."""

    def compute_pair(i, j):
        if algo_name == "sspd":
            return traj_dist_rs.sspd(trajs[i], trajs[j], dist_type)
        elif algo_name == "dtw":
            result = traj_dist_rs.dtw(
                trajs[i], trajs[j], dist_type, use_full_matrix=False
            )
            return result.distance
        elif algo_name == "hausdorff":
            return traj_dist_rs.hausdorff(trajs[i], trajs[j], dist_type)
        elif algo_name == "lcss":
            eps = kwargs.get("epsilon", 0.1)
            result = traj_dist_rs.lcss(
                trajs[i], trajs[j], dist_type, eps, use_full_matrix=False
            )
            return result.distance
        elif algo_name == "edr":
            eps = kwargs.get("epsilon", 0.1)
            result = traj_dist_rs.edr(
                trajs[i], trajs[j], dist_type, eps, use_full_matrix=False
            )
            return result.distance
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

    pairs = [(i, j) for i in range(len(trajs)) for j in range(i + 1, len(trajs))]
    results = Parallel(n_jobs=-1)(delayed(compute_pair)(i, j) for i, j in pairs)
    return np.array(results)


# Test a few algorithms in parallel
test_trajs = trajectories[:5]  # Use 5 trajectories for demo
algorithms = ["sspd", "dtw", "hausdorff", "lcss", "edr"]

for algo in algorithms:
    start_time = time.time()
    distances = compute_distance_pairwise(algo, test_trajs, epsilon=0.1)
    elapsed = time.time() - start_time
    print(
        f"  {algo:15s}: {len(distances)} distances, "
        f"mean={distances.mean():.6f}, time={elapsed:.3f}s"
    )

# Example 3: Parallel batch computation with pdist
print("\n3. Parallel Batch Computation vs Sequential")
print("-" * 60)

# Sequential pdist
metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
start_time = time.time()
distances_seq = traj_dist_rs.pdist(trajectories[:20], metric)
seq_time = time.time() - start_time
print(f"  Sequential pdist (20 trajectories): {seq_time:.3f}s")

# Parallel pdist (using Rust's Rayon)
start_time = time.time()
distances_par = traj_dist_rs.pdist(trajectories[:20], metric)
par_time = time.time() - start_time
print(f"  Parallel pdist (20 trajectories): {par_time:.3f}s")
print(f"  Speedup: {seq_time / par_time:.2f}x")
print(f"  Results match: {np.allclose(distances_seq, distances_par)}")

# Example 4: Parallel cross-distance computation
print("\n4. Parallel Cross-Distance Computation")
print("-" * 60)


def compute_cross_distance(set_a, set_b, dist_type):
    """Compute distance matrix between two trajectory sets."""

    def compute_cell(i, j):
        return traj_dist_rs.sspd(set_a[i], set_b[j], dist_type)

    results = Parallel(n_jobs=-1)(
        delayed(compute_cell)(i, j)
        for i in range(len(set_a))
        for j in range(len(set_b))
    )
    return np.array(results).reshape(len(set_a), len(set_b))


# Split trajectories into two sets
set_a = trajectories[:5]
set_b = trajectories[5:10]

dist_type = "euclidean"
start_time = time.time()
cross_matrix = compute_cross_distance(set_a, set_b, dist_type)
elapsed = time.time() - start_time

print(f"  Cross-distance matrix shape: {cross_matrix.shape}")
print(f"  Computation time: {elapsed:.3f}s")
print("  Matrix:")
print(cross_matrix)

# Example 5: Pickle serialization with joblib
print("\n5. Using DpResult with Joblib Parallel")
print("-" * 60)


def compute_dtw_with_matrix(i, j, trajs):
    """Compute DTW with full matrix and return DpResult."""
    result = traj_dist_rs.dtw(trajs[i], trajs[j], "euclidean", use_full_matrix=True)
    return result


# Note: DpResult objects can be pickled, enabling parallel processing
test_trajs = trajectories[:5]
pairs = [(i, j) for i in range(len(test_trajs)) for j in range(i + 1, len(test_trajs))]

start_time = time.time()
results = Parallel(n_jobs=-1)(
    delayed(compute_dtw_with_matrix)(i, j, test_trajs) for i, j in pairs
)
elapsed = time.time() - start_time

print(f"  Computed {len(results)} DTW distances with matrices")
print(f"  Time: {elapsed:.3f}s")
if results:
    print(f"  First result distance: {results[0].distance:.6f}")
    print(f"  First result matrix shape: {results[0].matrix.shape}")

print("\n" + "=" * 60)
print("All parallel processing examples completed successfully!")
print("=" * 60)
