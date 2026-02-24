"""
Metric API Examples

This example demonstrates how to use the Metric class and its factory methods
for type-safe and flexible distance calculation configuration.
"""

import numpy as np
import traj_dist_rs

print("=" * 60)
print("Metric API Examples")
print("=" * 60)

# Create sample trajectories
traj1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

traj2 = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [3.1, 3.1]])

traj3 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

# Example 1: Creating Metrics using factory methods
print("\n1. Creating Metrics using Factory Methods")
print("-" * 60)

# Create Metric with Euclidean distance
metric_sspd = traj_dist_rs.Metric.sspd(type_d="euclidean")
print("  Metric.sspd(type_d='euclidean'): Created successfully")

metric_dtw = traj_dist_rs.Metric.dtw(type_d="euclidean")
print("  Metric.dtw(type_d='euclidean'): Created successfully")

metric_hausdorff = traj_dist_rs.Metric.hausdorff(type_d="euclidean")
print("  Metric.hausdorff(type_d='euclidean'): Created successfully")

# Example 2: Creating Metrics with different distance types
print("\n2. Metrics with Different Distance Types")
print("-" * 60)

dist_types = ["euclidean", "spherical"]

for dist_type in dist_types:
    metric = traj_dist_rs.Metric.sspd(type_d=dist_type)
    distance = traj_dist_rs.sspd(traj1, traj2, dist_type)
    print(f"  {dist_type:12s}: SSPD distance = {distance:.6f}")

# Example 3: Creating Metrics with parameters
print("\n3. Metrics with Parameters")
print("-" * 60)

# LCSS with epsilon parameter
metric_lcss = traj_dist_rs.Metric.lcss(eps=0.1, type_d="euclidean")
result = traj_dist_rs.lcss(traj1, traj2, "euclidean", 0.1, use_full_matrix=False)
print(f"  LCSS (epsilon=0.1): distance = {result.distance:.6f}")

# EDR with epsilon parameter
metric_edr = traj_dist_rs.Metric.edr(eps=0.2, type_d="euclidean")
result = traj_dist_rs.edr(traj1, traj2, "euclidean", 0.2, use_full_matrix=False)
print(f"  EDR (epsilon=0.2): distance = {result.distance:.6f}")

# ERP with gap point
metric_erp = traj_dist_rs.Metric.erp(g=[0.0, 0.0], type_d="euclidean")
result = traj_dist_rs.erp_standard(
    traj1, traj2, "euclidean", [0.0, 0.0], use_full_matrix=False
)
print(f"  ERP (gap=[0,0]): distance = {result.distance:.6f}")

# Example 4: Using Metrics with batch computation
print("\n4. Using Metrics with Batch Computation")
print("-" * 60)

# Create a set of trajectories
trajectories = [np.random.rand(10, 2) * 5 for _ in range(10)]

# Compute pairwise distances using different metrics
batch_metrics = [
    ("SSPD", traj_dist_rs.Metric.sspd(type_d="euclidean")),
    ("DTW", traj_dist_rs.Metric.dtw(type_d="euclidean")),
    ("Hausdorff", traj_dist_rs.Metric.hausdorff(type_d="euclidean")),
]

for name, metric in batch_metrics:
    distances = traj_dist_rs.pdist(trajectories, metric)
    print(f"  {name:15s}: {len(distances)} distances, " f"mean={distances.mean():.6f}")

# Example 5: Using Metrics with cdist
print("\n5. Using Metrics with cdist")
print("-" * 60)

set_a = trajectories[:5]
set_b = trajectories[5:]

metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
distance_matrix = traj_dist_rs.cdist(set_a, set_b, metric)

print(f"  Distance matrix shape: {distance_matrix.shape}")
print("  Matrix:")
print(distance_matrix)

# Example 6: Comparing distances with different algorithms
print("\n6. Comparing Distances with Different Algorithms")
print("-" * 60)

print("  Comparing distances between traj1 and traj2:")
print(f"  traj1: {traj1[0]}, {traj1[1]}, {traj1[2]}, {traj1[3]}")
print(f"  traj2: {traj2[0]}, {traj2[1]}, {traj2[2]}, {traj2[3]}")
print()

# SSPD
distance = traj_dist_rs.sspd(traj1, traj2, "euclidean")
print(f"  {'SSPD':15s}: {distance:.6f}")

# DTW
result = traj_dist_rs.dtw(traj1, traj2, "euclidean", use_full_matrix=False)
print(f"  {'DTW':15s}: {result.distance:.6f}")

# Hausdorff
distance = traj_dist_rs.hausdorff(traj1, traj2, "euclidean")
print(f"  {'Hausdorff':15s}: {distance:.6f}")

# Example 7: Using Metrics with pdist for large datasets
print("\n7. Using Metrics with pdist for Large Datasets")
print("-" * 60)

# Create a larger set of trajectories
large_trajectories = [np.random.rand(20, 2) * 10 for _ in range(20)]

# Use different metrics with pdist
metrics = [
    ("SSPD", traj_dist_rs.Metric.sspd(type_d="euclidean")),
    ("DTW", traj_dist_rs.Metric.dtw(type_d="euclidean")),
    ("LCSS", traj_dist_rs.Metric.lcss(eps=0.5, type_d="euclidean")),
]

for name, metric in metrics:
    distances = traj_dist_rs.pdist(large_trajectories, metric)
    print(
        f"  {name:15s}: {len(distances)} distances, "
        f"mean={distances.mean():.6f}, "
        f"min={distances.min():.6f}, "
        f"max={distances.max():.6f}"
    )

# Example 8: Using Metrics with cdist for cross-distance computation
print("\n8. Using Metrics with cdist for Cross-Distance Computation")
print("-" * 60)

# Create two separate trajectory sets
set_a = large_trajectories[:10]
set_b = large_trajectories[10:]

metric = traj_dist_rs.Metric.hausdorff(type_d="euclidean")
distance_matrix = traj_dist_rs.cdist(set_a, set_b, metric)

print(f"  Cross-distance matrix shape: {distance_matrix.shape}")
print(f"  Mean distance: {distance_matrix.mean():.6f}")
print(f"  Min distance: {distance_matrix.min():.6f}")
print(f"  Max distance: {distance_matrix.max():.6f}")

print("\n" + "=" * 60)
print("All Metric API examples completed successfully!")
print("=" * 60)
