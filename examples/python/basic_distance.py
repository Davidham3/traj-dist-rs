"""
Basic Distance Calculation Examples

This example demonstrates how to calculate distances between trajectories
using various distance algorithms supported by traj-dist-rs.
"""

import numpy as np
import traj_dist_rs

# Define sample trajectories
# Trajectories can be defined as lists of [x, y] coordinates or numpy arrays
traj1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

traj2 = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [3.1, 3.1]])

traj3 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

print("=" * 60)
print("Basic Distance Calculation Examples")
print("=" * 60)

# SSPD (Symmetric Segment-Path Distance)
print("\n1. SSPD (Symmetric Segment-Path Distance)")
print("-" * 40)
dist_euclidean = traj_dist_rs.sspd(traj1, traj2, dist_type="euclidean")
dist_spherical = traj_dist_rs.sspd(traj1, traj2, dist_type="spherical")
print(f"  Euclidean: {dist_euclidean:.6f}")
print(f"  Spherical: {dist_spherical:.6f}")

# DTW (Dynamic Time Warping)
print("\n2. DTW (Dynamic Time Warping)")
print("-" * 40)
# DTW without matrix (returns DpResult with distance only)
result_no_matrix = traj_dist_rs.dtw(
    traj1, traj2, dist_type="euclidean", use_full_matrix=False
)
print(f"  Distance (no matrix): {result_no_matrix.distance:.6f}")

# DTW with matrix (returns DpResult with distance and matrix)
result_with_matrix = traj_dist_rs.dtw(
    traj1, traj2, dist_type="euclidean", use_full_matrix=True
)
print(f"  Distance (with matrix): {result_with_matrix.distance:.6f}")
print(f"  Matrix shape: {result_with_matrix.matrix.shape}")

# Hausdorff Distance
print("\n3. Hausdorff Distance")
print("-" * 40)
dist_euclidean = traj_dist_rs.hausdorff(traj1, traj3, dist_type="euclidean")
dist_spherical = traj_dist_rs.hausdorff(traj1, traj3, dist_type="spherical")
print(f"  Euclidean: {dist_euclidean:.6f}")
print(f"  Spherical: {dist_spherical:.6f}")

# LCSS (Longest Common Subsequence)
print("\n4. LCSS (Longest Common Subsequence)")
print("-" * 40)
# LCSS with epsilon parameter (similarity threshold)
result = traj_dist_rs.lcss(traj1, traj2, "euclidean", 0.1, use_full_matrix=True)
print(f"  Distance (epsilon=0.1): {result.distance:.6f}")
print(f"  Matrix shape: {result.matrix.shape}")

# LCSS with different epsilon
result = traj_dist_rs.lcss(traj1, traj3, "euclidean", 0.5, use_full_matrix=False)
print(f"  Distance (epsilon=0.5): {result.distance:.6f}")

# EDR (Edit Distance on Real sequence)
print("\n5. EDR (Edit Distance on Real sequence)")
print("-" * 40)
result = traj_dist_rs.edr(traj1, traj2, "euclidean", 0.1, use_full_matrix=True)
print(f"  Distance (epsilon=0.1): {result.distance:.6f}")
print(f"  Matrix shape: {result.matrix.shape}")

# Discret Frechet Distance
print("\n6. Discret Frechet Distance")
print("-" * 40)
result = traj_dist_rs.discret_frechet(traj1, traj2, "euclidean", use_full_matrix=True)
print(f"  Distance: {result.distance:.6f}")
print(f"  Matrix shape: {result.matrix.shape}")

# ERP (Edit distance with Real Penalty)
print("\n7. ERP (Edit distance with Real Penalty)")
print("-" * 40)
# Standard ERP implementation
gap_point = [0.0, 0.0]
result_standard = traj_dist_rs.erp_standard(
    traj1, traj2, "euclidean", gap_point, use_full_matrix=True
)
print(f"  Standard ERP distance: {result_standard.distance:.6f}")
print(f"  Standard ERP matrix shape: {result_standard.matrix.shape}")

# ERP compatible with traj-dist
result_compat = traj_dist_rs.erp_compat_traj_dist(
    traj1, traj2, "euclidean", gap_point, use_full_matrix=False
)
print(f"  Compatible ERP distance: {result_compat.distance:.6f}")

print("\n" + "=" * 60)
print("All distance calculations completed successfully!")
print("=" * 60)
