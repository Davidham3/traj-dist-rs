"""
Batch Computation Examples

This example demonstrates how to use batch computation functions (pdist and cdist)
to efficiently calculate distances between multiple trajectories.
"""

import numpy as np
import traj_dist_rs

print("=" * 60)
print("Batch Computation Examples")
print("=" * 60)

# Create a set of sample trajectories
num_trajectories = 10
trajectories = [
    np.random.rand(20, 2) * 10  # Each trajectory has 20 points in [0, 10] range
    for _ in range(num_trajectories)
]

print(f"\nCreated {num_trajectories} trajectories with 20 points each")

# Method 1: Using pdist (pairwise distances in compressed format)
print("\n1. pdist - Pairwise Distance Matrix (Compressed)")
print("-" * 60)

# Create a Metric object for pdist
metric = traj_dist_rs.Metric.sspd(type_d="euclidean")

# Calculate pairwise distances using pdist
# Returns compressed distance matrix (upper triangle in row-major order)
compressed_distances = traj_dist_rs.pdist(trajectories, metric)

print(f"  Number of pairwise distances: {len(compressed_distances)}")
print(f"  Expected: {num_trajectories * (num_trajectories - 1) // 2}")
print(f"  First 5 distances: {compressed_distances[:5]}")
print(f"  Min distance: {compressed_distances.min():.6f}")
print(f"  Max distance: {compressed_distances.max():.6f}")

# Method 2: Using pdist with different algorithms
print("\n2. pdist with Different Algorithms")
print("-" * 60)

# Create metrics using factory methods
metrics = [
    ("SSPD", traj_dist_rs.Metric.sspd(type_d="euclidean")),
    ("DTW", traj_dist_rs.Metric.dtw(type_d="euclidean")),
    ("Hausdorff", traj_dist_rs.Metric.hausdorff(type_d="euclidean")),
    ("LCSS", traj_dist_rs.Metric.lcss(eps=0.1, type_d="euclidean")),
    ("EDR", traj_dist_rs.Metric.edr(eps=0.1, type_d="euclidean")),
    ("ERP", traj_dist_rs.Metric.erp(g=[0.0, 0.0], type_d="euclidean")),
    ("Discret Frechet", traj_dist_rs.Metric.discret_frechet(type_d="euclidean")),
]

for name, metric in metrics:
    try:
        distances = traj_dist_rs.pdist(trajectories, metric)
        print(
            f"  {name:20s}: {len(distances)} distances, " f"mean={distances.mean():.6f}"
        )
    except Exception as e:
        print(f"  {name:20s}: Error - {e}")

# Method 3: Using cdist (full distance matrix)
print("\n3. cdist - Full Distance Matrix")
print("-" * 60)

# Create two sets of trajectories
traj_set_a = trajectories[:5]  # First 5 trajectories
traj_set_b = trajectories[5:]  # Last 5 trajectories

# Calculate distance matrix using cdist
metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
distance_matrix = traj_dist_rs.cdist(traj_set_a, traj_set_b, metric)

print(f"  Distance matrix shape: {distance_matrix.shape}")
print(f"  Expected: ({len(traj_set_a)}, {len(traj_set_b)})")
print("  Full matrix:")
print(distance_matrix)

# Method 4: cdist with spherical distance
print("\n4. cdist with Spherical Distance")
print("-" * 60)

# Create trajectories with geographic coordinates (latitude, longitude)
geo_trajectories = [
    np.array(
        [
            [40.7128, -74.0060],  # New York
            [40.7306, -73.9352],  # LaGuardia
            [40.6413, -73.7781],  # JFK
        ]
    ),
    np.array(
        [
            [34.0522, -118.2437],  # Los Angeles
            [34.0522, -118.2437],  # LAX (approximate)
            [33.9425, -118.4081],  # Another LA area point
        ]
    ),
    np.array(
        [
            [51.5074, -0.1278],  # London
            [51.4700, -0.4543],  # Heathrow
            [51.5055, -0.2799],  # Another London area point
        ]
    ),
]

metric = traj_dist_rs.Metric.sspd(type_d="spherical")
distance_matrix = traj_dist_rs.cdist(geo_trajectories, geo_trajectories, metric)

print("  Distance matrix (Haversine, in kilometers):")
print(distance_matrix / 1000)  # Convert to kilometers

# Method 5: Using Metric with parameters
print("\n5. pdist with Metric Parameters")
print("-" * 60)

# LCSS with epsilon parameter
metric_lcss = traj_dist_rs.Metric.lcss(eps=0.1, type_d="euclidean")
distances_lcss = traj_dist_rs.pdist(trajectories, metric_lcss)
print(f"  LCSS (epsilon=0.1): {distances_lcss.mean():.6f}")

# EDR with epsilon parameter
metric_edr = traj_dist_rs.Metric.edr(eps=0.2, type_d="euclidean")
distances_edr = traj_dist_rs.pdist(trajectories, metric_edr)
print(f"  EDR (epsilon=0.2): {distances_edr.mean():.6f}")

print("\n" + "=" * 60)
print("All batch computations completed successfully!")
print("=" * 60)
