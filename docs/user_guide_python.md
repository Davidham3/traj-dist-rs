---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# traj-dist-rs User Guide (Python)

This guide provides comprehensive usage examples for the traj-dist-rs library, a high-performance trajectory distance library implemented in Rust.

## Installation

First, install the package using pip:

```{code-cell} ipython3
# This would be the installation command (uncomment to run)
# !pip install traj-dist-rs
```

## Import the Library

```{code-cell} ipython3
import traj_dist_rs
import numpy as np
import matplotlib.pyplot as plt

print(f"traj_dist_rs version: {traj_dist_rs.__version__}")
```

## Basic Concepts

The library provides implementations of several trajectory distance algorithms:

- **SSPD**: Symmetric Segment-Path Distance
- **DTW**: Dynamic Time Warping
- **Hausdorff**: Hausdorff Distance
- **LCSS**: Longest Common Subsequence
- **EDR**: Edit Distance on Real sequence
- **ERP**: Edit distance with Real Penalty
- **Discret Frechet**: Discrete Fréchet Distance

Each algorithm supports both Euclidean (Cartesian) and Spherical (Haversine) distance calculations.

## Basic Usage Example

Let's start with a simple example using two trajectories:

```{code-cell} ipython3
# Define two simple trajectories as lists of [x, y] coordinates
traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

print("Trajectory 1:", traj1)
print("Trajectory 2:", traj2)
```

Now calculate distances using different algorithms:

```{code-cell} ipython3
# Calculate different trajectory distances
sspd_dist = traj_dist_rs.sspd(traj1, traj2, "euclidean")
dtw_dist = traj_dist_rs.dtw(traj1, traj2, "euclidean")
hausdorff_dist = traj_dist_rs.hausdorff(traj1, traj2, "euclidean")

print(f"SSPD distance: {sspd_dist:.6f}")
print(f"DTW distance: {dtw_dist:.6f}")
print(f"Hausdorff distance: {hausdorff_dist:.6f}")
```

## Using Spherical Distance for Geographic Coordinates

For geographic coordinates (latitude/longitude), use spherical distance:

```{code-cell} ipython3
# Geographic coordinates as [latitude, longitude]
# New York City trajectory
nyc_traj1 = [[40.7128, -74.0060], [40.7589, -73.9851], [40.7831, -73.9712]]
# Similar NYC trajectory with slight variations
nyc_traj2 = [[40.7228, -74.0160], [40.7689, -73.9951], [40.7931, -73.9812]]

print("NYC Trajectory 1:", nyc_traj1)
print("NYC Trajectory 2:", nyc_traj2)
```

```{code-cell} ipython3
# Calculate distances using spherical distance (Haversine formula)
sspd_dist = traj_dist_rs.sspd(nyc_traj1, nyc_traj2, "spherical")
dtw_dist = traj_dist_rs.dtw(nyc_traj1, nyc_traj2, "spherical")
hausdorff_dist = traj_dist_rs.hausdorff(nyc_traj1, nyc_traj2, "spherical")

print(f"SSPD spherical distance: {sspd_dist:.2f} meters")
print(f"DTW spherical distance: {dtw_dist:.2f} meters")
print(f"Hausdorff spherical distance: {hausdorff_dist:.2f} meters")
```

## Algorithms with Parameters

Some algorithms require additional parameters:

```{code-cell} ipython3
# Create some example trajectories
traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
traj2 = [[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]]

print("Parameterized Algorithms Example:")
print("Trajectory 1:", traj1)
print("Trajectory 2:", traj2)
```

```{code-cell} ipython3
# LCSS with epsilon parameter - threshold for point matching
lcss_dist = traj_dist_rs.lcss(traj1, traj2, "euclidean", 1.0)

# EDR with epsilon parameter - threshold for point matching
edr_dist = traj_dist_rs.edr(traj1, traj2, "euclidean", 1.0)

# ERP with gap point - reference point for penalties
erp_dist = traj_dist_rs.erp_compat_traj_dist(traj1, traj2, "euclidean", [0.0, 0.0])

print(f"LCSS distance (eps=1.0): {lcss_dist:.6f}")
print(f"EDR distance (eps=1.0): {edr_dist:.6f}")
print(f"ERP distance (g=[0,0]): {erp_dist:.6f}")
```

## Comparing All Available Algorithms

Let's compare all available algorithms on the same trajectories:

```{code-cell} ipython3
# Create sample trajectories
traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
traj2 = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]

print("Comparing All Algorithms:")
print("Trajectory 1:", traj1)
print("Trajectory 2:", traj2)
```

```{code-cell} ipython3
# Calculate all available distances
distances = {}

# Standard algorithms without parameters
distances['SSPD'] = traj_dist_rs.sspd(traj1, traj2, "euclidean")
distances['DTW'] = traj_dist_rs.dtw(traj1, traj2, "euclidean")
distances['Hausdorff'] = traj_dist_rs.hausdorff(traj1, traj2, "euclidean")
distances['Discret Frechet'] = traj_dist_rs.discret_frechet(traj1, traj2, "euclidean")

# Algorithms with parameters
distances['LCSS'] = traj_dist_rs.lcss(traj1, traj2, "euclidean", 2.0)
distances['EDR'] = traj_dist_rs.edr(traj1, traj2, "euclidean", 2.0)
distances['ERP (compat)'] = traj_dist_rs.erp_compat_traj_dist(traj1, traj2, "euclidean", [0.0, 0.0])
distances['ERP (standard)'] = traj_dist_rs.erp_standard(traj1, traj2, "euclidean", [0.0, 0.0])

# Print results
for algorithm, distance in distances.items():
    print(f"{algorithm}: {distance:.6f}")
```

## Visualizing Trajectories and Distances

Let's visualize the trajectories and compare algorithm results:

```{code-cell} ipython3
# Create trajectories for visualization
traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
traj2 = [[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, 2.0]]

# Calculate distances
sspd_dist = traj_dist_rs.sspd(traj1, traj2, "euclidean")
dtw_dist = traj_dist_rs.dtw(traj1, traj2, "euclidean")
hausdorff_dist = traj_dist_rs.hausdorff(traj1, traj2, "euclidean")

# Plot trajectories
plt.figure(figsize=(10, 6))
traj1_x, traj1_y = zip(*traj1)
traj2_x, traj2_y = zip(*traj2)

plt.plot(traj1_x, traj1_y, 'b-o', label=f'Trajectory 1', linewidth=2, markersize=8)
plt.plot(traj2_x, traj2_y, 'r-s', label=f'Trajectory 2', linewidth=2, markersize=8)

plt.title(f'Trajectory Comparison\nSSPD: {sspd_dist:.3f}, DTW: {dtw_dist:.3f}, Hausdorff: {hausdorff_dist:.3f}')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

## Performance Comparison

Let's compare the performance of different algorithms:

```{code-cell} ipython3
import time

# Create longer trajectories for performance testing
def create_random_trajectory(length, max_coord=10.0):
    return [[np.random.uniform(0, max_coord), np.random.uniform(0, max_coord)] 
            for _ in range(length)]

# Create test trajectories
traj_short = create_random_trajectory(10)
traj_medium = create_random_trajectory(50)

print("Performance Comparison (short trajectory - 10 points):")
algorithms = [
    ('SSPD', lambda t1, t2: traj_dist_rs.sspd(t1, t2, "euclidean")),
    ('DTW', lambda t1, t2: traj_dist_rs.dtw(t1, t2, "euclidean")),
    ('Hausdorff', lambda t1, t2: traj_dist_rs.hausdorff(t1, t2, "euclidean")),
    ('Discret Frechet', lambda t1, t2: traj_dist_rs.discret_frechet(t1, t2, "euclidean")),
]

for name, func in algorithms:
    start_time = time.time()
    result = func(traj_short, traj_short)
    end_time = time.time()
    print(f"{name}: {end_time - start_time:.6f}s, distance: {result:.6f}")
```

## Using with Real-World Data

Here's an example of how to use the library with real-world trajectory data:

```{code-cell} ipython3
# Example with GPS-like data
def create_gps_trajectory(start_lat, start_lon, num_points, lat_variation=0.01, lon_variation=0.01):
    """Create a GPS-like trajectory with some variation"""
    trajectory = []
    for i in range(num_points):
        lat = start_lat + i * 0.001 + np.random.uniform(-lat_variation, lat_variation)
        lon = start_lon + i * 0.001 + np.random.uniform(-lon_variation, lon_variation)
        trajectory.append([lat, lon])
    return trajectory

# Create two similar GPS trajectories
gps_traj1 = create_gps_trajectory(40.7128, -74.0060, 20)  # NYC-like
gps_traj2 = create_gps_trajectory(40.7130, -74.0062, 20)  # Similar, slightly different start

print(f"GPS Trajectory 1 (first 3 points): {gps_traj1[:3]}")
print(f"GPS Trajectory 2 (first 3 points): {gps_traj2[:3]}")
```

```{code-cell} ipython3
# Calculate distances using spherical distance for GPS coordinates
sspd_dist = traj_dist_rs.sspd(gps_traj1, gps_traj2, "spherical")
dtw_dist = traj_dist_rs.dtw(gps_traj1, gps_traj2, "spherical")
hausdorff_dist = traj_dist_rs.hausdorff(gps_traj1, gps_traj2, "spherical")

print(f"SSPD spherical distance: {sspd_dist:.2f} meters")
print(f"DTW spherical distance: {dtw_dist:.2f} meters")
print(f"Hausdorff spherical distance: {hausdorff_dist:.2f} meters")
```

## Error Handling

The library provides appropriate error handling:

```{code-cell} ipython3
# Example of error handling
try:
    # This will raise an error - invalid distance type
    invalid_dist = traj_dist_rs.sspd([[0, 0], [1, 1]], [[1, 1], [2, 2]], "invalid_type")
except ValueError as e:
    print(f"Caught expected error: {e}")

try:
    # This will raise an error - invalid trajectory format
    invalid_dist = traj_dist_rs.sspd([[0, 0, 0], [1, 1]], [[0, 0], [1, 1]], "euclidean")
except Exception as e:
    print(f"Caught expected error: {e}")
```

## Summary

This guide covered:

1. Basic installation and import
2. Different trajectory distance algorithms
3. Euclidean vs. Spherical distance calculations
4. Algorithms with parameters (LCSS, EDR, ERP)
5. Performance considerations
6. Real-world GPS data usage
7. Error handling

The `traj-dist-rs` library provides high-performance trajectory distance calculations with both Python and Rust APIs, making it suitable for a wide range of trajectory analysis applications.

For more information on each algorithm, refer to the API documentation.
