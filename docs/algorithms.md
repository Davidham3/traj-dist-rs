# Algorithms

This page describes the various trajectory distance algorithms implemented in `traj-dist-rs`.

## Available in Python and Rust

### SSPD (Symmetric Segment-Path Distance)

The Symmetric Segment-Path Distance measures the average distance between segments of two trajectories.

* **Function**: `sspd(traj1, traj2, type_d="euclidean")`
* **Distance Types**: euclidean, spherical
* **Parameters**: None
* **Description**: Calculates the average distance from points on one trajectory to the other trajectory and vice versa, then averages the two values.

### DTW (Dynamic Time Warping)

Dynamic Time Warping finds an optimal alignment between two sequences to measure their similarity.

* **Function**: `dtw(traj1, traj2, type_d="euclidean", use_full_matrix=False)`
* **Distance Types**: euclidean, spherical
* **Parameters**:
  * `use_full_matrix` (bool): Whether to return the full alignment matrix. Default: `False`
* **Description**: Computes the optimal alignment between two trajectories, allowing for temporal variations. Returns a `DpResult` object containing the distance and optionally the alignment matrix.

### Hausdorff Distance

The Hausdorff distance measures the maximum distance between any point on one trajectory to the closest point on the other.

* **Function**: `hausdorff(traj1, traj2, type_d="euclidean")`
* **Distance Types**: euclidean, spherical
* **Parameters**: None
* **Description**: Finds the maximum distance from any point in one trajectory to the closest point in the other trajectory.

### LCSS (Longest Common Subsequence)

Longest Common Subsequence similarity with a distance measure based on matching subsequences.

* **Function**: `lcss(traj1, traj2, type_d="euclidean", eps=200)`
* **Distance Types**: euclidean, spherical
* **Parameters**:
  * `eps` (float): Epsilon threshold for matching points
* **Description**: Finds the longest common subsequence of points within epsilon distance.

### EDR (Edit Distance on Real sequence)

Edit Distance on Real sequence measures similarity by counting edit operations needed to transform one trajectory to another.

* **Function**: `edr(traj1, traj2, type_d="euclidean", eps=200)`
* **Distance Types**: euclidean, spherical
* **Parameters**:
  * `eps` (float): Epsilon threshold for matching points
* **Description**: Calculates the minimum number of edit operations needed to transform one trajectory to match another.

### ERP (Edit distance with Real Penalty)

Edit distance with Real Penalty uses a gap penalty for unmatched points.

* **Functions**:
  * `erp_compat_traj_dist(traj1, traj2, type_d="euclidean", g=None)` - Compatible with original traj-dist
  * `erp_standard(traj1, traj2, type_d="euclidean", g=None)` - Standard implementation
* **Distance Types**: euclidean, spherical
* **Parameters**:
  * `g` (list[float]): Gap point for penalties. Default: `[0.0, 0.0]`
* **Description**: Two implementations: one compatible with traj-dist (`erp_compat_traj_dist`) and a standard implementation (`erp_standard`).

### Discret Frechet Distance

Discrete Fréchet distance measures similarity between polygonal curves.

* **Function**: `discret_frechet(traj1, traj2, type_d="euclidean")`
* **Distance Types**: euclidean (spherical not supported)
* **Parameters**: None
* **Description**: Computes the discrete Fréchet distance, a measure of similarity for polygonal curves.

## Batch Computation Functions

### pdist (Pairwise Distance Matrix)

Computes pairwise distances between all unique pairs in a trajectory collection.

* **Function**: `pdist(trajectories, metric, parallel=False)`
* **Parameters**:
  * `trajectories` (Sequence): Collection of trajectories
  * `metric` (Metric): Algorithm configuration object
  * `parallel` (bool): Whether to use parallel processing. Default: `False`
* **Returns**: Compressed distance matrix (1D array) containing all unique pairs (i < j)
* **Description**: Efficiently computes pairwise distances for symmetric distance metrics. Uses parallel processing with Rayon when enabled.

### cdist (Cross-Distance Matrix)

Computes distance matrix between two trajectory collections.

* **Function**: `cdist(trajectories_a, trajectories_b, metric, parallel=False)`
* **Parameters**:
  * `trajectories_a` (Sequence): First collection of trajectories
  * `trajectories_b` (Sequence): Second collection of trajectories
  * `metric` (Metric): Algorithm configuration object
  * `parallel` (bool): Whether to use parallel processing. Default: `False`
* **Returns**: Full distance matrix (2D array) of shape (len(trajectories_a), len(trajectories_b))
* **Description**: Computes cross-distances between two trajectory collections. Suitable for asymmetric distance metrics or cross-collection comparisons.

## Matrix-Based Computation

Some algorithms support computation with precomputed distance matrices for improved performance:

* **Functions**: `dtw_with_matrix`, `lcss_with_matrix`, `edr_with_matrix`, `discret_frechet_with_matrix`, `erp_compat_traj_dist_with_matrix`, `erp_standard_with_matrix`
* **Parameters**:
  * `distance_matrix` (numpy.ndarray): Precomputed distance matrix
  * `use_full_matrix` (bool): Whether to return the full alignment matrix
* **Description**: These functions accept a precomputed distance matrix for efficient batch computation, avoiding redundant distance calculations.
