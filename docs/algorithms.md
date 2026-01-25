# Algorithms

This page describes the various trajectory distance algorithms implemented in `traj-dist-rs`.

## SSPD (Symmetric Segment-Path Distance)

The Symmetric Segment-Path Distance measures the average distance between segments of two trajectories.

* **Function**: `sspd(traj1, traj2, type_d="euclidean")`
* **Distance Types**: euclidean, spherical
* **Parameters**: None
* **Description**: Calculates the average distance from points on one trajectory to the other trajectory and vice versa, then averages the two values.

## DTW (Dynamic Time Warping)

Dynamic Time Warping finds an optimal alignment between two sequences to measure their similarity.

* **Function**: `dtw(traj1, traj2, type_d="euclidean")`
* **Distance Types**: euclidean, spherical
* **Parameters**: None
* **Description**: Computes the optimal alignment between two trajectories, allowing for temporal variations.

## Hausdorff Distance

The Hausdorff distance measures the maximum distance between any point on one trajectory to the closest point on the other.

* **Function**: `hausdorff(traj1, traj2, type_d="euclidean")`
* **Distance Types**: euclidean, spherical
* **Parameters**: None
* **Description**: Finds the maximum distance from any point in one trajectory to the closest point in the other trajectory.

## LCSS (Longest Common Subsequence)

Longest Common Subsequence similarity with a distance measure based on matching subsequences.

* **Function**: `lcss(traj1, traj2, type_d="euclidean", eps=200)`
* **Distance Types**: euclidean, spherical
* **Parameters**: `eps` - Epsilon threshold for matching points
* **Description**: Finds the longest common subsequence of points within epsilon distance.

## EDR (Edit Distance on Real sequence)

Edit Distance on Real sequence measures similarity by counting edit operations needed to transform one trajectory to another.

* **Function**: `edr(traj1, traj2, type_d="euclidean", eps=200)`
* **Distance Types**: euclidean, spherical
* **Parameters**: `eps` - Epsilon threshold for matching points
* **Description**: Calculates the minimum number of edit operations needed to transform one trajectory to match another.

## ERP (Edit distance with Real Penalty)

Edit distance with Real Penalty uses a gap penalty for unmatched points.

* **Function**: `erp_compat_traj_dist(traj1, traj2, type_d="euclidean", g=None)` or `erp_standard(traj1, traj2, type_d="euclidean", g=None)`
* **Distance Types**: euclidean, spherical
* **Parameters**: `g` - Gap penalty value
* **Description**: Two implementations: one compatible with traj-dist (`erp_compat_traj_dist`) and a standard implementation (`erp_standard`).

## Discret Frechet Distance

Discrete Fréchet distance measures similarity between polygonal curves.

* **Function**: `discret_frechet(traj1, traj2, type_d="euclidean")`
* **Distance Types**: euclidean (spherical not supported)
* **Parameters**: None
* **Description**: Computes the discrete Fréchet distance, a measure of similarity for polygonal curves.

## Frechet Distance

Fréchet distance measures the similarity between curves by considering the location and ordering of points along the curves.

* **Function**: `frechet(traj1, traj2, type_d="euclidean")`
* **Distance Types**: euclidean (spherical not yet implemented)
* **Parameters**: None
* **Description**: Computes the Fréchet distance, the minimum length of a leash required to connect a dog and its owner walking along their respective curves.

## SOWD (One-Way Distance)

One-Way Distance measures the distance from one trajectory to another in a single direction.

* **Function**: `sowd_grid(traj1, traj2, type_d="euclidean", precision=None, converted=None)`
* **Distance Types**: spherical (euclidean not supported)
* **Parameters**: `precision` - Grid precision for computation
* **Description**: Calculates the one-way distance from one trajectory to another using a grid-based approach.