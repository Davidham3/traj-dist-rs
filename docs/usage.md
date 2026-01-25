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

# Usage Examples

## Basic Usage

Let's start with basic usage of the traj-dist-rs package:

```{code-cell} ipython3
import traj_dist_rs
import numpy as np

# Define two trajectories as lists of [x, y] coordinates
traj1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
traj2 = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])

print("Trajectory 1:", traj1)
print("Trajectory 2:", traj2)
```

Now we can calculate different trajectory distances:

```{code-cell} ipython3
# Calculate different trajectory distances
sspd_dist = traj_dist_rs.sspd(traj1, traj2, "euclidean")
dtw_dist = traj_dist_rs.dtw(traj1, traj2, "euclidean")
hausdorff_dist = traj_dist_rs.hausdorff(traj1, traj2, "euclidean")

print(f"SSPD distance: {sspd_dist}")
print(f"DTW distance: {dtw_dist}")
print(f"Hausdorff distance: {hausdorff_dist}")
```

## Using Spherical Distance

For geographic coordinates (latitude/longitude), use spherical distance:

```{code-cell} ipython3
import traj_dist_rs
import numpy as np

# Geographic coordinates as [latitude, longitude]
geo_traj1 = np.array([[40.7128, -74.0060], [40.7589, -73.9851], [40.7831, -73.9712]])  # NYC trajectory
geo_traj2 = np.array([[40.7228, -74.0160], [40.7689, -73.9951], [40.7931, -73.9812]])  # Similar NYC trajectory

print("Geographic Trajectory 1 (NYC):", geo_traj1)
print("Geographic Trajectory 2 (Similar to NYC):", geo_traj2)
```

Calculate distances using spherical distance (Haversine formula):

```{code-cell} ipython3
# Calculate distances using spherical distance (Haversine formula)
sspd_dist = traj_dist_rs.sspd(geo_traj1, geo_traj2, "spherical")
dtw_dist = traj_dist_rs.dtw(geo_traj1, geo_traj2, "spherical")

print(f"SSPD spherical distance: {sspd_dist}")
print(f"DTW spherical distance: {dtw_dist}")
```

## Using Algorithms with Parameters

Some algorithms require additional parameters:

```{code-cell} ipython3
import traj_dist_rs
import numpy as np

traj1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
traj2 = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])

print("Using trajectories with parameters:")
print("Trajectory 1:", traj1)
print("Trajectory 2:", traj2)
```

```{code-cell} ipython3
# LCSS with epsilon parameter - note the parameter order: t1, t2, dist_type, eps
lcss_dist = traj_dist_rs.lcss(traj1, traj2, "euclidean", 0.5)

# EDR with epsilon parameter - note the parameter order: t1, t2, dist_type, eps
edr_dist = traj_dist_rs.edr(traj1, traj2, "euclidean", 0.5)

# ERP with g parameter - note the parameter order: t1, t2, dist_type, g
erp_dist = traj_dist_rs.erp_compat_traj_dist(traj1, traj2, "euclidean", [0.0, 0.0])

print(f"LCSS distance: {lcss_dist}")
print(f"EDR distance: {edr_dist}")
print(f"ERP distance: {erp_dist}")
```

## Pairwise Distance Calculations

```{code-cell} ipython3
import traj_dist_rs
import numpy as np

# List of trajectories
trajectories = [
    np.array([[0.0, 0.0], [1.0, 1.0]]),
    np.array([[0.1, 0.1], [1.1, 1.1]]),
    np.array([[0.2, 0.2], [1.2, 1.2]])
]

print("List of trajectories:")
for i, traj in enumerate(trajectories):
    print(f"  Trajectory {i+1}: {traj}")

# Note: Pairwise distance calculations (cdist/pdist) are not yet implemented in current version
# This will be available in future releases
```