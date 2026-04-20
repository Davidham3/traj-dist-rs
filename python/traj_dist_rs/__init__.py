"""
traj-dist-rs: High-performance trajectory distance & similarity measures in Rust with Python bindings.

This package provides efficient implementations of various trajectory distance and similarity algorithms
including SSPD, DTW, Hausdorff, LCSS, EDR, ERP, Discret Frechet, Frechet, and EDwP.

Trajectory similarity is often measured via trajectory distances such as DTW, LCSS, EDR, ERP, Hausdorff, and Fréchet.
These algorithms are widely used for trajectory similarity search, clustering, and pattern mining.

Most algorithms support both Euclidean (Cartesian) and Spherical (Haversine) distance calculations.
Frechet and EDwP only support Euclidean distance.
The underlying algorithms are implemented in Rust for optimal performance, with Python bindings
generated using PyO3.

Supported Algorithms:
- SSPD (Symmetric Segment-Path Distance) - Euclidean & Spherical
- DTW (Dynamic Time Warping) - Euclidean & Spherical
- Hausdorff Distance - Euclidean & Spherical
- LCSS (Longest Common Subsequence) - Euclidean & Spherical
- EDR (Edit Distance on Real sequence) - Euclidean & Spherical
- ERP (Edit distance with Real Penalty) - Euclidean & Spherical
- Discret Frechet Distance - Euclidean & Spherical
- Frechet Distance (Continuous) - Euclidean only
- EDwP (Edit Distance with Projections) - Euclidean only

Batch Computation:
- pdist: Compute pairwise distances within a trajectory collection
- cdist: Compute distances between two trajectory collections
- Both support parallel processing and progress bar display

Use Cases:
- Trajectory similarity search and retrieval
- Nearest neighbor queries on trajectory databases
- Trajectory clustering and classification
- Mobility data analysis and GPS trace processing
- Route pattern mining
- Anomaly detection in movement data

Examples:
    >>> import traj_dist_rs
    >>>
    >>> # Define two trajectories as lists of [longitude, latitude] pairs
    >>> t1 = [[0.0, 0.0], [1.0, 1.0]]
    >>> t2 = [[0.0, 1.0], [1.0, 0.0]]
    >>>
    >>> # Calculate SSPD distance using Euclidean distance
    >>> distance = traj_dist_rs.sspd(t1, t2, "euclidean")
    >>> print(f"SSPD distance: {distance}")
    >>>
    >>> # Calculate DTW distance using Spherical distance (for geographic coordinates)
    >>> result = traj_dist_rs.dtw(t1, t2, "spherical")
    >>> print(f"DTW distance: {result.distance}")
    >>>
    >>> # Frechet distance (Euclidean only)
    >>> distance = traj_dist_rs.frechet(t1, t2)
    >>> print(f"Frechet distance: {distance}")
    >>>
    >>> # EDwP distance (Euclidean only)
    >>> distance = traj_dist_rs.edwp(t1, t2)
    >>> print(f"EDwP distance: {distance}")
    >>>
    >>> # Batch computation with Metric API
    >>> import numpy as np
    >>> trajectories = [np.array(t1), np.array(t2)]
    >>> metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
    >>> distances = traj_dist_rs.pdist(trajectories, metric=metric)
    >>> print(f"Pairwise distances: {distances}")
"""

from importlib.metadata import version

# Import all functions from the compiled Rust module
from ._lib import (DpResult, Metric, cdist, discret_frechet,
                   discret_frechet_with_matrix, dtw, dtw_with_matrix, edr,
                   edr_with_matrix, edwp, erp_compat_traj_dist,
                   erp_compat_traj_dist_with_matrix, erp_standard,
                   erp_standard_with_matrix, frechet, hausdorff, lcss,
                   lcss_with_matrix, pdist, sspd)

__version__ = version("traj-dist-rs")
__author__ = "traj-dist-rs contributors"
__all__ = [
    "sspd",
    "dtw",
    "dtw_with_matrix",
    "hausdorff",
    "lcss",
    "lcss_with_matrix",
    "edr",
    "edr_with_matrix",
    "discret_frechet",
    "discret_frechet_with_matrix",
    "erp_compat_traj_dist",
    "erp_compat_traj_dist_with_matrix",
    "erp_standard",
    "erp_standard_with_matrix",
    "edwp",
    "frechet",
    "pdist",
    "cdist",
    "Metric",
    "DpResult",
]
