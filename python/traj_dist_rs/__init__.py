"""
traj-dist-rs: High-performance trajectory distance & similarity measures in Rust with Python bindings.

This package provides efficient implementations of various trajectory distance and similarity algorithms
including SSPD, DTW, Hausdorff, LCSS, EDR, ERP, and Discret Frechet.

Trajectory similarity is often measured via trajectory distances such as DTW, LCSS, EDR, ERP, Hausdorff, and Fréchet.
These algorithms are widely used for trajectory similarity search, clustering, and pattern mining.

All algorithms support both Euclidean (Cartesian) and Spherical (Haversine) distance calculations.
The underlying algorithms are implemented in Rust for optimal performance, with Python bindings
generated using PyO3.

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
    >>> # Convert to similarity score
    >>> similarity = 1.0 / (1.0 + distance)
    >>> print(f"SSPD similarity: {similarity}")
    >>>
    >>> # Calculate DTW distance using Spherical distance (for geographic coordinates)
    >>> distance = traj_dist_rs.dtw(t1, t2, "spherical")
    >>> print(f"DTW distance: {distance}")
"""

# Import all functions from the compiled Rust module
from ._lib import (DpResult, Metric, cdist, discret_frechet,
                   discret_frechet_with_matrix, dtw, dtw_with_matrix, edr,
                   edr_with_matrix, erp_compat_traj_dist,
                   erp_compat_traj_dist_with_matrix, erp_standard,
                   erp_standard_with_matrix, hausdorff, lcss, lcss_with_matrix,
                   pdist, sspd)

__version__ = "1.0.0b4"
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
    "pdist",
    "cdist",
    "Metric",
    "DpResult",
]
