"""
traj-dist-rs: High-performance trajectory distance calculations in Rust with Python bindings.

This package provides efficient implementations of various trajectory distance algorithms
including SSPD, DTW, Hausdorff, LCSS, EDR, ERP, and Discret Frechet.

All algorithms support both Euclidean (Cartesian) and Spherical (Haversine) distance calculations.
The underlying algorithms are implemented in Rust for optimal performance, with Python bindings
generated using PyO3.

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
    >>> distance = traj_dist_rs.dtw(t1, t2, "spherical")
    >>> print(f"DTW distance: {distance}")
"""

# Import all functions from the compiled Rust module
from ._lib import (DpResult, Metric, __dp_result_from_pickle, cdist,
                   discret_frechet, discret_frechet_with_matrix, dtw,
                   dtw_with_matrix, edr, edr_with_matrix, erp_compat_traj_dist,
                   erp_compat_traj_dist_with_matrix, erp_standard,
                   erp_standard_with_matrix, hausdorff, lcss, lcss_with_matrix,
                   pdist, sspd)

__version__ = "1.0.0-beta.1"
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
    "__dp_result_from_pickle",
]
