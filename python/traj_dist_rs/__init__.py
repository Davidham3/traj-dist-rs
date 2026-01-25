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
try:
    from ._lib import (discret_frechet, dtw, edr, erp_compat_traj_dist,
                       erp_standard, hausdorff, lcss, sspd)
except ImportError:
    # For documentation generation or when the Rust extension isn't built
    pass

# Add docstrings to imported functions to provide documentation in Python
if 'sspd' in locals():
    sspd.__doc__ = """Compute the SSPD (Symmetric Segment-Path Distance) between two trajectories.

    SSPD measures the distance between two trajectories by computing the average distance from 
    each point in one trajectory to the other trajectory, and then symmetrizing the result.

    Args:
        t1 (list[list[float]]): First trajectory as a list of [longitude, latitude] pairs.
        t2 (list[list[float]]): Second trajectory as a list of [longitude, latitude] pairs.
        dist_type (str): Distance type - "euclidean" for 2D Cartesian space or 
                        "spherical" for Great circle distance on Earth.

    Returns:
        float: SSPD distance between the two trajectories.

    Raises:
        ValueError: If an invalid distance type is provided.

    Examples:
        >>> import traj_dist_rs
        >>> t1 = [[0.0, 0.0], [1.0, 1.0]]
        >>> t2 = [[0.0, 1.0], [1.0, 0.0]]
        >>> distance = traj_dist_rs.sspd(t1, t2, "euclidean")
        >>> print(f"SSPD distance: {distance}")
    """

if 'dtw' in locals():
    dtw.__doc__ = """Compute the DTW (Dynamic Time Warping) distance between two trajectories.

    DTW finds the optimal alignment between two sequences by dynamically warping the time axis,
    allowing for similar patterns that are out of phase to be matched.

    Args:
        t1 (list[list[float]]): First trajectory as a list of [longitude, latitude] pairs.
        t2 (list[list[float]]): Second trajectory as a list of [longitude, latitude] pairs.
        dist_type (str): Distance type - "euclidean" for 2D Cartesian space or 
                        "spherical" for Great circle distance on Earth.

    Returns:
        float: DTW distance between the two trajectories.

    Raises:
        ValueError: If an invalid distance type is provided.

    Examples:
        >>> import traj_dist_rs
        >>> t1 = [[0.0, 0.0], [1.0, 1.0]]
        >>> t2 = [[0.0, 1.0], [1.0, 0.0]]
        >>> distance = traj_dist_rs.dtw(t1, t2, "euclidean")
        >>> print(f"DTW distance: {distance}")
    """

if 'hausdorff' in locals():
    hausdorff.__doc__ = """Compute the Hausdorff distance between two trajectories.

    The Hausdorff distance measures how far two subsets of a metric space are from each other.
    It is defined as the maximum of all distances from a point in one set to the closest point in the other set.

    Args:
        t1 (list[list[float]]): First trajectory as a list of [longitude, latitude] pairs.
        t2 (list[list[float]]): Second trajectory as a list of [longitude, latitude] pairs.
        dist_type (str): Distance type - "euclidean" for 2D Cartesian space or 
                        "spherical" for Great circle distance on Earth.

    Returns:
        float: Hausdorff distance between the two trajectories.

    Raises:
        ValueError: If an invalid distance type is provided.

    Examples:
        >>> import traj_dist_rs
        >>> t1 = [[0.0, 0.0], [1.0, 1.0]]
        >>> t2 = [[0.0, 1.0], [1.0, 0.0]]
        >>> distance = traj_dist_rs.hausdorff(t1, t2, "euclidean")
        >>> print(f"Hausdorff distance: {distance}")
    """

if 'lcss' in locals():
    lcss.__doc__ = """Compute the LCSS (Longest Common Subsequence) distance between two trajectories.

    LCSS is a similarity measure that finds the longest subsequence common to both trajectories.
    Two points are considered matching if their distance is less than a given threshold (epsilon).
    The LCSS distance is calculated as 1 - (length of longest common subsequence) / min(len(t0), len(t1)).

    Args:
        t1 (list[list[float]]): First trajectory as a list of [longitude, latitude] pairs.
        t2 (list[list[float]]): Second trajectory as a list of [longitude, latitude] pairs.
        dist_type (str): Distance type - "euclidean" for 2D Cartesian space or 
                        "spherical" for Great circle distance on Earth.
        eps (float): Epsilon threshold for considering two points as matching.

    Returns:
        float: LCSS distance between the two trajectories (value between 0 and 1).

    Raises:
        ValueError: If an invalid distance type is provided.

    Examples:
        >>> import traj_dist_rs
        >>> t1 = [[0.0, 0.0], [1.0, 1.0]]
        >>> t2 = [[0.0, 1.0], [1.0, 0.0]]
        >>> distance = traj_dist_rs.lcss(t1, t2, "euclidean", eps=0.5)
        >>> print(f"LCSS distance: {distance}")
    """

if 'edr' in locals():
    edr.__doc__ = """Compute the EDR (Edit Distance on Real sequence) distance between two trajectories.

    EDR is a distance measure for trajectories that allows for gaps in the matching.
    It uses a threshold `eps` to determine if two points match.
    The distance is normalized by the maximum length of the two trajectories.

    Args:
        t1 (list[list[float]]): First trajectory as a list of [longitude, latitude] pairs.
        t2 (list[list[float]]): Second trajectory as a list of [longitude, latitude] pairs.
        dist_type (str): Distance type - "euclidean" for 2D Cartesian space or 
                        "spherical" for Great circle distance on Earth.
        eps (float): Epsilon threshold for considering two points as matching.

    Returns:
        float: EDR distance between the two trajectories (normalized to [0, 1]).

    Raises:
        ValueError: If an invalid distance type is provided.

    Examples:
        >>> import traj_dist_rs
        >>> t1 = [[0.0, 0.0], [1.0, 1.0]]
        >>> t2 = [[0.0, 1.0], [1.0, 0.0]]
        >>> distance = traj_dist_rs.edr(t1, t2, "euclidean", eps=0.5)
        >>> print(f"EDR distance: {distance}")
    """

if 'discret_frechet' in locals():
    discret_frechet.__doc__ = """Compute the Discret Frechet distance between two trajectories.

    The discrete Frechet distance is a measure of similarity between two curves
    that takes into account the location and ordering of the points along the curves.
    Only Euclidean distance is supported for this algorithm.

    Args:
        t1 (list[list[float]]): First trajectory as a list of [longitude, latitude] pairs.
        t2 (list[list[float]]): Second trajectory as a list of [longitude, latitude] pairs.
        dist_type (str): Distance type - "euclidean" (only Euclidean is supported for Discret Frechet).

    Returns:
        float: Discrete Frechet distance between the two trajectories.

    Raises:
        ValueError: If an invalid distance type is provided (only "euclidean" is supported).

    Examples:
        >>> import traj_dist_rs
        >>> t1 = [[0.0, 0.0], [1.0, 1.0]]
        >>> t2 = [[0.0, 1.0], [1.0, 0.0]]
        >>> distance = traj_dist_rs.discret_frechet(t1, t2, "euclidean")
        >>> print(f"Discrete Frechet distance: {distance}")
    """

if 'erp_compat_traj_dist' in locals():
    erp_compat_traj_dist.__doc__ = """Compute the ERP (Edit distance with Real Penalty) distance between two trajectories.

    This is the **traj-dist compatible** implementation that matches the (buggy) implementation
    in traj-dist. This version should be used when compatibility with traj-dist is required.

    ERP is a distance measure for trajectories that uses a gap point `g` as a penalty
    for insertions and deletions. The distance is computed using dynamic programming.

    Note: This implementation has a bug in the DP matrix initialization where it uses
    the total sum of distances to g instead of cumulative sums. This matches the bug in
    traj-dist's Python implementation.

    Args:
        t1 (list[list[float]]): First trajectory as a list of [longitude, latitude] pairs.
        t2 (list[list[float]]): Second trajectory as a list of [longitude, latitude] pairs.
        dist_type (str): Distance type - "euclidean" for 2D Cartesian space or 
                        "spherical" for Great circle distance on Earth.
        g (list[float] or None): Gap point for penalty as [longitude, latitude] or None for centroid.

    Returns:
        float: ERP distance between the two trajectories.

    Raises:
        ValueError: If an invalid distance type is provided or if gap point has wrong dimensions.

    Examples:
        >>> import traj_dist_rs
        >>> t1 = [[0.0, 0.0], [1.0, 1.0]]
        >>> t2 = [[0.0, 1.0], [1.0, 0.0]]
        >>> distance = traj_dist_rs.erp_compat_traj_dist(t1, t2, "euclidean", g=[0.0, 0.0])
        >>> print(f"ERP distance (compat): {distance}")
    """

if 'erp_standard' in locals():
    erp_standard.__doc__ = """Compute the ERP (Edit distance with Real Penalty) distance between two trajectories.

    This is the **standard** ERP implementation with correct cumulative initialization.
    This version should be used for new applications where correctness is more important
    than compatibility with traj-dist.

    ERP is a distance measure for trajectories that uses a gap point `g` as a penalty
    for insertions and deletions. The distance is computed using dynamic programming.

    Note: This implementation correctly accumulates distances by index in the DP matrix
    initialization, unlike the buggy implementation in traj-dist.

    Args:
        t1 (list[list[float]]): First trajectory as a list of [longitude, latitude] pairs.
        t2 (list[list[float]]): Second trajectory as a list of [longitude, latitude] pairs.
        dist_type (str): Distance type - "euclidean" for 2D Cartesian space or 
                        "spherical" for Great circle distance on Earth.
        g (list[float] or None): Gap point for penalty as [longitude, latitude] or None for centroid.

    Returns:
        float: ERP distance between the two trajectories.

    Raises:
        ValueError: If an invalid distance type is provided or if gap point has wrong dimensions.

    Examples:
        >>> import traj_dist_rs
        >>> t1 = [[0.0, 0.0], [1.0, 1.0]]
        >>> t2 = [[0.0, 1.0], [1.0, 0.0]]
        >>> distance = traj_dist_rs.erp_standard(t1, t2, "euclidean", g=[0.0, 0.0])
        >>> print(f"ERP distance (standard): {distance}")
    """

__version__ = "0.1.0"
__author__ = "traj-dist-rs contributors"
__all__ = [
    "sspd",
    "dtw",
    "hausdorff",
    "lcss",
    "edr",
    "discret_frechet",
    "erp_compat_traj_dist",
    "erp_standard",
]
