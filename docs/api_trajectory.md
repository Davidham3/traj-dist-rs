# API Reference - Trajectory Types

This section documents the trajectory-related types and input formats supported by `traj-dist-rs`.

## Input Format

The trajectory distance functions accept trajectory data in the following formats:

* List of coordinate pairs: `[[x1, y1], [x2, y2], ...]`
* NumPy arrays: `numpy.array([[x1, y1], [x2, y2], ...])`

Coordinate pairs should be in the format `[longitude, latitude]` for spherical distances,
or `[x, y]` for euclidean distances.