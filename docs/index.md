# traj-dist-rs Documentation

Welcome to the documentation for `traj-dist-rs`, a high-performance Rust reimplementation 
of the `traj-dist` Python package for trajectory distance calculations.

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
usage
api
algorithms
performance
```

## Installation

To install `traj-dist-rs`, simply run:

```bash
pip install traj-dist-rs
```

Or if you're building from source:

```bash
pip install maturin
maturin develop
```

## Usage

Basic usage example:

```python
import traj_dist_rs

# Example trajectories (Nx2 numpy arrays)
traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

# Calculate distance using different algorithms
distance = traj_dist_rs.sspd(traj1, traj2, type_d="euclidean")
print(f"SSPD distance: {distance}")
```

## API Reference

```{toctree}
:maxdepth: 2

api_distance
api_trajectory
```

## Algorithms

The package implements several trajectory distance algorithms:

* **SSPD** (Symmetric Segment-Path Distance)
* **DTW** (Dynamic Time Warping)
* **Hausdorff**
* **LCSS** (Longest Common Subsequence)
* **EDR** (Edit Distance on Real sequence)
* **ERP** (Edit distance with Real Penalty)
* **Discret Frechet**
* **Frechet**
* **SOWD** (One-Way Distance)

## Performance

For detailed performance benchmarks and comparisons, see the [Performance Report](performance).

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`