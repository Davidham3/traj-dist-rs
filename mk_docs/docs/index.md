# traj-dist-rs

**High-performance trajectory distance & similarity measures in Rust and Python.**

A high-performance Rust implementation of trajectory distance algorithms with Python bindings, offering significant speed improvements over the original [traj-dist](https://github.com/bguillouet/traj-dist) library.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Rust Version](https://img.shields.io/badge/rust-1.85%2B-orange)](https://www.rust-lang.org/)

## About

**traj-dist-rs** is a high-performance trajectory distance calculation library written in Rust, providing both native Rust APIs and Python bindings via PyO3. It is based on the original [traj-dist](https://github.com/bguillouet/traj-dist) library with additional algorithms (e.g., EDwP), focusing on performance optimization and modern language features.

### Why traj-dist-rs?

- **Performance**: **~231x faster** than Python implementation and **~15.3x faster** than Cython implementation on average
- **Batch Computation**: Native `pdist` and `cdist` functions with parallel support up to **61.1x** faster than `traj-dist`
- **Zero Dependencies**: Only requires **numpy >= 1.21** - no heavy dependencies like polars, pyarrow, pandas, or shapely
- **Safety**: Rust's memory safety guarantees eliminate common runtime errors
- **Cross-platform**: Supports Linux, macOS, and Windows with native binaries
- **Dual API**: Use it from Python or Rust with minimal overhead
- **Accuracy**: All algorithms verified against original implementation with < 1e-8 error margin

## Features

### Supported Distance Algorithms

| Algorithm | Full Name | Best For |
|-----------|-----------|----------|
| **SSPD** | Symmetric Segment-Path Distance | General similarity, noise tolerance |
| **DTW** | Dynamic Time Warping | Similarity with time warping, flexible alignment |
| **Frechet** | Fréchet Distance (Continuous) | Exact geometric similarity, considers all curve points |
| **Discret Frechet** | Discrete Fréchet Distance | Geometric similarity, path-based matching |
| **Hausdorff** | Hausdorff Distance | Maximum distance, outlier-sensitive similarity |
| **LCSS** | Longest Common Subsequence | Robust similarity with noise tolerance |
| **EDR** | Edit Distance on Real sequence | Similarity with noise and outlier tolerance |
| **ERP** | Edit distance with Real Penalty | Robust similarity with gap handling |
| **EDwP** | Edit Distance with Projections | Inconsistent sampling rates, projection-based matching |

### Distance Types

- **Euclidean** - 2D Euclidean distance (all algorithms)
- **Spherical** - Haversine distance for geographic coordinates (all algorithms except Frechet, Discret Frechet, and EDwP)

### Batch Computation

- **`pdist`** - Pairwise distance matrix for trajectory collections (compressed format)
- **`cdist`** - Cross-distance matrix between two trajectory collections
- **Parallel processing** - Automatic parallelization using Rayon for large datasets
- **Metric API** - Type-safe configuration with factory methods

### Additional Features

- Matrix return for DP-based algorithms (DTW, LCSS, EDR, ERP, Discret Frechet, EDwP)
- Precomputed distance matrix support for efficient batch computations
- Zero-copy NumPy array support for optimal performance
- Pickle serialization for `DpResult` objects (compatible with joblib)
- Comprehensive error handling for invalid inputs
- Full Python type hints for better IDE support

### Keywords / Search Terms

Common search terms related to this library:

- **Core concepts**: trajectory similarity, trajectory distance, similarity measures, trajectory analysis
- **Algorithms**: DTW, LCSS, EDR, ERP, Fréchet distance, Hausdorff distance, SSPD
- **Applications**: trajectory clustering, trajectory similarity search, nearest neighbor retrieval, mobility data analysis, GPS trace analysis
- **Domains**: time series similarity, spatiotemporal data, movement pattern mining, anomaly detection

### Migration from traj-dist

If you used the original `traj-dist` library for trajectory similarity measurement, this library is compatible and offers significant performance improvements:

- **Algorithm compatibility**: Core algorithms (SSPD, DTW, Hausdorff, LCSS, EDR, ERP, Frechet, Discret Frechet) supported, plus EDwP (not in original traj-dist)
- **Performance**: 3-10x faster than Cython implementation
