# traj-dist-rs

A high-performance Rust implementation of trajectory distance algorithms with Python bindings, offering significant speed improvements over the original [traj-dist](https://github.com/bguillouet/traj-dist) library.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Rust Version](https://img.shields.io/badge/rust-1.70%2B-orange)](https://www.rust-lang.org/)

## 📖 About

**traj-dist-rs** is a high-performance trajectory distance calculation library written in Rust, providing both native Rust APIs and Python bindings via PyO3. It is a complete rewrite of the original [traj-dist](https://github.com/bguillouet/traj-dist) library, focusing on performance optimization and modern language features.

### Why traj-dist-rs?

- **🚀 Performance**: **~82x faster** than Python implementation and **~3x faster** than Cython implementation on average
- **⚡ Batch Computation**: Native `pdist` and `cdist` functions with parallel support up to **130x** faster than `traj-dist`
- **📦 Zero Dependencies**: Only requires **numpy >= 1.21** - no heavy dependencies like polars, pyarrow, pandas, or shapely
- **🔒 Safety**: Rust's memory safety guarantees eliminate common runtime errors
- **📦 Cross-platform**: Supports Linux, macOS, and Windows with native binaries
- **🔗 Dual API**: Use it from Python or Rust with minimal overhead
- **🎯 Accuracy**: All algorithms verified against original implementation with < 1e-8 error margin

## ✨ Features

### Supported Distance Algorithms

- **SSPD** - Symmetric Segment-Path Distance
- **DTW** - Dynamic Time Warping (with optional matrix return)
- **Discret Frechet** - Discrete Fréchet Distance
- **Hausdorff** - Hausdorff Distance
- **LCSS** - Longest Common Subsequence
- **EDR** - Edit Distance on Real sequence
- **ERP** - Edit distance with Real Penalty (standard & traj-dist compatible)

### Distance Types

- **Euclidean** - 2D Euclidean distance
- **Spherical** - Haversine distance for geographic coordinates

### Batch Computation

- **`pdist`** - Pairwise distance matrix for trajectory collections (compressed format)
- **`cdist`** - Cross-distance matrix between two trajectory collections
- **Parallel processing** - Automatic parallelization using Rayon for large datasets
- **Metric API** - Type-safe configuration with factory methods

### Additional Features

- Matrix return for DP-based algorithms (DTW, LCSS, EDR, ERP, Discret Frechet)
- Precomputed distance matrix support for efficient batch computations
- Zero-copy NumPy array support for optimal performance
- Pickle serialization for `DpResult` objects (compatible with joblib)
- Comprehensive error handling for invalid inputs
- Full Python type hints for better IDE support