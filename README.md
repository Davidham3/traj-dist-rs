# traj-dist-rs

**High-performance trajectory distance & similarity measures in Rust and Python.**

A high-performance Rust implementation of trajectory distance algorithms with Python bindings, offering significant speed improvements over the original [traj-dist](https://github.com/bguillouet/traj-dist) library.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Rust Version](https://img.shields.io/badge/rust-1.85%2B-orange)](https://www.rust-lang.org/)

## About

**traj-dist-rs** is a high-performance trajectory distance calculation library written in Rust, providing both native Rust APIs and Python bindings via PyO3. It is based on the original [traj-dist](https://github.com/bguillouet/traj-dist) library with additional algorithms (e.g., EDwP), focusing on performance optimization and modern language features.

### Why traj-dist-rs?

- **Performance**: **~220x faster** than `traj-dist`'s Python implementation and **~6.5x faster** than Cython implementation on average
- **Batch Computation**: Native `pdist` and `cdist` functions with parallel support up to **130x** faster than `traj-dist`
- **Zero Dependencies**: Only requires **numpy >= 1.21** - no heavy dependencies like polars, pyarrow, pandas, or shapely
- **Safety**: Rust's memory safety guarantees eliminate common runtime errors
- **Cross-platform**: Supports Linux, macOS, and Windows with native binaries
- **Dual API**: Use it from Python or Rust with minimal overhead
- **Accuracy**: All algorithms verified against original implementation with < 1.5e-8 error margin

## Performance Overview

![traj-dist-rs benchmark speedup](mk_docs/docs/assets/benchmark_speedup_readme.svg)

**Median benchmark summary**:
- **~231x faster than `traj-dist (Python)`** on average
- **~15.3x faster than `traj-dist (Cython)`** on average
- Parallel batch `pdist` / `cdist` reaches up to **~61.1x speedup** on large inputs

See [performance.md](mk_docs/docs/performance.md) for the full benchmark report and additional plots.

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

## Quick Start

### Python

```python
import numpy as np
import traj_dist_rs

# Define trajectories as list of [x, y] coordinates or numpy arrays
traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

# Calculate SSPD distance
distance = traj_dist_rs.sspd(traj1, traj2, dist_type="euclidean")
print(f"SSPD distance: {distance}")

# Calculate DTW distance (returns DpResult with distance and optional matrix)
result = traj_dist_rs.dtw(traj1, traj2, dist_type="euclidean", use_full_matrix=False)
print(f"DTW distance: {result.distance}")

# Calculate Hausdorff distance
distance = traj_dist_rs.hausdorff(traj1, traj2, dist_type="spherical")
print(f"Hausdorff distance: {distance}")

# Batch computation with pdist (pairwise distances)
trajectories = [np.array([[0.0, 0.0], [1.0, 1.0]]) for _ in range(10)]
metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
distances = traj_dist_rs.pdist(trajectories, metric=metric, parallel=True)
print(f"Computed {len(distances)} pairwise distances")

# Cross-distance computation with cdist
dist_matrix = traj_dist_rs.cdist(trajectories[:5], trajectories[5:], metric=metric)
print(f"Distance matrix shape: {dist_matrix.shape}")
```

### Rust

```rust
use traj_dist_rs::distance::sspd::sspd;
use traj_dist_rs::distance::dtw::dtw;
use traj_dist_rs::distance::base::TrajectoryCalculator;
use traj_dist_rs::distance::distance_type::DistanceType;
use traj_dist_rs::distance::batch::{pdist, Metric, DistanceAlgorithm};

fn main() {
    let traj1 = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2 = vec![[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]];

    // Calculate SSPD distance
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    println!("SSPD distance: {}", dist);

    // Calculate DTW distance
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    println!("DTW distance: {}", result.distance);

    // Batch computation with pdist
    let trajectories = vec![
        vec![[0.0, 0.0], [1.0, 1.0]],
        vec![[0.0, 1.0], [1.0, 0.0]],
        vec![[0.5, 0.5], [1.5, 1.5]],
    ];
    let metric = Metric::new(DistanceAlgorithm::SSPD, DistanceType::Euclidean);
    let distances = pdist(&trajectories, &metric, true).unwrap();
    println!("Computed {} pairwise distances", distances.len());
}
```

## Installation

### From PyPI (Python)

```bash
pip install traj-dist-rs
```

**Minimal Dependencies**: traj-dist-rs only requires **numpy >= 1.21** to function. This makes it extremely lightweight and easy to install compared to alternatives that depend on pandas, shapely, or other heavy libraries.

### Requirements

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **NumPy**: >= 1.21 (the only runtime dependency)
- **Platform**: Linux, macOS, or Windows

### From crates.io (Python)

```bash
cargo add traj-dist-rs --features parallel
```

### Installation Options

**Basic Installation** (minimal dependencies):
```bash
pip install traj-dist-rs
```

**Installation with Test Dependencies** (for development):
```bash
pip install traj-dist-rs[test]
```

**From Source** (requires Rust toolchain):

**Prerequisites:**
- Rust 1.85 or later
- uv

**Build and install:**
```bash
# Clone the repository
git clone https://github.com/Davidham3/traj-dist-rs.git
cd traj-dist-rs

# Compile and install via uv
uv pip install .
```

**Rust-only build:**
```bash
cargo build --release --features parallel
```

## Performance

Compared to the original traj-dist implementation (based on median values from K=1000 trajectory pairs):

### Overall Performance

| Implementation | Average Speedup |
|---------------|-----------------|
| Rust vs Python | **~220x** faster |
| Rust vs Cython | **~6.5x** faster |

### By Distance Type

**Euclidean Distance:**
- Rust vs Python: **~329x** faster (range: 169x - 517x)
- Rust vs Cython: **~8.9x** faster (range: 6.3x - 12.9x)

**Spherical Distance:**
- Rust vs Python: **~93x** faster (range: 47x - 195x)
- Rust vs Cython: **~3.6x** faster (range: 2.3x - 6.8x)

### Batch Computation Performance

**pdist (DTW, 5 trajectories, varying lengths):**

| Trajectory Length | Rust Seq vs `traj-dist` | Rust Par vs `traj-dist` |
|-------------------|-------------------|-------------------|
| 10 points | 9.15x | 0.21x (parallel overhead) |
| 100 points | 11.58x | 9.73x |
| 1000 points | 12.47x | **71.24x** |

**cdist (DTW, 5×5, varying lengths):**

| Trajectory Length | Rust Seq vs `traj-dist` | Rust Par vs `traj-dist` |
|-------------------|-------------------|-------------------|
| 10 points | 10.77x | 0.55x (parallel overhead) |
| 100 points | 14.45x | 34.81x |
| 1000 points | 12.27x | **50.36x** |

**Real-world Example: TrajCL Data Preprocessing**
- Dataset: 7,000 trajectories (Porto dataset)
- Task: DTW distance matrix computation
- Performance: **31.8x** faster than traj-dist baseline (2933s → 92s)

*For detailed performance analysis with statistics, see [performance.md](mk_docs/docs/performance.md).*

## Documentation

- **Installation Guide**: [installation.md](mk_docs/docs/installation.md)
- **Python User Guide**: [usage_for_python.ipynb](mk_docs/docs/usage_for_python.ipynb)
- **Rust User Guide**: [usage_for_rust.ipynb](mk_docs/docs/usage_for_rust.ipynb)
- **Performance Report**: [performance.md](mk_docs/docs/performance.md)
- **Examples**: [examples/](examples/) - Python and Rust example code

## Testing

Run comprehensive integration tests:

```bash
bash scripts/pre_build.sh
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure they pass via `bash scripts/pre_build.sh`
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Workflow

For daily development, use the pre-build script:
```bash
bash scripts/pre_build.sh
```

This script will:
- Format Rust and Python code
- Run linting (clippy, ruff)
- Run all tests (Rust + Python)
- Generate Python stub files
- Build Python bindings

## Project Structure

```
traj-dist-rs/
├── src/
│   ├── distance/       # Distance algorithm implementations
│   ├── binding/        # Python bindings (PyO3)
│   └── lib.rs          # Library entry point
├── tests/              # Rust integration tests
├── py_tests/           # Python integration tests
├── python/             # Python package source
├── docs/               # Documentation
└── scripts/            # Build and utility scripts
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original [traj-dist](https://github.com/bguillouet/traj-dist) library for algorithm reference
- [PyO3](https://github.com/PyO3/pyo3) for Python bindings
- The Rust community for excellent tooling and libraries

## Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join discussions about usage and development
- **Documentation**: Check the [docs](docs/) directory for detailed guides
