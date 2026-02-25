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

## 🚀 Quick Start

### Python

```python
import traj_dist_rs
import numpy as np

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

## 📦 Installation

### From PyPI (Python)

```bash
pip install traj-dist-rs
```

**Minimal Dependencies**: traj-dist-rs only requires **numpy >= 1.21** to function. This makes it extremely lightweight and easy to install compared to alternatives that depend on pandas, shapely, or other heavy libraries.

### Requirements

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **NumPy**: >= 1.21 (the only runtime dependency)
- **Platform**: Linux, macOS, or Windows

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
- Rust 1.70 or later
- Python 3.10, 3.11, 3.12, or 3.13
- maturin

**Build and install:**
```bash
# Clone the repository
git clone <repository-url>
cd traj-dist-rs

# Install development dependencies
pip install maturin

# Build and install in development mode
maturin develop

# Or build a release wheel
maturin build --release
pip install target/wheels/*.whl
```

**Rust-only build:**
```bash
cargo build --release
```

## 📦 Dependency Comparison

One of the biggest advantages of traj-dist-rs is its minimal dependency footprint. Compare with alternatives:

| Library | Core Dependencies | Total Size* |
|---------|-----------------|-------------|
| **traj-dist-rs** | **numpy >= 1.21** | **~2 MB** |
| traj-dist | numpy, Cython, Shapely, geohash2, pandas, scipy | ~200 MB |
| Similar libraries | numpy, pandas, scikit-learn, etc. | ~300-500 MB |

*Estimated total size including all transitive dependencies

**Benefits of Minimal Dependencies:**
- ✅ **Faster Installation**: Only needs numpy, which is likely already installed
- ✅ **Smaller Disk Footprint**: ~2 MB vs ~200+ MB for alternatives
- ✅ **Fewer Conflicts**: Less likely to have version conflicts with other packages
- ✅ **Better for Production**: Smaller attack surface, faster startup time
- ✅ **Ideal for Containers**: Smaller Docker images, faster build times

## 📊 Performance

Compared to the original traj-dist implementation (based on median values from K=1000 trajectory pairs):

### Overall Performance

| Implementation | Average Speedup |
|---------------|-----------------|
| Rust vs Python | **~82x** faster |
| Rust vs Cython | **~3x** faster |

### By Distance Type

**Euclidean Distance:**
- Rust vs Python: **~388x** faster (range: 169x - 612x)
- Rust vs Cython: **~9.7x** faster (range: 6.2x - 13.7x)

**Spherical Distance:**
- Rust vs Python: **~87x** faster (range: 47x - 194x)
- Rust vs Cython: **~3.5x** faster (range: 1.8x - 8.6x)

### Best Performing Algorithms

**Rust vs Cython (Euclidean):**
- ERP: **12.15x** faster
- Hausdorff: **13.67x** faster
- SSPD: **12.65x** faster

**Rust vs Python (Euclidean):**
- DTW: **612x** faster
- Discret Frechet: **489x** faster
- SSPD: **386x** faster

### Batch Computation Performance

**pdist (DTW, 5 trajectories, varying lengths):**

| Trajectory Length | Rust Seq vs Cython | Rust Par vs Cython |
|-------------------|-------------------|-------------------|
| 10 points | 8.02x | 0.14x (parallel overhead) |
| 100 points | 15.55x | 10.52x |
| 1000 points | 15.76x | **83.41x** |

**cdist (DTW, 5×5, varying lengths):**

| Trajectory Length | Rust Seq vs Cython | Rust Par vs Cython |
|-------------------|-------------------|-------------------|
| 10 points | 15.85x | 1.00x (parallel overhead) |
| 100 points | 15.21x | 15.15x |
| 1000 points | 15.20x | **60.97x** |

**Real-world Example: TrajCL Data Preprocessing**
- Dataset: 7,000 trajectories (Porto dataset)
- Task: DTW distance matrix computation
- Performance: **18.8x** faster than traj-dist baseline (3111s → 166s)

*For detailed performance analysis with statistics, see [docs/performance.md](docs/performance.md).*

## 📚 Documentation

- **Installation Guide**: [docs/installation.md](docs/installation.md)
- **Usage Examples**: [docs/usage.md](docs/usage.md)
- **Python API**: [docs/api.md](docs/api.md)
- **Rust API**: [docs/user_guide_rust.md](docs/user_guide_rust.md)
- **Algorithm Details**: [docs/algorithms.md](docs/algorithms.md)
- **Performance Report**: [docs/performance.md](docs/performance.md)
- **Examples**: [examples/](examples/) - Python and Rust example code

## 🧪 Testing

### Python Tests

```bash
cd traj-dist-rs
uv sync --dev
pytest py_tests/
```

### Rust Tests

```bash
cd traj-dist-rs
cargo test
```

### Integration Tests

Run comprehensive integration tests:
```bash
bash scripts/pre_build.sh
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure they pass
5. Format your code (`cargo fmt` for Rust, `black` for Python)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

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

## 🔧 Project Structure

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original [traj-dist](https://github.com/bguillouet/traj-dist) library for algorithm reference
- [PyO3](https://github.com/PyO3/pyo3) for Python bindings
- The Rust community for excellent tooling and libraries

## 📮 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join discussions about usage and development
- **Documentation**: Check the [docs](docs/) directory for detailed guides