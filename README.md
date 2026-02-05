# traj-dist-rs

A high-performance Rust implementation of trajectory distance algorithms with Python bindings, offering significant speed improvements over the original [traj-dist](https://github.com/bguillouet/traj-dist) library.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Rust Version](https://img.shields.io/badge/rust-1.70%2B-orange)](https://www.rust-lang.org/)

## 📖 About

**traj-dist-rs** is a high-performance trajectory distance calculation library written in Rust, providing both native Rust APIs and Python bindings via PyO3. It is a complete rewrite of the original [traj-dist](https://github.com/bguillouet/traj-dist) library, focusing on performance optimization and modern language features.

### Why traj-dist-rs?

- **🚀 Performance**: **~73x faster** than Python implementation and **~2.7x faster** than Cython implementation on average
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

### Additional Features

- Matrix return for DP-based algorithms (DTW, LCSS, EDR, ERP, Discret Frechet)
- Precomputed distance matrix support for efficient batch computations
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
```

### Rust

```rust
use traj_dist_rs::distance::sspd::sspd;
use traj_dist_rs::distance::dtw::dtw;
use traj_dist_rs::distance::base::TrajectoryCalculator;
use traj_dist_rs::distance::distance_type::DistanceType;

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
}
```

## 📦 Installation

### From PyPI (Python)

```bash
pip install traj-dist-rs
```

### From Source

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

## 📊 Performance

Compared to the original traj-dist implementation (based on median values from K=1000 trajectory pairs):

### Overall Performance

| Implementation | Average Speedup |
|---------------|-----------------|
| Rust vs Python | **~73x** faster |
| Rust vs Cython | **~2.7x** faster |

### By Distance Type

**Euclidean Distance:**
- Rust vs Python: **~389x** faster (range: 187x - 595x)
- Rust vs Cython: **~10x** faster (range: 6x - 16x)

**Spherical Distance:**
- Rust vs Python: **~76x** faster (range: 41x - 167x)
- Rust vs Cython: **~2.7x** faster (range: 1.6x - 5.2x)

### Best Performing Algorithms

**Rust vs Cython (Euclidean):**
- SSPD: **16.13x** faster
- Hausdorff: **14.07x** faster
- ERP: **12.03x** faster

**Rust vs Python (Euclidean):**
- DTW: **595x** faster
- Discret Frechet: **554x** faster
- LCSS: **381x** faster

*For detailed performance analysis with statistics, see [docs/performance.md](docs/performance.md).*

## 📚 Documentation

- **Installation Guide**: [docs/installation.md](docs/installation.md)
- **Usage Examples**: [docs/usage.md](docs/usage.md)
- **Python API**: [docs/api.md](docs/api.md)
- **Rust API**: [docs/user_guide_rust.md](docs/user_guide_rust.md)
- **Algorithm Details**: [docs/algorithms.md](docs/algorithms.md)
- **Performance Report**: [docs/performance.md](docs/performance.md)

## 🧪 Testing

### Python Tests

```bash
cd traj-dist-rs
pip install pytest numpy polars pyarrow
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

## 🗺️ Roadmap

For information about upcoming features and releases, see [roadmap_0.1.0a1.md](../roadmap_0.1.0a1.md).

---

**Version**: 0.1.0-alpha.1  
**Last Updated**: 2026-02-05