# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha.1] - 2026-02-05

### Added

#### Core Features
- Initial release of traj-dist-rs, a high-performance Rust implementation of trajectory distance algorithms
- Python bindings via PyO3 for seamless integration with Python projects
- Native Rust API for high-performance applications
- Support for 7 distance algorithms:
  - SSPD (Symmetric Segment-Path Distance)
  - DTW (Dynamic Time Warping)
  - Discret Frechet
  - Hausdorff
  - LCSS (Longest Common Subsequence)
  - EDR (Edit Distance on Real sequence)
  - ERP (Edit distance with Real Penalty) - both standard and traj-dist compatible variants
- Support for two distance types:
  - Euclidean distance for 2D coordinates
  - Spherical (Haversine) distance for geographic coordinates

#### Advanced Features
- Matrix return for DP-based algorithms (DTW, LCSS, EDR, ERP, Discret Frechet)
- Precomputed distance matrix support for efficient batch computations
- Comprehensive error handling for invalid inputs (NaN, Infinity, negative epsilon, etc.)
- Full Python type hints for better IDE support and autocompletion
- Support for numpy arrays and Python lists as trajectory inputs

#### Testing & Validation
- 36 Rust integration tests covering:
  - Boundary cases (empty trajectories, single-point trajectories)
  - Error handling (NaN, Infinity, invalid parameters)
  - Matrix return verification
  - Algorithm correctness
- 111 Python integration tests across all algorithms and distance types
- User acceptance tests (17 Python tests, 15 Rust tests)
- All algorithms verified against original traj-dist implementation with < 1e-8 error margin

#### Performance
- **~73x faster** than original Python implementation
- **~2.7x faster** than Cython implementation on average
- **~389x faster** than Python for Euclidean distance (up to 595x for DTW)
- **~76x faster** than Python for Spherical distance
- **~10x faster** than Cython for Euclidean distance (up to 16x for SSPD)

#### Build & Distribution
- Python 3.10, 3.11, 3.12, and 3.13 wheel packages for Linux
- PyO3 bindings with automatic type stub generation
- cibuildwheel configuration for multi-platform builds
- Pre-build script for automated testing and code quality checks

#### Documentation
- Comprehensive API documentation (Python and Rust)
- Installation guide with multiple methods
- Usage examples for both Python and Rust
- Algorithm documentation with mathematical details
- Performance benchmark report with detailed statistics
- User guides for Python and Rust developers

### Changed

#### Improvements
- Upgraded precision from float to double for spherical distance calculations (compared to original traj-dist)
- Enhanced error handling with descriptive error messages
- Optimized memory usage for large trajectory datasets

### Known Limitations

- Discret Frechet algorithm only supports Euclidean distance, not Spherical
- Frechet and SOWD algorithms are only available in Rust API (not exposed to Python)
- Batch distance calculation functions (cdist/pdist) are not yet implemented
- Currently only Linux wheels are available (Windows and macOS coming soon)

### Planned Features (Future Releases)

- [ ] Expose Frechet algorithm to Python
- [ ] Expose SOWD algorithm to Python
- [ ] Implement batch distance calculation (cdist/pdist)
- [ ] Windows and macOS wheel packages
- [ ] Parallel computation support
- [ ] SIMD optimizations for distance calculations
- [ ] Additional distance metrics
- [ ] Jupyter notebook tutorials
- [ ] Performance optimizations for specific use cases

### Performance Details

#### Algorithm Performance (Median Values, Euclidean Distance)

| Algorithm | Rust vs Python | Rust vs Cython |
|-----------|----------------|----------------|
| SSPD | 345.70x | 16.13x |
| DTW | 594.74x | 8.53x |
| Discret Frechet | 554.07x | 6.52x |
| Hausdorff | 320.79x | 14.07x |
| LCSS | 381.35x | 6.56x |
| EDR | 339.62x | 6.06x |
| ERP | 187.20x | 12.03x |

*For detailed performance analysis, see [docs/performance.md](docs/performance.md)*

### Installation

```bash
pip install traj-dist-rs
```

### Quick Example

```python
import traj_dist_rs

traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]

# Calculate DTW distance
result = traj_dist_rs.dtw(traj1, traj2, type_d="euclidean", use_full_matrix=False)
print(f"DTW distance: {result.distance}")
```

### Acknowledgments

- Original [traj-dist](https://github.com/bguillouet/traj-dist) library for algorithm reference and validation
- [PyO3](https://github.com/PyO3/pyo3) for Python bindings
- The Rust community for excellent tooling and libraries

---

[Unreleased]: https://github.com/Davidham3/traj-dist-rs/compare/v0.1.0-alpha.1...HEAD
[0.1.0-alpha.1]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v0.1.0-alpha.1