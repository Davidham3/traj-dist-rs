# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [1.0.0-beta.1] - 2026-02-25

### Added

#### Cross-Platform Support
- Complete multi-platform wheel support for Linux, macOS, and Windows
- Linux: Support for glibc (manylinux) C library
  - x86_64 and aarch64 architectures
  - 8 wheels total (manylinux only)
- macOS: Support for Intel (x86_64) and Apple Silicon (arm64) architectures
  - 8 wheels total
- Windows: Support for AMD64 architecture
  - 4 wheels total
- Total: 21 distribution packages (20 wheels + 1 source)

#### Testing Infrastructure
- Added optional test dependencies via `[project.optional-dependencies.test]`
  - polars>=1.8.2 for efficient parquet file reading
  - pydantic>=2.12.5 for data validation
  - pytest>=8.3.5 for test framework
- Removed pyarrow dependency from test requirements
  - Replaced with polars for better performance and smaller dependency footprint
  - Simplified test environment setup
  - Reduced wheel build complexity

#### Comprehensive Testing
- Full pytest integration for all platform builds
- 164 test cases covering:
  - All distance algorithms (SSPD, DTW, Hausdorff, LCSS, EDR, ERP, Discret Frechet)
  - Both Euclidean and Spherical distance types
  - Batch computation (pdist, cdist)
  - Pickle serialization
  - Precomputed distance matrices
  - Boundary cases and error handling
- Automated testing in CI for all platforms
- Test data automatically copied to build environment

#### GitHub Actions CI/CD
- Multi-platform wheel building workflow
- Automated testing on Linux, macOS, and Windows
- Support for Python 3.10, 3.11, 3.12, 3.13
- Configured cibuildwheel for reliable cross-platform builds

### Changed

#### Dependencies
- Removed pyarrow from test dependencies
  - Replaced with polars for parquet file reading in tests
  - Reduced build complexity and dependency size
  - Improved cross-platform compatibility
- Added optional test dependencies via `[project.optional-dependencies.test]`
  - Users can now install test dependencies with: `pip install traj-dist-rs[test]`
  - Cleaner separation between runtime and test dependencies

### Fixed

#### Test Dependencies
- Fixed test dependency issues in cross-platform builds
  - Resolved pyarrow compilation failures in manylinux containers
  - Used precompiled pyarrow in build environment when needed
  - Simplified test setup to avoid complex build-time dependencies

### Added

#### Internationalization
- Complete internationalization of the entire codebase
  - All documentation translated to English
  - All source code comments translated to English
  - All scripts and configuration files updated to English

#### Batch Computation
- `pdist()` function for pairwise distance computation (compressed matrix)
- `cdist()` function for distance matrix computation (full matrix)
- Support for all 7 distance algorithms in batch mode
- Strategy Pattern implementation with `Distance<T>` trait
- Type-safe parameter binding via `DistanceAlgorithm` enum
- Zero-copy NumPy array support for Python bindings

#### Architecture Improvements
- **Batch Computation Design (2026-02-12)**: Simplified architecture by removing `Distance<T>` trait
  - Changed from Strategy Pattern with generic `Distance<T>` trait to direct `Metric` type
  - `Metric::distance` changed from trait implementation to ordinary method
  - Simplified `pdist` and `cdist` function signatures:
    - From: `pdist<T, D>(..., calculator: &D) where D: Distance<T>`
    - To: `pdist<T>(..., metric: &Metric)`
  - Benefits: Reduced code complexity, lower understanding cost, maintained extensibility
  - Adding new algorithms now only requires:
    1. Adding variant to `DistanceAlgorithm` enum
    2. Adding match branch in `Metric::distance()` method
    3. Adding factory method in Python binding `PyMetric`
- Rayon's global thread pool for efficient parallelization
- Zero-copy NumPy array support
- Zero-copy PrecomputedDistanceCalculator (2026-02-12)
  - Changed `PrecomputedDistanceCalculator` to use flat array `&[f64]` instead of `Vec<Vec<f64>>`
  - Added `seq1_len` and `seq2_len` parameters to track matrix dimensions
  - Updated `seq1_extra_dists` and `seq2_extra_dists` to use `Option<&[f64]>`
  - Implemented zero-copy NumPy integration for all `*_with_matrix` functions
  - Flat indexing: `distance_matrix[idx1 * seq2_len + idx2]`
  - Benefits: Zero-copy, better performance, cache efficiency, consistent API
  - Affected functions: `dtw_with_matrix`, `lcss_with_matrix`, `edr_with_matrix`, `discret_frechet_with_matrix`, `erp_compat_traj_dist_with_matrix`, `erp_standard_with_matrix`

#### Parallel Processing
- Rayon integration for automatic parallelization
- Global thread pool for efficient parallel execution

#### Pickle Serialization
- Binary serialization support for `DpResult` objects using bincode
- `__reduce__` method implementation in `PyDpResult` class
- Compatible with `joblib.Parallel` for parallel processing
- High-performance binary serialization (better than JSON/serde_json)

#### Metric API
- `Metric` class with factory methods for type-safe configuration
- `Metric.sspd(type_d="euclidean")`
- `Metric.dtw(type_d="euclidean")`
- `Metric.hausdorff(type_d="euclidean")`
- `Metric.discret_frechet(type_d="euclidean")`
- `Metric.lcss(eps=5.0, type_d="euclidean")`
- `Metric.edr(eps=5.0, type_d="euclidean")`
- `Metric.erp(g=[0.0, 0.0], type_d="euclidean")`

#### Examples
- Comprehensive examples directory (`examples/`)
- 4 Python examples demonstrating all APIs:
  - `basic_distance.py` - Basic distance calculation examples
  - `batch_computation.py` - Batch computation (pdist/cdist) examples
  - `parallel_processing.py` - Parallel processing with joblib examples
  - `metric_api.py` - Metric API examples
- 3 Rust examples demonstrating all APIs:
  - `basic_distance.rs` - Basic distance calculation examples
  - `batch_computation.rs` - Batch computation examples with Rayon
  - `precomputed_matrix.rs` - Multiple distance calculations examples
- All examples verified and tested with 0 warnings

#### Testing
- 7 pickle serialization test cases added
- 22 batch computation test cases added
- Total: 305 tests passing (141 Rust + 157 Python + 7 pickle)
- All examples verified and tested

### Changed

#### Dependencies
- Added `rayon` (v1.8) for parallel processing (optional)
- Added `num_cpus` (v1.16) for CPU core detection (optional)
- Added `strum` (v0.26) for enum parsing and serialization
- Added `strum_macros` (v0.26) for enum derive macros
- Added `bincode` (v2) for binary serialization (better performance than serde_json)
- Added `rand` (v0.8) for example code generation (dev-dependency)
- All dependencies now use semantic versioning (`^`) for better safety and compatibility

#### Features
- `parallel` feature added to enable Rayon-based parallel processing
- `pyproject.toml` updated to include `parallel` feature in build configuration

#### Code Quality
- Resolved all warnings (0 warnings in build process)
- `strum` integration eliminated ~50 lines of duplicate parsing code
- Improved type safety and error messages
- All examples pass clippy with 0 warnings

### Fixed

#### Critical Bugs

#### Python Stub Files
- Fixed Metric class import issue causing Pylance "unknown import symbol" errors
  - Removed `module = "traj_dist_rs"` from `#[pyclass]` attributes
  - All types now automatically exposed in `_lib` module
  - Pylance type hints work correctly
- Fixed type annotation inconsistency in trajectory parameters
  - Updated all 7 algorithm functions to support `typing.List[typing.List[float]] | numpy.ndarray`
  - Users can now pass numpy arrays without Pylance errors
- Fixed pdist and cdist type annotations for better compatibility
  - Changed from `List` to `Sequence` (covariant type)
  - User's `list[ndarray]` can match `Sequence[List[List[float]] | ndarray]`
- Fixed critical .gitignore bug (2026-02-12)
  - `_lib.pyi` stub file was excluded from built wheel packages due to overly broad `.gitignore` rules
  - Removed problematic rules: `python/traj_dist_rs/_lib.*` and `python/traj_dist_rs/_lib/`
  - Wheel packages now include `_lib.pyi` for proper IDE type hints
  - Verified with `unzip -l` after building with `uvx cibuildwheel`
  - Ensures all users installing from PyPI or wheel packages get proper IDE support

#### Code Quality
- Fixed `useless_conversion` clippy warning in `src/binding/mod.rs`
  - Removed unnecessary `.into()` call
- Fixed `multiple_bound_locations` clippy warnings in `src/distance/batch.rs`
- Fixed boolean comparison warnings in benchmark scripts

### Documentation

#### pdist/cdist Symmetry
- Updated pdist docstring to clarify symmetry assumption
  - Assumes `distance(A, B) == distance(B, A)`
  - All standard algorithms are symmetric
  - Only computes upper triangle (i < j)
- Updated cdist docstring to explain when to use pdist vs cdist
  - Use cdist for asymmetric distances
  - Use cdist for cross-collection computations
  - Use pdist for symmetric distances (saves memory)

#### Development Tools
- Enhanced `pre_build.sh` to cover examples and scripts directories
  - Code formatting for `examples/python/` and `scripts/`
  - Code linting for all Python directories
  - Running all Python examples for validation
- Added `--release` mode to `maturin develop` for better performance
  - Example execution time reduced by 15-20x
  - Overall pre_build.sh time significantly reduced

### Performance

#### Batch Computation Performance (DTW, 5 trajectories, varying lengths [10, 100, 1000])

**pdist Performance (Cython vs Rust)**:
- Euclidean distance:
  - Rust sequential: Average 13.11x speedup (range: 8.02x - 15.76x)
  - Rust parallel: Average 31.36x speedup (range: 0.14x - 83.41x)
- Spherical distance:
  - Rust sequential: Average 1.80x speedup (range: 1.18x - 2.12x)
  - Rust parallel: Average 5.59x speedup (range: 0.21x - 10.99x)

**cdist Performance (Cython vs Rust, 5×5)**:
- Euclidean distance:
  - Rust sequential: Average 15.42x speedup (range: 15.20x - 15.85x)
  - Rust parallel: Average 25.70x speedup (range: 1.00x - 60.97x)
- Spherical distance:
  - Rust sequential: Average 3.87x speedup (range: 1.94x - 7.34x)
  - Rust parallel: Average 5.49x speedup (range: 3.41x - 7.16x)

**Key Findings**:
- Euclidean distance shows significant speedup benefits from Rust implementation
- Spherical distance has smaller speedup ratios due to complex haversine calculations
- Parallel execution provides better performance for larger trajectory lengths (length=1000)
- Parallel overhead is significant for small trajectories (length=10)
- Parallel efficiency improves with trajectory length: best results at length=1000
- Proper warmup (1 run) significantly improves measurement accuracy

**Overall Performance (Single trajectory pairs, K=1000, WARMUP_RUNS=10, NUM_RUNS=50)**:
- **Rust vs Python**: ~80x average performance improvement
- **Rust vs Cython**: ~3x average performance improvement

**By Distance Type**:
- Euclidean: Rust ~10x faster than Cython, ~378x faster than Python
- Spherical: Rust ~3x faster than Cython, ~83x faster than Python

### Migration Guide

#### Batch Computation

```python
import traj_dist_rs
import numpy as np

# Create trajectories
trajectories = [np.array([[0.0, 0.0], [1.0, 1.0]]) for _ in range(10)]

# Create metric configuration
metric = traj_dist_rs.Metric.sspd(type_d="euclidean")

# Compute pairwise distances (compressed matrix)
distances = traj_dist_rs.pdist(trajectories, metric=metric)

# Compute full distance matrix
distances_matrix = traj_dist_rs.cdist(trajectories, trajectories, metric=metric)

# Parallel computation
distances_parallel = traj_dist_rs.pdist(trajectories, metric=metric, parallel=True)
```

#### Parallel Processing with joblib

```python
import traj_dist_rs
from joblib import Parallel, delayed

def compute_distance(traj1, traj2):
    return traj_dist_rs.dtw(traj1, traj2, "euclidean", use_full_matrix=False)

# Parallel computation
trajs = [[[i*0.1, i*0.2] for i in range(3)] for _ in range(10)]
results = Parallel(n_jobs=-1)(
    delayed(compute_distance)(t1, t2) for i, t1 in enumerate(trajs) for j, t2 in enumerate(trajs)
)
```

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
result = traj_dist_rs.dtw(traj1, traj2, "euclidean", use_full_matrix=False)
print(f"DTW distance: {result.distance}")

# Batch computation
import numpy as np
trajectories = [np.array([[0.0, 0.0], [1.0, 1.0]]) for _ in range(10)]
metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
distances = traj_dist_rs.pdist(trajectories, metric=metric, parallel=True)
print(f"Computed {len(distances)} distances")
```

### Acknowledgments

- Original [traj-dist](https://github.com/bguillouet/traj-dist) library for algorithm reference and validation
- [PyO3](https://github.com/PyO3/pyo3) for Python bindings
- [Rayon](https://github.com/rayon-rs/rayon) for parallel processing
- The Rust community for excellent tooling and libraries

---

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

[Unreleased]: https://github.com/Davidham3/traj-dist-rs/compare/v1.0.0-beta.1...HEAD
[1.0.0]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v1.0.0
[1.0.0-beta.1]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v1.0.0-beta.1
[1.0.0-alpha.1]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v1.0.0-alpha.1
[0.1.0-alpha.1]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v0.1.0-alpha.1
