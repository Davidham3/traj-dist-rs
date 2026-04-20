# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [1.0.0-rc.5] - 2026-04-20

### Added

#### Batch Computation
- Added non-TTY environment support for progress display in `pdist()` and `cdist()`
  - Automatically falls back to periodic logging (every 10 seconds) when `stderr` is not attached to a terminal
  - Uses a lightweight background monitor thread and atomic counters for thread-safe progress tracking
  - Prevents log spam in CI/CD pipelines or redirected outputs while still providing computation status

### Fixed

#### Documentation Updates
- Fixed `pdist` and `cdist` doc comments in Rust source (`src/binding/batch.rs`) to include missing `show_progress` parameter documentation
- Regenerated Python stub file (`_lib.pyi`) to reflect the updated API documentation
- Updated `__init__.py` module docstring:
  - Added Frechet and EDwP to the supported algorithms list
  - Documented distance type support per algorithm (Euclidean & Spherical vs Euclidean only)
  - Added Batch Computation section describing `pdist`/`cdist` with parallel and progress bar support
  - Added usage examples for `frechet()`, `edwp()`, and batch computation with `Metric` API

---

## [1.0.0-rc.4] - 2026-04-19

### Added

#### New Algorithm: Frechet Distance (Continuous)
- Implemented continuous Fréchet distance algorithm in Rust (`src/distance/frechet.rs`)
  - Considers all continuous points along curve segments (unlike Discrete Fréchet which only considers vertices)
  - Uses binary search over critical values with a free-space decision procedure
  - Result is always ≤ the Discrete Fréchet distance for the same trajectories
  - Only supports Euclidean distance (by design, as the algorithm relies on circle-line intersections)
  - No Python implementation in upstream traj-dist (only Cython); no Haversine/Spherical support
  - Time complexity: O(n*m * log(n*m))
- Python bindings for Frechet (`src/binding/distance/frechet.rs`)
  - `traj_dist_rs.frechet(traj1, traj2)` returns `float`
  - Supports both list and NumPy array inputs
- Batch computation support via `Metric.frechet()` factory method
  - Works with `pdist()` and `cdist()` for batch distance computation

#### New Algorithm: EDwP (Edit Distance with Projections)
- Implemented EDwP algorithm in Rust (`src/distance/edwp.rs`)
  - Designed for trajectories with inconsistent sampling rates
  - Uses point-to-segment projections to handle different sampling densities
  - Only supports Euclidean distance (by design, as projections are geometry-dependent)
  - Supports both full DP matrix return and memory-optimized rolling array mode
- Python bindings for EDwP (`src/binding/distance/edwp.rs`)
  - `traj_dist_rs.edwp(traj1, traj2, use_full_matrix=False)` returns `DpResult`
  - Supports both list and NumPy array inputs
- Batch computation support via `Metric.edwp()` factory method

#### Examples & Scripts
- Added Frechet and EDwP to batch computation examples (`batch_computation.rs`, `batch_computation.py`)
- Added Rust examples execution to `pre_build.sh` (Step 14-15)
- Added Frechet to `benchmark_guide.md` supported algorithms list
  - Works with `pdist()` and `cdist()` for batch distance computation
- Python reference implementation from [TrajCL](https://github.com/changyanchuan/TrajCL) included in `scripts/benchmark/extra_algos/edwp.py` for validation

#### Progress Display for Batch Computation
- Added `show_progress` parameter to `pdist()` and `cdist()` (both Rust and Python APIs)
  - Displays a real-time progress bar during batch distance computation using `indicatif` crate
  - Progress bar is rendered to stderr to avoid interfering with stdout
  - Implemented at the Rust layer to preserve Rayon's parallel processing capability
  - Uses `ParallelProgressIterator` for thread-safe progress updates in parallel mode
  - Progress granularity: per distance pair for `pdist`, per row (`n_b` distances) for `cdist`
  - Progress bar format: `pdist [SSPD/euclidean]  ████████░░░░  45/100  [00:02:00<00:01:00, 22.5/s]`
  - Zero-overhead design: no `ProgressBar` created when `show_progress=false`
  - Conditional compilation via `progress` feature flag for zero dependency cost when disabled
  - Python binding uses `py.detach()` to release GIL, allowing Rust to freely parallelize and output progress
- Added `progress` feature flag in `Cargo.toml`
  - `progress = ["indicatif"]` — opt-in progress bar support
  - Automatically enabled by `python-binding` feature
- Added `Display` trait implementation for `Metric` type
  - Formats as `"AlgorithmName/distance_type"` (e.g., `"SSPD/euclidean"`)
  - Used in progress bar prefix for clear identification


#### Utility Functions
- Added `project_point_to_segment()` in `distance::euclidean` module
  - Projects a point onto a line segment, returning the closest point on the segment
  - Supports independent generic type parameters (`<C, D, E: AsCoord>`) for flexible usage with mixed coordinate types
  - Used by EDwP algorithm for point-to-segment projection calculations
- Added `point_to_segment_distance()` in `distance::euclidean` module
  - Simplified version without precomputed parameters, used by Frechet algorithm
  - Uses threshold `u <= 0.00001` matching the original traj-dist implementation
- Added `circle_line_intersection()` in `distance::euclidean` module
  - Computes intersection points between a circle and a line
  - Used by Frechet algorithm for free-space computation

### Changed

#### Dependencies
- Added `indicatif` (v0.17, with `rayon` feature) as optional dependency for progress bar display
- Added `progress` feature flag: `progress = ["indicatif"]`
- `python-binding` feature now automatically enables `progress`
- Updated `pyproject.toml` maturin features to include `"progress"`

#### API Changes
- `pdist()` and `cdist()` signatures updated with new `show_progress` parameter (default: `false`)
  - Rust: `pdist(trajectories, metric, parallel, show_progress)`
  - Python: `pdist(trajectories, metric, parallel=True, show_progress=False)`
- Replaced deprecated `py.allow_threads()` with `py.detach()` in Python bindings (PyO3 0.26+)


#### Code Quality
- EDwP projection logic uses shared `project_point_to_segment()` function instead of inline implementation
  - Eliminates code duplication across 4 projection sites (2 in full_matrix, 2 in rolling_array)
  - Ensures consistent projection behavior across the codebase

#### Documentation
- Updated `README.md` and `mk_docs/docs/index.md` algorithm compatibility list to include Frechet and EDwP
- Updated `src/lib.rs` module-level documentation to list all 9 supported algorithms including Frechet and EDwP
- Updated `examples/python/basic_distance.py` and `examples/basic_distance.rs` with Frechet usage examples
- Updated `mk_docs/docs/performance.md` Algorithms covered list to include FRECHET

#### Benchmark Scripts
- Added Frechet to `scripts/benchmark/algorithms_config.json`
- Added Frechet function mapping in `benchmark_traj_dist.py` (Cython: `tdist.c_frechet`) and `benchmark_traj_dist_rs.py`
- Added `FRECHET` to `ALGORITHM_ORDER` and `norm_algo()` in `analyze_benchmark_results.py`
- Added FRECHET to `render_scope()` Algorithms covered list in `analyze_benchmark_results.py`
- Fixed `plot_overview()` NaN annotation handling for algorithms with partial baseline coverage (Frechet has no Python impl, EDwP has no Cython impl)

### Testing

#### EDwP Test Coverage
- Rust unit tests: empty trajectories, single-point, identical, symmetry, matrix dimensions, matrix consistency
- Rust integration tests in `tests/algorithms.rs`, `tests/boundary_cases.rs`, `tests/matrix_return.rs`
- Python integration tests (`py_tests/test_edwp.py`) validated against Python reference implementation with 0 error margin
- EDwP test case generation script (`scripts/generate_edwp_test_cases.py`)

#### Frechet Test Coverage
- Cython test samples generated (`py_tests/data/cython_samples/frechet_euclidean.parquet`, 1225 trajectory pairs)
- Python integration tests (`py_tests/test_frechet.py`): accuracy validation, mathematical properties, boundary cases, batch computation
- Frechet test case generation added to `scripts/generate_all_test_cases.sh` and `scripts/generate_test_case.py`
- Test framework (`py_tests/test_framework.py`) updated with Frechet-specific handling (no `type_d` parameter, returns `float` directly)

---

## [1.0.0-rc.3] - 2026-03-30

### Added

#### Performance
- Added `SphericalTrajectoryCache` struct in `distance::spherical` module for precomputing per-trajectory spherical distance data
  - Precomputes longitude in radians (`lng_rad`), latitude in radians (`lat_rad`), and cosine of latitude (`cos_lat`) for each trajectory point
  - Eliminates redundant trigonometric calculations when computing many distances between the same trajectory pair
  - Reduces computation per distance call from 6 trig ops to 2 (only `sin(Δlat/2)` and `sin(Δlon/2)`)
- Added `great_circle_distance_cached()` inline function for cache-accelerated Haversine distance computation
- Added `point_to_path_cached()` for cache-accelerated point-to-segment distance used by SSPD and Hausdorff

#### Optimization
- `TrajectoryCalculator` now automatically creates `SphericalTrajectoryCache` for both trajectories when `DistanceType::Spherical` is selected
  - All 5 DP-based algorithms (DTW, LCSS, EDR, ERP, Discret Frechet) benefit automatically with no API changes
- SSPD and Hausdorff algorithms now use `SphericalTrajectoryCache` internally for spherical distance computation
- `precompute_distance_matrix()` in `distance::utils` now uses cache for spherical distance computation
- Optimized `great_circle_distance()` to use `x * x` multiplication instead of `powi(2)` for minor additional gains

#### API
- Re-exported `DistanceCalculator` trait from `traj_dist_rs::traits` module for convenient access
  - Trait definition remains in `traj_dist_rs::distance::base` (no breaking change)
  - Users can now import via `use traj_dist_rs::traits::DistanceCalculator`

#### Documentation
- Added "Performance Optimization: SphericalTrajectoryCache" section to `usage_for_rust.ipynb`
- Added "Core Traits and Types" section to `usage_for_rust.ipynb` covering all 5 key types:
  - `AsCoord` — coordinate representation trait with usage demo
  - `CoordSequence` — trajectory sequence trait with iteration demo
  - `DistanceCalculator` — DP algorithm interface, demonstrated via `TrajectoryCalculator` and `PrecomputedDistanceCalculator`
  - `TrajectoryCalculator` — on-the-fly distance computation with Euclidean/Spherical support
  - `PrecomputedDistanceCalculator` — zero-copy precomputed matrix lookup
- Added module-level documentation (`//!`) to previously undocumented modules:
  - `traj_dist_rs::err` — error types and categories
  - `traj_dist_rs::traits` — core traits overview with cross-references
  - `traj_dist_rs::distance::base` — DP algorithm architecture and usage example
  - `traj_dist_rs::distance::euclidean` — formula and function list
  - `traj_dist_rs::distance::spherical` — Haversine formula and coordinate format

### Changed

#### Performance Results
- Spherical distance average speedup vs Cython improved from ~3.1x to **~3.6x** (+16%)
- Individual algorithm improvements vs previous benchmark:
  - DTW (spherical): 2.97x → 3.40x (+14%)
  - SSPD (spherical): 1.72x → 2.25x (+31%)
  - Hausdorff (spherical): 2.41x → 2.91x (+21%)
  - LCSS (spherical): 2.68x → 3.00x (+12%)
  - EDR (spherical): 2.58x → 3.32x (+29%)
  - ERP (spherical): 6.30x → 6.77x (+7%)
- Batch `cdist` spherical (length=1000): 7.73x → **12.01x** (+55%) parallel speedup vs Cython
- Batch `pdist` spherical (length=1000): 10.38x → **17.81x** (+72%) parallel speedup vs Cython

#### Documentation
- Updated `README.md` Performance section with new benchmark numbers
- Fixed incorrect link in `README.md`
- Added visualization charts to `performance.md` and `README.md`
- Removed overly detailed benchmark data from `performance.md` to improve readability
- Streamlined performance documentation to focus on key results and visual summaries

---

## [1.0.0-rc.2] - 2026-03-17

### Added

#### Development Environment
- Added `.devcontainer` configuration for GitHub Codespaces and VS Code Remote - Containers
  - Dockerfile based on Ubuntu with pre-installed development tools
  - Rust toolchain (stable) with rustup
  - uv package manager for Python dependency management
  - Pre-configured VS Code extensions (rust-analyzer, Python, Ruff, Even Better TOML)
  - Automated environment setup with postCreateCommand

#### Documentation
- Added Google Search Console verification meta tag for SEO
  - Meta tag added to MkDocs main.html override template
  - Enables Google Search Console ownership verification

### Changed

#### Linux Wheel Compatibility
- Updated `build-wheels-reusable.yml` to use manylinux2014 image for Linux builds
  - Added `CIBW_MANYLINUX_X86_64_IMAGE: manylinux2014` for x86_64 builds
  - Added `CIBW_MANYLINUX_AARCH64_IMAGE: manylinux2014` for arm64 builds
  - Wheel packages downgraded from manylinux_2_28 to manylinux_2_17
  - Ensures compatibility with older Linux distributions (glibc 2.17+)
  - Supported distributions now include CentOS 7, RHEL 7, Ubuntu 14.04, and other older systems

---

## [1.0.0-rc.1] - 2026-03-15

### Changed

#### Version Update
- Updated version from 1.0.0-beta.5 to 1.0.0-rc.1 for release candidate
- Python package version updated from 1.0.0b5 to 1.0.0rc1
- Ready for RC release with all features tested and verified

---

## [1.0.0-beta.5] - 2026-03-09

### Fixed

#### Documentation
- Fixed incorrect section reference in CHANGELOG.md 1.0.0-beta.4 entry
  - Changed "Quick Start section" to "Installation section" to accurately reflect the documentation section where the fix was applied

---

## [1.0.0-beta.4] - 2026-03-09

### Fixed

#### Documentation
- Fixed incorrect rust version in README.md Installation section

#### Release Workflow
- Renamed GitHub Actions workflow file from `create-prerelease.yml` to `create-release.yml`
  - Updated workflow to support both pre-release and official release automation
  - Workflow automatically detects version type (alpha, beta, rc, or stable)
  - Creates appropriate GitHub Release with correct pre-release flag

---

## [1.0.0-beta.3] - 2026-03-05

### Fixed

#### Release Issues
- Fixed download wheels issue in create-prerelease.yml GitHub Action workflow
  - Corrected wheel download path and artifact handling
  - Ensures pre-release packages are correctly built and distributed

#### Version Number Standardization
- Standardized version number format across the codebase
  - `Cargo.toml`: Uses semantic versioning format `1.0.0-beta.3` (Rust convention)
  - `pyproject.toml`: Uses PEP 440 format `1.0.0b3` (Python convention)
  - `python/traj_dist_rs/__init__.py`: Uses PEP 440 format `1.0.0b3` (Python convention)
  - `CHANGELOG.md`: Uses semantic versioning format `1.0.0-beta.3` (documentation convention)
  - Ensures proper version recognition by pip and other Python tools

---

## [1.0.0-beta.2] - 2026-03-05

### Fixed

#### Release Issues
- Fixed crates.io publication failure caused by keyword length exceeding 20 characters limit
  - Changed keyword from "trajectory-similarity" (22 characters) to "similarity" (11 characters)
  - Ensures compliance with crates.io keyword length constraints

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

#### MkDocs Documentation
- Integrated MkDocs for modern, lightweight static site generation
- Comprehensive documentation structure with:
  - API reference pages automatically generated from Python docstrings
  - User guides for Python and Rust APIs
  - Installation and quick start guides
  - Performance benchmark reports with detailed statistics
  - Migration guides for version upgrades
- Live preview support with `mkdocs serve` for local development
- Production builds with `mkdocs build` for static website deployment
- Custom theme configuration for polished, professional appearance
- Navigation structure optimized for easy content discovery

### Changed

#### Dependencies
- Removed pyarrow from test dependencies
  - Replaced with polars for parquet file reading in tests
  - Reduced build complexity and dependency size
  - Improved cross-platform compatibility
- Added optional test dependencies via `[project.optional-dependencies.test]`
  - Users can now install test dependencies with: `pip install traj-dist-rs[test]`
  - Cleaner separation between runtime and test dependencies
- Added optional documentation dependencies via `[project.optional-dependencies.docs]`
  - Users can now install documentation dependencies with: `pip install traj-dist-rs[docs]`
  - Enables local documentation preview and production builds with MkDocs

---

## [1.0.0-alpha.1] - 2026-02-12

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

[Unreleased]: https://github.com/Davidham3/traj-dist-rs/compare/v1.0.0-rc.4...HEAD
[1.0.0-rc.4]: https://github.com/Davidham3/traj-dist-rs/compare/v1.0.0-rc.3...v1.0.0-rc.4
[1.0.0-rc.3]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v1.0.0-rc.3
[1.0.0-rc.2]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v1.0.0-rc.2
[1.0.0-rc.1]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v1.0.0-rc.1
[1.0.0-beta.5]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v1.0.0-beta.5
[1.0.0-beta.4]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v1.0.0-beta.4
[1.0.0-beta.3]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v1.0.0-beta.3
[1.0.0-beta.2]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v1.0.0-beta.2
[1.0.0-beta.1]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v1.0.0-beta.1
[1.0.0-alpha.1]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v1.0.0-alpha.1
[0.1.0-alpha.1]: https://github.com/Davidham3/traj-dist-rs/releases/tag/v0.1.0-alpha.1