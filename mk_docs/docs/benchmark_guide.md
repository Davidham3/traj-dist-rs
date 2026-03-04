# Benchmark Reproduction Guide

This document provides complete steps for reproducing benchmark experiments. Anyone can follow this guide to easily run performance tests.

## 0. Prerequisites

Before starting, ensure you have the following tools installed:

### 0.1. Install uv (Python Package Manager)

`uv` is a fast Python package manager used in this project for dependency management.

**Install on Linux/macOS**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install on Windows**:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:
```bash
uv --version
```

### 0.2. Install Rust (Optional for traj-dist, Required for traj-dist-rs)

If you only want to run the original traj-dist benchmark, Rust is not required. However, to build and test traj-dist-rs, you need to install Rust.

**Install Rust**:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Verify installation:
```bash
rustc --version
cargo --version
```

## 1. Directory Structure

First, the benchmark workspace structure is as follows:

```
traj-dist-dev/
├── traj-dist/     # Python/Cython original implementation (with precision improvements)
└── traj-dist-rs/  # Rust reimplementation
```

Both projects must be in the same level directory because the benchmark script needs to access traj-dist's Python environment.

### 1.1. Clone traj-dist (Precision-Improved Version)

```bash
cd /path/to/traj-dist-dev
git clone https://github.com/Davidham3/traj-dist.git
```

**Note**: This version of traj-dist has upgraded the precision of spherical distance calculations from float to double, making it a more reasonable baseline implementation.

### 1.2. Clone traj-dist-rs

```bash
cd /path/to/traj-dist-dev
git clone https://github.com/Davidham3/traj-dist-rs.git
```

Ensure both repositories are in the same level directory:

```bash
ls -la
# You should see:
# traj-dist/
# traj-dist-rs/
```

## 2. Environment Installation

### 2.1. Install traj-dist Environment

Navigate to the traj-dist directory and use the provided script to quickly build the environment:

```bash
cd traj-dist
bash scripts/setup_traj_dist_env.sh
```

This script will automatically complete the following operations:
1. Use uv to create a Python 3.8 virtual environment
2. Install all dependencies (numpy, Cython, Shapely, etc.)
3. Clean up old build files
4. Build and install traj-dist

**Expected time**: 3-5 minutes (depending on network speed)

**Verify installation**:

```bash
source .venv/bin/activate
cd /tmp
python -c "import traj_dist; print('traj-dist installed successfully')"
```

### 2.2. Install traj-dist-rs Environment

Navigate to the traj-dist-rs directory and install the Rust development environment and Python bindings:

```bash
cd traj-dist-rs
uv sync --dev --all-extras
maturin develop --release
```

**Explanation**:

- `uv sync --dev --all-extras`: Install all development dependencies
- `maturin develop --release`: Build and install Python bindings in release mode (for best performance)

**Expected time**: 3-5 minutes (first run requires compiling Rust code)

**Verify installation**:

```bash
source .venv/bin/activate
python -c "import traj_dist_rs; print('traj-dist-rs installed successfully')"
```

## 3. Running Benchmark

### 3.1. Basic Command

Run performance tests in the traj-dist-rs directory:

```bash
cd traj-dist-rs
K=1000 WARMUP_RUNS=10 NUM_RUNS=50 bash scripts/benchmark/run_benchmark.sh
```

**Parameter explanation**:
- `K=1000`: Number of trajectory pairs to test (randomly extract 1000 pairs from baseline data)
- `WARMUP_RUNS=10`: Number of warmup runs (10 warmups to avoid cold start effects)
- `NUM_RUNS=50`: Number of measurement runs (50 measurements to calculate statistical metrics)

### 3.2. Custom Parameters (Optional)

If you need to use different test scales, you can adjust the parameters:

```bash
# Quick test (suitable for debugging)
K=50 WARMUP_RUNS=3 NUM_RUNS=10 bash scripts/benchmark/run_benchmark.sh

# Large-scale test (suitable for production environment performance evaluation)
K=5000 WARMUP_RUNS=20 NUM_RUNS=100 bash scripts/benchmark/run_benchmark.sh
```

### 3.3. Benchmark Execution Flow

The script will automatically complete the following steps:

1. **Generate baseline trajectory data**
   - Randomly extract K trajectory pairs from traj-dist/data/benchmark_trajectories.pkl
   - Save to benchmark_output/baseline_trajectories.parquet

2. **Test traj-dist (Cython implementation)**
   - Single trajectory pair distance calculation performance
   - Batch computation performance (pdist/cdist)

3. **Test traj-dist (Python implementation)**
   - Single trajectory pair distance calculation performance

4. **Test traj-dist-rs (Rust implementation)**
   - Single trajectory pair distance calculation performance
   - Batch computation performance (pdist/cdist, including sequential and parallel)

5. **Generate performance comparison report**
   - Analyze all test results
   - Generate Markdown format report to mk_docs/docs/performance.md

**Expected time**:
- K=1000, WARMUP_RUNS=10, NUM_RUNS=50: Approximately 10-15 minutes
- K=50, WARMUP_RUNS=3, NUM_RUNS=10: Approximately 1-2 minutes

## 4. Viewing Results

### 4.1. Performance Report

After the test is completed, view the generated performance report:

```bash
cat traj-dist-rs/mk_docs/docs/performance.md
```

The report includes:
- Single trajectory pair performance comparison (Rust vs Python vs Cython)
- Batch computation performance comparison (pdist/cdist)
- Performance analysis by distance type (Euclidean vs Spherical)
- Parallel efficiency analysis
- Statistical metrics (median, standard deviation, coefficient of variation)

### 4.2. Raw Data

All raw test data is saved in the `traj-dist-rs/benchmark_output/` directory:

```
benchmark_output/
├── baseline_trajectories.parquet              # Baseline trajectory data
├── traj_dist_cython_benchmark.parquet         # Cython single trajectory pair performance
├── traj_dist_python_benchmark.parquet         # Python single trajectory pair performance
├── traj_dist_rs_rust_benchmark.parquet        # Rust single trajectory pair performance
├── traj_dist_cython_batch_benchmark.parquet   # Cython batch computation performance
└── traj_dist_rs_rust_batch_benchmark.parquet  # Rust batch computation performance
```

You can use Python to read these data for custom analysis:

```python
import polars as pl

# Read Rust performance data
rust_data = pl.read_parquet("traj-dist-rs/benchmark_output/traj_dist_rs_rust_benchmark.parquet")
print(rust_data.head())
```

## 5. Supported Algorithms

The benchmark script tests the following distance algorithms:

1. **SSPD** (Symmetric Segment-Path Distance)
2. **DTW** (Dynamic Time Warping)
3. **Hausdorff Distance**
4. **LCSS** (Longest Common Subsequence)
5. **EDR** (Edit Distance on Real sequence)
6. **ERP** (Edit distance with Real Penalty)
7. **Discret Frechet Distance**

All algorithms support two distance types:
- **Euclidean**: Euclidean distance based on 2D coordinates
- **Spherical**: Haversine distance based on 2D coordinates (spherical distance)

## 6. References

- **traj-dist**: Original Python/Cython implementation, with improved spherical distance precision
  - GitHub: https://github.com/Davidham3/traj-dist
- **traj-dist-rs**: Rust high-performance reimplementation
  - GitHub: https://github.com/Davidham3/traj-dist-rs

## 7. Feedback and Contributions

If you encounter issues during the benchmark reproduction process, or have suggestions for improvements, please:
- Submit an Issue to the traj-dist-rs repository
- Submit a Pull Request to improve documentation or scripts

---

**Last Updated**: 2026-03-03