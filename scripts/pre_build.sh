#!/bin/bash
# Pre-build script: Code formatting, compilation, and full-chain testing
# This script should be run after any code changes

set -e  # Exit immediately on error

# Switch to parent directory of script location (i.e., project root)
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

echo "=========================================="
echo "Starting pre-build process"
echo "=========================================="

# ==========================================
# Step 1: Rust code formatting
# ==========================================
echo ""
echo "[1/16] Rust code formatting (cargo fmt)..."
cargo fmt

# ==========================================
# Step 2: Rust code checking
# ==========================================
echo ""
echo "[2/16] Rust code checking (cargo clippy)..."
cargo clippy -- -D warnings

# ==========================================
# Step 3: Rust unit tests
# ==========================================
echo ""
echo "[3/16] Rust unit tests (cargo test)..."
cargo test

# ==========================================
# Step 3.5: README Rust examples test
# ==========================================
echo ""
echo "[3.5/16] README Rust examples test..."
cargo test --test readme_examples

# ==========================================
# Step 4: Generate Python stub files
# ==========================================
echo ""
echo "[4/16] Generate Python stub files (stub_gen)..."
# If stub_gen build fails, continue (stub files may already exist)
cargo run --features python-binding --bin stub_gen

# ==========================================
# Step 5: Remove interfering _lib directory
# ==========================================
echo ""
echo "[5/16] Remove interfering _lib directory..."
rm -rf python/traj_dist_rs/_lib

# ==========================================
# Step 6: Python binding build
# ==========================================
echo ""
echo "[6/16] Python binding build (maturin develop --release)..."
maturin develop --release

# ==========================================
# Step 7: Python code cleanup
# ==========================================
echo ""
echo "[7/16] Python code cleanup (autoflake)..."
find python/ py_tests/ examples/python/ scripts/ -name "*.py" -exec autoflake --in-place --remove-all-unused-imports --remove-unused-variables {} \;

# ==========================================
# Step 8: Python code formatting
# ==========================================
echo ""
echo "[8/16] Python code formatting (black)..."
black python/ py_tests/ examples/python/ scripts/

# ==========================================
# Step 9: Python import sorting
# ==========================================
echo ""
echo "[9/16] Python import sorting (isort)..."
isort python/ py_tests/ examples/python/ scripts/

# ==========================================
# Step 10: Python code checking and fixing
# ==========================================
echo ""
echo "[10/16] Python code checking and fixing (ruff check --fix)..."
ruff check --fix python/ py_tests/ examples/python/ scripts/

# ==========================================
# Step 11: Python integration tests
# ==========================================
echo ""
echo "[11/16] Python integration tests (pytest)..."
pytest py_tests/

# ==========================================
# Step 12: README Python examples test
# ==========================================
echo ""
echo "[12/16] README Python examples test..."
pytest py_tests/test_readme_examples.py -v

# ==========================================
# Step 13: Run all Python examples
# ==========================================
echo ""
echo "[13/16] Running all Python examples..."
cd examples/python

echo "  Running basic_distance.py..."
python basic_distance.py

echo "  Running batch_computation.py..."
python batch_computation.py

echo "  Running metric_api.py..."
python metric_api.py

echo "  Running parallel_processing.py..."
python parallel_processing.py

cd ../..

# ==========================================
# Step 14: Run all Rust examples
# ==========================================
echo ""
echo "[14/16] Running Rust examples..."

echo "  Running basic_distance example..."
cargo run --example basic_distance

echo "  Running batch_computation example..."
cargo run --example batch_computation --features parallel

echo "  Running precomputed_matrix example..."
cargo run --example precomputed_matrix

# ==========================================
# Step 15: Compile-check all Rust examples
# ==========================================
echo ""
echo "[15/16] Compile-checking all Rust examples (cargo build --examples)..."
cargo build --examples --features parallel

# ==========================================
# Step 16: Complete
# ==========================================
echo ""
echo "=========================================="
echo "Pre-build process completed successfully!"
echo "=========================================="