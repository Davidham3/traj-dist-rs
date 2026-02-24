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
echo "[1/9] Rust code formatting (cargo fmt)..."
cargo fmt

# ==========================================
# Step 2: Rust code checking
# ==========================================
echo ""
echo "[2/9] Rust code checking (cargo clippy)..."
cargo clippy -- -D warnings

# ==========================================
# Step 3: Rust unit tests
# ==========================================
echo ""
echo "[3/11] Rust unit tests (cargo test)..."
cargo test

# ==========================================
# Step 3.5: README Rust examples test
# ==========================================
echo ""
echo "[3.5/11] README Rust examples test..."
cargo test --test readme_examples

# ==========================================
# Step 4: Generate Python stub files
# ==========================================
echo ""
echo "[4/11] Generate Python stub files (stub_gen)..."
# If stub_gen build fails, continue (stub files may already exist)
cargo run --features python-binding --bin stub_gen

# ==========================================
# Step 5: Remove interfering _lib directory
# ==========================================
echo ""
echo "[5/11] Remove interfering _lib directory..."
rm -rf python/traj_dist_rs/_lib

# ==========================================
# Step 6: Python binding build
# ==========================================
echo ""
echo "[6/11] Python binding build (maturin develop --release)..."
maturin develop --release

# ==========================================
# Step 7: Python code cleanup
# ==========================================
echo ""
echo "[7/14] Python code cleanup (autoflake)..."
find python/ py_tests/ examples/python/ scripts/ -name "*.py" -exec autoflake --in-place --remove-all-unused-imports --remove-unused-variables {} \;

# ==========================================
# Step 8: Python code formatting
# ==========================================
echo ""
echo "[8/14] Python code formatting (black)..."
black python/ py_tests/ examples/python/ scripts/

# ==========================================
# Step 9: Python import sorting
# ==========================================
echo ""
echo "[9/14] Python import sorting (isort)..."
isort python/ py_tests/ examples/python/ scripts/

# ==========================================
# Step 10: Python code checking and fixing
# ==========================================
echo ""
echo "[10/14] Python code checking and fixing (ruff check --fix)..."
ruff check --fix python/ py_tests/ examples/python/ scripts/

# ==========================================
# Step 11: Python integration tests
# ==========================================
echo ""
echo "[11/14] Python integration tests (pytest)..."
pytest py_tests/

# ==========================================
# Step 12: README Python examples test
# ==========================================
echo ""
echo "[12/14] README Python examples test..."
pytest py_tests/test_readme_examples.py -v

# ==========================================
# Step 13: Run all Python examples
# ==========================================
echo ""
echo "[13/14] Running all Python examples..."
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
# Step 14: Complete
# ==========================================
echo ""
echo "=========================================="
echo "Pre-build process completed successfully!"
echo "=========================================="