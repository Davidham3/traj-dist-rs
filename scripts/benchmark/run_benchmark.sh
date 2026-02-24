#!/bin/bash
# Performance testing main script

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAJ_DIST_ROOT="$PROJECT_ROOT/traj-dist"
TRAJ_DIST_RS_ROOT="$PROJECT_ROOT/traj-dist-rs"

# Number of trajectory pairs (K value)
K=${K:-50}

# Number of warmup runs
WARMUP_RUNS=${WARMUP_RUNS:-5}

# Number of test runs
NUM_RUNS=${NUM_RUNS:-10}

# Output directory (placed in traj-dist-rs root directory)
OUTPUT_DIR="$TRAJ_DIST_RS_ROOT/benchmark_output"

# Clean old output
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Performance Testing Main Script"
echo "=========================================="
echo "Number of trajectory pairs (K): $K"
echo "Warmup runs: $WARMUP_RUNS"
echo "Test runs: $NUM_RUNS"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Step 1: Generate baseline trajectory data
echo "=========================================="
echo "Step 1: Generate baseline trajectory data"
echo "=========================================="
cd "$TRAJ_DIST_RS_ROOT"
source .venv/bin/activate

cd "$SCRIPT_DIR"
python generate_baseline_trajectories.py \
    --pkl-file "../traj-dist/data/benchmark_trajectories.pkl" \
    --k "$K" \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Baseline trajectory data generation failed"
    exit 1
fi
echo ""

deactivate

# Step 2: Test traj-dist (Cython implementation)
echo "=========================================="
echo "Step 2: Test traj-dist (Cython implementation)"
echo "=========================================="
cd "$TRAJ_DIST_ROOT"
source .venv/bin/activate

cd "$SCRIPT_DIR"
python benchmark_traj_dist.py \
    --baseline-file "$OUTPUT_DIR/baseline_trajectories.parquet" \
    --config-file "$SCRIPT_DIR/algorithms_config.json" \
    --implementation cython \
    --warmup-runs "$WARMUP_RUNS" \
    --num-runs "$NUM_RUNS" \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: traj-dist (Cython) performance test failed"
    exit 1
fi
echo ""

deactivate

# Step 3: Test traj-dist (Python implementation)
echo "=========================================="
echo "Step 3: Test traj-dist (Python implementation)"
echo "=========================================="
cd "$TRAJ_DIST_ROOT"
source .venv/bin/activate

cd "$SCRIPT_DIR"
python benchmark_traj_dist.py \
    --baseline-file "$OUTPUT_DIR/baseline_trajectories.parquet" \
    --config-file "$SCRIPT_DIR/algorithms_config.json" \
    --implementation python \
    --warmup-runs "$WARMUP_RUNS" \
    --num-runs "$NUM_RUNS" \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: traj-dist (Python) performance test failed"
    exit 1
fi
echo ""

deactivate

# Step 4: Test traj-dist-rs (Rust implementation)
echo "=========================================="
echo "Step 4: Test traj-dist-rs (Rust implementation)"
echo "=========================================="
cd "$TRAJ_DIST_RS_ROOT"
source .venv/bin/activate

cd "$SCRIPT_DIR"
python benchmark_traj_dist_rs.py \
    --baseline-file "$OUTPUT_DIR/baseline_trajectories.parquet" \
    --config-file "$SCRIPT_DIR/algorithms_config.json" \
    --warmup-runs "$WARMUP_RUNS" \
    --num-runs "$NUM_RUNS" \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: traj-dist-rs (Rust) performance test failed"
    exit 1
fi
echo ""

deactivate

# Step 5: Test batch computation (pdist/cdist) - Cython implementation
echo "=========================================="
echo "Step 5: Test batch computation - Cython implementation"
echo "=========================================="
cd "$TRAJ_DIST_ROOT"
source .venv/bin/activate

cd "$SCRIPT_DIR"
python benchmark_batch_traj_dist.py \
    --config-file "$SCRIPT_DIR/algorithms_config.json" \
    --warmup-runs 1 \
    --num-runs 1 \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Batch computation (Cython) performance test failed"
    exit 1
fi
echo ""

deactivate

# Step 6: Test batch computation (pdist/cdist) - Rust implementation
echo "=========================================="
echo "Step 6: Test batch computation - Rust implementation"
echo "=========================================="
cd "$TRAJ_DIST_RS_ROOT"
source .venv/bin/activate

cd "$SCRIPT_DIR"
python benchmark_batch_traj_dist_rs.py \
    --config-file "$SCRIPT_DIR/algorithms_config.json" \
    --warmup-runs 1 \
    --num-runs 1 \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Batch computation (Rust) performance test failed"
    exit 1
fi
echo ""

deactivate

# Step 7: Analyze results and generate report directly to docs directory
echo "=========================================="
echo "Step 7: Analyze results"
echo "=========================================="
cd "$TRAJ_DIST_RS_ROOT"
source .venv/bin/activate

cd "$SCRIPT_DIR"
python analyze_benchmark_results.py \
    --output-dir "$OUTPUT_DIR" \
    --output-file "../docs/performance.md"

deactivate

if [ $? -ne 0 ]; then
    echo "Error: Result analysis failed"
    exit 1
fi

# Complete
echo "=========================================="
echo "Performance testing completed!"
echo "=========================================="
echo "Report location: $TRAJ_DIST_RS_ROOT/docs/performance.md"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"
echo ""