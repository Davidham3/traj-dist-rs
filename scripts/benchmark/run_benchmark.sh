#!/bin/bash
# 性能测试主脚本

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAJ_DIST_ROOT="$PROJECT_ROOT/traj-dist"
TRAJ_DIST_RS_ROOT="$PROJECT_ROOT/traj-dist-rs"

# 轨迹对数量（K值）
K=${K:-50}

# 预热次数
WARMUP_RUNS=${WARMUP_RUNS:-5}

# 测试次数
NUM_RUNS=${NUM_RUNS:-10}

# 输出目录（放在 traj-dist-rs 根目录下）
OUTPUT_DIR="$TRAJ_DIST_RS_ROOT/benchmark_output"

# 清理旧的输出
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "性能测试主脚本"
echo "=========================================="
echo "轨迹对数量 (K): $K"
echo "预热次数: $WARMUP_RUNS"
echo "测试次数: $NUM_RUNS"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
echo ""

# 步骤 1: 生成基准轨迹数据
echo "=========================================="
echo "步骤 1: 生成基准轨迹数据"
echo "=========================================="
cd "$TRAJ_DIST_RS_ROOT"
source .venv/bin/activate

cd "$SCRIPT_DIR"
python generate_baseline_trajectories.py \
    --pkl-file "../traj-dist/data/benchmark_trajectories.pkl" \
    --k "$K" \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "错误: 基准轨迹数据生成失败"
    exit 1
fi
echo ""

deactivate

# 步骤 2: 测试 traj-dist (Cython 实现)
echo "=========================================="
echo "步骤 2: 测试 traj-dist (Cython 实现)"
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
    echo "错误: traj-dist (Cython) 性能测试失败"
    exit 1
fi
echo ""

deactivate

# 步骤 3: 测试 traj-dist (Python 实现)
echo "=========================================="
echo "步骤 3: 测试 traj-dist (Python 实现)"
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
    echo "错误: traj-dist (Python) 性能测试失败"
    exit 1
fi
echo ""

deactivate

# 步骤 4: 测试 traj-dist-rs (Rust 实现)
echo "=========================================="
echo "步骤 4: 测试 traj-dist-rs (Rust 实现)"
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
    echo "错误: traj-dist-rs (Rust) 性能测试失败"
    exit 1
fi
echo ""

deactivate

# 步骤 5: 分析结果并直接生成报告到 docs 目录
echo "=========================================="
echo "步骤 5: 分析结果"
echo "=========================================="
cd "$TRAJ_DIST_RS_ROOT"
source .venv/bin/activate

cd "$SCRIPT_DIR"
python analyze_benchmark_results.py \
    --output-dir "$OUTPUT_DIR" \
    --output-file "../docs/performance.md"

deactivate

if [ $? -ne 0 ]; then
    echo "错误: 结果分析失败"
    exit 1
fi

# 完成
echo "=========================================="
echo "性能测试完成!"
echo "=========================================="
echo "报告位置: $TRAJ_DIST_RS_ROOT/docs/performance.md"
echo ""
echo "生成的文件:"
ls -lh "$OUTPUT_DIR"
echo ""