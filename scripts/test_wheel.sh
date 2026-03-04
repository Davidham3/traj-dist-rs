#!/bin/bash
# 测试构建的 wheel 包

set -e

echo "==================================="
echo "测试 traj-dist-rs wheel 包"
echo "==================================="

# 检查 wheelhouse 目录是否存在
if [ ! -d "wheelhouse" ]; then
    echo "错误: wheelhouse 目录不存在"
    echo "请先运行构建脚本: bash scripts/build_wheels.sh"
    exit 1
fi

# 检查是否有 wheel 包
WHEELS=$(find wheelhouse -name "*.whl" 2>/dev/null || true)
if [ -z "$WHEELS" ]; then
    echo "错误: wheelhouse 目录中没有 wheel 包"
    exit 1
fi

echo "找到以下 wheel 包:"
ls -lh wheelhouse/*.whl
echo ""

# 获取第一个 wheel 包用于测试
FIRST_WHEEL=$(ls wheelhouse/*.whl | head -n 1)
echo "测试 wheel 包: $FIRST_WHEEL"
echo ""

# 创建临时虚拟环境
TEST_ENV="test_wheel_env_$(date +%s)"
echo "创建临时虚拟环境: $TEST_ENV"

# 检测 Python 版本
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "错误: 未找到 Python"
    exit 1
fi

$PYTHON -m venv "$TEST_ENV"
source "$TEST_ENV/bin/activate"

echo ""
echo "安装 wheel 包..."
pip install "$FIRST_WHEEL"

echo ""
echo "运行导入测试..."
python -c 'import traj_dist_rs; print("✓ 导入成功")'

echo ""
echo "运行简单功能测试..."
python -c '
import numpy as np
import traj_dist_rs

# 测试 DTW 算法
t1 = np.array([[0.0, 0.0], [1.0, 1.0]])
t2 = np.array([[0.0, 1.0], [1.0, 0.0]])
dist = traj_dist_rs.dtw(t1, t2, "euclidean")
print(f"✓ DTW 距离: {dist}")

# 测试球面距离
geo_t1 = np.array([[37.7749, -122.4194], [37.8044, -122.2711]])
geo_t2 = np.array([[37.7849, -122.4094], [37.7944, -122.3811]])
dist = traj_dist_rs.sspd(geo_t1, geo_t2, "spherical")
print(f"✓ SSPD 球面距离: {dist} km")

print("✓ 所有测试通过")
'

echo ""
echo "==================================="
echo "测试完成！"
echo "==================================="

# 清理临时环境
echo "清理临时环境..."
deactivate
rm -rf "$TEST_ENV"

echo "wheel 包测试成功！"