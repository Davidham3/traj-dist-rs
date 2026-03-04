#!/bin/bash
# 本地构建 wheel 包的脚本
# 使用 cibuildwheel 进行跨平台构建

set -e

echo "==================================="
echo "本地构建 traj-dist-rs wheel 包"
echo "==================================="

# 检查是否安装了 cibuildwheel
if ! command -v cibuildwheel &> /dev/null; then
    echo "cibuildwheel 未安装，正在安装..."
    pip install cibuildwheel
fi

# 检查 Docker 是否运行（Linux/macOS 需要）
if ! docker info &> /dev/null; then
    echo "错误: Docker 未运行或未安装"
    echo "请先启动 Docker 服务"
    echo ""
    echo "Linux: sudo systemctl start docker"
    echo "macOS: 打开 Docker Desktop 应用"
    exit 1
fi

# 检测当前操作系统
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    MSYS_NT*)   MACHINE=Git;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "当前操作系统: ${MACHINE}"

# 根据操作系统设置 cibuildwheel 参数
if [ "${MACHINE}" = "Linux" ]; then
    PLATFORM=linux
    ARCHS=x86_64
elif [ "${MACHINE}" = "Mac" ]; then
    PLATFORM=macos
    ARCHS=x86_64
else
    echo "错误: 不支持的操作系统 ${MACHINE}"
    echo "Linux 和 macOS 支持 cibuildwheel 本地构建"
    echo "Windows 用户请在 GitHub Actions 中构建"
    exit 1
fi

# 设置 Python 版本
PYTHON_VERSIONS="cp310-* cp311-* cp312-* cp313-*"

echo "Python 版本: ${PYTHON_VERSIONS}"
echo "架构: ${ARCHS}"
echo ""

# 清理旧的 wheel 包
if [ -d "wheelhouse" ]; then
    echo "清理旧的 wheel 包..."
    rm -rf wheelhouse
fi

# 构建 wheel 包
echo "开始构建 wheel 包..."
echo "这可能需要几分钟时间，请耐心等待..."
echo ""

cibuildwheel \
    --platform ${PLATFORM} \
    --archs ${ARCHS} \
    --build "${PYTHON_VERSIONS}" \
    --output-dir wheelhouse

echo ""
echo "==================================="
echo "构建完成！"
echo "==================================="
echo "wheel 包已保存在 wheelhouse/ 目录"

# 列出生成的 wheel 包
ls -lh wheelhouse/

echo ""
echo "测试 wheel 包:"
echo "  python3 -m venv test_env"
echo "  source test_env/bin/activate"
echo "  pip install wheelhouse/*.whl"
echo "  python -c 'import traj_dist_rs; print(\"Import successful\")'"