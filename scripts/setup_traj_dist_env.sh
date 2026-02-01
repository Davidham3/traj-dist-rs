#!/bin/bash
#
# traj-dist 环境安装脚本
# 使用 pyenv Python 3.8 和最低版本依赖
#

set -e  # 遇到错误立即退出

echo "========================================"
echo "开始安装 traj-dist 开发环境"
echo "========================================"

# 检查 pyenv 是否可用
if ! command -v pyenv &> /dev/null; then
    echo "错误: pyenv 未安装"
    exit 1
fi

# 设置 Python 3.8.20 为本地版本
echo ""
echo "步骤 1: 设置 Python 3.8.20 为本地版本..."
# pyenv install 3.8.20
pyenv local 3.8.20
echo "✓ Python 版本设置为 $(python --version)"

# 创建虚拟环境
echo ""
echo "步骤 2: 创建虚拟环境..."
if [ -d "venv" ]; then
    echo "删除旧的虚拟环境..."
    rm -rf venv
fi
python -m venv venv
echo "✓ 虚拟环境创建成功"

# 升级 pip 和基础工具
echo ""
echo "步骤 3: 升级 pip 和基础工具..."
./venv/bin/pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade pip setuptools wheel
echo "✓ pip、setuptools 和 wheel 已升级"

# 安装核心依赖（使用与 Python 3.8 兼容的最低版本）
echo ""
echo "步骤 4: 安装核心依赖（numpy 和 Cython）..."
./venv/bin/pip install -i https://mirrors.aliyun.com/pypi/simple 'numpy==1.17.5' 'Cython==0.29.21'
echo "✓ numpy 1.17.5 和 Cython 0.29.21 已安装"

# 安装其他依赖（使用与 Python 3.8 兼容的最低版本）
echo ""
echo "步骤 5: 安装其他依赖..."
./venv/bin/pip install -i https://mirrors.aliyun.com/pypi/simple 'Shapely==1.7.1' 'geohash2==1.1' 'pandas==0.25.3' 'scipy==1.3.3' 'pyarrow==2.0.0' 'pydantic'
echo "✓ 所有依赖已安装"

# 清理旧的构建文件
echo ""
echo "步骤 6: 清理旧的构建文件..."
rm -rf build/ dist/ traj_dist.egg-info/ traj_dist/cydist/*.c traj_dist/cydist/__pycache__/*.so traj_dist/cydist/*.so
echo "✓ 旧的构建文件已清理"

# 编译并安装 traj-dist
echo ""
echo "步骤 7: 编译并安装 traj-dist..."
./venv/bin/python setup.py install
echo "✓ traj-dist 已成功编译并安装"

# 验证安装
echo ""
echo "步骤 8: 验证安装..."
source ./venv/bin/activate
cd /tmp
python -c "import traj_dist.distance as tdist"