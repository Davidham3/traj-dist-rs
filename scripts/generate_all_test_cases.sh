#!/bin/bash
# 批量生成所有算法的测试用例
# 超参数定义在此脚本中
# 支持生成 pydist 或 cydist 版本的测试用例

cd "$(dirname "$0")/.."

# 激活虚拟环境
source .venv/bin/activate

# 从命令行参数获取实现类型，默认为 pydist
IMPLEMENTATION_TYPE="${1:-pydist}"

if [ "$IMPLEMENTATION_TYPE" != "pydist" ] && [ "$IMPLEMENTATION_TYPE" != "cydist" ]; then
    echo "错误: 实现类型必须是 'pydist' 或 'cydist'"
    echo "用法: $0 [pydist|cydist]"
    exit 1
fi

echo "使用实现类型: $IMPLEMENTATION_TYPE"

# 确保输出目录存在
mkdir -p py_tests/data/samples py_tests/data/cython_samples py_tests/data/metainfo

# 清空现有的元数据文件
rm -f py_tests/data/metainfo/*.jsonl

# 根据实现类型清空相应的样本文件
if [ "$IMPLEMENTATION_TYPE" = "pydist" ]; then
    rm -f py_tests/data/samples/*.parquet
else
    rm -f py_tests/data/cython_samples/*.parquet
fi

echo "=========================================="
echo "开始生成测试用例 (实现类型: $IMPLEMENTATION_TYPE)"
echo "=========================================="

# ==========================================
# 无超参数的算法
# ==========================================

if [ "$IMPLEMENTATION_TYPE" = "pydist" ]; then
    echo "生成 SSPD (Euclidean) 测试用例..."
    python scripts/generate_test_case.py \
        --algorithm sspd \
        --type_d euclidean \
        --output-dir py_tests/data

    echo "生成 SSPD (Spherical) 测试用例..."
    python scripts/generate_test_case.py \
        --algorithm sspd \
        --type_d spherical \
        --output-dir py_tests/data

    echo "生成 Hausdorff (Euclidean) 测试用例..."
    python scripts/generate_test_case.py \
        --algorithm hausdorff \
        --type_d euclidean \
        --output-dir py_tests/data

    echo "生成 Hausdorff (Spherical) 测试用例..."
    python scripts/generate_test_case.py \
        --algorithm hausdorff \
        --type_d spherical \
        --output-dir py_tests/data

    echo "生成 DTW (Euclidean) 测试用例..."
    python scripts/generate_test_case.py \
        --algorithm dtw \
        --type_d euclidean \
        --output-dir py_tests/data

    echo "生成 DTW (Spherical) 测试用例..."
    python scripts/generate_test_case.py \
        --algorithm dtw \
        --type_d spherical \
        --output-dir py_tests/data

    echo "生成 Frechet 测试用例..."
    python scripts/generate_test_case.py \
        --algorithm frechet \
        --type_d euclidean \
        --output-dir py_tests/data

    echo "生成 Discret Frechet 测试用例..."
    python scripts/generate_test_case.py \
        --algorithm discret_frechet \
        --type_d euclidean \
        --output-dir py_tests/data
else
    echo "生成 SSPD (Euclidean) 测试用例 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm sspd \
        --type_d euclidean \
        --output-dir py_tests/data

    echo "生成 SSPD (Spherical) 测试用例 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm sspd \
        --type_d spherical \
        --output-dir py_tests/data

    echo "生成 Hausdorff (Euclidean) 测试用例 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm hausdorff \
        --type_d euclidean \
        --output-dir py_tests/data

    echo "生成 Hausdorff (Spherical) 测试用例 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm hausdorff \
        --type_d spherical \
        --output-dir py_tests/data

    echo "生成 DTW (Euclidean) 测试用例 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm dtw \
        --type_d euclidean \
        --output-dir py_tests/data

    echo "生成 DTW (Spherical) 测试用例 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm dtw \
        --type_d spherical \
        --output-dir py_tests/data

    echo "生成 Frechet 测试用例 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm frechet \
        --type_d euclidean \
        --output-dir py_tests/data

    echo "生成 Discret Frechet 测试用例 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm discret_frechet \
        --type_d euclidean \
        --output-dir py_tests/data
fi

# ==========================================
# 有超参数的算法（每组 3 个超参数值）
# ==========================================

# LCSS 算法 - eps 参数
if [ "$IMPLEMENTATION_TYPE" = "pydist" ]; then
    echo "生成 LCSS (Euclidean) 测试用例 - eps=0.0..."
    python scripts/generate_test_case.py \
        --algorithm lcss \
        --type_d euclidean \
        --eps 0.0 \
        --output-dir py_tests/data

    echo "生成 LCSS (Euclidean) 测试用例 - eps=0.01..."
    python scripts/generate_test_case.py \
        --algorithm lcss \
        --type_d euclidean \
        --eps 0.01 \
        --output-dir py_tests/data

    echo "生成 LCSS (Euclidean) 测试用例 - eps=0.02..."
    python scripts/generate_test_case.py \
        --algorithm lcss \
        --type_d euclidean \
        --eps 0.02 \
        --output-dir py_tests/data

    echo "生成 LCSS (Spherical) 测试用例 - eps=0.01..."
    python scripts/generate_test_case.py \
        --algorithm lcss \
        --type_d spherical \
        --eps 0.01 \
        --output-dir py_tests/data

    echo "生成 LCSS (Spherical) 测试用例 - eps=0.02..."
    python scripts/generate_test_case.py \
        --algorithm lcss \
        --type_d spherical \
        --eps 0.02 \
        --output-dir py_tests/data

    echo "生成 LCSS (Spherical) 测试用例 - eps=0.05..."
    python scripts/generate_test_case.py \
        --algorithm lcss \
        --type_d spherical \
        --eps 0.05 \
        --output-dir py_tests/data
else
    echo "生成 LCSS (Euclidean) 测试用例 - eps=0.0 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm lcss \
        --type_d euclidean \
        --eps 0.0 \
        --output-dir py_tests/data

    echo "生成 LCSS (Euclidean) 测试用例 - eps=0.01 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm lcss \
        --type_d euclidean \
        --eps 0.01 \
        --output-dir py_tests/data

    echo "生成 LCSS (Euclidean) 测试用例 - eps=0.02 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm lcss \
        --type_d euclidean \
        --eps 0.02 \
        --output-dir py_tests/data

    echo "生成 LCSS (Spherical) 测试用例 - eps=0.01 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm lcss \
        --type_d spherical \
        --eps 0.01 \
        --output-dir py_tests/data

    echo "生成 LCSS (Spherical) 测试用例 - eps=0.02 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm lcss \
        --type_d spherical \
        --eps 0.02 \
        --output-dir py_tests/data

    echo "生成 LCSS (Spherical) 测试用例 - eps=0.05 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm lcss \
        --type_d spherical \
        --eps 0.05 \
        --output-dir py_tests/data
fi

# EDR 算法 - eps 参数
if [ "$IMPLEMENTATION_TYPE" = "pydist" ]; then
    echo "生成 EDR (Euclidean) 测试用例 - eps=0.0..."
    python scripts/generate_test_case.py \
        --algorithm edr \
        --type_d euclidean \
        --eps 0.0 \
        --output-dir py_tests/data

    echo "生成 EDR (Euclidean) 测试用例 - eps=0.01..."
    python scripts/generate_test_case.py \
        --algorithm edr \
        --type_d euclidean \
        --eps 0.01 \
        --output-dir py_tests/data

    echo "生成 EDR (Euclidean) 测试用例 - eps=0.02..."
    python scripts/generate_test_case.py \
        --algorithm edr \
        --type_d euclidean \
        --eps 0.02 \
        --output-dir py_tests/data

    echo "生成 EDR (Spherical) 测试用例 - eps=0.01..."
    python scripts/generate_test_case.py \
        --algorithm edr \
        --type_d spherical \
        --eps 0.01 \
        --output-dir py_tests/data

    echo "生成 EDR (Spherical) 测试用例 - eps=0.02..."
    python scripts/generate_test_case.py \
        --algorithm edr \
        --type_d spherical \
        --eps 0.02 \
        --output-dir py_tests/data

    echo "生成 EDR (Spherical) 测试用例 - eps=0.05..."
    python scripts/generate_test_case.py \
        --algorithm edr \
        --type_d spherical \
        --eps 0.05 \
        --output-dir py_tests/data
else
    echo "生成 EDR (Euclidean) 测试用例 - eps=0.0 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm edr \
        --type_d euclidean \
        --eps 0.0 \
        --output-dir py_tests/data

    echo "生成 EDR (Euclidean) 测试用例 - eps=0.01 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm edr \
        --type_d euclidean \
        --eps 0.01 \
        --output-dir py_tests/data

    echo "生成 EDR (Euclidean) 测试用例 - eps=0.02 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm edr \
        --type_d euclidean \
        --eps 0.02 \
        --output-dir py_tests/data

    echo "生成 EDR (Spherical) 测试用例 - eps=0.01 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm edr \
        --type_d spherical \
        --eps 0.01 \
        --output-dir py_tests/data

    echo "生成 EDR (Spherical) 测试用例 - eps=0.02 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm edr \
        --type_d spherical \
        --eps 0.02 \
        --output-dir py_tests/data

    echo "生成 EDR (Spherical) 测试用例 - eps=0.05 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm edr \
        --type_d spherical \
        --eps 0.05 \
        --output-dir py_tests/data
fi

# ERP 算法 - g 参数（使用四分位数）
if [ "$IMPLEMENTATION_TYPE" = "pydist" ]; then
    echo "生成 ERP (Euclidean) 测试用例 - g=[-122.41443,37.77646]..."
    python scripts/generate_test_case.py \
        --algorithm erp \
        --type_d euclidean \
        --g -122.41443 37.77646 \
        --output-dir py_tests/data

    echo "生成 ERP (Euclidean) 测试用例 - g=[-122.40382,37.78038]..."
    python scripts/generate_test_case.py \
        --algorithm erp \
        --type_d euclidean \
        --g -122.40382 37.78038 \
        --output-dir py_tests/data

    echo "生成 ERP (Euclidean) 测试用例 - g=[-122.39607,37.78732]..."
    python scripts/generate_test_case.py \
        --algorithm erp \
        --type_d euclidean \
        --g -122.39607 37.78732 \
        --output-dir py_tests/data

    echo "生成 ERP (Spherical) 测试用例 - g=[-122.41443,37.77646]..."
    python scripts/generate_test_case.py \
        --algorithm erp \
        --type_d spherical \
        --g -122.41443 37.77646 \
        --output-dir py_tests/data

    echo "生成 ERP (Spherical) 测试用例 - g=[-122.40382,37.78038]..."
    python scripts/generate_test_case.py \
        --algorithm erp \
        --type_d spherical \
        --g -122.40382 37.78038 \
        --output-dir py_tests/data

    echo "生成 ERP (Spherical) 测试用例 - g=[-122.39607,37.78732]..."
    python scripts/generate_test_case.py \
        --algorithm erp \
        --type_d spherical \
        --g -122.39607 37.78732 \
        --output-dir py_tests/data
else
    echo "生成 ERP (Euclidean) 测试用例 - g=[-122.41443,37.77646] (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm erp \
        --type_d euclidean \
        --g -122.41443 37.77646 \
        --output-dir py_tests/data

    echo "生成 ERP (Euclidean) 测试用例 - g=[-122.40382,37.78038] (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm erp \
        --type_d euclidean \
        --g -122.40382 37.78038 \
        --output-dir py_tests/data

    echo "生成 ERP (Euclidean) 测试用例 - g=[-122.39607,37.78732] (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm erp \
        --type_d euclidean \
        --g -122.39607 37.78732 \
        --output-dir py_tests/data

    echo "生成 ERP (Spherical) 测试用例 - g=[-122.41443,37.77646] (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm erp \
        --type_d spherical \
        --g -122.41443 37.77646 \
        --output-dir py_tests/data

    echo "生成 ERP (Spherical) 测试用例 - g=[-122.40382,37.78038] (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm erp \
        --type_d spherical \
        --g -122.40382 37.78038 \
        --output-dir py_tests/data

    echo "生成 ERP (Spherical) 测试用例 - g=[-122.39607,37.78732] (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm erp \
        --type_d spherical \
        --g -122.39607 37.78732 \
        --output-dir py_tests/data
fi

# SOWD 算法 - precision 参数
if [ "$IMPLEMENTATION_TYPE" = "pydist" ]; then
    echo "生成 SOWD (Spherical) 测试用例 - precision=4..."
    python scripts/generate_test_case.py \
        --algorithm sowd_grid \
        --type_d spherical \
        --precision 4 \
        --output-dir py_tests/data

    echo "生成 SOWD (Spherical) 测试用例 - precision=5..."
    python scripts/generate_test_case.py \
        --algorithm sowd_grid \
        --type_d spherical \
        --precision 5 \
        --output-dir py_tests/data

    echo "生成 SOWD (Spherical) 测试用例 - precision=6..."
    python scripts/generate_test_case.py \
        --algorithm sowd_grid \
        --type_d spherical \
        --precision 6 \
        --output-dir py_tests/data
else
    echo "生成 SOWD (Spherical) 测试用例 - precision=4 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm sowd_grid \
        --type_d spherical \
        --precision 4 \
        --output-dir py_tests/data

    echo "生成 SOWD (Spherical) 测试用例 - precision=5 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm sowd_grid \
        --type_d spherical \
        --precision 5 \
        --output-dir py_tests/data

    echo "生成 SOWD (Spherical) 测试用例 - precision=6 (Cython)..."
    python scripts/generate_test_case_cython.py \
        --algorithm sowd_grid \
        --type_d spherical \
        --precision 6 \
        --output-dir py_tests/data
fi

echo "=========================================="
echo "测试用例生成完成"
echo "=========================================="

# 列出生成的文件
if [ "$IMPLEMENTATION_TYPE" = "pydist" ]; then
    echo "生成的元数据文件："
    ls -lh py_tests/data/metainfo/*.jsonl

    echo ""
    echo "生成的样本文件："
    ls -lh py_tests/data/samples/*.parquet
else
    echo "生成的元数据文件："
    ls -lh py_tests/data/metainfo/*.jsonl

    echo ""
    echo "生成的 Cython 样本文件："
    ls -lh py_tests/data/cython_samples/*.parquet
fi
