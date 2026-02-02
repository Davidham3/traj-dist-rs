#!/bin/bash
# 批量生成所有算法的测试用例（支持 Cython 和 Python 实现）
# 超参数定义在此脚本中

cd "$(dirname "$0")/.."

# 激活虚拟环境（在 ../traj_dist/.venv 中）
source ../traj_dist/.venv/bin/activate

# 确保输出目录存在
mkdir -p py_tests/data/cython_samples py_tests/data/python_samples py_tests/data/metainfo

# 清空现有的元数据文件和样本文件
rm -f py_tests/data/metainfo/*.jsonl
rm -f py_tests/data/cython_samples/*.parquet
rm -f py_tests/data/python_samples/*.parquet

echo "=========================================="
echo "开始生成测试用例"
echo "=========================================="

# 解析命令行参数
IMPLEMENTATION="both"
while [[ $# -gt 0 ]]; do
    case $1 in
        --implementation)
            IMPLEMENTATION="$2"
            shift 2
            ;;
        --cython)
            IMPLEMENTATION="cython"
            shift
            ;;
        --python)
            IMPLEMENTATION="python"
            shift
            ;;
        --both)
            IMPLEMENTATION="both"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "实现类型: $IMPLEMENTATION"

# ==========================================
# 无超参数的算法
# ==========================================

echo "生成 SSPD (Euclidean) 测试用例..."
python scripts/generate_test_case.py \
    --algorithm sspd \
    --type_d euclidean \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 SSPD (Spherical) 测试用例..."
python scripts/generate_test_case.py \
    --algorithm sspd \
    --type_d spherical \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 Hausdorff (Euclidean) 测试用例..."
python scripts/generate_test_case.py \
    --algorithm hausdorff \
    --type_d euclidean \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 Hausdorff (Spherical) 测试用例..."
python scripts/generate_test_case.py \
    --algorithm hausdorff \
    --type_d spherical \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 DTW (Euclidean) 测试用例..."
python scripts/generate_test_case.py \
    --algorithm dtw \
    --type_d euclidean \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 DTW (Spherical) 测试用例..."
python scripts/generate_test_case.py \
    --algorithm dtw \
    --type_d spherical \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 Discret Frechet 测试用例..."
python scripts/generate_test_case.py \
    --algorithm discret_frechet \
    --type_d euclidean \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

# ==========================================
# 有超参数的算法（每组 3 个超参数值）
# ==========================================

# LCSS 算法 - eps 参数

echo "生成 LCSS (Euclidean) 测试用例 - eps=0.0 ..."
python scripts/generate_test_case.py \
    --algorithm lcss \
    --type_d euclidean \
    --eps 0.0 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 LCSS (Euclidean) 测试用例 - eps=0.01 ..."
python scripts/generate_test_case.py \
    --algorithm lcss \
    --type_d euclidean \
    --eps 0.01 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 LCSS (Euclidean) 测试用例 - eps=0.02 ..."
python scripts/generate_test_case.py \
    --algorithm lcss \
    --type_d euclidean \
    --eps 0.02 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 LCSS (Spherical) 测试用例 - eps=0.01 ..."
python scripts/generate_test_case.py \
    --algorithm lcss \
    --type_d spherical \
    --eps 0.01 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 LCSS (Spherical) 测试用例 - eps=0.02 ..."
python scripts/generate_test_case.py \
    --algorithm lcss \
    --type_d spherical \
    --eps 0.02 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 LCSS (Spherical) 测试用例 - eps=0.05 ..."
python scripts/generate_test_case.py \
    --algorithm lcss \
    --type_d spherical \
    --eps 0.05 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

# EDR 算法 - eps 参数
echo "生成 EDR (Euclidean) 测试用例 - eps=0.0 ..."
python scripts/generate_test_case.py \
    --algorithm edr \
    --type_d euclidean \
    --eps 0.0 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 EDR (Euclidean) 测试用例 - eps=0.01 ..."
python scripts/generate_test_case.py \
    --algorithm edr \
    --type_d euclidean \
    --eps 0.01 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 EDR (Euclidean) 测试用例 - eps=0.02 ..."
python scripts/generate_test_case.py \
    --algorithm edr \
    --type_d euclidean \
    --eps 0.02 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 EDR (Spherical) 测试用例 - eps=0.01 ..."
python scripts/generate_test_case.py \
    --algorithm edr \
    --type_d spherical \
    --eps 0.01 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 EDR (Spherical) 测试用例 - eps=0.02 ..."
python scripts/generate_test_case.py \
    --algorithm edr \
    --type_d spherical \
    --eps 0.02 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 EDR (Spherical) 测试用例 - eps=0.05 ..."
python scripts/generate_test_case.py \
    --algorithm edr \
    --type_d spherical \
    --eps 0.05 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

# ERP 算法 - g 参数（使用四分位数）
echo "生成 ERP (Euclidean) 测试用例 - g=[-122.41443,37.77646]..."
python scripts/generate_test_case.py \
    --algorithm erp \
    --type_d euclidean \
    --g -122.41443 37.77646 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 ERP (Euclidean) 测试用例 - g=[-122.40382,37.78038]..."
python scripts/generate_test_case.py \
    --algorithm erp \
    --type_d euclidean \
    --g -122.40382 37.78038 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 ERP (Euclidean) 测试用例 - g=[-122.39607,37.78732]..."
python scripts/generate_test_case.py \
    --algorithm erp \
    --type_d euclidean \
    --g -122.39607 37.78732 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 ERP (Spherical) 测试用例 - g=[-122.41443,37.77646]..."
python scripts/generate_test_case.py \
    --algorithm erp \
    --type_d spherical \
    --g -122.41443 37.77646 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 ERP (Spherical) 测试用例 - g=[-122.40382,37.78038]..."
python scripts/generate_test_case.py \
    --algorithm erp \
    --type_d spherical \
    --g -122.40382 37.78038 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data

echo "生成 ERP (Spherical) 测试用例 - g=[-122.39607,37.78732]..."
python scripts/generate_test_case.py \
    --algorithm erp \
    --type_d spherical \
    --g -122.39607 37.78732 \
    --implementation "$IMPLEMENTATION" \
    --output-dir py_tests/data


echo "=========================================="
echo "测试用例生成完成"
echo "=========================================="

# 列出生成的文件
if [ "$IMPLEMENTATION" = "both" ] || [ "$IMPLEMENTATION" = "cython" ]; then
    echo "生成的 Cython 样本文件："
    ls -lh py_tests/data/cython_samples/*.parquet 2>/dev/null || echo "无 Cython 样本文件"
    echo ""
fi

if [ "$IMPLEMENTATION" = "both" ] || [ "$IMPLEMENTATION" = "python" ]; then
    echo "生成的 Python 样本文件："
    ls -lh py_tests/data/python_samples/*.parquet 2>/dev/null || echo "无 Python 样本文件"
    echo ""
fi

echo "生成的元数据文件："
ls -lh py_tests/data/metainfo/*.jsonl 2>/dev/null || echo "无元数据文件"