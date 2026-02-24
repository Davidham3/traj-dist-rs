#!/bin/bash
# Batch generate test cases for all algorithms (supports Cython and Python implementations)
# Hyperparameters are defined in this script

cd "$(dirname "$0")/.."

# Activate virtual environment (in ../traj-dist/.venv)
source ../traj-dist/.venv/bin/activate

# Ensure output directories exist
mkdir -p py_tests/data/cython_samples py_tests/data/metainfo

# Clear existing metadata files and sample files
rm -f py_tests/data/metainfo/*.jsonl
rm -f py_tests/data/cython_samples/*.parquet

echo "=========================================="
echo "Starting test case generation"
echo "=========================================="

# Correctness validation only uses Cython implementation as baseline
echo "Implementation type: cython (for correctness validation only)"

# ==========================================
# Algorithms without hyperparameters
# ==========================================

echo "Generating SSPD (Euclidean) test cases..."
python scripts/generate_test_case.py \
    --algorithm sspd \
    --type_d euclidean \
    --output-dir py_tests/data

echo "Generating SSPD (Spherical) test cases..."
python scripts/generate_test_case.py \
    --algorithm sspd \
    --type_d spherical \
    --output-dir py_tests/data

echo "Generating Hausdorff (Euclidean) test cases..."
python scripts/generate_test_case.py \
    --algorithm hausdorff \
    --type_d euclidean \
    --output-dir py_tests/data

echo "Generating Hausdorff (Spherical) test cases..."
python scripts/generate_test_case.py \
    --algorithm hausdorff \
    --type_d spherical \
    --output-dir py_tests/data

echo "Generating DTW (Euclidean) test cases..."
python scripts/generate_test_case.py \
    --algorithm dtw \
    --type_d euclidean \
    --output-dir py_tests/data

echo "Generating DTW (Spherical) test cases..."
python scripts/generate_test_case.py \
    --algorithm dtw \
    --type_d spherical \
    --output-dir py_tests/data

echo "Generating Discret Frechet test cases..."
python scripts/generate_test_case.py \
    --algorithm discret_frechet \
    --type_d euclidean \
    --output-dir py_tests/data

# ==========================================
# Algorithms with hyperparameters (3 hyperparameter values per group)
# ==========================================

# LCSS algorithm - eps parameter

echo "Generating LCSS (Euclidean) test cases - eps=0.0 ..."
python scripts/generate_test_case.py \
    --algorithm lcss \
    --type_d euclidean \
    --eps 0.0 \
    --output-dir py_tests/data

echo "Generating LCSS (Euclidean) test cases - eps=0.01 ..."
python scripts/generate_test_case.py \
    --algorithm lcss \
    --type_d euclidean \
    --eps 0.01 \
    --output-dir py_tests/data

echo "Generating LCSS (Euclidean) test cases - eps=0.02 ..."
python scripts/generate_test_case.py \
    --algorithm lcss \
    --type_d euclidean \
    --eps 0.02 \
    --output-dir py_tests/data

echo "Generating LCSS (Spherical) test cases - eps=0.01 ..."
python scripts/generate_test_case.py \
    --algorithm lcss \
    --type_d spherical \
    --eps 0.01 \
    --output-dir py_tests/data

echo "Generating LCSS (Spherical) test cases - eps=0.02 ..."
python scripts/generate_test_case.py \
    --algorithm lcss \
    --type_d spherical \
    --eps 0.02 \
    --output-dir py_tests/data

echo "Generating LCSS (Spherical) test cases - eps=0.05 ..."
python scripts/generate_test_case.py \
    --algorithm lcss \
    --type_d spherical \
    --eps 0.05 \
    --output-dir py_tests/data

# EDR algorithm - eps parameter
echo "Generating EDR (Euclidean) test cases - eps=0.0 ..."
python scripts/generate_test_case.py \
    --algorithm edr \
    --type_d euclidean \
    --eps 0.0 \
    --output-dir py_tests/data

echo "Generating EDR (Euclidean) test cases - eps=0.01 ..."
python scripts/generate_test_case.py \
    --algorithm edr \
    --type_d euclidean \
    --eps 0.01 \
    --output-dir py_tests/data

echo "Generating EDR (Euclidean) test cases - eps=0.02 ..."
python scripts/generate_test_case.py \
    --algorithm edr \
    --type_d euclidean \
    --eps 0.02 \
    --output-dir py_tests/data

echo "Generating EDR (Spherical) test cases - eps=0.01 ..."
python scripts/generate_test_case.py \
    --algorithm edr \
    --type_d spherical \
    --eps 0.01 \
    --output-dir py_tests/data

echo "Generating EDR (Spherical) test cases - eps=0.02 ..."
python scripts/generate_test_case.py \
    --algorithm edr \
    --type_d spherical \
    --eps 0.02 \
    --output-dir py_tests/data

echo "Generating EDR (Spherical) test cases - eps=0.05 ..."
python scripts/generate_test_case.py \
    --algorithm edr \
    --type_d spherical \
    --eps 0.05 \
    --output-dir py_tests/data

# ERP algorithm - g parameter (using quartiles)
echo "Generating ERP (Euclidean) test cases - g=[-122.41443,37.77646]..."
python scripts/generate_test_case.py \
    --algorithm erp \
    --type_d euclidean \
    --g -122.41443 37.77646 \
    --output-dir py_tests/data

echo "Generating ERP (Euclidean) test cases - g=[-122.40382,37.78038]..."
python scripts/generate_test_case.py \
    --algorithm erp \
    --type_d euclidean \
    --g -122.40382 37.78038 \
    --output-dir py_tests/data

echo "Generating ERP (Euclidean) test cases - g=[-122.39607,37.78732]..."
python scripts/generate_test_case.py \
    --algorithm erp \
    --type_d euclidean \
    --g -122.39607 37.78732 \
    --output-dir py_tests/data

echo "Generating ERP (Spherical) test cases - g=[-122.41443,37.77646]..."
python scripts/generate_test_case.py \
    --algorithm erp \
    --type_d spherical \
    --g -122.41443 37.77646 \
    --output-dir py_tests/data

echo "Generating ERP (Spherical) test cases - g=[-122.40382,37.78038]..."
python scripts/generate_test_case.py \
    --algorithm erp \
    --type_d spherical \
    --g -122.40382 37.78038 \
    --output-dir py_tests/data

echo "Generating ERP (Spherical) test cases - g=[-122.39607,37.78732]..."
python scripts/generate_test_case.py \
    --algorithm erp \
    --type_d spherical \
    --g -122.39607 37.78732 \
    --output-dir py_tests/data


echo "=========================================="
echo "Test case generation completed"
echo "=========================================="

# List generated files
echo "Generated Cython sample files:"
ls -lh py_tests/data/cython_samples/*.parquet 2>/dev/null || echo "No Cython sample files"
echo ""

echo "Generated metadata files:"
ls -lh py_tests/data/metainfo/*.jsonl 2>/dev/null || echo "No metadata files"
