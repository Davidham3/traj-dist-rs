# Performance Benchmark Report

**Statistical Metric**: Median

## Summary Table

| Algorithm | Distance Type | Hyperparameters | Rust | Cython | Python | Rust/Cython | Rust/Python |
|-----------|---------------|-----------------|------|--------|--------|-------------|-------------|
| discret_frechet | euclidean | eps=None; g=None | 0.0008ms | 0.0086ms | 0.4842ms | 10.75x | 605.25x |
| dtw | euclidean | eps=None; g=None | 0.0008ms | 0.0082ms | 0.4890ms | 10.25x | 611.19x |
| dtw | spherical | eps=None; g=None | 0.0038ms | 0.0109ms | 0.3908ms | 2.87x | 102.86x |
| edr | euclidean | eps=0.01; g=None | 0.0011ms | 0.0074ms | 0.3277ms | 6.73x | 297.96x |
| edr | spherical | eps=0.01; g=None | 0.0038ms | 0.0103ms | 0.2216ms | 2.71x | 58.32x |
| erp | euclidean | eps=None; g=[-122.41443, 37.77646] | 0.0017ms | 0.0224ms | 0.2953ms | 13.21x | 173.71x |
| erp | spherical | eps=None; g=[-122.41443, 37.77646] | 0.0050ms | 0.0297ms | 0.8735ms | 5.93x | 174.71x |
| hausdorff | euclidean | eps=None; g=None | 0.0013ms | 0.0284ms | 0.4785ms | 21.85x | 368.12x |
| hausdorff | spherical | eps=None; g=None | 0.0316ms | 0.0929ms | 1.4050ms | 2.94x | 44.53x |
| lcss | euclidean | eps=0.01; g=None | 0.0006ms | 0.0072ms | 0.3646ms | 12.00x | 607.58x |
| lcss | spherical | eps=0.01; g=None | 0.0030ms | 0.0102ms | 0.1888ms | 3.34x | 61.90x |
| sspd | euclidean | eps=None; g=None | 0.0023ms | 0.0199ms | 0.4758ms | 8.65x | 206.87x |
| sspd | spherical | eps=None; g=None | 0.0451ms | 0.1005ms | 1.9232ms | 2.23x | 42.64x |

## Detailed Statistics

### DISCRET_FRECHET (euclidean)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0008 | 0.0009 | 0.0004 | 0.0005 | 0.0081 | 46.89 |
| CYTHON | 0.0086 | 0.0095 | 0.0037 | 0.0039 | 0.0401 | 38.85 |
| PYTHON | 0.4842 | 0.5182 | 0.2172 | 0.1910 | 1.3931 | 41.90 |

#### Performance Improvement

- **Rust vs Cython**: 10.75x
- Rust vs Python: 605.25x

### DTW (euclidean)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0008 | 0.0008 | 0.0003 | 0.0005 | 0.0046 | 32.95 |
| CYTHON | 0.0082 | 0.0093 | 0.0053 | 0.0043 | 0.0888 | 57.23 |
| PYTHON | 0.4890 | 0.5229 | 0.2236 | 0.1874 | 1.3917 | 42.75 |

#### Performance Improvement

- **Rust vs Cython**: 10.25x
- Rust vs Python: 611.19x

### DTW (spherical)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0038 | 0.0039 | 0.0014 | 0.0017 | 0.0122 | 36.69 |
| CYTHON | 0.0109 | 0.0124 | 0.0058 | 0.0054 | 0.0684 | 46.80 |
| PYTHON | 0.3908 | 0.4301 | 0.1957 | 0.1475 | 1.4195 | 45.51 |

#### Performance Improvement

- **Rust vs Cython**: 2.87x
- Rust vs Python: 102.86x

### EDR (euclidean)
Hyperparameters: eps=0.01; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0011 | 0.0012 | 0.0011 | 0.0006 | 0.0208 | 89.89 |
| CYTHON | 0.0074 | 0.0082 | 0.0033 | 0.0036 | 0.0289 | 40.67 |
| PYTHON | 0.3277 | 0.3677 | 0.1702 | 0.1394 | 0.9708 | 46.29 |

#### Performance Improvement

- **Rust vs Cython**: 6.73x
- Rust vs Python: 297.96x

### EDR (spherical)
Hyperparameters: eps=0.01; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0038 | 0.0042 | 0.0017 | 0.0018 | 0.0139 | 40.37 |
| CYTHON | 0.0103 | 0.0118 | 0.0075 | 0.0048 | 0.1410 | 64.00 |
| PYTHON | 0.2216 | 0.2452 | 0.1143 | 0.0852 | 0.7367 | 46.60 |

#### Performance Improvement

- **Rust vs Cython**: 2.71x
- Rust vs Python: 58.32x

### ERP (euclidean)
Hyperparameters: eps=None; g=[-122.41443, 37.77646]

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0017 | 0.0019 | 0.0013 | 0.0013 | 0.0234 | 65.97 |
| CYTHON | 0.0224 | 0.0249 | 0.0132 | 0.0107 | 0.1869 | 52.80 |
| PYTHON | 0.2953 | 0.3671 | 0.4784 | 0.1266 | 6.5803 | 130.30 |

#### Performance Improvement

- **Rust vs Cython**: 13.21x
- Rust vs Python: 173.71x

### ERP (spherical)
Hyperparameters: eps=None; g=[-122.41443, 37.77646]

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0050 | 0.0055 | 0.0020 | 0.0028 | 0.0172 | 35.80 |
| CYTHON | 0.0297 | 0.0328 | 0.0147 | 0.0131 | 0.1469 | 44.86 |
| PYTHON | 0.8735 | 1.0177 | 0.5379 | 0.3410 | 4.6360 | 52.86 |

#### Performance Improvement

- **Rust vs Cython**: 5.93x
- Rust vs Python: 174.71x

### HAUSDORFF (euclidean)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0013 | 0.0015 | 0.0012 | 0.0008 | 0.0241 | 81.42 |
| CYTHON | 0.0284 | 0.0286 | 0.0092 | 0.0121 | 0.0703 | 32.28 |
| PYTHON | 0.4785 | 0.5642 | 0.3286 | 0.1530 | 3.9326 | 58.25 |

#### Performance Improvement

- **Rust vs Cython**: 21.85x
- Rust vs Python: 368.12x

### HAUSDORFF (spherical)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0316 | 0.0359 | 0.0173 | 0.0119 | 0.0940 | 48.27 |
| CYTHON | 0.0929 | 0.1066 | 0.0588 | 0.0355 | 0.3418 | 55.15 |
| PYTHON | 1.4050 | 1.5799 | 0.7753 | 0.4739 | 5.8188 | 49.07 |

#### Performance Improvement

- **Rust vs Cython**: 2.94x
- Rust vs Python: 44.53x

### LCSS (euclidean)
Hyperparameters: eps=0.01; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0006 | 0.0007 | 0.0011 | 0.0005 | 0.0250 | 149.19 |
| CYTHON | 0.0072 | 0.0084 | 0.0042 | 0.0036 | 0.0370 | 50.36 |
| PYTHON | 0.3646 | 0.4130 | 0.2455 | 0.1461 | 3.1617 | 59.45 |

#### Performance Improvement

- **Rust vs Cython**: 12.00x
- Rust vs Python: 607.58x

### LCSS (spherical)
Hyperparameters: eps=0.01; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0030 | 0.0034 | 0.0013 | 0.0015 | 0.0081 | 39.18 |
| CYTHON | 0.0102 | 0.0127 | 0.0345 | 0.0047 | 0.7768 | 271.50 |
| PYTHON | 0.1888 | 0.2150 | 0.1201 | 0.0665 | 0.9492 | 55.84 |

#### Performance Improvement

- **Rust vs Cython**: 3.34x
- Rust vs Python: 61.90x

### SSPD (euclidean)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0023 | 0.0026 | 0.0013 | 0.0011 | 0.0179 | 50.55 |
| CYTHON | 0.0199 | 0.0215 | 0.0076 | 0.0123 | 0.1061 | 35.40 |
| PYTHON | 0.4758 | 0.5536 | 0.3039 | 0.1809 | 2.2450 | 54.89 |

#### Performance Improvement

- **Rust vs Cython**: 8.65x
- Rust vs Python: 206.87x

### SSPD (spherical)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0451 | 0.0502 | 0.0259 | 0.0190 | 0.2020 | 51.47 |
| CYTHON | 0.1005 | 0.1133 | 0.0502 | 0.0474 | 0.3043 | 44.28 |
| PYTHON | 1.9232 | 2.1274 | 0.9684 | 0.6251 | 5.6920 | 45.52 |

#### Performance Improvement

- **Rust vs Cython**: 2.23x
- Rust vs Python: 42.64x

## Analysis by Algorithm

Performance comparison across different implementations for each algorithm:

### DISCRET_FRECHET

- **Rust vs Cython**: Average improvement 10.75x (range: 10.75x - 10.75x)
- Rust vs Python: Average improvement 605.25x (range: 605.25x - 605.25x)

### DTW

- **Rust vs Cython**: Average improvement 6.56x (range: 2.87x - 10.25x)
- Rust vs Python: Average improvement 357.02x (range: 102.86x - 611.19x)

### EDR

- **Rust vs Cython**: Average improvement 4.72x (range: 2.71x - 6.73x)
- Rust vs Python: Average improvement 178.14x (range: 58.32x - 297.96x)

### ERP

- **Rust vs Cython**: Average improvement 9.57x (range: 5.93x - 13.21x)
- Rust vs Python: Average improvement 174.21x (range: 173.71x - 174.71x)

### HAUSDORFF

- **Rust vs Cython**: Average improvement 12.40x (range: 2.94x - 21.85x)
- Rust vs Python: Average improvement 206.32x (range: 44.53x - 368.12x)

### LCSS

- **Rust vs Cython**: Average improvement 7.67x (range: 3.34x - 12.00x)
- Rust vs Python: Average improvement 334.74x (range: 61.90x - 607.58x)

### SSPD

- **Rust vs Cython**: Average improvement 5.44x (range: 2.23x - 8.65x)
- Rust vs Python: Average improvement 124.76x (range: 42.64x - 206.87x)

## Analysis by Distance Type

Performance comparison across different distance types:

### EUCLIDEAN Distance

- **Rust vs Cython**: Average improvement 11.92x (range: 6.73x - 21.85x)
- Rust vs Python: Average improvement 410.09x (range: 173.71x - 611.19x)
- **Best Performance Improvement Algorithm**: hausdorff (21.85x)

### SPHERICAL Distance

- **Rust vs Cython**: Average improvement 3.34x (range: 2.23x - 5.93x)
- Rust vs Python: Average improvement 80.83x (range: 42.64x - 174.71x)
- **Best Performance Improvement Algorithm**: erp (5.93x)

## Overall Statistics

- RUST overall average time: 0.0078 ms
- CYTHON overall average time: 0.0274 ms
- PYTHON overall average time: 0.6091 ms

- Rust vs Cython overall average improvement: 3.53x
- Rust vs Python overall average improvement: 78.47x

## Batch Computation Performance

Performance comparison for batch distance computation (pdist and cdist):

### Test Configuration

- **Algorithm**: dtw
- **Number of trajectories**: 5 (fixed)
- **pdist computation**: 10 distances (5×4/2)
- **Trajectory lengths tested**: 10, 100, 1000 points
- **Distance types**: euclidean, spherical

### pdist Performance

Performance for pairwise distance computation (compressed distance matrix) with varying trajectory lengths:

| Distance Type | Traj Length | Distances | Cython (ms) | Rust Seq (ms) | Rust Par (ms) | Speedup (Seq) | Speedup (Par) | Parallel Efficiency |
|---------------|-------------|-----------|-------------|--------------|--------------|--------------|--------------|---------------------|
| euclidean | 10 | 10 | 0.0902 | 0.0054 | 0.6141 | 16.70x | 0.15x | 0.009x |
| euclidean | 100 | 10 | 6.3308 | 0.4340 | 0.8140 | 14.59x | 7.78x | 0.533x |
| euclidean | 1000 | 10 | 626.1093 | 42.7020 | 7.4122 | 14.66x | 84.47x | 5.761x |
| spherical | 10 | 10 | 0.1348 | 0.0649 | 0.6308 | 2.08x | 0.21x | 0.103x |
| spherical | 100 | 10 | 13.2197 | 5.4938 | 1.7841 | 2.41x | 7.41x | 3.079x |
| spherical | 1000 | 10 | 1029.9823 | 471.6477 | 85.0858 | 2.18x | 12.11x | 5.543x |

### cdist Performance

Performance for distance computation between two trajectory collections with varying trajectory lengths:

| Distance Type | Traj Length | Distances | Cython (ms) | Rust Seq (ms) | Rust Par (ms) | Speedup (Seq) | Speedup (Par) | Parallel Efficiency |
|---------------|-------------|-----------|-------------|--------------|--------------|--------------|--------------|---------------------|
| euclidean | 10 | 25 | 0.1924 | 0.0120 | 0.1381 | 16.03x | 1.39x | 0.087x |
| euclidean | 100 | 25 | 15.3446 | 1.0641 | 0.6353 | 14.42x | 24.15x | 1.675x |
| euclidean | 1000 | 25 | 1521.2707 | 92.2849 | 26.5950 | 16.48x | 57.20x | 3.470x |
| spherical | 10 | 25 | 0.8238 | 0.1291 | 0.4425 | 6.38x | 1.86x | 0.292x |
| spherical | 100 | 25 | 25.5498 | 13.2348 | 4.2448 | 1.93x | 6.02x | 3.118x |
| spherical | 1000 | 25 | 2553.6417 | 1186.0862 | 316.9027 | 2.15x | 8.06x | 3.743x |

### Batch Computation Summary

**Euclidean Distance**:

- **pdist** - Rust (sequential) vs Cython: Average 15.32x speedup (range: 14.59x - 16.70x)
- **pdist** - Rust (parallel) vs Cython: Average 30.80x speedup (range: 0.15x - 84.47x)
- **cdist** - Rust (sequential) vs Cython: Average 15.65x speedup (range: 14.42x - 16.48x)
- **cdist** - Rust (parallel) vs Cython: Average 27.58x speedup (range: 1.39x - 57.20x)

**Spherical Distance**:

- **pdist** - Rust (sequential) vs Cython: Average 2.22x speedup (range: 2.08x - 2.41x)
- **pdist** - Rust (parallel) vs Cython: Average 6.58x speedup (range: 0.21x - 12.11x)
- **cdist** - Rust (sequential) vs Cython: Average 3.49x speedup (range: 1.93x - 6.38x)
- **cdist** - Rust (parallel) vs Cython: Average 5.31x speedup (range: 1.86x - 8.06x)

**Note**: Parallel efficiency measures how much faster the parallel implementation is compared to the sequential implementation. For small datasets, parallel overhead may outweigh benefits.