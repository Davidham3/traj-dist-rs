# Performance Benchmark Report

**Statistical Metric**: Median

## Summary Table

| Algorithm | Distance Type | Hyperparameters | Rust | Cython | Python | Rust/Cython | Rust/Python |
|-----------|---------------|-----------------|------|--------|--------|-------------|-------------|
| discret_frechet | euclidean | eps=None; g=None | 0.0011ms | 0.0075ms | 0.5376ms | 6.82x | 488.73x |
| dtw | euclidean | eps=None; g=None | 0.0009ms | 0.0088ms | 0.5509ms | 9.78x | 612.06x |
| dtw | spherical | eps=None; g=None | 0.0040ms | 0.0124ms | 0.4879ms | 3.10x | 121.99x |
| edr | euclidean | eps=0.01; g=None | 0.0012ms | 0.0074ms | 0.4072ms | 6.17x | 339.33x |
| edr | spherical | eps=0.01; g=None | 0.0044ms | 0.0108ms | 0.2607ms | 2.45x | 59.25x |
| erp | euclidean | eps=None; g=[-122.41443, 37.77646] | 0.0020ms | 0.0243ms | 0.3381ms | 12.15x | 169.07x |
| erp | spherical | eps=None; g=[-122.41443, 37.77646] | 0.0057ms | 0.0490ms | 1.1078ms | 8.60x | 194.34x |
| hausdorff | euclidean | eps=None; g=None | 0.0015ms | 0.0205ms | 0.5484ms | 13.67x | 365.60x |
| hausdorff | spherical | eps=None; g=None | 0.0377ms | 0.0902ms | 1.8017ms | 2.39x | 47.79x |
| lcss | euclidean | eps=0.01; g=None | 0.0010ms | 0.0068ms | 0.3532ms | 6.75x | 353.20x |
| lcss | spherical | eps=0.01; g=None | 0.0041ms | 0.0100ms | 0.2031ms | 2.44x | 49.52x |
| sspd | euclidean | eps=None; g=None | 0.0017ms | 0.0215ms | 0.6564ms | 12.65x | 386.15x |
| sspd | spherical | eps=None; g=None | 0.0543ms | 0.0964ms | 2.5505ms | 1.78x | 46.97x |

## Detailed Statistics

### DISCRET_FRECHET (euclidean)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0011 | 0.0012 | 0.0006 | 0.0007 | 0.0097 | 52.78 |
| CYTHON | 0.0075 | 0.0083 | 0.0032 | 0.0040 | 0.0308 | 38.89 |
| PYTHON | 0.5376 | 0.6172 | 0.3100 | 0.2249 | 2.6063 | 50.23 |

#### Performance Improvement

- **Rust vs Cython**: 6.82x
- Rust vs Python: 488.73x

### DTW (euclidean)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0009 | 0.0009 | 0.0003 | 0.0006 | 0.0048 | 29.96 |
| CYTHON | 0.0088 | 0.0097 | 0.0043 | 0.0044 | 0.0664 | 44.85 |
| PYTHON | 0.5509 | 0.6453 | 0.2961 | 0.2404 | 2.0142 | 45.89 |

#### Performance Improvement

- **Rust vs Cython**: 9.78x
- Rust vs Python: 612.06x

### DTW (spherical)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0040 | 0.0043 | 0.0016 | 0.0020 | 0.0092 | 37.56 |
| CYTHON | 0.0124 | 0.0133 | 0.0053 | 0.0057 | 0.0365 | 40.00 |
| PYTHON | 0.4879 | 0.5314 | 0.2376 | 0.1848 | 1.6567 | 44.72 |

#### Performance Improvement

- **Rust vs Cython**: 3.10x
- Rust vs Python: 121.99x

### EDR (euclidean)
Hyperparameters: eps=0.01; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0012 | 0.0013 | 0.0005 | 0.0008 | 0.0080 | 36.11 |
| CYTHON | 0.0074 | 0.0082 | 0.0034 | 0.0035 | 0.0253 | 41.01 |
| PYTHON | 0.4072 | 0.4476 | 0.2148 | 0.1481 | 1.6577 | 48.00 |

#### Performance Improvement

- **Rust vs Cython**: 6.17x
- Rust vs Python: 339.33x

### EDR (spherical)
Hyperparameters: eps=0.01; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0044 | 0.0048 | 0.0019 | 0.0022 | 0.0170 | 38.89 |
| CYTHON | 0.0108 | 0.0124 | 0.0058 | 0.0048 | 0.0393 | 47.01 |
| PYTHON | 0.2607 | 0.2836 | 0.1280 | 0.0981 | 0.9082 | 45.14 |

#### Performance Improvement

- **Rust vs Cython**: 2.45x
- Rust vs Python: 59.25x

### ERP (euclidean)
Hyperparameters: eps=None; g=[-122.41443, 37.77646]

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0020 | 0.0025 | 0.0042 | 0.0016 | 0.0677 | 169.55 |
| CYTHON | 0.0243 | 0.0256 | 0.0108 | 0.0107 | 0.0990 | 42.32 |
| PYTHON | 0.3381 | 0.4181 | 0.5273 | 0.1536 | 7.0076 | 126.11 |

#### Performance Improvement

- **Rust vs Cython**: 12.15x
- Rust vs Python: 169.07x

### ERP (spherical)
Hyperparameters: eps=None; g=[-122.41443, 37.77646]

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0057 | 0.0065 | 0.0038 | 0.0033 | 0.0591 | 58.90 |
| CYTHON | 0.0490 | 0.0561 | 0.0293 | 0.0174 | 0.2268 | 52.23 |
| PYTHON | 1.1078 | 1.2283 | 0.5478 | 0.4133 | 3.4123 | 44.60 |

#### Performance Improvement

- **Rust vs Cython**: 8.60x
- Rust vs Python: 194.34x

### HAUSDORFF (euclidean)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0015 | 0.0018 | 0.0023 | 0.0009 | 0.0457 | 129.17 |
| CYTHON | 0.0205 | 0.0218 | 0.0053 | 0.0128 | 0.0529 | 24.38 |
| PYTHON | 0.5484 | 0.5882 | 0.2619 | 0.2034 | 1.8606 | 44.52 |

#### Performance Improvement

- **Rust vs Cython**: 13.67x
- Rust vs Python: 365.60x

### HAUSDORFF (spherical)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0377 | 0.0431 | 0.0236 | 0.0152 | 0.3098 | 54.78 |
| CYTHON | 0.0902 | 0.0987 | 0.0421 | 0.0374 | 0.2439 | 42.68 |
| PYTHON | 1.8017 | 1.9683 | 0.9336 | 0.6499 | 6.9395 | 47.43 |

#### Performance Improvement

- **Rust vs Cython**: 2.39x
- Rust vs Python: 47.79x

### LCSS (euclidean)
Hyperparameters: eps=0.01; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0010 | 0.0011 | 0.0010 | 0.0007 | 0.0232 | 96.04 |
| CYTHON | 0.0068 | 0.0076 | 0.0031 | 0.0032 | 0.0340 | 41.23 |
| PYTHON | 0.3532 | 0.4059 | 0.1907 | 0.1425 | 1.2419 | 46.98 |

#### Performance Improvement

- **Rust vs Cython**: 6.75x
- Rust vs Python: 353.20x

### LCSS (spherical)
Hyperparameters: eps=0.01; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0041 | 0.0045 | 0.0018 | 0.0021 | 0.0139 | 39.18 |
| CYTHON | 0.0100 | 0.0113 | 0.0057 | 0.0047 | 0.0868 | 50.60 |
| PYTHON | 0.2031 | 0.2240 | 0.0989 | 0.0831 | 0.6964 | 44.13 |

#### Performance Improvement

- **Rust vs Cython**: 2.44x
- Rust vs Python: 49.52x

### SSPD (euclidean)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0017 | 0.0019 | 0.0015 | 0.0009 | 0.0310 | 78.81 |
| CYTHON | 0.0215 | 0.0235 | 0.0100 | 0.0136 | 0.1476 | 42.70 |
| PYTHON | 0.6564 | 0.8227 | 0.6013 | 0.2140 | 5.7540 | 73.09 |

#### Performance Improvement

- **Rust vs Cython**: 12.65x
- Rust vs Python: 386.15x

### SSPD (spherical)
Hyperparameters: eps=None; g=None

#### Time Statistics

| Implementation | Median (ms) | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | CV (%) |
|----------------|-------------|-----------|--------------|----------|----------|--------|
| RUST | 0.0543 | 0.0615 | 0.0311 | 0.0217 | 0.2733 | 50.57 |
| CYTHON | 0.0964 | 0.1108 | 0.0528 | 0.0388 | 0.4146 | 47.69 |
| PYTHON | 2.5505 | 2.8694 | 1.4265 | 0.9137 | 9.8789 | 49.71 |

#### Performance Improvement

- **Rust vs Cython**: 1.78x
- Rust vs Python: 46.97x

## Analysis by Algorithm

Performance comparison across different implementations for each algorithm:

### DISCRET_FRECHET

- **Rust vs Cython**: Average improvement 6.82x (range: 6.82x - 6.82x)
- Rust vs Python: Average improvement 488.73x (range: 488.73x - 488.73x)

### DTW

- **Rust vs Cython**: Average improvement 6.44x (range: 3.10x - 9.78x)
- Rust vs Python: Average improvement 367.02x (range: 121.99x - 612.06x)

### EDR

- **Rust vs Cython**: Average improvement 4.31x (range: 2.45x - 6.17x)
- Rust vs Python: Average improvement 199.29x (range: 59.25x - 339.33x)

### ERP

- **Rust vs Cython**: Average improvement 10.37x (range: 8.60x - 12.15x)
- Rust vs Python: Average improvement 181.71x (range: 169.07x - 194.34x)

### HAUSDORFF

- **Rust vs Cython**: Average improvement 8.03x (range: 2.39x - 13.67x)
- Rust vs Python: Average improvement 206.70x (range: 47.79x - 365.60x)

### LCSS

- **Rust vs Cython**: Average improvement 4.59x (range: 2.44x - 6.75x)
- Rust vs Python: Average improvement 201.36x (range: 49.52x - 353.20x)

### SSPD

- **Rust vs Cython**: Average improvement 7.21x (range: 1.78x - 12.65x)
- Rust vs Python: Average improvement 216.56x (range: 46.97x - 386.15x)

## Analysis by Distance Type

Performance comparison across different distance types:

### EUCLIDEAN Distance

- **Rust vs Cython**: Average improvement 9.71x (range: 6.17x - 13.67x)
- Rust vs Python: Average improvement 387.74x (range: 169.07x - 612.06x)
- **Best Performance Improvement Algorithm**: hausdorff (13.67x)

### SPHERICAL Distance

- **Rust vs Cython**: Average improvement 3.46x (range: 1.78x - 8.60x)
- Rust vs Python: Average improvement 86.64x (range: 46.97x - 194.34x)
- **Best Performance Improvement Algorithm**: erp (8.60x)

## Overall Statistics

- RUST overall average time: 0.0092 ms
- CYTHON overall average time: 0.0281 ms
- PYTHON overall average time: 0.7541 ms

- Rust vs Cython overall average improvement: 3.06x
- Rust vs Python overall average improvement: 81.97x

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
| euclidean | 10 | 10 | 0.1042 | 0.0130 | 0.7254 | 8.02x | 0.14x | 0.018x |
| euclidean | 100 | 10 | 8.2472 | 0.5305 | 0.7843 | 15.55x | 10.52x | 0.676x |
| euclidean | 1000 | 10 | 798.9234 | 50.7034 | 9.5785 | 15.76x | 83.41x | 5.293x |
| spherical | 10 | 10 | 0.1425 | 0.1204 | 0.6776 | 1.18x | 0.21x | 0.178x |
| spherical | 100 | 10 | 12.0262 | 5.7171 | 2.1539 | 2.10x | 5.58x | 2.654x |
| spherical | 1000 | 10 | 1262.5977 | 595.2448 | 114.9133 | 2.12x | 10.99x | 5.180x |

### cdist Performance

Performance for distance computation between two trajectory collections with varying trajectory lengths:

| Distance Type | Traj Length | Distances | Cython (ms) | Rust Seq (ms) | Rust Par (ms) | Speedup (Seq) | Speedup (Par) | Parallel Efficiency |
|---------------|-------------|-----------|-------------|--------------|--------------|--------------|--------------|---------------------|
| euclidean | 10 | 25 | 0.2187 | 0.0138 | 0.2192 | 15.85x | 1.00x | 0.063x |
| euclidean | 100 | 25 | 18.2095 | 1.1974 | 1.2021 | 15.21x | 15.15x | 0.996x |
| euclidean | 1000 | 25 | 1897.6417 | 124.8654 | 31.1265 | 15.20x | 60.97x | 4.012x |
| spherical | 10 | 25 | 1.4174 | 0.1932 | 0.4156 | 7.34x | 3.41x | 0.465x |
| spherical | 100 | 25 | 34.5021 | 17.7861 | 5.8351 | 1.94x | 5.91x | 3.048x |
| spherical | 1000 | 25 | 3302.5918 | 1417.0407 | 461.1969 | 2.33x | 7.16x | 3.073x |

### Batch Computation Summary

**Euclidean Distance**:
- **pdist** - Rust (sequential) vs Cython: Average 13.11x speedup (range: 8.02x - 15.76x)
- **pdist** - Rust (parallel) vs Cython: Average 31.36x speedup (range: 0.14x - 83.41x)
- **cdist** - Rust (sequential) vs Cython: Average 15.42x speedup (range: 15.20x - 15.85x)
- **cdist** - Rust (parallel) vs Cython: Average 25.70x speedup (range: 1.00x - 60.97x)

**Spherical Distance**:
- **pdist** - Rust (sequential) vs Cython: Average 1.80x speedup (range: 1.18x - 2.12x)
- **pdist** - Rust (parallel) vs Cython: Average 5.59x speedup (range: 0.21x - 10.99x)
- **cdist** - Rust (sequential) vs Cython: Average 3.87x speedup (range: 1.94x - 7.34x)
- **cdist** - Rust (parallel) vs Cython: Average 5.49x speedup (range: 3.41x - 7.16x)

**Note**: Parallel efficiency measures how much faster the parallel implementation is compared to the sequential implementation. For small datasets, parallel overhead may outweigh benefits.