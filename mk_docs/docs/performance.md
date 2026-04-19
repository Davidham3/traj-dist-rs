# Performance Benchmark Report

Benchmarks are summarized using the **median runtime**.

## TL;DR

- `traj-dist-rs` is on average **~231x faster** than `traj-dist (Python)`
- `traj-dist-rs` is on average **~15.3x faster** than `traj-dist (Cython)`
- Parallel batch execution reaches up to **~61.1x speedup** over `traj-dist (Cython)` on large inputs

## Benchmark Scope

This report compares `traj-dist-rs`, `traj-dist (Cython)`, and `traj-dist (Python)`.

Algorithms covered:

- DISCRET_FRECHET
- DTW
- EDR
- EDWP (Euclidean only, no Cython implementation; Python baseline from [TrajCL](https://github.com/changyanchuan/TrajCL))
- ERP
- FRECHET (Euclidean only, Cython implementation only; no Python implementation in upstream traj-dist)
- HAUSDORFF
- LCSS
- SSPD

Distance types:

- euclidean
- spherical

All summary values use the **median runtime** across benchmark samples.
Batch benchmarks compare Rust and Cython implementations.

## Figures

![README overview figure](assets/benchmark_speedup_readme.svg)

![Speedup by algorithm and distance type](assets/benchmark_speedup_by_distance.svg)

![Batch computation speedup](assets/benchmark_batch_speedup.svg)

## Summary Table

| algorithm       | distance_type   | hyperparam                         | traj-dist-rs   | traj-dist (Cython)   | traj-dist (Python)   | traj-dist-rs vs traj-dist (Cython)   | traj-dist-rs vs traj-dist (Python)   |
|:----------------|:----------------|:-----------------------------------|:---------------|:---------------------|:---------------------|:-------------------------------------|:-------------------------------------|
| DISCRET_FRECHET | euclidean       | eps=None; g=None                   | 0.0009 ms      | 0.0061 ms            | 0.4330 ms            | 6.78x                                | 481.11x                              |
| DTW             | euclidean       | eps=None; g=None                   | 0.0008 ms      | 0.0069 ms            | 0.4282 ms            | 8.63x                                | 535.25x                              |
| DTW             | spherical       | eps=None; g=None                   | 0.0026 ms      | 0.0090 ms            | 0.3717 ms            | 3.46x                                | 142.96x                              |
| EDR             | euclidean       | eps=0.01; g=None                   | 0.0010 ms      | 0.0077 ms            | 0.2953 ms            | 7.70x                                | 295.30x                              |
| EDR             | spherical       | eps=0.01; g=None                   | 0.0029 ms      | 0.0114 ms            | 0.2057 ms            | 3.93x                                | 70.93x                               |
| EDWP            | euclidean       | eps=None; g=None                   | 0.0029 ms      | N/A                  | 1.1206 ms            | N/A                                  | 386.41x                              |
| ERP             | euclidean       | eps=None; g=[-122.41443, 37.77646] | 0.0016 ms      | 0.0240 ms            | 0.2574 ms            | 15.00x                               | 160.88x                              |
| ERP             | spherical       | eps=None; g=[-122.41443, 37.77646] | 0.0040 ms      | 0.0334 ms            | 0.8215 ms            | 8.35x                                | 205.38x                              |
| FRECHET         | euclidean       | eps=None; g=None                   | 0.0152 ms      | 1.9191 ms            | N/A                  | 126.25x                              | N/A                                  |
| HAUSDORFF       | euclidean       | eps=None; g=None                   | 0.0014 ms      | 0.0160 ms            | 0.3951 ms            | 11.43x                               | 282.18x                              |
| HAUSDORFF       | spherical       | eps=None; g=None                   | 0.0249 ms      | 0.0735 ms            | 1.2738 ms            | 2.95x                                | 51.16x                               |
| LCSS            | euclidean       | eps=0.01; g=None                   | 0.0009 ms      | 0.0058 ms            | 0.2816 ms            | 6.44x                                | 312.89x                              |
| LCSS            | spherical       | eps=0.01; g=None                   | 0.0028 ms      | 0.0085 ms            | 0.1661 ms            | 3.04x                                | 59.32x                               |
| SSPD            | euclidean       | eps=None; g=None                   | 0.0020 ms      | 0.0170 ms            | 0.4019 ms            | 8.50x                                | 200.93x                              |
| SSPD            | spherical       | eps=None; g=None                   | 0.0325 ms      | 0.0746 ms            | 1.7528 ms            | 2.30x                                | 53.93x                               |

## Key Findings

- Against `traj-dist (Cython)`, the largest single-case speedup is **126.25x** on **FRECHET (euclidean)**.
- Against `traj-dist (Python)`, the largest single-case speedup is **535.25x** on **DTW (euclidean)**.
- On **euclidean** benchmarks, `traj-dist-rs` is on average **23.84x** faster than `traj-dist (Cython)` and **331.87x** faster than `traj-dist (Python)`.
- On **spherical** benchmarks, `traj-dist-rs` is on average **4.00x** faster than `traj-dist (Cython)` and **97.28x** faster than `traj-dist (Python)`.
- In batch mode, parallel Rust reaches up to **61.10x** speedup on `pdist` and **48.46x** on `cdist`.

## Batch Computation

### Configuration

- **Algorithm**: dtw
- **Number of trajectories**: 5 (fixed)
- **pdist computation**: 10 distances
- **Trajectory lengths tested**: 10, 100, 1000
- **Distance types**: euclidean, spherical

### Results

| Function   | Distance Type   |   Traj Length |   Distances | traj-dist (Cython) (ms)   | traj-dist-rs Seq (ms)   | traj-dist-rs Par (ms)   | Speedup (Seq)   | Speedup (Par)   |
|:-----------|:----------------|--------------:|------------:|:--------------------------|:------------------------|:------------------------|:----------------|:----------------|
| cdist      | euclidean       |            10 |          25 | 0.1842 ms                 | 0.0414 ms               | 0.3811 ms               | 4.45x           | 0.48x           |
| cdist      | euclidean       |           100 |          25 | 16.4476 ms                | 0.9475 ms               | 0.8907 ms               | 17.36x          | 18.47x          |
| cdist      | euclidean       |          1000 |          25 | 1467.9222 ms              | 126.4213 ms             | 30.2903 ms              | 11.61x          | 48.46x          |
| cdist      | spherical       |            10 |          25 | 0.2605 ms                 | 0.1122 ms               | 0.5688 ms               | 2.32x           | 0.46x           |
| cdist      | spherical       |           100 |          25 | 23.7994 ms                | 10.0824 ms              | 2.7441 ms               | 2.36x           | 8.67x           |
| cdist      | spherical       |          1000 |          25 | 2691.1131 ms              | 857.7901 ms             | 239.6442 ms             | 3.14x           | 11.23x          |
| pdist      | euclidean       |            10 |          10 | 0.1301 ms                 | 0.0090 ms               | 0.3618 ms               | 14.46x          | 0.36x           |
| pdist      | euclidean       |           100 |          10 | 6.6328 ms                 | 0.6956 ms               | 1.1937 ms               | 9.54x           | 5.56x           |
| pdist      | euclidean       |          1000 |          10 | 574.1788 ms               | 40.9131 ms              | 9.3981 ms               | 14.03x          | 61.10x          |
| pdist      | spherical       |            10 |          10 | 0.1453 ms                 | 0.0478 ms               | 0.3555 ms               | 3.04x           | 0.41x           |
| pdist      | spherical       |           100 |          10 | 10.3956 ms                | 4.3205 ms               | 2.5783 ms               | 2.41x           | 4.03x           |
| pdist      | spherical       |          1000 |          10 | 1082.8202 ms              | 371.9546 ms             | 62.9440 ms              | 2.91x           | 17.20x          |

### Notes

- `traj-dist-rs` sequential already outperforms `traj-dist (Cython)` across the tested batch cases.
- Parallel Rust provides the largest gains on longer trajectories.
- For small inputs, parallel overhead can outweigh the benefits of parallel execution.
