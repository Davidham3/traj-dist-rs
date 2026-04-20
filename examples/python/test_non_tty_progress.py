"""
To test TTY environment:
    uv run examples/python/test_non_tty_progress.py

To test non-TTY environment:
    uv run examples/python/test_non_tty_progress.py 2>&1 | cat
"""

import sys
import time

import numpy as np
import traj_dist_rs


def main():
    print("Generating test data...", file=sys.stderr)
    # Generate 1000 trajectories, 200 points each
    # 1000 * 999 / 2 = 499,500 pairs, should take > 10s
    np.random.seed(42)
    trajectories = [np.random.rand(200, 2) for _ in range(1000)]

    metric = traj_dist_rs.Metric.dtw(type_d="euclidean")

    print("Starting DTW computation...", file=sys.stderr)
    start_time = time.time()

    distances = traj_dist_rs.pdist(
        trajectories, metric=metric, parallel=True, show_progress=True
    )

    end_time = time.time()

    print(
        f"Computation finished in {end_time - start_time:.2f} seconds.", file=sys.stderr
    )
    print(f"Computed {len(distances)} distances.", file=sys.stderr)


if __name__ == "__main__":
    main()
