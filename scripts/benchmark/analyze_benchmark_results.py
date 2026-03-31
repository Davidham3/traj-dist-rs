from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

LABEL_RS = "traj-dist-rs"
LABEL_PY = "traj-dist (Python)"
LABEL_CY = "traj-dist (Cython)"

COMPARE_RS_PY = f"{LABEL_RS} vs {LABEL_PY}"
COMPARE_RS_CY = f"{LABEL_RS} vs {LABEL_CY}"

LEGEND_PY = "vs Python"
LEGEND_CY = "vs Cython"
LEGEND_SEQ = "Rust seq vs Cython"
LEGEND_PAR = "Rust par vs Cython"

BLUE_DARK = "#1f77b4"
BLUE_LIGHT = "#6baed6"
BLUE_MID = "#4C78A8"

ALGORITHM_ORDER = [
    "DISCRET_FRECHET",
    "DTW",
    "EDR",
    "ERP",
    "HAUSDORFF",
    "LCSS",
    "SSPD",
]

REPORT_DIR = Path("../../mk_docs/docs")
FIGURE_DIR = REPORT_DIR / "assets"
REPORT_FILE = REPORT_DIR / "performance.md"

README_IMAGE_PATH = "mk_docs/docs/assets/benchmark_speedup_readme.svg"
README_PERFORMANCE_PATH = "mk_docs/docs/performance.md"
FIGURE_REL_DIR = "assets"

IMPL_LABELS = {
    "rust": LABEL_RS,
    "cython": LABEL_CY,
    "python": LABEL_PY,
}


def parse_args():
    p = ArgumentParser()
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def norm_algo(x: str) -> str:
    x = str(x).strip().lower().replace("_", " ").replace("-", " ")
    x = " ".join(x.split())
    return {
        "sspd": "SSPD",
        "dtw": "DTW",
        "discret frechet": "DISCRET_FRECHET",
        "discrete frechet": "DISCRET_FRECHET",
        "frechet": "DISCRET_FRECHET",
        "hausdorff": "HAUSDORFF",
        "hausdorf": "HAUSDORFF",
        "lcss": "LCSS",
        "edr": "EDR",
        "erp": "ERP",
    }.get(x, x.upper())


def algo_key(x: str):
    x = norm_algo(x)
    return ALGORITHM_ORDER.index(x) if x in ALGORITHM_ORDER else 999


def median_ms(xs) -> float:
    xs = np.asarray(xs, dtype=float)
    return float(np.median(xs) * 1000)


def load_main_results(output_dir: Path):
    return {
        "rust": pl.read_parquet(output_dir / "traj_dist_rs_rust_benchmark.parquet"),
        "cython": pl.read_parquet(output_dir / "traj_dist_cython_benchmark.parquet"),
        "python": pl.read_parquet(output_dir / "traj_dist_python_benchmark.parquet"),
    }


def load_batch_results(output_dir: Path):
    return {
        "rust": pl.read_parquet(
            output_dir / "traj_dist_rs_rust_batch_benchmark.parquet"
        ),
        "cython": pl.read_parquet(
            output_dir / "traj_dist_cython_batch_benchmark.parquet"
        ),
    }


def build_summary_df(results):
    grouped = {}

    for impl, df in results.items():
        hyper_cols = [c for c in df.columns if c.startswith("hyperparam_")]
        for row in df.iter_rows(named=True):
            hp = (
                "; ".join(
                    f"{c.removeprefix('hyperparam_')}={row[c]}" for c in hyper_cols
                )
                if hyper_cols
                else "N/A"
            )
            key = (norm_algo(row["algorithm"]), row["distance_type"], hp, impl)
            grouped.setdefault(key, []).extend(row["times"])

    rows = []
    for (algorithm, distance_type, hyperparam, impl), times in grouped.items():
        rows.append(
            {
                "algorithm": algorithm,
                "distance_type": distance_type,
                "hyperparam": hyperparam,
                "impl": impl,
                "impl_name": IMPL_LABELS[impl],
                "median_ms": median_ms(times),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(
        ["algorithm", "distance_type"],
        key=lambda s: s.map(algo_key) if s.name == "algorithm" else s,
    )


def build_comparison_df(summary_df: pd.DataFrame):
    wide = summary_df.pivot_table(
        index=["algorithm", "distance_type", "hyperparam"],
        columns="impl_name",
        values="median_ms",
        aggfunc="mean",
    ).reset_index()

    wide[COMPARE_RS_CY] = wide[LABEL_CY] / wide[LABEL_RS]
    wide[COMPARE_RS_PY] = wide[LABEL_PY] / wide[LABEL_RS]

    cols = [
        "algorithm",
        "distance_type",
        "hyperparam",
        LABEL_RS,
        LABEL_CY,
        LABEL_PY,
        COMPARE_RS_CY,
        COMPARE_RS_PY,
    ]
    return wide[cols].sort_values(
        ["algorithm", "distance_type"],
        key=lambda s: s.map(algo_key) if s.name == "algorithm" else s,
    )


def build_batch_df(batch_results):
    cy = batch_results["cython"].to_pandas()
    rs = batch_results["rust"].to_pandas()

    cy_rows = []
    for (func, dist, length), g in cy.groupby(
        ["function", "distance_type", "trajectory_length"]
    ):
        times = np.concatenate(g["times"].to_numpy())
        cy_rows.append(
            {
                "function": func,
                "distance_type": dist,
                "trajectory_length": int(length),
                "num_distances": int(g["num_distances"].iloc[0]),
                f"{LABEL_CY} (ms)": median_ms(times),
            }
        )
    cy_df = pd.DataFrame(cy_rows)

    def collect_rust(parallel, col):
        rows = []
        gdf = rs[rs["parallel"] == parallel]
        for (func, dist, length), g in gdf.groupby(
            ["function", "distance_type", "trajectory_length"]
        ):
            times = np.concatenate(g["times"].to_numpy())
            rows.append(
                {
                    "function": func,
                    "distance_type": dist,
                    "trajectory_length": int(length),
                    col: median_ms(times),
                }
            )
        return pd.DataFrame(rows)

    rs_seq = collect_rust(False, f"{LABEL_RS} Seq (ms)")
    rs_par = collect_rust(True, f"{LABEL_RS} Par (ms)")

    df = cy_df.merge(
        rs_seq, on=["function", "distance_type", "trajectory_length"]
    ).merge(rs_par, on=["function", "distance_type", "trajectory_length"])
    df["Speedup (Seq)"] = df[f"{LABEL_CY} (ms)"] / df[f"{LABEL_RS} Seq (ms)"]
    df["Speedup (Par)"] = df[f"{LABEL_CY} (ms)"] / df[f"{LABEL_RS} Par (ms)"]

    return df.sort_values(["function", "distance_type", "trajectory_length"])


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, pad_inches=0.15)
    plt.close(fig)


def plot_overview(comparison_df: pd.DataFrame):
    df = (
        comparison_df.groupby("algorithm")[[COMPARE_RS_PY, COMPARE_RS_CY]]
        .mean()
        .reset_index()
    )
    df["order"] = df["algorithm"].map(algo_key)
    df = df.sort_values("order")

    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(16, 6.8))

    b1 = ax.bar(
        x - width / 2,
        df[COMPARE_RS_PY],
        width,
        color=BLUE_DARK,
        label=LEGEND_PY,
        edgecolor="black",
        linewidth=0.5,
    )
    b2 = ax.bar(
        x + width / 2,
        df[COMPARE_RS_CY],
        width,
        color=BLUE_LIGHT,
        label=LEGEND_CY,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title(f"{LABEL_RS} Speedup by Algorithm", fontsize=20, weight="bold", pad=14)
    ax.set_ylabel("Speedup (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["algorithm"], rotation=15, ha="right")
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}x",
                (bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / "benchmark_speedup_readme.svg")


def plot_by_distance(comparison_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    handles, labels = None, None
    ordered_algos = sorted(comparison_df["algorithm"].unique(), key=algo_key)

    for ax, dist in zip(axes, ["euclidean", "spherical"]):
        grouped = (
            comparison_df[comparison_df["distance_type"] == dist]
            .groupby("algorithm")[[COMPARE_RS_PY, COMPARE_RS_CY]]
            .mean()
        )

        df = grouped.reindex(ordered_algos).reset_index()

        x = np.arange(len(df))
        width = 0.36

        b1 = ax.bar(
            x - width / 2,
            df[COMPARE_RS_PY],
            width,
            color=BLUE_DARK,
            label=LEGEND_PY,
            edgecolor="black",
            linewidth=0.4,
        )
        b2 = ax.bar(
            x + width / 2,
            df[COMPARE_RS_CY],
            width,
            color=BLUE_LIGHT,
            label=LEGEND_CY,
            edgecolor="black",
            linewidth=0.4,
        )

        ax.set_title(f"{dist.capitalize()} Distance", fontsize=16, weight="bold", pad=8)
        ax.set_ylabel("Speedup (log scale)")
        ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                if pd.notna(h) and h > 0:
                    ax.annotate(
                        f"{h:.1f}x",
                        (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    axes[-1].set_xticks(np.arange(len(ordered_algos)))
    axes[-1].set_xticklabels(ordered_algos, rotation=15, ha="right")

    fig.suptitle(
        "Speedup by Algorithm and Distance Type", fontsize=18, weight="bold", y=0.975
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=True,
    )

    fig.subplots_adjust(top=0.86, bottom=0.14, hspace=0.32)
    savefig(fig, FIGURE_DIR / "benchmark_speedup_by_distance.svg")


def plot_batch(batch_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.8))
    handles, labels = None, None
    width = 0.36

    for i, (ax, func) in enumerate(zip(axes, ["pdist", "cdist"])):
        df = (
            batch_df[batch_df["function"] == func]
            .groupby("trajectory_length")[["Speedup (Seq)", "Speedup (Par)"]]
            .mean()
            .reset_index()
            .sort_values("trajectory_length")
        )

        if df.empty:
            ax.set_title(f"{func} Speedup", fontsize=16, weight="bold", pad=4)
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_xlabel("Trajectory Length")
            ax.set_ylabel("Speedup (log scale)" if i == 0 else "")
            ax.set_yscale("log")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            continue

        x = np.arange(len(df))

        seq_vals = df["Speedup (Seq)"].to_numpy()
        par_vals = df["Speedup (Par)"].to_numpy()

        # 关键：补上 bar
        ax.bar(
            x - width / 2,
            seq_vals,
            width,
            color=BLUE_DARK,
            label=LEGEND_SEQ,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar(
            x + width / 2,
            par_vals,
            width,
            color=BLUE_LIGHT,
            label=LEGEND_PAR,
            edgecolor="black",
            linewidth=0.4,
        )

        # 只在存在有效值时标注最大值
        if np.isfinite(seq_vals).any():
            seq_pos = int(np.nanargmax(seq_vals))
            ax.annotate(
                f"max seq: {seq_vals[seq_pos]:.1f}x",
                (x[seq_pos] - width / 2, seq_vals[seq_pos]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=BLUE_DARK,
                weight="bold",
            )

        if np.isfinite(par_vals).any():
            par_pos = int(np.nanargmax(par_vals))
            ax.annotate(
                f"max par: {par_vals[par_pos]:.1f}x",
                (x[par_pos] + width / 2, par_vals[par_pos]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=BLUE_LIGHT,
                weight="bold",
            )

        ax.set_title(f"{func} Speedup", fontsize=16, weight="bold", pad=4)
        ax.set_xlabel("Trajectory Length")
        ax.set_ylabel("Speedup (log scale)" if i == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(df["trajectory_length"].astype(str))
        ax.set_yscale("log")
        ax.margins(y=0.15)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    fig.suptitle("Batch Computation Performance", fontsize=18, weight="bold", y=0.985)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.948),
            ncol=2,
            frameon=True,
        )

    fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.18)
    savefig(fig, FIGURE_DIR / "benchmark_batch_speedup.svg")


def fmt_ms(x):
    return f"{x:.4f} ms"


def fmt_x(x):
    return f"{x:.2f}x"


def md(df: pd.DataFrame):
    return df.to_markdown(index=False)


def render_tldr(comparison_df: pd.DataFrame, batch_df: pd.DataFrame):
    rs_py = comparison_df[COMPARE_RS_PY].mean()
    rs_cy = comparison_df[COMPARE_RS_CY].mean()
    best_batch = batch_df["Speedup (Par)"].max()

    lines = [
        "## TL;DR",
        "",
        f"- `{LABEL_RS}` is on average **~{rs_py:.0f}x faster** than `{LABEL_PY}`",
        f"- `{LABEL_RS}` is on average **~{rs_cy:.1f}x faster** than `{LABEL_CY}`",
        f"- Parallel batch execution reaches up to **~{best_batch:.1f}x speedup** over `{LABEL_CY}` on large inputs",
        "",
    ]
    return "\n".join(lines)


def render_scope():
    return "\n".join(
        [
            "## Benchmark Scope",
            "",
            f"This report compares `{LABEL_RS}`, `{LABEL_CY}`, and `{LABEL_PY}`.",
            "",
            "Algorithms covered:",
            "",
            "- DISCRET_FRECHET",
            "- DTW",
            "- EDR",
            "- ERP",
            "- HAUSDORFF",
            "- LCSS",
            "- SSPD",
            "",
            "Distance types:",
            "",
            "- euclidean",
            "- spherical",
            "",
            "All summary values use the **median runtime** across benchmark samples.",
            "Batch benchmarks compare Rust and Cython implementations.",
            "",
        ]
    )


def render_summary_table(comparison_df: pd.DataFrame):
    summary = comparison_df.copy()
    summary[LABEL_RS] = summary[LABEL_RS].map(fmt_ms)
    summary[LABEL_CY] = summary[LABEL_CY].map(fmt_ms)
    summary[LABEL_PY] = summary[LABEL_PY].map(fmt_ms)
    summary[COMPARE_RS_CY] = summary[COMPARE_RS_CY].map(fmt_x)
    summary[COMPARE_RS_PY] = summary[COMPARE_RS_PY].map(fmt_x)

    return "\n".join(
        [
            "## Summary Table",
            "",
            md(summary),
            "",
        ]
    )


def render_key_findings(comparison_df: pd.DataFrame, batch_df: pd.DataFrame):
    cy_best = comparison_df.loc[comparison_df[COMPARE_RS_CY].idxmax()]
    py_best = comparison_df.loc[comparison_df[COMPARE_RS_PY].idxmax()]

    eu = comparison_df[comparison_df["distance_type"] == "euclidean"]
    sp = comparison_df[comparison_df["distance_type"] == "spherical"]

    batch_best_pdist = batch_df[batch_df["function"] == "pdist"]["Speedup (Par)"].max()
    batch_best_cdist = batch_df[batch_df["function"] == "cdist"]["Speedup (Par)"].max()

    lines = [
        "## Key Findings",
        "",
        f"- Against `{LABEL_CY}`, the largest single-case speedup is **{cy_best[COMPARE_RS_CY]:.2f}x** on **{cy_best['algorithm']} ({cy_best['distance_type']})**.",
        f"- Against `{LABEL_PY}`, the largest single-case speedup is **{py_best[COMPARE_RS_PY]:.2f}x** on **{py_best['algorithm']} ({py_best['distance_type']})**.",
        f"- On **euclidean** benchmarks, `{LABEL_RS}` is on average **{eu[COMPARE_RS_CY].mean():.2f}x** faster than `{LABEL_CY}` and **{eu[COMPARE_RS_PY].mean():.2f}x** faster than `{LABEL_PY}`.",
        f"- On **spherical** benchmarks, `{LABEL_RS}` is on average **{sp[COMPARE_RS_CY].mean():.2f}x** faster than `{LABEL_CY}` and **{sp[COMPARE_RS_PY].mean():.2f}x** faster than `{LABEL_PY}`.",
        f"- In batch mode, parallel Rust reaches up to **{batch_best_pdist:.2f}x** speedup on `pdist` and **{batch_best_cdist:.2f}x** on `cdist`.",
        "",
    ]
    return "\n".join(lines)


def render_batch_section(batch_df: pd.DataFrame, batch_results):
    first = batch_results["rust"].row(0, named=True)
    traj_lengths = sorted(batch_results["rust"]["trajectory_length"].unique().to_list())
    dist_types = sorted(batch_results["rust"]["distance_type"].unique().to_list())

    out = batch_df.copy()
    out = out.rename(
        columns={
            "function": "Function",
            "distance_type": "Distance Type",
            "trajectory_length": "Traj Length",
            "num_distances": "Distances",
        }
    )

    for col in [f"{LABEL_CY} (ms)", f"{LABEL_RS} Seq (ms)", f"{LABEL_RS} Par (ms)"]:
        out[col] = out[col].map(fmt_ms)
    for col in ["Speedup (Seq)", "Speedup (Par)"]:
        out[col] = out[col].map(fmt_x)

    batch_table = out[
        [
            "Function",
            "Distance Type",
            "Traj Length",
            "Distances",
            f"{LABEL_CY} (ms)",
            f"{LABEL_RS} Seq (ms)",
            f"{LABEL_RS} Par (ms)",
            "Speedup (Seq)",
            "Speedup (Par)",
        ]
    ]

    notes = [
        "### Notes",
        "",
        f"- `{LABEL_RS}` sequential already outperforms `{LABEL_CY}` across the tested batch cases.",
        "- Parallel Rust provides the largest gains on longer trajectories.",
        "- For small inputs, parallel overhead can outweigh the benefits of parallel execution.",
        "",
    ]

    return "\n".join(
        [
            "## Batch Computation",
            "",
            "### Configuration",
            "",
            f"- **Algorithm**: {first['algorithm']}",
            f"- **Number of trajectories**: {first['num_trajectories']} (fixed)",
            f"- **pdist computation**: {first['num_trajectories'] * (first['num_trajectories'] - 1) // 2} distances",
            f"- **Trajectory lengths tested**: {', '.join(map(str, traj_lengths))}",
            f"- **Distance types**: {', '.join(dist_types)}",
            "",
            "### Results",
            "",
            md(batch_table),
            "",
            *notes,
        ]
    )


def render_report(comparison_df: pd.DataFrame, batch_df: pd.DataFrame, batch_results):
    parts = [
        "# Performance Benchmark Report",
        "",
        "Benchmarks are summarized using the **median runtime**.",
        "",
        render_tldr(comparison_df, batch_df),
        render_scope(),
        "## Figures",
        "",
        f"![README overview figure]({FIGURE_REL_DIR}/benchmark_speedup_readme.svg)",
        "",
        f"![Speedup by algorithm and distance type]({FIGURE_REL_DIR}/benchmark_speedup_by_distance.svg)",
        "",
        f"![Batch computation speedup]({FIGURE_REL_DIR}/benchmark_batch_speedup.svg)",
        "",
        render_summary_table(comparison_df),
        render_key_findings(comparison_df, batch_df),
        render_batch_section(batch_df, batch_results),
    ]
    return "\n".join(parts)


def readme_snippet(comparison_df: pd.DataFrame, batch_df: pd.DataFrame):
    rs_py = comparison_df[COMPARE_RS_PY].mean()
    rs_cy = comparison_df[COMPARE_RS_CY].mean()
    batch_best = batch_df["Speedup (Par)"].max()

    return f"""## Performance Overview

![traj-dist-rs benchmark speedup]({README_IMAGE_PATH})

**Median benchmark summary**:
- **~{rs_py:.0f}x faster than `{LABEL_PY}`** on average
- **~{rs_cy:.1f}x faster than `{LABEL_CY}`** on average
- Parallel batch `pdist` / `cdist` reaches up to **~{batch_best:.1f}x speedup** on large inputs

See [performance.md]({README_PERFORMANCE_PATH}) for the full benchmark report and additional plots.
"""


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    results = load_main_results(output_dir)
    batch_results = load_batch_results(output_dir)

    summary_df = build_summary_df(results)
    comparison_df = build_comparison_df(summary_df)
    batch_df = build_batch_df(batch_results)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    plot_overview(comparison_df)
    plot_by_distance(comparison_df)
    plot_batch(batch_df)

    REPORT_FILE.write_text(
        render_report(comparison_df, batch_df, batch_results),
        encoding="utf-8",
    )

    print("\nREADME MARKDOWN SNIPPET\n")
    print(readme_snippet(comparison_df, batch_df))


if __name__ == "__main__":
    main()
