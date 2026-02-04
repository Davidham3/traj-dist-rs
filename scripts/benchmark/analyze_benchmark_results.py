#!/usr/bin/env python3
"""
结果汇总分析脚本

读取三种类型的 parquet 文件（traj_dist_cython_benchmark.parquet、
traj_dist_python_benchmark.parquet、traj_dist_rs_rust_benchmark.parquet），
汇总耗时明细，并给出数据分析结果。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import polars as pl


def parse_args():
    parser = argparse.ArgumentParser(
        description="结果汇总分析脚本"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="输出目录",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="benchmark_analysis_report.md",
        help="输出报告文件名",
    )
    parser.add_argument(
        "--use-median",
        action="store_true",
        default=True,
        help="使用中位数进行统计（默认）",
    )
    parser.add_argument(
        "--use-mean",
        action="store_true",
        default=False,
        help="使用平均值进行统计",
    )
    return parser.parse_args()


def load_benchmark_results(output_dir: Path) -> Dict[str, pl.DataFrame]:
    """加载所有 benchmark 结果"""
    results = {}

    # 尝试加载各个实现的结果
    files = {
        "cython": "traj_dist_cython_benchmark.parquet",
        "python": "traj_dist_python_benchmark.parquet",
        "rust": "traj_dist_rs_rust_benchmark.parquet",
    }

    for impl, filename in files.items():
        filepath = output_dir / filename
        if filepath.exists():
            print(f"加载 {impl} 结果: {filepath}")
            results[impl] = pl.read_parquet(filepath)
        else:
            print(f"警告: {impl} 结果文件不存在: {filepath}")

    return results


def calculate_statistics(
    times_list: List[float], use_median: bool = True
) -> Dict[str, float]:
    """计算统计指标"""
    times_array = np.array(times_list)

    if use_median:
        central_value = np.median(times_array)
    else:
        central_value = np.mean(times_array)

    return {
        "mean": float(np.mean(times_array)),
        "median": float(np.median(times_array)),
        "std": float(np.std(times_array)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
        "central": float(central_value),  # 用于比较的中心值
        "cv": float(np.std(times_array) / np.mean(times_array)) if np.mean(times_array) > 0 else 0.0,
    }


def group_and_analyze(
    df: pl.DataFrame, use_median: bool = True
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    按算法、距离类型、超参数分组并分析

    Returns:
        {(algorithm, distance_type, hyperparam_str): {"times": [...], "stats": {...}}}
    """
    grouped = {}

    # 获取超参数列
    hyperparam_cols = [col for col in df.columns if col.startswith("hyperparam_")]

    if not hyperparam_cols:
        # 没有超参数的情况
        for row in df.iter_rows(named=True):
            key = (row["algorithm"], row["distance_type"], "")
            if key not in grouped:
                grouped[key] = {"times": []}
            grouped[key]["times"].extend(row["times"])
    else:
        # 有超参数的情况
        for row in df.iter_rows(named=True):
            # 构建超参数字符串（使用分号分隔，避免与markdown表格冲突）
            hyperparam_str = "; ".join(
                f"{col.replace('hyperparam_', '')}={row[col]}"
                for col in hyperparam_cols
            )
            key = (row["algorithm"], row["distance_type"], hyperparam_str)
            if key not in grouped:
                grouped[key] = {"times": []}
            grouped[key]["times"].extend(row["times"])

    # 计算统计信息
    for key in grouped:
        grouped[key]["stats"] = calculate_statistics(grouped[key]["times"], use_median)

    return grouped


def compare_implementations(
    results: Dict[str, pl.DataFrame],
    use_median: bool = True,
) -> List[Dict[str, Any]]:
    """
    比较不同实现的性能

    Returns:
        比较结果列表
    """
    # 获取所有唯一的 (algorithm, distance_type, hyperparam) 组合
    all_keys = set()

    for impl, df in results.items():
        grouped = group_and_analyze(df, use_median)
        all_keys.update(grouped.keys())

    # 比较结果
    comparison_results = []

    for key in sorted(all_keys):
        algorithm, distance_type, hyperparam_str = key

        result = {
            "algorithm": algorithm,
            "distance_type": distance_type,
            "hyperparam": hyperparam_str if hyperparam_str else "N/A",
            "implementations": {},
        }

        # 获取各实现的统计信息
        for impl, df in results.items():
            grouped = group_and_analyze(df, use_median)
            if key in grouped:
                stats = grouped[key]["stats"]
                result["implementations"][impl] = stats

        # 计算性能提升
        impl_stats = result["implementations"]

        if "rust" in impl_stats and "cython" in impl_stats:
            rust_time = impl_stats["rust"]["central"]
            cython_time = impl_stats["cython"]["central"]
            if rust_time > 0:
                result["rust_vs_cython"] = cython_time / rust_time

        if "rust" in impl_stats and "python" in impl_stats:
            rust_time = impl_stats["rust"]["central"]
            python_time = impl_stats["python"]["central"]
            if rust_time > 0:
                result["rust_vs_python"] = python_time / rust_time

        if "cython" in impl_stats and "python" in impl_stats:
            cython_time = impl_stats["cython"]["central"]
            python_time = impl_stats["python"]["central"]
            if cython_time > 0:
                result["cython_vs_python"] = python_time / cython_time

        comparison_results.append(result)

    return comparison_results


def generate_markdown_report(
    comparison_results: List[Dict[str, Any]],
    use_median: bool = True,
) -> str:
    """生成 Markdown 格式的报告"""
    central_metric = "中位数" if use_median else "平均值"

    lines = []
    lines.append("# 性能测试报告")
    lines.append("")
    lines.append(f"**统计指标**: {central_metric}")
    lines.append("")

    # 汇总表
    lines.append("## 汇总表")
    lines.append("")
    # 汇总表列顺序：算法、距离类型、超参数、Rust、Cython、Python、Rust/Cython、Rust/Python
    lines.append("| 算法 | 距离类型 | 超参数 | Rust | Cython | Python | Rust/Cython | Rust/Python |")
    lines.append("|------|----------|--------|------|--------|--------|-------------|-------------|")

    for result in comparison_results:
        impl_stats = result["implementations"]

        rust_str = "N/A"
        cython_str = "N/A"
        python_str = "N/A"
        rust_c_str = "N/A"
        rust_p_str = "N/A"

        if "rust" in impl_stats:
            stats = impl_stats["rust"]
            rust_str = f"{stats['central']*1000:.4f}ms"
        if "cython" in impl_stats:
            stats = impl_stats["cython"]
            cython_str = f"{stats['central']*1000:.4f}ms"
        if "python" in impl_stats:
            stats = impl_stats["python"]
            python_str = f"{stats['central']*1000:.4f}ms"

        if "rust_vs_cython" in result:
            rust_c_str = f"{result['rust_vs_cython']:.2f}x"
        if "rust_vs_python" in result:
            rust_p_str = f"{result['rust_vs_python']:.2f}x"

        lines.append(
            f"| {result['algorithm']} | {result['distance_type']} | {result['hyperparam']} | "
            f"{rust_str} | {cython_str} | {python_str} | {rust_c_str} | {rust_p_str} |"
        )

    lines.append("")

    # 详细统计表
    lines.append("## 详细统计")
    lines.append("")

    for result in comparison_results:
        algorithm = result["algorithm"]
        distance_type = result["distance_type"]
        hyperparam = result["hyperparam"]

        lines.append(f"### {algorithm.upper()} ({distance_type})")
        if hyperparam != "N/A":
            lines.append(f"超参数: {hyperparam}")
        lines.append("")

        impl_stats = result["implementations"]

        # 创建耗时统计表格 - 顺序：Rust、Cython、Python
        lines.append("#### 耗时统计")
        lines.append("")
        lines.append("| 实现 | 中位数 (ms) | 平均值 (ms) | 标准差 (ms) | 最小值 (ms) | 最大值 (ms) | 变异系数 (%) |")
        lines.append("|------|-------------|-------------|-------------|-------------|-------------|--------------|")

        for impl in ["rust", "cython", "python"]:
            if impl in impl_stats:
                stats = impl_stats[impl]
                lines.append(
                    f"| {impl.upper()} | {stats['median']*1000:.4f} | {stats['mean']*1000:.4f} | "
                    f"{stats['std']*1000:.4f} | {stats['min']*1000:.4f} | {stats['max']*1000:.4f} | {stats['cv']*100:.2f} |"
                )

        lines.append("")

        # 性能提升 - 重点突出 Rust vs Cython，删除 Cython vs Python
        lines.append("#### 性能提升")
        lines.append("")

        if "rust_vs_cython" in result:
            lines.append(f"- **Rust vs Cython**: {result['rust_vs_cython']:.2f}x")
        if "rust_vs_python" in result:
            lines.append(f"- Rust vs Python: {result['rust_vs_python']:.2f}x")

        lines.append("")

    # 按算法分析
    lines.append("## 按算法分析")
    lines.append("")
    lines.append("各算法在不同实现下的性能表现对比：")
    lines.append("")

    # 按算法分组
    algorithm_stats = {}
    for result in comparison_results:
        algorithm = result["algorithm"]
        if algorithm not in algorithm_stats:
            algorithm_stats[algorithm] = []
        algorithm_stats[algorithm].append(result)

    for algorithm in sorted(algorithm_stats.keys()):
        lines.append(f"### {algorithm.upper()}")
        lines.append("")

        rust_vs_cython_speedups = []
        rust_vs_python_speedups = []

        for result in algorithm_stats[algorithm]:
            if "rust_vs_cython" in result:
                rust_vs_cython_speedups.append(result["rust_vs_cython"])
            if "rust_vs_python" in result:
                rust_vs_python_speedups.append(result["rust_vs_python"])

        if rust_vs_cython_speedups:
            avg = np.mean(rust_vs_cython_speedups)
            lines.append(f"- **Rust vs Cython**: 平均提升 {avg:.2f}x (范围: {np.min(rust_vs_cython_speedups):.2f}x - {np.max(rust_vs_cython_speedups):.2f}x)")
        if rust_vs_python_speedups:
            avg = np.mean(rust_vs_python_speedups)
            lines.append(f"- Rust vs Python: 平均提升 {avg:.2f}x (范围: {np.min(rust_vs_python_speedups):.2f}x - {np.max(rust_vs_python_speedups):.2f}x)")

        lines.append("")

    # 按距离类型分析
    lines.append("## 按距离类型分析")
    lines.append("")
    lines.append("不同距离类型下的性能表现对比：")
    lines.append("")

    # 按距离类型分组
    distance_type_stats = {
        "euclidean": [],
        "spherical": []
    }

    for result in comparison_results:
        distance_type = result["distance_type"]
        if distance_type in distance_type_stats:
            distance_type_stats[distance_type].append(result)

    for distance_type in ["euclidean", "spherical"]:
        lines.append(f"### {distance_type.upper()} 距离")
        lines.append("")

        rust_vs_cython_speedups = []
        rust_vs_python_speedups = []

        for result in distance_type_stats[distance_type]:
            if "rust_vs_cython" in result:
                rust_vs_cython_speedups.append(result["rust_vs_cython"])
            if "rust_vs_python" in result:
                rust_vs_python_speedups.append(result["rust_vs_python"])

        if rust_vs_cython_speedups:
            avg = np.mean(rust_vs_cython_speedups)
            lines.append(f"- **Rust vs Cython**: 平均提升 {avg:.2f}x (范围: {np.min(rust_vs_cython_speedups):.2f}x - {np.max(rust_vs_cython_speedups):.2f}x)")
        if rust_vs_python_speedups:
            avg = np.mean(rust_vs_python_speedups)
            lines.append(f"- Rust vs Python: 平均提升 {avg:.2f}x (范围: {np.min(rust_vs_python_speedups):.2f}x - {np.max(rust_vs_python_speedups):.2f}x)")

        # 找出该距离类型下性能提升最高的算法（基于 Rust vs Cython）
        if rust_vs_cython_speedups:
            max_speedup = np.max(rust_vs_cython_speedups)
            best_algo = None
            for result in distance_type_stats[distance_type]:
                if "rust_vs_cython" in result and result["rust_vs_cython"] == max_speedup:
                    best_algo = result["algorithm"]
                    break
            lines.append(f"- **最佳性能提升算法**: {best_algo} ({max_speedup:.2f}x)")

        lines.append("")

    # 总体统计
    lines.append("## 总体统计")
    lines.append("")

    # 计算各实现的总体平均时间
    overall_stats = {
        "python": {"times": []},
        "cython": {"times": []},
        "rust": {"times": []},
    }

    for result in comparison_results:
        for impl, stats in result["implementations"].items():
            if impl in overall_stats:
                overall_stats[impl]["times"].append(stats["central"])

    # 按顺序输出：Rust、Cython、Python
    for impl in ["rust", "cython", "python"]:
        if overall_stats[impl]["times"]:
            avg_time = np.mean(overall_stats[impl]["times"])
            lines.append(f"- {impl.upper()} 总体平均时间: {avg_time*1000:.4f} ms")

    lines.append("")

    # 计算总体性能提升：Rust vs Cython、Rust vs Python
    if "rust" in overall_stats and "cython" in overall_stats:
        if overall_stats["rust"]["times"] and overall_stats["cython"]["times"]:
            rust_avg = np.mean(overall_stats["rust"]["times"])
            cython_avg = np.mean(overall_stats["cython"]["times"])
            if rust_avg > 0:
                lines.append(f"- Rust vs Cython 总体平均提升: {cython_avg/rust_avg:.2f}x")

    if "rust" in overall_stats and "python" in overall_stats:
        if overall_stats["rust"]["times"] and overall_stats["python"]["times"]:
            rust_avg = np.mean(overall_stats["rust"]["times"])
            python_avg = np.mean(overall_stats["python"]["times"])
            if rust_avg > 0:
                lines.append(f"- Rust vs Python 总体平均提升: {python_avg/rust_avg:.2f}x")

    return "\n".join(lines)


def main():
    args = parse_args()

    # 确定使用中位数还是平均值
    use_median = args.use_median and not args.use_mean

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有 benchmark 结果
    results = load_benchmark_results(output_dir)

    if not results:
        raise ValueError("错误: 没有找到任何 benchmark 结果文件")

    # 比较实现
    comparison_results = compare_implementations(results, use_median)

    # 生成报告
    report = generate_markdown_report(comparison_results, use_median)

    # 保存报告
    output_path = output_dir / args.output_file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n报告已保存到: {output_path}")

    # 打印汇总表
    print("\n" + "=" * 100)
    print("性能对比汇总")
    print("=" * 100)
    print(f"\n{'算法':<15} {'距离类型':<12} {'超参数':<20} {'Python(ms)':<15} {'Cython(ms)':<15} {'Rust(ms)':<15} {'Rust/C':<10} {'Rust/P':<10} {'C/P':<10}")
    print("-" * 140)

    for result in comparison_results:
        impl_stats = result["implementations"]

        python_str = "N/A"
        cython_str = "N/A"
        rust_str = "N/A"
        rust_c_str = "N/A"
        rust_p_str = "N/A"
        c_p_str = "N/A"

        if "python" in impl_stats:
            stats = impl_stats["python"]
            python_str = f"{stats['central']*1000:.4f}"
        if "cython" in impl_stats:
            stats = impl_stats["cython"]
            cython_str = f"{stats['central']*1000:.4f}"
        if "rust" in impl_stats:
            stats = impl_stats["rust"]
            rust_str = f"{stats['central']*1000:.4f}"

        if "rust_vs_cython" in result:
            rust_c_str = f"{result['rust_vs_cython']:.2f}x"
        if "rust_vs_python" in result:
            rust_p_str = f"{result['rust_vs_python']:.2f}x"
        if "cython_vs_python" in result:
            c_p_str = f"{result['cython_vs_python']:.2f}x"

        hyperparam_str = result["hyperparam"] if result["hyperparam"] != "N/A" else ""

        print(
            f"{result['algorithm']:<15} {result['distance_type']:<12} {hyperparam_str:<20} "
            f"{python_str:<15} {cython_str:<15} {rust_str:<15} {rust_c_str:<10} {rust_p_str:<10} {c_p_str:<10}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()