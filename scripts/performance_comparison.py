#!/usr/bin/env python3
"""
性能对比基准测试脚本

比较 Rust、Python 和 Cython 实现的性能
"""

import timeit
from tqdm import trange
import traceback
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "py_tests"))

from test_framework import (
    load_all_metainfo_from_data_dir,
    get_algorithm_config_from_metainfo,
    get_hyperparameter_value_from_metainfo,
    get_sample_path,
)
from schemas import Metainfo
import traj_dist_rs._lib as traj_dist_rs

DATA_DIR = Path(__file__).parent.parent / "py_tests" / "data"

# 加载所有元数据
ALL_METAINFO = load_all_metainfo_from_data_dir(DATA_DIR)


def load_test_data(sample_path: Path) -> pl.DataFrame:
    """加载测试数据"""
    return pl.read_parquet(sample_path)


def get_rust_function(
    algorithm_name: str,
    distance_type: str,
    hyperparameter_value: Any = None,
    use_standard: bool = False,
):
    """获取 Rust 算法函数"""
    config = get_algorithm_config_from_metainfo(algorithm_name)
    function_name = config["function_name"]

    # 获取 Rust 函数
    # ERP 算法有两个版本：compat_traj_dist 和 standard
    if algorithm_name == "erp":
        if use_standard:
            rust_func = getattr(traj_dist_rs, "erp_standard")
        else:
            rust_func = getattr(traj_dist_rs, "erp_compat_traj_dist")
    else:
        rust_func = getattr(traj_dist_rs, function_name)

    def wrapper(traj1, traj2):
        if config["has_hyperparameters"]:
            if config["hyperparameter_name"] == "g":
                # ERP 算法有两个版本：compat_traj_dist 和 standard
                if algorithm_name == "erp":
                    if use_standard:
                        return traj_dist_rs.erp_standard(
                            traj1, traj2, distance_type, g=hyperparameter_value
                        )
                    else:
                        return traj_dist_rs.erp_compat_traj_dist(
                            traj1, traj2, distance_type, g=hyperparameter_value
                        )
                else:
                    return rust_func(
                        traj1, traj2, distance_type, g=hyperparameter_value
                    )
            elif config["hyperparameter_name"] == "eps":
                return rust_func(traj1, traj2, distance_type, eps=hyperparameter_value)
            elif config["hyperparameter_name"] == "precision":
                return rust_func(
                    traj1, traj2, distance_type, precision=hyperparameter_value
                )
        else:
            return rust_func(traj1, traj2, distance_type)

    return wrapper


def benchmark_implementation(
    metainfo: Metainfo, num_runs: int = 10
) -> Dict[str, Any]:
    """
    对单个算法的特定配置进行三种实现的性能 benchmark

    Args:
        metainfo: 元数据对象（Metainfo）
        num_runs: 每个测试用例运行次数（用于计算平均耗时）

    Returns:
        benchmark 结果字典
    """
    algorithm_name = metainfo.algorithm
    distance_type = metainfo.type_d  # 由于 use_enum_values=True，这已经是字符串
    hyperparameter_value = get_hyperparameter_value_from_metainfo(metainfo)

    # 加载测试数据
    sample_path = get_sample_path(metainfo, DATA_DIR)
    df = load_test_data(sample_path)

    rust_func = get_rust_function(algorithm_name, distance_type, hyperparameter_value)

    # 初始化统计变量
    cython_times = []
    rust_times = []
    num_test_cases = len(df)

    print(f"\n{'='*80}")
    print(f"算法: {algorithm_name.upper()} ({distance_type})")
    if hyperparameter_value is not None:
        print(f"超参数: {hyperparameter_value}")
    print(f"测试用例数: {num_test_cases}")
    print(f"{'='*80}")

    # 运行 benchmark
    for idx in trange(len(df)):
        row = df[idx]
        traj1 = row["traj1"].item().to_numpy()
        traj2 = row["traj2"].item().to_numpy()
        cython_time = row["time"].item()
        cython_times.append(cython_time)

        def tmp_test():
            return rust_func(traj1, traj2)

        rust_time = timeit.timeit(tmp_test, number=num_runs)
        rust_times.append(rust_time)

    # 计算统计结果
    avg_cython_time = np.mean(cython_times)
    max_cython_time = np.max(cython_times)
    avg_rust_time = np.mean(rust_times)
    max_rust_time = np.max(rust_times)

    # 计算性能提升倍数
    cython_vs_rust = avg_cython_time / avg_rust_time if avg_rust_time > 0 else 0

    result = {
        "algorithm": algorithm_name,
        "distance_type": distance_type,
        "hyperparameter": hyperparameter_value,
        "num_test_cases": num_test_cases,
        "cython_time": {
            "avg": avg_cython_time,
            "max": max_cython_time,
            "all": cython_times,
        },
        "rust_time": {"avg": avg_rust_time, "max": max_rust_time, "all": rust_times},
        "speedup": {
            "cython_vs_rust": cython_vs_rust,
        },
    }

    # 打印结果
    print("\n耗时统计 (ms):")
    print(f"  Cython 平均耗时: {avg_cython_time * 1000:.6f} ms")
    print(f"  Rust 平均耗时: {avg_rust_time * 1000:.6f} ms")

    print("\n性能提升:")
    print(f"  Rust vs Cython: {cython_vs_rust:.2f}x")

    return result


def run_performance_comparison(num_runs: int = 10):
    """运行完整的性能对比"""
    print("=" * 80)
    print("开始运行性能对比: Python vs Cython vs Rust")
    print("=" * 80)

    # 过滤掉未实现的算法
    implemented_algorithms = [
        "sspd",
        "dtw",
        "hausdorff",
        "lcss",
        "edr",
        "discret_frechet",
        "erp",
    ]

    # 运行 benchmark
    results = []
    for algorithm_name, metainfo_list in ALL_METAINFO.items():
        if algorithm_name not in implemented_algorithms:
            continue

        for metainfo in metainfo_list:
            try:
                results.append(benchmark_implementation(metainfo, num_runs))
            except Exception as e:
                print(f"\n错误: {algorithm_name} benchmark 失败: {e}")
                traceback.print_exc()

    # 生成汇总报告
    print("\n" + "=" * 80)
    print("性能对比汇总报告")
    print("=" * 80)

    print(
        f"\n{'函数名':<20} {'距离类型':<12} {'超参数':<20} {'Cython平均':<12} {'Rust平均':<12} {'Rust/C':<10}"
    )
    print("-" * 100)

    for result in results:
        algorithm = result["algorithm"]
        distance_type = result["distance_type"]
        avg_cython_time = result["cython_time"]["avg"]
        avg_rust_time = result["rust_time"]["avg"]
        c_vs_r = result["speedup"]["cython_vs_rust"]

        config = get_algorithm_config_from_metainfo(algorithm)
        function_name = config["function_name"]

        # 对于 ERP 算法，显示实际使用的函数名
        if algorithm == "erp":
            function_name = "erp_compat_traj_dist"

        # 从 result 中获取超参数值
        hyperparameter_value = result["hyperparameter"]

        # 格式化超参数
        if hyperparameter_value is not None:
            if isinstance(hyperparameter_value, list):
                hyperparam_str = (
                    f"[{hyperparameter_value[0]:.5f}, {hyperparameter_value[1]:.5f}]"
                )
            else:
                hyperparam_str = f"{hyperparameter_value}"
        else:
            hyperparam_str = "N/A"

        print(
            f"{function_name:<20} {distance_type:<12} {hyperparam_str:<20} {avg_cython_time*1000:>10.4f}ms {avg_rust_time*1000:>10.4f}ms {c_vs_r:>8.2f}x"
        )

    # 计算总体统计
    all_cython_times = [r["cython_time"]["avg"] for r in results]
    all_rust_times = [r["rust_time"]["avg"] for r in results]
    all_c_vs_r = [r["speedup"]["cython_vs_rust"] for r in results]

    print("\n" + "-" * 100)
    print(f"{'总体平均 Cython 时间:':<30} {np.mean(all_cython_times)*1000:.4f} ms")
    print(f"{'总体平均 Rust 时间:':<30} {np.mean(all_rust_times)*1000:.4f} ms")
    print(f"{'Rust vs Cython 平均提升:':<30} {np.mean(all_c_vs_r):.2f}x")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="运行轨迹距离算法性能对比 (Python vs Cython vs Rust)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="每个测试用例运行次数（用于计算平均耗时）",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="指定要测试的算法（不指定则测试所有已实现的算法）",
    )
    parser.add_argument(
        "--distance-type",
        type=str,
        default=None,
        help="指定距离类型（euclidean 或 spherical）",
    )

    args = parser.parse_args()

    if args.algorithm:
        metainfo_list = ALL_METAINFO.get(args.algorithm)
        if not metainfo_list:
            raise ValueError(f"算法 {args.algorithm} 不存在或未实现")

        distance_types = list({m.type_d for m in metainfo_list})

        if args.distance_type:
            metainfo_list = [
                m for m in metainfo_list if m.type_d == args.distance_type
            ]

        for metainfo in metainfo_list:
            try:
                benchmark_implementation(metainfo, args.num_runs)
            except Exception as e:
                print(f"\n错误: {args.algorithm} benchmark 失败: {e}")
                traceback.print_exc()

    else:
        # 测试所有已实现的算法
        run_performance_comparison(args.num_runs)
