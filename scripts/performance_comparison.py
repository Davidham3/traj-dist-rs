#!/usr/bin/env python3
"""
性能对比基准测试脚本

比较 Rust、Python 和 Cython 实现的性能
"""

import time
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "py_tests"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "traj-dist"))

from test_framework import (
    load_all_metainfo_from_data_dir,
    get_algorithm_config_from_metainfo,
    get_hyperparameter_value_from_metainfo,
    load_test_data_by_metainfo,
    get_sample_path,
)
import traj_dist_rs._lib as traj_dist_rs
import traj_dist.distance as traj_dist_cython


def load_test_data(sample_path: Path) -> pl.DataFrame:
    """加载测试数据"""
    try:
        return pl.read_parquet(sample_path)
    except Exception:
        # 如果 polars 失败，使用 pandas 作为备用方案
        df_pandas = pd.read_parquet(sample_path)
        return pl.from_pandas(df_pandas)


def get_python_function(algorithm_name: str, distance_type: str, hyperparameter_value: Any = None):
    """获取 traj-dist Python 实现函数"""
    config = get_algorithm_config_from_metainfo(algorithm_name)
    
    # 根据算法映射到 Python 实现
    if algorithm_name == "sspd":
        from traj_dist.pydist.sspd import e_sspd, s_sspd
        func = e_sspd if distance_type == "euclidean" else s_sspd
    elif algorithm_name == "dtw":
        from traj_dist.pydist.dtw import e_dtw, s_dtw
        func = e_dtw if distance_type == "euclidean" else s_dtw
    elif algorithm_name == "hausdorff":
        from traj_dist.pydist.hausdorff import e_hausdorff, s_hausdorff
        func = e_hausdorff if distance_type == "euclidean" else s_hausdorff
    elif algorithm_name == "lcss":
        from traj_dist.pydist.lcss import e_lcss, s_lcss
        func = e_lcss if distance_type == "euclidean" else s_lcss
    elif algorithm_name == "edr":
        from traj_dist.pydist.edr import e_edr, s_edr
        func = e_edr if distance_type == "euclidean" else s_edr
    elif algorithm_name == "erp":
        from traj_dist.pydist.erp import e_erp, s_erp
        func = e_erp if distance_type == "euclidean" else s_erp
    elif algorithm_name == "discret_frechet":
        from traj_dist.pydist.discret_frechet import discret_frechet
        func = discret_frechet
    elif algorithm_name == "frechet":
        from traj_dist.pydist.frechet import frechet
        func = frechet
    else:
        raise ValueError(f"未知算法: {algorithm_name}")

    def wrapper(traj1, traj2):
        # traj_dist 函数需要 NumPy 数组
        traj1_np = np.array(traj1, dtype=np.float64)
        traj2_np = np.array(traj2, dtype=np.float64)

        if config["has_hyperparameters"]:
            if config["hyperparameter_name"] == "g":
                return func(traj1_np, traj2_np, g=hyperparameter_value)
            elif config["hyperparameter_name"] == "eps":
                return func(traj1_np, traj2_np, eps=hyperparameter_value)
            elif config["hyperparameter_name"] == "precision":
                return func(traj1_np, traj2_np, precision=hyperparameter_value)
        else:
            return func(traj1_np, traj2_np)

    return wrapper


def get_cython_function(algorithm_name: str, distance_type: str, hyperparameter_value: Any = None):
    """获取 traj-dist Cython 实现函数"""
    config = get_algorithm_config_from_metainfo(algorithm_name)
    
    # 获取 Cython 函数
    cython_func_map = {
        'sspd': traj_dist_cython.sspd,
        'dtw': traj_dist_cython.dtw,
        'hausdorff': traj_dist_cython.hausdorff,
        'lcss': traj_dist_cython.lcss,
        'edr': traj_dist_cython.edr,
        'erp': traj_dist_cython.erp,
        'discret_frechet': traj_dist_cython.discret_frechet,
        'frechet': traj_dist_cython.frechet,
        'sowd_grid': traj_dist_cython.sowd_grid,
    }
    
    cython_func = cython_func_map.get(algorithm_name)
    if not cython_func:
        raise ValueError(f"Cython 中没有找到算法: {algorithm_name}")

    def wrapper(traj1, traj2):
        # Cython 实现也需要 NumPy 数组
        traj1_np = np.array(traj1, dtype=np.float64)
        traj2_np = np.array(traj2, dtype=np.float64)
        
        if config["has_hyperparameters"]:
            if config["hyperparameter_name"] == "g":
                return cython_func(traj1_np, traj2_np, type_d=distance_type, g=hyperparameter_value)
            elif config["hyperparameter_name"] == "eps":
                return cython_func(traj1_np, traj2_np, type_d=distance_type, eps=hyperparameter_value)
            elif config["hyperparameter_name"] == "precision":
                return cython_func(traj1_np, traj2_np, type_d=distance_type, precision=hyperparameter_value, converted=True)
        else:
            return cython_func(traj1_np, traj2_np, type_d=distance_type)

    return wrapper


def get_rust_function(algorithm_name: str, distance_type: str, hyperparameter_value: Any = None, use_standard: bool = False):
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
                        return traj_dist_rs.erp_standard(traj1, traj2, distance_type, g=hyperparameter_value)
                    else:
                        return traj_dist_rs.erp_compat_traj_dist(traj1, traj2, distance_type, g=hyperparameter_value)
                else:
                    return rust_func(traj1, traj2, distance_type, g=hyperparameter_value)
            elif config["hyperparameter_name"] == "eps":
                return rust_func(traj1, traj2, distance_type, eps=hyperparameter_value)
            elif config["hyperparameter_name"] == "precision":
                return rust_func(traj1, traj2, distance_type, precision=hyperparameter_value)
        else:
            return rust_func(traj1, traj2, distance_type)

    return wrapper


def benchmark_implementation(
    metainfo: Dict[str, Any],
    data_dir: Path,
    num_runs: int = 10
) -> Dict[str, Any]:
    """
    对单个算法的特定配置进行三种实现的性能 benchmark

    Args:
        metainfo: 元数据字典
        data_dir: 数据目录
        num_runs: 每个测试用例运行次数（用于计算平均耗时）

    Returns:
        benchmark 结果字典
    """
    algorithm_name = metainfo["algorithm"]
    distance_type = metainfo["type_d"]
    hyperparameter_value = get_hyperparameter_value_from_metainfo(metainfo)

    # 加载测试数据
    sample_path = get_sample_path(metainfo, data_dir)
    df = load_test_data(sample_path)

    # 获取三种实现的函数
    python_func = get_python_function(algorithm_name, distance_type, hyperparameter_value)
    cython_func = get_cython_function(algorithm_name, distance_type, hyperparameter_value)
    rust_func = get_rust_function(algorithm_name, distance_type, hyperparameter_value)

    # 初始化统计变量
    python_times = []
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
    for idx, row in enumerate(df.iter_rows(named=True)):
        traj1 = row["traj_1"]
        traj2 = row["traj_2"]

        # 测量 Python 耗时
        python_times_run = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            python_result = python_func(traj1, traj2)
            end_time = time.perf_counter()
            python_times_run.append(end_time - start_time)

        python_time = np.mean(python_times_run)
        python_times.append(python_time)

        # 测量 Cython 耗时
        cython_times_run = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            cython_result = cython_func(traj1, traj2)
            end_time = time.perf_counter()
            cython_times_run.append(end_time - start_time)

        cython_time = np.mean(cython_times_run)
        cython_times.append(cython_time)

        # 测量 Rust 耗时
        rust_times_run = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            rust_result = rust_func(traj1, traj2)
            end_time = time.perf_counter()
            rust_times_run.append(end_time - start_time)

        rust_time = np.mean(rust_times_run)
        rust_times.append(rust_time)

        # 打印进度
        if (idx + 1) % 10 == 0 or idx == num_test_cases - 1:
            print(f"进度: {idx + 1}/{num_test_cases} ({(idx + 1) / num_test_cases * 100:.1f}%)")

    # 计算统计结果
    avg_python_time = np.mean(python_times)
    max_python_time = np.max(python_times)
    avg_cython_time = np.mean(cython_times)
    max_cython_time = np.max(cython_times)
    avg_rust_time = np.mean(rust_times)
    max_rust_time = np.max(rust_times)

    # 计算性能提升倍数
    python_vs_cython = avg_python_time / avg_cython_time if avg_cython_time > 0 else 0
    python_vs_rust = avg_python_time / avg_rust_time if avg_rust_time > 0 else 0
    cython_vs_rust = avg_cython_time / avg_rust_time if avg_rust_time > 0 else 0

    result = {
        "algorithm": algorithm_name,
        "distance_type": distance_type,
        "hyperparameter": hyperparameter_value,
        "num_test_cases": num_test_cases,
        "python_time": {
            "avg": avg_python_time,
            "max": max_python_time,
            "all": python_times
        },
        "cython_time": {
            "avg": avg_cython_time,
            "max": max_cython_time,
            "all": cython_times
        },
        "rust_time": {
            "avg": avg_rust_time,
            "max": max_rust_time,
            "all": rust_times
        },
        "speedup": {
            "python_vs_cython": python_vs_cython,
            "python_vs_rust": python_vs_rust,
            "cython_vs_rust": cython_vs_rust
        }
    }

    # 打印结果
    print(f"\n耗时统计 (ms):")
    print(f"  Python 平均耗时: {avg_python_time * 1000:.6f} ms")
    print(f"  Cython 平均耗时: {avg_cython_time * 1000:.6f} ms") 
    print(f"  Rust 平均耗时: {avg_rust_time * 1000:.6f} ms")

    print(f"\n性能提升:")
    print(f"  Cython vs Python: {python_vs_cython:.2f}x")
    print(f"  Rust vs Python: {python_vs_rust:.2f}x")
    print(f"  Rust vs Cython: {cython_vs_rust:.2f}x")

    return result


def run_performance_comparison(num_runs: int = 10):
    """运行完整的性能对比"""
    print("="*80)
    print("开始运行性能对比: Python vs Cython vs Rust")
    print("="*80)

    # 获取数据目录
    data_dir = Path(__file__).parent.parent / "py_tests" / "data"

    # 加载所有元数据
    all_metainfo = load_all_metainfo_from_data_dir(data_dir)

    # 过滤掉未实现的算法
    implemented_algorithms = ["sspd", "dtw", "hausdorff", "lcss", "edr", "discret_frechet", "erp"]

    # 运行 benchmark
    results = []
    for algorithm_name, metainfo_list in all_metainfo.items():
        if algorithm_name not in implemented_algorithms:
            continue

        for metainfo in metainfo_list:
            try:
                result = benchmark_implementation(metainfo, data_dir, num_runs)
                results.append(result)
            except Exception as e:
                print(f"\n错误: {algorithm_name} benchmark 失败: {e}")
                import traceback
                traceback.print_exc()

    # 生成汇总报告
    print("\n" + "="*80)
    print("性能对比汇总报告")
    print("="*80)

    print(f"\n{'函数名':<20} {'距离类型':<12} {'超参数':<20} {'Python平均':<12} {'Cython平均':<12} {'Rust平均':<12} {'Cython/P':<10} {'Rust/P':<10} {'Rust/C':<10}")
    print("-"*100)

    for result in results:
        algorithm = result["algorithm"]
        distance_type = result["distance_type"]
        hyperparameter_value = result["hyperparameter"]
        avg_python_time = result["python_time"]["avg"]
        avg_cython_time = result["cython_time"]["avg"]
        avg_rust_time = result["rust_time"]["avg"]
        p_vs_c = result["speedup"]["python_vs_cython"]
        p_vs_r = result["speedup"]["python_vs_rust"]
        c_vs_r = result["speedup"]["cython_vs_rust"]

        config = get_algorithm_config_from_metainfo(algorithm)
        function_name = config["function_name"]

        # 对于 ERP 算法，显示实际使用的函数名
        if algorithm == "erp":
            function_name = "erp_compat_traj_dist"

        # 格式化超参数
        if hyperparameter_value is not None:
            if isinstance(hyperparameter_value, list):
                hyperparam_str = f"[{hyperparameter_value[0]:.5f}, {hyperparameter_value[1]:.5f}]"
            else:
                hyperparam_str = f"{hyperparameter_value}"
        else:
            hyperparam_str = "N/A"

        print(f"{function_name:<20} {distance_type:<12} {hyperparam_str:<20} {avg_python_time*1000:>10.4f}ms {avg_cython_time*1000:>10.4f}ms {avg_rust_time*1000:>10.4f}ms {p_vs_c:>8.2f}x {p_vs_r:>8.2f}x {c_vs_r:>8.2f}x")

    # 计算总体统计
    all_python_times = [r["python_time"]["avg"] for r in results]
    all_cython_times = [r["cython_time"]["avg"] for r in results]
    all_rust_times = [r["rust_time"]["avg"] for r in results]
    all_p_vs_c = [r["speedup"]["python_vs_cython"] for r in results]
    all_p_vs_r = [r["speedup"]["python_vs_rust"] for r in results]
    all_c_vs_r = [r["speedup"]["cython_vs_rust"] for r in results]

    print("\n" + "-"*100)
    print(f"{'总体平均 Python 时间:':<30} {np.mean(all_python_times)*1000:.4f} ms")
    print(f"{'总体平均 Cython 时间:':<30} {np.mean(all_cython_times)*1000:.4f} ms")
    print(f"{'总体平均 Rust 时间:':<30} {np.mean(all_rust_times)*1000:.4f} ms")
    print(f"{'Cython vs Python 平均提升:':<30} {np.mean(all_p_vs_c):.2f}x")
    print(f"{'Rust vs Python 平均提升:':<30} {np.mean(all_p_vs_r):.2f}x")
    print(f"{'Rust vs Cython 平均提升:':<30} {np.mean(all_c_vs_r):.2f}x")
    print("="*80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行轨迹距离算法性能对比 (Python vs Cython vs Rust)")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="每个测试用例运行次数（用于计算平均耗时）"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="指定要测试的算法（不指定则测试所有已实现的算法）"
    )
    parser.add_argument(
        "--distance-type",
        type=str,
        default=None,
        help="指定距离类型（euclidean 或 spherical）"
    )

    args = parser.parse_args()

    if args.algorithm:
        # 测试单个算法
        data_dir = Path(__file__).parent.parent / "py_tests" / "data"
        all_metainfo = load_all_metainfo_from_data_dir(data_dir)

        if args.algorithm in all_metainfo:
            metainfo_list = all_metainfo[args.algorithm]
            distance_types = list(set(m["type_d"] for m in metainfo_list))

            if args.distance_type:
                metainfo_list = [m for m in metainfo_list if m["type_d"] == args.distance_type]

            for metainfo in metainfo_list:
                try:
                    benchmark_implementation(metainfo, data_dir, args.num_runs)
                except Exception as e:
                    print(f"\n错误: {args.algorithm} benchmark 失败: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"算法 {args.algorithm} 不存在或未实现")
    else:
        # 测试所有已实现的算法
        run_performance_comparison(args.num_runs)