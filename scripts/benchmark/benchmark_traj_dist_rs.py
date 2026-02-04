#!/usr/bin/env python3
"""
traj-dist-rs 性能测试脚本

支持命令行参数指定预热次数和测试次数。
将结果保存为 parquet 文件，包含算法、距离类型、超参数等元信息。
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

import traj_dist_rs._lib as traj_dist_rs


def parse_args():
    parser = argparse.ArgumentParser(
        description="traj-dist-rs 性能测试脚本"
    )
    parser.add_argument(
        "--baseline-file",
        type=str,
        default="output/baseline_trajectories.parquet",
        help="基准轨迹文件路径",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="algorithms_config.json",
        help="算法配置文件路径",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="预热次数，默认 5 次",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="测试次数，默认 10 次",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="输出目录",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    """加载算法配置"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_baseline_trajectories(baseline_path: Path) -> pl.DataFrame:
    """加载基准轨迹数据"""
    print(f"加载基准轨迹数据: {baseline_path}")
    df = pl.read_parquet(baseline_path)
    print(f"共加载 {len(df)} 个轨迹对")
    return df


def get_rust_function(algorithm_name: str):
    """获取 Rust 算法函数"""
    # 算法名称到函数名的映射
    function_map = {
        "sspd": "sspd",
        "dtw": "dtw",
        "hausdorff": "hausdorff",
        "discret_frechet": "discret_frechet",
        "frechet": "frechet",
        "lcss": "lcss",
        "edr": "edr",
        "erp": "erp_compat_traj_dist",
        "sowd_grid": "sowd_grid",
    }

    if algorithm_name not in function_map:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    function_name = function_map[algorithm_name]
    return getattr(traj_dist_rs, function_name)


def benchmark_algorithm(
    algorithm_config: Dict[str, Any],
    distance_type: str,
    baseline_df: pl.DataFrame,
    warmup_runs: int,
    num_runs: int,
) -> List[Dict[str, Any]]:
    """
    对单个算法的特定配置进行性能测试

    Returns:
        测试结果列表，每个元素包含算法名称、距离类型、超参数、时间列表等
    """
    algorithm_name = algorithm_config["name"]
    has_hyperparameters = algorithm_config["has_hyperparameters"]

    # 获取距离函数
    try:
        rust_func = get_rust_function(algorithm_name)
    except ValueError:
        print(f"警告: {algorithm_name} 在 Rust 实现中不可用，跳过")
        return []

    # 构建调用参数
    call_params = {}
    hyperparameter_value = None

    if has_hyperparameters:
        hyperparam = algorithm_config["hyperparameter"]
        hyperparam_name = hyperparam["name"]
        hyperparam_type = hyperparam["type"]
        hyperparameter_value = hyperparam["values"][0]  # 只取第一个超参数值

        if hyperparam_type == "list":
            call_params[hyperparam_name] = np.array(hyperparameter_value, dtype=np.float64)
        else:
            call_params[hyperparam_name] = hyperparameter_value

    results = []

    print(f"\n测试: {algorithm_name} ({distance_type}) - Rust")
    if hyperparameter_value is not None:
        print(f"  超参数: {hyperparam_name} = {hyperparameter_value}")

    for row_idx in range(len(baseline_df)):
        row = baseline_df[row_idx]
        traj1 = row["traj1"].item().to_numpy()
        traj2 = row["traj2"].item().to_numpy()

        # 预热
        for _ in range(warmup_runs):
            rust_func(traj1, traj2, distance_type, **call_params)

        # 测量时间
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            result = rust_func(traj1, traj2, distance_type, **call_params)
            end = time.perf_counter()
            times.append(end - start)

        # 存储结果
        result_dict = {
            "algorithm": algorithm_name,
            "distance_type": distance_type,
            "implementation": "rust",
            "traj_index": row_idx,
        }

        if has_hyperparameters:
            hyperparam = algorithm_config["hyperparameter"]
            hyperparam_name = hyperparam["name"]
            hyperparam_type = hyperparam["type"]
            hyperparameter_value = hyperparam["values"][0]
            if hyperparam_type == "list":
                result_dict[f"hyperparam_{hyperparam_name}"] = json.dumps(hyperparameter_value)
            else:
                result_dict[f"hyperparam_{hyperparam_name}"] = str(hyperparameter_value)

        result_dict["times"] = times

        results.append(result_dict)

    return results


def save_results(results: List[Dict[str, Any]], output_path: Path):
    """将结果保存为 parquet 文件"""
    # 准备数据
    data = {}

    # 提取所有可能的列名
    columns = set()
    for result in results:
        columns.update(result.keys())
        columns.discard("times")

    # 添加 times 列
    columns.add("times")

    # 按列整理数据
    for col in columns:
        if col == "times":
            data[col] = [r.get(col, []) for r in results]
        else:
            data[col] = [r.get(col, None) for r in results]

    # 定义 schema
    schema_fields = []
    for col in sorted(columns):
        if col == "times":
            schema_fields.append((col, pa.list_(pa.float64())))
        elif col.startswith("hyperparam_"):
            schema_fields.append((col, pa.string()))  # 超参数作为字符串存储
        elif col == "algorithm":
            schema_fields.append((col, pa.string()))
        elif col == "distance_type":
            schema_fields.append((col, pa.string()))
        elif col == "implementation":
            schema_fields.append((col, pa.string()))
        elif col == "traj_index":
            schema_fields.append((col, pa.int64()))
        else:
            schema_fields.append((col, pa.float64()))

    schema = pa.schema(schema_fields)

    # 创建表格
    table = pa.table(data, schema=schema)

    # 保存
    pq.write_table(table, output_path)
    print(f"已保存 {len(results)} 个结果到: {output_path}")


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    config_path = Path(args.config_file)
    if not config_path.exists():
        config_path = Path(__file__).parent / args.config_file

    config = load_config(config_path)

    # 加载基准轨迹
    baseline_path = Path(args.baseline_file)
    if not baseline_path.exists():
        baseline_path = output_dir / args.baseline_file

    baseline_df = load_baseline_trajectories(baseline_path)

    # 运行所有算法测试
    all_results = []

    for algorithm_config in config["algorithms"]:
        algorithm_name = algorithm_config["name"]

        for distance_type in algorithm_config["distance_types"]:
            results = benchmark_algorithm(
                algorithm_config,
                distance_type,
                baseline_df,
                args.warmup_runs,
                args.num_runs,
            )
            all_results.extend(results)

    # 保存结果
    output_file = "traj_dist_rs_rust_benchmark.parquet"
    output_path = output_dir / output_file
    save_results(all_results, output_path)

    print(f"\n测试完成! 共测试 {len(all_results)} 个用例")


if __name__ == "__main__":
    main()