#!/usr/bin/env python3
"""
基准轨迹生成脚本

从 traj-dist/data/benchmark_trajectories.pkl 随机抽取 k 条轨迹 pair，
以 parquet 格式存储，使用 pa.large_list(pa.list_(pa.float64(), 2)) 类型。
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 pkl 文件生成基准轨迹数据"
    )
    parser.add_argument(
        "--pkl-file",
        type=str,
        default="../traj-dist/data/benchmark_trajectories.pkl",
        help="轨迹数据 pkl 文件路径",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="抽取的轨迹对数量（轨迹数量约为 sqrt(2*k)）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，确保结果可重现",
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
        default="baseline_trajectories.parquet",
        help="输出文件名",
    )
    return parser.parse_args()


def load_trajectories(pkl_path: Path) -> list:
    """加载轨迹数据"""
    print(f"加载轨迹数据: {pkl_path}")
    with open(pkl_path, "rb") as f:
        traj_list = pickle.load(f, encoding="latin1")
    print(f"共加载 {len(traj_list)} 条轨迹")
    return traj_list


def generate_trajectory_pairs(
    traj_list: list, num_pairs: int, seed: int
) -> list:
    """
    生成轨迹对

    Args:
        traj_list: 轨迹列表
        num_pairs: 需要生成的轨迹对数量
        seed: 随机种子

    Returns:
        轨迹对列表 [(traj1, traj2), ...]
    """
    random.seed(seed)
    np.random.seed(seed)

    # 计算需要的轨迹数量
    # 对于 n 条轨迹，可以生成 n*(n-1)/2 个不同的轨迹对
    # 所以如果需要 k 个轨迹对，大约需要 sqrt(2*k) 条轨迹
    num_traj = int(np.sqrt(2 * num_pairs)) + 1

    # 确保不超过可用轨迹数量
    num_traj = min(num_traj, len(traj_list))

    print(f"从 {len(traj_list)} 条轨迹中随机选择 {num_traj} 条")
    selected_indices = random.sample(range(len(traj_list)), num_traj)
    selected_trajectories = [traj_list[i] for i in selected_indices]

    # 生成所有可能的轨迹对（i < j，避免重复）
    pairs = []
    for i in range(num_traj):
        for j in range(i + 1, num_traj):
            pairs.append((selected_trajectories[i], selected_trajectories[j]))

    # 如果生成的对数超过需要的数量，随机抽取
    if len(pairs) > num_pairs:
        pairs = random.sample(pairs, num_pairs)

    print(f"生成了 {len(pairs)} 个轨迹对")
    return pairs


def save_to_parquet(pairs: list, output_path: Path):
    """将轨迹对保存为 parquet 文件"""
    traj1_list = []
    traj2_list = []

    for traj1, traj2 in pairs:
        traj1_list.append(traj1.tolist() if isinstance(traj1, np.ndarray) else traj1)
        traj2_list.append(traj2.tolist() if isinstance(traj2, np.ndarray) else traj2)

    # 定义 schema
    traj_type = pa.large_list(pa.list_(pa.float64(), 2))

    schema = pa.schema([
        ("traj1", traj_type),
        ("traj2", traj_type),
    ])

    # 创建表格
    table = pa.table({
        "traj1": traj1_list,
        "traj2": traj2_list,
    }, schema=schema)

    # 保存
    pq.write_table(table, output_path)
    print(f"已保存 {len(pairs)} 个轨迹对到: {output_path}")


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始的 pkl_file 参数（用于 metadata）
    original_pkl_file = args.pkl_file

    # 加载轨迹数据
    pkl_path = Path(args.pkl_file)
    # 如果是相对路径，相对于脚本所在目录解析
    if not pkl_path.is_absolute():
        pkl_path = (Path(__file__).parent / pkl_path).resolve()

    if not pkl_path.exists():
        # 尝试默认路径
        pkl_path = (
            Path(__file__).parent.parent.parent.parent
            / "traj-dist"
            / "data"
            / "benchmark_trajectories.pkl"
        ).resolve()
        original_pkl_file = "../traj-dist/data/benchmark_trajectories.pkl"

    traj_list = load_trajectories(pkl_path)

    # 生成轨迹对
    pairs = generate_trajectory_pairs(traj_list, args.k, args.seed)

    # 保存为 parquet
    output_path = output_dir / args.output_file
    save_to_parquet(pairs, output_path)

    # 保存元数据（使用相对路径）
    metadata = {
        "num_pairs": len(pairs),
        "seed": args.seed,
        "pkl_file": original_pkl_file,
    }
    metadata_path = output_dir / f"{args.output_file}.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"元数据已保存到: {metadata_path}")


if __name__ == "__main__":
    main()