#!/usr/bin/env python3
"""
生成单个算法的测试用例脚本（使用 Cython 实现）

通过 argparse 指定算法名、超参数值，生成 Parquet 文件和元数据 JSONL 文件。
Parquet 文件名包含算法名、距离类型、超参数值。
文件包含 4 列：2 列轨迹数据、1 列距离值、1 列计算时间。
"""

import sys
import argparse
import pickle
import timeit
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import traj_dist.distance as tdist

sys.path.append(str(Path(__file__).parent.parent))
from py_tests.schemas import Metainfo, ImplementationType


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成单个算法的测试用例（使用Cython实现）"
    )
    parser.add_argument("--algorithm", type=str, required=True, help="算法名称")
    parser.add_argument(
        "--type_d",
        type=str,
        default="euclidean",
        choices=["euclidean", "spherical"],
        help="距离类型",
    )
    parser.add_argument(
        "--eps", type=float, default=None, help="LCSS 和 EDR 算法的 eps 参数"
    )
    parser.add_argument(
        "--g", type=float, nargs=2, default=None, help="ERP 算法的 g 参数（两个值）"
    )
    parser.add_argument(
        "--precision", type=int, default=None, help="SOWD 算法的 precision 参数"
    )
    parser.add_argument(
        "--converted", type=bool, default=None, help="SOWD 算法的 converted 参数"
    )
    parser.add_argument(
        "--output-dir", type=str, default="py_tests/data", help="输出目录"
    )
    parser.add_argument(
        "--traj-data",
        type=str,
        default="../traj-dist/data/benchmark_trajectories.pkl",
        help="轨迹数据文件",
    )
    parser.add_argument("--num-traj", type=int, default=50, help="使用的轨迹数量")
    return parser.parse_args()


def build_sample_filename(algorithm, type_d, eps=None, g=None, precision=None):
    """
    构建样本文件名，包含算法名、距离类型和超参数

    格式: {algorithm}_{type_d}_{param}_{value}.parquet
    例如: edr_euclidean_eps_0.01.parquet
    """
    parts = [algorithm, type_d]
    if eps is not None:
        parts.append(f"eps_{eps}")
    if g is not None:
        # 对于 g 参数，使用特定的格式
        parts.append(f"g_{g[0]}_{g[1]}")
    if precision is not None:
        parts.append(f"precision_{precision}")
    return "_".join(parts) + ".parquet"


def build_metainfo_filename(algorithm):
    """
    构建元数据文件名

    格式: {algorithm}.jsonl
    例如: edr.jsonl
    """
    return f"{algorithm}.jsonl"


def main():
    args = parse_args()

    # 加载轨迹数据
    traj_data_path = Path(args.traj_data)
    if not traj_data_path.exists():
        # 尝试相对路径
        traj_data_path = (
            Path(__file__).parent.parent.parent
            / "traj-dist"
            / "data"
            / "benchmark_trajectories.pkl"
        )

    print(f"Loading trajectory data from: {traj_data_path}")
    with open(traj_data_path, "rb") as f:
        traj_list = pickle.load(f, encoding="latin1")

    # 使用指定数量的轨迹
    traj_list = traj_list[: args.num_traj]
    print(f"Using {len(traj_list)} trajectories")

    # 构建参数字典
    params = {"type_d": args.type_d}
    if args.eps is not None:
        params["eps"] = args.eps
    if args.g is not None:
        params["g"] = np.array(args.g, dtype=np.float64)
    if args.precision is not None:
        params["precision"] = args.precision
    if args.converted is not None:
        params["converted"] = args.converted

    # 构建输出目录
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "cython_samples"  # Cython 实现的样本放在单独目录
    metainfo_dir = output_dir / "metainfo"
    samples_dir.mkdir(parents=True, exist_ok=True)
    metainfo_dir.mkdir(parents=True, exist_ok=True)

    # 构建样本文件名和路径
    sample_filename = build_sample_filename(
        args.algorithm, args.type_d, args.eps, args.g, args.precision
    )
    sample_path = samples_dir / sample_filename

    print(f"Generating test cases for: {args.algorithm} (using Cython implementation)")
    print(f"Parameters: {params}")
    print(f"Sample file: {sample_path}")

    # 获取对应的距离函数（使用 Cython 实现）
    if args.algorithm == "sspd":
        dist_func = tdist.c_e_sspd if args.type_d == "euclidean" else tdist.c_g_sspd
    elif args.algorithm == "dtw":
        dist_func = tdist.c_e_dtw if args.type_d == "euclidean" else tdist.c_g_dtw
    elif args.algorithm == "hausdorff":
        dist_func = (
            tdist.c_e_hausdorff if args.type_d == "euclidean" else tdist.c_g_hausdorff
        )
    elif args.algorithm == "frechet":
        dist_func = tdist.c_frechet  # frechet只支持欧几里得距离
    elif args.algorithm == "discret_frechet":
        dist_func = tdist.c_discret_frechet  # discret_frechet只支持欧几里得距离
    elif args.algorithm == "lcss":
        dist_func = tdist.c_e_lcss if args.type_d == "euclidean" else tdist.c_g_lcss
    elif args.algorithm == "edr":
        dist_func = tdist.c_e_edr if args.type_d == "euclidean" else tdist.c_g_edr
    elif args.algorithm == "erp":
        dist_func = tdist.c_e_erp if args.type_d == "euclidean" else tdist.c_g_erp
    elif args.algorithm == "sowd_grid":
        dist_func = tdist.c_sowd_grid  # SOWD only supports spherical/geographical
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # 生成测试用例
    results = []
    for i, traj1_orig in enumerate(traj_list):
        for j, traj2_orig in enumerate(traj_list):
            if i >= j:  # 避免重复计算
                continue

            traj1 = traj1_orig
            traj2 = traj2_orig

            # SOWD 算法需要特殊处理：转换为 cell 格式
            call_params = {}
            if args.algorithm == "sowd_grid":
                from traj_dist.pydist.linecell import trajectory_set_grid

                precision = params.get("precision", 7)
                cells_list, _, _, _, _ = trajectory_set_grid([traj1, traj2], precision)
                traj1 = np.array([[c[0], c[1]] for c in cells_list[0]], dtype=np.int64)
                traj2 = np.array([[c[0], c[1]] for c in cells_list[1]], dtype=np.int64)
                # sowd_grid 函数不接受 extra 参数

            # 添加超参数
            if args.eps is not None:
                call_params["eps"] = args.eps
            if args.g is not None:
                call_params["g"] = np.array(args.g, dtype=np.float64)
            # SOWD 算法的 precision 参数已经在转换为 cell 格式时使用，不需要传递给 sowd_grid 函数

            # 测量计算时间
            time_elapsed = timeit.timeit(
                lambda: dist_func(traj1, traj2, **call_params), number=10
            )

            # 计算距离
            distance = dist_func(traj1, traj2, **call_params)

            results.append(
                (
                    traj1.tolist(),
                    traj2.tolist(),
                    distance,
                    time_elapsed,
                )
            )
    traj1, traj2, distance, running_time = zip(*results)

    traj_type = pa.large_list(pa.list_(pa.float64(), 2))

    table = pa.table(
        {"traj1": traj1, "traj2": traj2, "distance": distance, "time": running_time},
        schema=pa.schema(
            [
                ("traj1", traj_type),
                ("traj2", traj_type),
                ("distance", pa.float64()),
                ("time", pa.float64()),
            ]
        ),
    )

    pq.write_table(table, sample_path)
    print(f"Generated {len(results)} test cases")
    print(f"Saved to: {sample_path}")

    try:
        metainfo = Metainfo(
            algorithm=args.algorithm,
            type_d=args.type_d,
            implemented_by=ImplementationType.CYTHON,
            eps=args.eps,
            g=args.g,  # Pydantic 会自动处理 list
            precision=args.precision,
            converted=args.converted,
            # 路径相对于 metainfo_dir 的父目录
            sample_file=f"cython_samples/{sample_filename}",
        )
    except Exception as e:
        print(f"Error creating Pydantic model: {e}")
        return

    # 构建元数据文件名和路径
    metainfo_filename = build_metainfo_filename(args.algorithm)
    metainfo_path = metainfo_dir / metainfo_filename

    # 使用模型的 .json() 方法序列化并追加到元数据文件
    with open(metainfo_path, "a") as f:
        # metainfo.json() 会生成一个紧凑的 JSON 字符串
        f.write(metainfo.model_dump_json() + "\n")

    print(f"Metadata appended to: {metainfo_path}")


if __name__ == "__main__":
    main()
