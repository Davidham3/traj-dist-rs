"""
pytest 配置文件

全局加载测试数据，使用 pyarrow 读取 Parquet 文件然后转换为 pl.DataFrame
"""

import pytest
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq
from test_framework import load_all_metainfo_from_data_dir, get_sample_path


# 测试数据目录
DATA_DIR = Path(__file__).parent / "data"
SAMPLES_DIR = DATA_DIR / "cython_samples"
METAINFO_DIR = DATA_DIR / "metainfo"


def pytest_configure(config):
    """pytest 配置钩子，在测试会话开始时执行"""
    # 确保数据目录存在
    if not DATA_DIR.exists():
        print(f"警告: 测试数据目录不存在: {DATA_DIR}")

    if not SAMPLES_DIR.exists():
        print(f"警告: 测试样本目录不存在: {SAMPLES_DIR}")

    if not METAINFO_DIR.exists():
        print(f"警告: 测试元数据目录不存在: {METAINFO_DIR}")


@pytest.fixture(scope="session")
def data_dir():
    """返回数据目录"""
    return DATA_DIR


@pytest.fixture(scope="session")
def samples_dir():
    """返回样本目录"""
    return SAMPLES_DIR


@pytest.fixture(scope="session")
def metainfo_dir():
    """返回元数据目录"""
    return METAINFO_DIR


@pytest.fixture(scope="session")
def all_metainfo(data_dir):
    """
    全局 fixture，加载所有算法的元数据

    返回一个字典，键为算法名称，值为元数据列表
    """
    return load_all_metainfo_from_data_dir(data_dir)


@pytest.fixture(scope="session")
def algorithm_names(all_metainfo):
    """
    返回所有算法名称列表

    用于参数化测试
    """
    return list(all_metainfo.keys())


# 特定算法的 fixtures
@pytest.fixture(scope="session")
def sspd_euclidean(all_metainfo, data_dir):
    """SSPD 欧几里得距离测试数据"""

    sspd_metainfo = all_metainfo.get("sspd", [])
    euclidean_metainfo = [m for m in sspd_metainfo if m.type_d == "spherical"]

    if not euclidean_metainfo:
        pytest.skip("SSPD 欧几里得距离测试数据不存在")

    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 SSPD 欧几里得距离测试数据: {e}")


@pytest.fixture(scope="session")
def sspd_spherical(all_metainfo, data_dir):
    """SSPD 球面距离测试数据"""

    sspd_metainfo = all_metainfo.get("sspd", [])
    spherical_metainfo = [m for m in sspd_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        pytest.skip("SSPD 球面距离测试数据不存在")

    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 SSPD 球面距离测试数据: {e}")


@pytest.fixture(scope="session")
def dtw_euclidean(all_metainfo, data_dir):
    """DTW 欧几里得距离测试数据"""

    dtw_metainfo = all_metainfo.get("dtw", [])
    euclidean_metainfo = [m for m in dtw_metainfo if m["type_d"] == "euclidean"]

    if not euclidean_metainfo:
        pytest.skip("DTW 欧几里得距离测试数据不存在")

    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 DTW 欧几里得距离测试数据: {e}")


@pytest.fixture(scope="session")
def dtw_spherical(all_metainfo, data_dir):
    """DTW 球面距离测试数据"""

    dtw_metainfo = all_metainfo.get("dtw", [])
    spherical_metainfo = [m for m in dtw_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        pytest.skip("DTW 球面距离测试数据不存在")

    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 DTW 球面距离测试数据: {e}")


@pytest.fixture(scope="session")
def hausdorff_euclidean(all_metainfo, data_dir):
    """Hausdorff 欧几里得距离测试数据"""

    hausdorff_metainfo = all_metainfo.get("hausdorff", [])
    euclidean_metainfo = [m for m in hausdorff_metainfo if m["type_d"] == "euclidean"]

    if not euclidean_metainfo:
        pytest.skip("Hausdorff 欧几里得距离测试数据不存在")

    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 Hausdorff 欧几里得距离测试数据: {e}")


@pytest.fixture(scope="session")
def hausdorff_spherical(all_metainfo, data_dir):
    """Hausdorff 球面距离测试数据"""

    hausdorff_metainfo = all_metainfo.get("hausdorff", [])
    spherical_metainfo = [m for m in hausdorff_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        pytest.skip("Hausdorff 球面距离测试数据不存在")

    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 Hausdorff 球面距离测试数据: {e}")


@pytest.fixture(scope="session")
def frechet_euclidean(all_metainfo, data_dir):
    """Frechet 欧几里得距离测试数据"""

    frechet_metainfo = all_metainfo.get("frechet", [])
    euclidean_metainfo = [m for m in frechet_metainfo if m["type_d"] == "euclidean"]

    if not euclidean_metainfo:
        pytest.skip("Frechet 欧几里得距离测试数据不存在")

    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 Frechet 欧几里得距离测试数据: {e}")


@pytest.fixture(scope="session")
def discret_frechet_euclidean(all_metainfo, data_dir):
    """Discret Frechet 欧几里得距离测试数据"""

    discret_frechet_metainfo = all_metainfo.get("discret_frechet", [])
    euclidean_metainfo = [
        m for m in discret_frechet_metainfo if m["type_d"] == "euclidean"
    ]

    if not euclidean_metainfo:
        pytest.skip("Discret Frechet 欧几里得距离测试数据不存在")

    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 Discret Frechet 欧几里得距离测试数据: {e}")


@pytest.fixture(scope="session")
def lcss_euclidean(all_metainfo, data_dir):
    """LCSS 欧几里得距离测试数据（eps=0.0）"""

    lcss_metainfo = all_metainfo.get("lcss", [])
    euclidean_metainfo = [m for m in lcss_metainfo if m["type_d"] == "euclidean"]

    if not euclidean_metainfo:
        pytest.skip("LCSS 欧几里得距离测试数据不存在")

    # 使用第一个 eps 值
    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 LCSS 欧几里得距离测试数据: {e}")


@pytest.fixture(scope="session")
def lcss_spherical(all_metainfo, data_dir):
    """LCSS 球面距离测试数据（eps=0.01）"""

    lcss_metainfo = all_metainfo.get("lcss", [])
    spherical_metainfo = [m for m in lcss_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        pytest.skip("LCSS 球面距离测试数据不存在")

    # 使用第一个 eps 值
    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 LCSS 球面距离测试数据: {e}")


@pytest.fixture(scope="session")
def edr_euclidean(all_metainfo, data_dir):
    """EDR 欧几里得距离测试数据（eps=0.0）"""

    edr_metainfo = all_metainfo.get("edr", [])
    euclidean_metainfo = [m for m in edr_metainfo if m["type_d"] == "euclidean"]

    if not euclidean_metainfo:
        pytest.skip("EDR 欧几里得距离测试数据不存在")

    # 使用第一个 eps 值
    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 EDR 欧几里得距离测试数据: {e}")


@pytest.fixture(scope="session")
def edr_spherical(all_metainfo, data_dir):
    """EDR 球面距离测试数据（eps=0.01）"""

    edr_metainfo = all_metainfo.get("edr", [])
    spherical_metainfo = [m for m in edr_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        pytest.skip("EDR 球面距离测试数据不存在")

    # 使用第一个 eps 值
    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 EDR 球面距离测试数据: {e}")


@pytest.fixture(scope="session")
def erp_euclidean(all_metainfo, data_dir):
    """ERP 欧几里得距离测试数据（g=[-122.41443,37.77646]）"""

    erp_metainfo = all_metainfo.get("erp", [])
    euclidean_metainfo = [m for m in erp_metainfo if m["type_d"] == "euclidean"]

    if not euclidean_metainfo:
        pytest.skip("ERP 欧几里得距离测试数据不存在")

    # 使用第一个 g 值
    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 ERP 欧几里得距离测试数据: {e}")


@pytest.fixture(scope="session")
def erp_spherical(all_metainfo, data_dir):
    """ERP 球面距离测试数据（g=[-122.41443,37.77646]）"""

    erp_metainfo = all_metainfo.get("erp", [])
    spherical_metainfo = [m for m in erp_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        pytest.skip("ERP 球面距离测试数据不存在")

    # 使用第一个 g 值
    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 ERP 球面距离测试数据: {e}")


@pytest.fixture(scope="session")
def sowd_spherical(all_metainfo, data_dir):
    """SOWD 球面距离测试数据（precision=4）"""

    sowd_metainfo = all_metainfo.get("sowd_grid", [])
    spherical_metainfo = [m for m in sowd_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        pytest.skip("SOWD 球面距离测试数据不存在")

    # 使用第一个 precision 值
    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        table = pq.read_table(sample_path)
        df = pl.from_arrow(table)
        return df
    except Exception as e:
        pytest.skip(f"无法读取 SOWD 球面距离测试数据: {e}")


# 辅助函数：获取算法的所有测试数据
def get_algorithm_test_data(algorithm_name, all_metainfo, data_dir):
    """
    获取指定算法的所有测试数据

    Args:
        algorithm_name: 算法名称
        all_metainfo: 所有元数据
        data_dir: 数据目录

    Returns:
        字典，键为元数据，值为对应的 DataFrame
    """

    metainfo_list = all_metainfo.get(algorithm_name, [])
    test_data = {}

    for metainfo in metainfo_list:
        sample_path = get_sample_path(metainfo, data_dir)
        try:
            table = pq.read_table(sample_path)
            df = pl.from_arrow(table)
            test_data[metainfo] = df
        except Exception as e:
            print(f"Warning: Failed to read {sample_path}: {e}")

    return test_data
