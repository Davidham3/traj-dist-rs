"""
pytest configuration file

Globally load test data, use pyarrow to read Parquet files and convert to pl.DataFrame
"""

from pathlib import Path

import polars as pl
import pytest
from test_framework import get_sample_path, load_all_metainfo_from_data_dir

# Test data directory
DATA_DIR = Path(__file__).parent / "data"
SAMPLES_DIR = DATA_DIR / "cython_samples"
METAINFO_DIR = DATA_DIR / "metainfo"


def pytest_configure(config):
    """pytest configuration hook, executed at the start of test session"""
    # Ensure data directory exists, fail if not
    if not DATA_DIR.exists():
        raise RuntimeError(
            f"Test data directory not found: {DATA_DIR}\n"
            f"Please ensure test data has been committed to git repository."
        )

    if not SAMPLES_DIR.exists():
        raise RuntimeError(
            f"Test sample directory not found: {SAMPLES_DIR}\n"
            f"Please run 'bash scripts/generate_all_test_cases.sh' to generate test data."
        )

    if not METAINFO_DIR.exists():
        raise RuntimeError(
            f"Test metadata directory not found: {METAINFO_DIR}\n"
            f"Please run 'bash scripts/generate_all_test_cases.sh' to generate test data."
        )


@pytest.fixture(scope="session")
def data_dir():
    """Return data directory"""
    return DATA_DIR


@pytest.fixture(scope="session")
def samples_dir():
    """Return samples directory"""
    return SAMPLES_DIR


@pytest.fixture(scope="session")
def metainfo_dir():
    """Return metadata directory"""
    return METAINFO_DIR


@pytest.fixture(scope="session")
def all_metainfo(data_dir):
    """
    Global fixture, loads metadata for all algorithms

    Returns a dictionary with algorithm names as keys and metadata lists as values
    """
    return load_all_metainfo_from_data_dir(data_dir)


@pytest.fixture(scope="session")
def algorithm_names(all_metainfo):
    """
    Return list of all algorithm names

    Used for parameterized testing
    """
    return list(all_metainfo.keys())


# Specific algorithm fixtures
@pytest.fixture(scope="session")
def sspd_euclidean(all_metainfo, data_dir):
    """SSPD Euclidean distance test data"""

    sspd_metainfo = all_metainfo.get("sspd", [])
    euclidean_metainfo = [m for m in sspd_metainfo if m.type_d == "euclidean"]

    if not euclidean_metainfo:
        raise RuntimeError("SSPD Euclidean distance test data not found")

    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read SSPD Euclidean distance test data: {e}")


@pytest.fixture(scope="session")
def sspd_spherical(all_metainfo, data_dir):
    """SSPD Spherical distance test data"""

    sspd_metainfo = all_metainfo.get("sspd", [])
    spherical_metainfo = [m for m in sspd_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        raise RuntimeError("SSPD Spherical distance test data not found")

    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read SSPD Spherical distance test data: {e}")


@pytest.fixture(scope="session")
def dtw_euclidean(all_metainfo, data_dir):
    """DTW Euclidean distance test data"""

    dtw_metainfo = all_metainfo.get("dtw", [])
    euclidean_metainfo = [m for m in dtw_metainfo if m["type_d"] == "euclidean"]

    if not euclidean_metainfo:
        raise RuntimeError("DTW Euclidean distance test data not found")

    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read DTW Euclidean distance test data: {e}")


@pytest.fixture(scope="session")
def dtw_spherical(all_metainfo, data_dir):
    """DTW Spherical distance test data"""

    dtw_metainfo = all_metainfo.get("dtw", [])
    spherical_metainfo = [m for m in dtw_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        raise RuntimeError("DTW Spherical distance test data not found")

    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read DTW Spherical distance test data: {e}")


@pytest.fixture(scope="session")
def hausdorff_euclidean(all_metainfo, data_dir):
    """Hausdorff Euclidean distance test data"""

    hausdorff_metainfo = all_metainfo.get("hausdorff", [])
    euclidean_metainfo = [m for m in hausdorff_metainfo if m["type_d"] == "euclidean"]

    if not euclidean_metainfo:
        raise RuntimeError("Hausdorff Euclidean distance test data not found")

    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to read Hausdorff Euclidean distance test data: {e}"
        )


@pytest.fixture(scope="session")
def hausdorff_spherical(all_metainfo, data_dir):
    """Hausdorff Spherical distance test data"""

    hausdorff_metainfo = all_metainfo.get("hausdorff", [])
    spherical_metainfo = [m for m in hausdorff_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        raise RuntimeError("Hausdorff Spherical distance test data not found")

    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to read Hausdorff Spherical distance test data: {e}"
        )


@pytest.fixture(scope="session")
def frechet_euclidean(all_metainfo, data_dir):
    """Frechet Euclidean distance test data"""

    frechet_metainfo = all_metainfo.get("frechet", [])
    euclidean_metainfo = [m for m in frechet_metainfo if m["type_d"] == "euclidean"]

    if not euclidean_metainfo:
        raise RuntimeError("Frechet Euclidean distance test data not found")

    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read Frechet Euclidean distance test data: {e}")


@pytest.fixture(scope="session")
def discret_frechet_euclidean(all_metainfo, data_dir):
    """Discret Frechet Euclidean distance test data"""

    discret_frechet_metainfo = all_metainfo.get("discret_frechet", [])
    euclidean_metainfo = [
        m for m in discret_frechet_metainfo if m["type_d"] == "euclidean"
    ]

    if not euclidean_metainfo:
        raise RuntimeError("Discret Frechet Euclidean distance test data not found")

    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to read Discret Frechet Euclidean distance test data: {e}"
        )


@pytest.fixture(scope="session")
def lcss_euclidean(all_metainfo, data_dir):
    """LCSS Euclidean distance test data (eps=0.0)"""

    lcss_metainfo = all_metainfo.get("lcss", [])
    euclidean_metainfo = [m for m in lcss_metainfo if m["type_d"] == "euclidean"]

    if not euclidean_metainfo:
        raise RuntimeError("LCSS Euclidean distance test data not found")

    # Use first eps value
    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read LCSS Euclidean distance test data: {e}")


@pytest.fixture(scope="session")
def lcss_spherical(all_metainfo, data_dir):
    """LCSS Spherical distance test data (eps=0.01)"""

    lcss_metainfo = all_metainfo.get("lcss", [])
    spherical_metainfo = [m for m in lcss_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        raise RuntimeError("LCSS Spherical distance test data not found")

    # Use first eps value
    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read LCSS Spherical distance test data: {e}")


@pytest.fixture(scope="session")
def edr_euclidean(all_metainfo, data_dir):
    """EDR Euclidean distance test data (eps=0.0)"""

    edr_metainfo = all_metainfo.get("edr", [])
    euclidean_metainfo = [m for m in edr_metainfo if m["type_d"] == "euclidean"]

    if not euclidean_metainfo:
        raise RuntimeError("EDR Euclidean distance test data not found")

    # Use first eps value
    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read EDR Euclidean distance test data: {e}")


@pytest.fixture(scope="session")
def edr_spherical(all_metainfo, data_dir):
    """EDR Spherical distance test data (eps=0.01)"""

    edr_metainfo = all_metainfo.get("edr", [])
    spherical_metainfo = [m for m in edr_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        raise RuntimeError("EDR Spherical distance test data not found")

    # Use first eps value
    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read EDR Spherical distance test data: {e}")


@pytest.fixture(scope="session")
def erp_euclidean(all_metainfo, data_dir):
    """ERP Euclidean distance test data (g=[-122.41443,37.77646])"""

    erp_metainfo = all_metainfo.get("erp", [])
    euclidean_metainfo = [m for m in erp_metainfo if m["type_d"] == "euclidean"]

    if not euclidean_metainfo:
        raise RuntimeError("ERP Euclidean distance test data not found")

    # Use first g value
    sample_path = get_sample_path(euclidean_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read ERP Euclidean distance test data: {e}")


@pytest.fixture(scope="session")
def erp_spherical(all_metainfo, data_dir):
    """ERP Spherical distance test data (g=[-122.41443,37.77646])"""

    erp_metainfo = all_metainfo.get("erp", [])
    spherical_metainfo = [m for m in erp_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        raise RuntimeError("ERP Spherical distance test data not found")

    # Use first g value
    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read ERP Spherical distance test data: {e}")


@pytest.fixture(scope="session")
def sowd_spherical(all_metainfo, data_dir):
    """SOWD Spherical distance test data (precision=4)"""

    sowd_metainfo = all_metainfo.get("sowd_grid", [])
    spherical_metainfo = [m for m in sowd_metainfo if m["type_d"] == "spherical"]

    if not spherical_metainfo:
        raise RuntimeError("SOWD Spherical distance test data not found")

    # Use first precision value
    sample_path = get_sample_path(spherical_metainfo[0], data_dir)

    try:
        df = pl.read_parquet(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read SOWD Spherical distance test data: {e}")


# Helper function: get all test data for an algorithm
def get_algorithm_test_data(algorithm_name, all_metainfo, data_dir):
    """
    Get all test data for a specified algorithm

    Args:
        algorithm_name: Algorithm name
        all_metainfo: All metadata
        data_dir: Data directory

    Returns:
        Dictionary with metadata as keys and DataFrames as values
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
            raise RuntimeError(f"Failed to read {sample_path}: {e}")

    return test_data
