"""
CI 专用测试

这些测试不依赖外部测试数据，适用于 GitHub Actions CI 环境。
这些测试主要验证基本功能是否正常工作，不验证正确性。
"""

import pytest
import numpy as np
import traj_dist_rs


def test_sspd_basic():
    """测试 SSPD 基本功能"""
    traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]
    
    dist = traj_dist_rs.sspd(traj1, traj2, "euclidean")
    
    # 只验证返回值是有效的数字
    assert isinstance(dist, float)
    assert not np.isnan(dist)
    assert dist >= 0


def test_sspd_spherical():
    """测试 SSPD 球面距离"""
    traj1 = [[-122.4, 37.7], [-122.3, 37.8]]
    traj2 = [[-122.4, 37.8], [-122.3, 37.7]]
    
    dist = traj_dist_rs.sspd(traj1, traj2, "spherical")
    
    assert isinstance(dist, float)
    assert not np.isnan(dist)
    assert dist >= 0


def test_dtw_basic():
    """测试 DTW 基本功能"""
    traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]
    
    result = traj_dist_rs.dtw(traj1, traj2, "euclidean", use_full_matrix=False)
    
    # 检查结果有 distance 和 matrix 属性
    assert hasattr(result, 'distance')
    assert hasattr(result, 'matrix')
    assert isinstance(result.distance, float)
    assert not np.isnan(result.distance)
    assert result.distance >= 0
    assert result.matrix is None


def test_dtw_with_matrix():
    """测试 DTW 返回矩阵"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]
    
    result = traj_dist_rs.dtw(traj1, traj2, "euclidean", use_full_matrix=True)
    
    assert hasattr(result, 'distance')
    assert hasattr(result, 'matrix')
    assert isinstance(result.distance, float)
    assert result.matrix is not None
    assert isinstance(result.matrix, np.ndarray)


def test_hausdorff_basic():
    """测试 Hausdorff 基本功能"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]
    
    dist = traj_dist_rs.hausdorff(traj1, traj2, "euclidean")
    
    assert isinstance(dist, float)
    assert not np.isnan(dist)
    assert dist >= 0


def test_lcss_basic():
    """测试 LCSS 基本功能"""
    traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]
    
    result = traj_dist_rs.lcss(traj1, traj2, "euclidean", eps=0.5, use_full_matrix=False)
    
    assert hasattr(result, 'distance')
    assert hasattr(result, 'matrix')
    assert isinstance(result.distance, float)
    assert 0 <= result.distance <= 1


def test_edr_basic():
    """测试 EDR 基本功能"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]
    
    result = traj_dist_rs.edr(traj1, traj2, "euclidean", eps=0.5, use_full_matrix=False)
    
    assert hasattr(result, 'distance')
    assert hasattr(result, 'matrix')
    assert isinstance(result.distance, float)
    assert 0 <= result.distance <= 1


def test_erp_standard_basic():
    """测试 ERP standard 基本功能"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]
    
    result = traj_dist_rs.erp_standard(traj1, traj2, "euclidean", g=[0.0, 0.0], use_full_matrix=False)
    
    assert hasattr(result, 'distance')
    assert hasattr(result, 'matrix')
    assert isinstance(result.distance, float)
    assert result.distance >= 0


def test_erp_compat_basic():
    """测试 ERP compat_traj_dist 基本功能"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]
    
    result = traj_dist_rs.erp_compat_traj_dist(traj1, traj2, "euclidean", g=[0.0, 0.0], use_full_matrix=False)
    
    assert hasattr(result, 'distance')
    assert hasattr(result, 'matrix')
    assert isinstance(result.distance, float)
    assert result.distance >= 0


def test_discret_frechet_basic():
    """测试 Discret Frechet 基本功能"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]
    
    result = traj_dist_rs.discret_frechet(traj1, traj2, "euclidean", use_full_matrix=False)
    
    assert hasattr(result, 'distance')
    assert hasattr(result, 'matrix')
    assert isinstance(result.distance, float)
    assert result.distance >= 0


def test_same_trajectory_zero_distance():
    """测试相同轨迹的距离应该接近 0"""
    traj = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    
    # SSPD
    dist = traj_dist_rs.sspd(traj, traj, "euclidean")
    assert dist < 0.01  # 允许小的浮点误差
    
    # Hausdorff
    dist = traj_dist_rs.hausdorff(traj, traj, "euclidean")
    assert dist < 0.01


def test_numpy_array_input():
    """测试 numpy 数组输入"""
    traj1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    traj2 = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])
    
    dist = traj_dist_rs.sspd(traj1, traj2, "euclidean")
    
    assert isinstance(dist, float)
    assert not np.isnan(dist)
    assert dist >= 0


def test_empty_trajectory_handling():
    """测试空轨迹处理"""
    # 单点轨迹（SSPD 应该返回 inf）
    traj1 = [[0.0, 0.0]]
    traj2 = [[0.0, 1.0]]
    
    dist = traj_dist_rs.sspd(traj1, traj2, "euclidean")
    assert dist == float('inf')


def test_invalid_distance_type():
    """测试无效的距离类型"""
    traj1 = [[0.0, 0.0], [1.0, 1.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1]]
    
    # 应该抛出异常或返回错误
    with pytest.raises(Exception):
        traj_dist_rs.sspd(traj1, traj2, "invalid_distance_type")


def test_readme_examples():
    """测试 README 中的示例代码"""
    traj1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    traj2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]
    
    # SSPD
    distance = traj_dist_rs.sspd(traj1, traj2, dist_type="euclidean")
    assert isinstance(distance, float)
    assert not np.isnan(distance)
    
    # DTW
    result = traj_dist_rs.dtw(traj1, traj2, dist_type="euclidean", use_full_matrix=False)
    assert isinstance(result.distance, float)
    
    # Hausdorff
    distance = traj_dist_rs.hausdorff(traj1, traj2, dist_type="spherical")
    assert isinstance(distance, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])