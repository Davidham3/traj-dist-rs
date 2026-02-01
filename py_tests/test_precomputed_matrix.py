"""
预计算距离矩阵接口测试用例

测试使用预计算距离矩阵的距离计算功能
"""

import pytest
import numpy as np
import traj_dist_rs


class TestDTWWithMatrix:
    """测试 DTW 使用预计算距离矩阵"""

    def test_dtw_with_matrix_basic(self):
        """测试基本的预计算距离矩阵功能"""
        t1 = [[0.0, 0.0], [1.0, 1.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0]]

        # 使用标准接口计算
        result_standard = traj_dist_rs.dtw(t1, t2, "euclidean")

        # 手动预计算距离矩阵
        dist_matrix = np.zeros((2, 2))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # 使用预计算矩阵计算
        result_matrix = traj_dist_rs.dtw_with_matrix(dist_matrix)

        # 两种方法的结果应该一致
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10

    def test_dtw_with_matrix_and_full_matrix(self):
        """测试预计算矩阵并返回完整DP矩阵"""
        dist_matrix = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ])

        # 不返回完整矩阵
        result_no_matrix = traj_dist_rs.dtw_with_matrix(dist_matrix, use_full_matrix=False)
        assert result_no_matrix.matrix is None

        # 返回完整矩阵
        result_with_matrix = traj_dist_rs.dtw_with_matrix(dist_matrix, use_full_matrix=True)
        assert result_with_matrix.matrix is not None

        # 距离值应该一致
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-10

    def test_dtw_with_matrix_consistency(self):
        """测试预计算矩阵和标准接口的一致性"""
        t1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]]

        # 使用标准接口
        result_standard = traj_dist_rs.dtw(t1, t2, "euclidean")

        # 预计算距离矩阵
        dist_matrix = np.zeros((3, 3))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # 使用预计算矩阵
        result_matrix = traj_dist_rs.dtw_with_matrix(dist_matrix)

        # 两种方法的结果应该一致
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10


class TestLCSSWithMatrix:
    """测试 LCSS 使用预计算距离矩阵"""

    def test_lcss_with_matrix_basic(self):
        """测试基本的预计算距离矩阵功能"""
        t1 = [[0.0, 0.0], [1.0, 1.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0]]

        # 使用标准接口计算
        result_standard = traj_dist_rs.lcss(t1, t2, "euclidean", eps=0.5)

        # 手动预计算距离矩阵
        dist_matrix = np.zeros((2, 2))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # 使用预计算矩阵计算
        result_matrix = traj_dist_rs.lcss_with_matrix(dist_matrix, eps=0.5)

        # 两种方法的结果应该一致
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10

    def test_lcss_with_matrix_and_full_matrix(self):
        """测试预计算矩阵并返回完整DP矩阵"""
        dist_matrix = np.array([
            [0.1, 0.1],
            [0.1, 0.1],
        ])

        eps = 0.5

        # 不返回完整矩阵
        result_no_matrix = traj_dist_rs.lcss_with_matrix(dist_matrix, eps, use_full_matrix=False)
        assert result_no_matrix.matrix is None

        # 返回完整矩阵
        result_with_matrix = traj_dist_rs.lcss_with_matrix(dist_matrix, eps, use_full_matrix=True)
        assert result_with_matrix.matrix is not None

        # 距离值应该一致
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-10


class TestEDRWithMatrix:
    """测试 EDR 使用预计算距离矩阵"""

    def test_edr_with_matrix_basic(self):
        """测试基本的预计算距离矩阵功能"""
        t1 = [[0.0, 0.0], [1.0, 1.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0]]

        # 使用标准接口计算
        result_standard = traj_dist_rs.edr(t1, t2, "euclidean", eps=0.5)

        # 手动预计算距离矩阵
        dist_matrix = np.zeros((2, 2))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # 使用预计算矩阵计算
        result_matrix = traj_dist_rs.edr_with_matrix(dist_matrix, eps=0.5)

        # 两种方法的结果应该一致
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10

    def test_edr_with_matrix_and_full_matrix(self):
        """测试预计算矩阵并返回完整DP矩阵"""
        dist_matrix = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ])

        eps = 0.5

        # 不返回完整矩阵
        result_no_matrix = traj_dist_rs.edr_with_matrix(dist_matrix, eps, use_full_matrix=False)
        assert result_no_matrix.matrix is None

        # 返回完整矩阵
        result_with_matrix = traj_dist_rs.edr_with_matrix(dist_matrix, eps, use_full_matrix=True)
        assert result_with_matrix.matrix is not None

        # 距离值应该一致
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-10


class TestDiscretFrechetWithMatrix:
    """测试 Discret Frechet 使用预计算距离矩阵"""

    def test_discret_frechet_with_matrix_basic(self):
        """测试基本的预计算距离矩阵功能"""
        t1 = [[0.0, 0.0], [1.0, 1.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0]]

        # 使用标准接口计算
        result_standard = traj_dist_rs.discret_frechet(t1, t2, "euclidean")

        # 手动预计算距离矩阵
        dist_matrix = np.zeros((2, 2))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # 使用预计算矩阵计算
        result_matrix = traj_dist_rs.discret_frechet_with_matrix(dist_matrix)

        # 两种方法的结果应该一致
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10

    def test_discret_frechet_with_matrix_and_full_matrix(self):
        """测试预计算矩阵并返回完整DP矩阵"""
        dist_matrix = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ])

        # 不返回完整矩阵
        result_no_matrix = traj_dist_rs.discret_frechet_with_matrix(
            dist_matrix, use_full_matrix=False
        )
        assert result_no_matrix.matrix is None

        # 返回完整矩阵
        result_with_matrix = traj_dist_rs.discret_frechet_with_matrix(
            dist_matrix, use_full_matrix=True
        )
        assert result_with_matrix.matrix is not None

        # 距离值应该一致
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-10


class TestERPStandardWithMatrix:
    """测试 ERP standard 使用预计算距离矩阵"""

    def test_erp_standard_with_matrix_basic(self):
        """测试基本的预计算距离矩阵功能"""
        t1 = [[0.0, 0.0], [1.0, 1.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0]]
        g = [0.0, 0.0]

        # 使用标准接口计算
        result_standard = traj_dist_rs.erp_standard(t1, t2, "euclidean", g)

        # 手动预计算距离矩阵
        dist_matrix = np.zeros((2, 2))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # 预计算 gap 距离
        seq0_gap_dists = np.zeros(2)
        for i in range(len(t1)):
            dx = t1[i][0] - g[0]
            dy = t1[i][1] - g[1]
            seq0_gap_dists[i] = np.sqrt(dx * dx + dy * dy)

        seq1_gap_dists = np.zeros(2)
        for j in range(len(t2)):
            dx = t2[j][0] - g[0]
            dy = t2[j][1] - g[1]
            seq1_gap_dists[j] = np.sqrt(dx * dx + dy * dy)

        # 使用预计算矩阵计算
        result_matrix = traj_dist_rs.erp_standard_with_matrix(
            dist_matrix, seq0_gap_dists, seq1_gap_dists
        )

        # 两种方法的结果应该一致
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10

    def test_erp_standard_with_matrix_and_full_matrix(self):
        """测试预计算矩阵并返回完整DP矩阵"""
        dist_matrix = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        seq0_gap_dists = np.array([1.0, 1.0])
        seq1_gap_dists = np.array([1.0, 1.0])

        # 不返回完整矩阵
        result_no_matrix = traj_dist_rs.erp_standard_with_matrix(
            dist_matrix, seq0_gap_dists, seq1_gap_dists, use_full_matrix=False
        )
        assert result_no_matrix.matrix is None

        # 返回完整矩阵
        result_with_matrix = traj_dist_rs.erp_standard_with_matrix(
            dist_matrix, seq0_gap_dists, seq1_gap_dists, use_full_matrix=True
        )
        assert result_with_matrix.matrix is not None

        # 距离值应该一致
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-10


class TestERPCompatWithMatrix:
    """测试 ERP compat 使用预计算距离矩阵"""

    def test_erp_compat_with_matrix_basic(self):
        """测试基本的预计算距离矩阵功能"""
        t1 = [[0.0, 0.0], [1.0, 1.0]]
        t2 = [[0.0, 1.0], [1.0, 0.0]]
        g = [0.0, 0.0]

        # 使用标准接口计算
        result_standard = traj_dist_rs.erp_compat_traj_dist(t1, t2, "euclidean", g)

        # 手动预计算距离矩阵
        dist_matrix = np.zeros((2, 2))
        for i in range(len(t1)):
            for j in range(len(t2)):
                dx = t1[i][0] - t2[j][0]
                dy = t1[i][1] - t2[j][1]
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # 预计算 gap 距离
        seq0_gap_dists = np.zeros(2)
        for i in range(len(t1)):
            dx = t1[i][0] - g[0]
            dy = t1[i][1] - g[1]
            seq0_gap_dists[i] = np.sqrt(dx * dx + dy * dy)

        seq1_gap_dists = np.zeros(2)
        for j in range(len(t2)):
            dx = t2[j][0] - g[0]
            dy = t2[j][1] - g[1]
            seq1_gap_dists[j] = np.sqrt(dx * dx + dy * dy)

        # 使用预计算矩阵计算
        result_matrix = traj_dist_rs.erp_compat_traj_dist_with_matrix(
            dist_matrix, seq0_gap_dists, seq1_gap_dists
        )

        # 两种方法的结果应该一致
        assert abs(result_standard.distance - result_matrix.distance) < 1e-10

    def test_erp_compat_with_matrix_and_full_matrix(self):
        """测试预计算矩阵并返回完整DP矩阵"""
        dist_matrix = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        seq0_gap_dists = np.array([1.0, 1.0])
        seq1_gap_dists = np.array([1.0, 1.0])

        # 不返回完整矩阵
        result_no_matrix = traj_dist_rs.erp_compat_traj_dist_with_matrix(
            dist_matrix, seq0_gap_dists, seq1_gap_dists, use_full_matrix=False
        )
        assert result_no_matrix.matrix is None

        # 返回完整矩阵
        result_with_matrix = traj_dist_rs.erp_compat_traj_dist_with_matrix(
            dist_matrix, seq0_gap_dists, seq1_gap_dists, use_full_matrix=True
        )
        assert result_with_matrix.matrix is not None

        # 距离值应该一致
        assert abs(result_no_matrix.distance - result_with_matrix.distance) < 1e-10


class TestMatrixValidation:
    """测试输入验证"""

    def test_invalid_matrix_shape(self):
        """测试无效的矩阵形状"""
        # 1D 数组应该失败
        with pytest.raises(ValueError, match="must be a 2D numpy array"):
            traj_dist_rs.dtw_with_matrix(np.array([1.0, 2.0, 3.0]))

        # 3D 数组应该失败
        with pytest.raises(ValueError, match="must be a 2D numpy array"):
            traj_dist_rs.dtw_with_matrix(np.array([[[1.0]]]))

    def test_invalid_matrix_dtype(self):
        """测试无效的矩阵数据类型"""
        # 整数数组应该失败
        with pytest.raises(ValueError, match="must be a 2D numpy array of float64 values"):
            traj_dist_rs.dtw_with_matrix(np.array([[1, 2], [3, 4]]))


class TestEmptyMatrix:
    """测试空矩阵的情况"""

    def test_dtw_empty_matrix(self):
        """测试空矩阵"""
        dist_matrix = np.zeros((0, 2))
        result = traj_dist_rs.dtw_with_matrix(dist_matrix)
        # 空轨迹应该返回 f64::MAX
        assert result.distance == 1.7976931348623157e+308

    def test_dtw_matrix_with_empty_row(self):
        """测试空行的矩阵"""
        dist_matrix = np.zeros((2, 0))
        result = traj_dist_rs.dtw_with_matrix(dist_matrix)
        # 空轨迹应该返回 f64::MAX
        assert result.distance == 1.7976931348623157e+308