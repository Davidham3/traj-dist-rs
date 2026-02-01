use crate::{
    distance::distance_type::DistanceType,
    traits::{AsCoord, CoordSequence},
};

/// 所有距离计算器必须实现的 trait
///
/// 这个 trait 适用于只需要点对点距离的算法（如 DTW、LCSS、EDR、ERP、Discret Frechet）。
/// 对于需要计算点到线段距离的算法（如 Hausdorff 和 SSPD），应该直接使用 CoordSequence。
pub trait DistanceCalculator {
    /// 计算两个序列中对应元素之间的距离
    fn dis_between(&self, seq_a_idx: usize, seq_b_idx: usize) -> f64;

    /// 计算某个序列中的点与外部"锚点"之间的距离
    /// seq_id: 0 表示第一个序列，1 表示第二个
    fn compute_dis_for_extra_point<C: AsCoord>(
        &self,
        seq_id: usize,
        point_idx: usize,
        anchor: Option<&C>,
    ) -> f64;

    /// 获取第一个序列的长度
    fn len_seq1(&self) -> usize;

    /// 获取第二个序列的长度
    fn len_seq2(&self) -> usize;
}

/// 基于轨迹的距离计算器
pub struct TrajectoryCalculator<'a, T, U>
where
    T: CoordSequence + 'a,
    U: CoordSequence + 'a,
{
    traj1: &'a T,
    traj2: &'a U,
    metric: DistanceType,
}

impl<'a, T, U> TrajectoryCalculator<'a, T, U>
where
    T: CoordSequence + 'a,
    U: CoordSequence + 'a,
{
    pub fn new(traj1: &'a T, traj2: &'a U, metric: DistanceType) -> Self {
        Self {
            traj1,
            traj2,
            metric,
        }
    }
}

impl<'a, T, U> DistanceCalculator for TrajectoryCalculator<'a, T, U>
where
    T: CoordSequence + 'a,
    U: CoordSequence + 'a,
{
    fn dis_between(&self, idx1: usize, idx2: usize) -> f64 {
        let p1 = self.traj1.get(idx1);
        let p2 = self.traj2.get(idx2);
        self.metric.distance(&p1, &p2)
    }

    fn compute_dis_for_extra_point<C: AsCoord>(
        &self,
        seq_id: usize,
        idx: usize,
        anchor: Option<&C>,
    ) -> f64 {
        let anchor = anchor.expect("anchor must not be None");
        match seq_id {
            0 => {
                let p = self.traj1.get(idx);
                self.metric.distance(anchor, &p)
            }
            1 => {
                let p = self.traj2.get(idx);
                self.metric.distance(anchor, &p)
            }
            _ => panic!("Invalid seq_id"),
        }
    }

    fn len_seq1(&self) -> usize {
        self.traj1.len()
    }

    fn len_seq2(&self) -> usize {
        self.traj2.len()
    }
}

/// 基于预计算距离的距离计算器
pub struct PrecomputedDistanceCalculator<'a> {
    distance_matrix: &'a Vec<Vec<f64>>,
    seq1_extra_dists: Option<&'a Vec<f64>>,
    seq2_extra_dists: Option<&'a Vec<f64>>,
}

impl<'a> PrecomputedDistanceCalculator<'a> {
    pub fn new(distance_matrix: &'a Vec<Vec<f64>>) -> Self {
        Self {
            seq1_extra_dists: None,
            seq2_extra_dists: None,
            distance_matrix,
        }
    }

    pub fn with_extra_distances(
        distance_matrix: &'a Vec<Vec<f64>>,
        seq1_dists: Option<&'a Vec<f64>>,
        seq2_dists: Option<&'a Vec<f64>>,
    ) -> Self {
        Self {
            distance_matrix,
            seq1_extra_dists: seq1_dists,
            seq2_extra_dists: seq2_dists,
        }
    }
}

impl<'a> DistanceCalculator for PrecomputedDistanceCalculator<'a> {
    fn dis_between(&self, idx1: usize, idx2: usize) -> f64 {
        self.distance_matrix[idx1][idx2]
    }

    fn compute_dis_for_extra_point<C: AsCoord>(
        &self,
        seq_id: usize,
        point_idx: usize,
        _anchor: Option<&C>,
    ) -> f64 {
        match (seq_id, &self.seq1_extra_dists, &self.seq2_extra_dists) {
            (0, Some(dists), _) => dists[point_idx],
            (1, _, Some(dists)) => dists[point_idx],
            _ => panic!("Extra distance not available for seq_id={}", seq_id),
        }
    }

    fn len_seq1(&self) -> usize {
        self.distance_matrix.len()
    }

    fn len_seq2(&self) -> usize {
        if self.distance_matrix.is_empty() {
            0
        } else {
            self.distance_matrix[0].len()
        }
    }
}
