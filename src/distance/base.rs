use crate::{
    distance::distance_type::DistanceType,
    traits::{AsCoord, CoordSequence},
};

/// Trait that all distance calculators must implement
///
/// This trait is suitable for algorithms that only need point-to-point distances (e.g., DTW, LCSS, EDR, ERP, Discret Frechet).
/// For algorithms that need to calculate point-to-segment distances (e.g., Hausdorff and SSPD), use CoordSequence directly.
pub trait DistanceCalculator {
    /// Calculate the distance between corresponding elements in two sequences
    fn dis_between(&self, seq_a_idx: usize, seq_b_idx: usize) -> f64;

    /// Calculate the distance between a point in a sequence and an external "anchor" point
    /// seq_id: 0 indicates the first sequence, 1 indicates the second
    fn compute_dis_for_extra_point<C: AsCoord>(
        &self,
        seq_id: usize,
        point_idx: usize,
        anchor: Option<&C>,
    ) -> f64;

    /// Get the length of the first sequence
    fn len_seq1(&self) -> usize;

    /// Get the length of the second sequence
    fn len_seq2(&self) -> usize;
}

/// Distance calculator based on trajectories
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

/// Distance calculator based on precomputed distances
///
/// This calculator uses a flat array representation for zero-copy NumPy integration.
/// The distance matrix is stored as a contiguous 1D array with row-major ordering.
pub struct PrecomputedDistanceCalculator<'a> {
    distance_matrix: &'a [f64],
    seq1_len: usize,
    seq2_len: usize,
    seq1_extra_dists: Option<&'a [f64]>,
    seq2_extra_dists: Option<&'a [f64]>,
}

impl<'a> PrecomputedDistanceCalculator<'a> {
    pub fn new(distance_matrix: &'a [f64], seq1_len: usize, seq2_len: usize) -> Self {
        Self {
            distance_matrix,
            seq1_len,
            seq2_len,
            seq1_extra_dists: None,
            seq2_extra_dists: None,
        }
    }

    pub fn with_extra_distances(
        distance_matrix: &'a [f64],
        seq1_len: usize,
        seq2_len: usize,
        seq1_dists: Option<&'a [f64]>,
        seq2_dists: Option<&'a [f64]>,
    ) -> Self {
        Self {
            distance_matrix,
            seq1_len,
            seq2_len,
            seq1_extra_dists: seq1_dists,
            seq2_extra_dists: seq2_dists,
        }
    }
}

impl<'a> DistanceCalculator for PrecomputedDistanceCalculator<'a> {
    fn dis_between(&self, idx1: usize, idx2: usize) -> f64 {
        // Row-major ordering: index = idx1 * seq2_len + idx2
        self.distance_matrix[idx1 * self.seq2_len + idx2]
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
        self.seq1_len
    }

    fn len_seq2(&self) -> usize {
        self.seq2_len
    }
}
