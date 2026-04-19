//! # Frechet Distance Algorithm
//!
//! This module implements the continuous Fréchet distance algorithm for comparing
//! two polygonal curves. Unlike the Discrete Fréchet distance which only considers
//! vertices, this algorithm considers all points along the curve segments.
//!
//! ## Algorithm Description
//!
//! The algorithm works by:
//! 1. Computing critical values (candidate distances)
//! 2. Binary searching over critical values using a decision procedure
//! 3. The decision procedure checks if a monotone path exists in the free space diagram
//!
//! ## Reference
//!
//! "Computing the Fréchet distance between two polygonal curves"
//! International Journal of Computational Geometry & Applications, Vol 5, 1995
//!
//! ## Complexity
//!
//! - Time complexity: O(n*m * log(n*m)) where n and m are trajectory lengths
//! - Space complexity: O(n*m)
//!
//! ## Limitation
//!
//! This implementation only supports Euclidean distance in 2D Cartesian space.
//! Spherical distance is not supported.

use crate::distance::euclidean::{
    circle_line_intersection, euclidean_distance, point_to_segment_distance,
};
use crate::traits::{AsCoord, CoordSequence};

/// Compute the free space interval on a segment from a point
///
/// Returns the fraction `(a, b)` of the segment that lies within distance `eps`
/// of the given point. Returns `None` if no part of the segment is within range.
fn free_line<C: AsCoord, D: AsCoord, E: AsCoord>(
    point: &C,
    seg_start: &D,
    seg_end: &E,
    eps: f64,
) -> Option<(f64, f64)> {
    let px = point.x();
    let py = point.y();
    let s1x = seg_start.x();
    let s1y = seg_start.y();
    let s2x = seg_end.x();
    let s2y = seg_end.y();

    // Degenerate segment (single point)
    if s1x == s2x && s1y == s2y {
        if euclidean_distance(point, seg_start) > eps {
            return None;
        } else {
            return Some((0.0, 1.0));
        }
    }

    // Check if the closest point on the segment is within eps
    if point_to_segment_distance(point, seg_start, seg_end) > eps {
        return None;
    }

    // Compute circle-line intersection
    let intersections = circle_line_intersection(px, py, s1x, s1y, s2x, s2y, eps);
    let (i1x, i1y) = intersections[0];
    let (i2x, i2y) = intersections[1];

    let dx = s2x - s1x;
    let dy = s2y - s1y;
    let segl_sq = dx * dx + dy * dy;

    if i1x != i2x || i1y != i2y {
        // Two distinct intersection points
        let u1 = ((i1x - s1x) * dx + (i1y - s1y) * dy) / segl_sq;
        let u2 = ((i2x - s1x) * dx + (i2y - s1y) * dy) / segl_sq;

        // Sort [0, 1, u1, u2] and take middle two
        let mut sorted = [0.0_f64, 1.0, u1, u2];
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some((sorted[1], sorted[2]))
    } else {
        // Single intersection point (tangent)
        if px == s1x && py == s1y {
            Some((0.0, 0.0))
        } else if px == s2x && py == s2y {
            Some((1.0, 1.0))
        } else {
            let u1 = ((i1x - s1x) * dx + (i1y - s1y) * dy) / segl_sq;
            if (0.0..=1.0).contains(&u1) {
                Some((u1, u1))
            } else {
                None
            }
        }
    }
}

/// Check if a monotone path exists from origin to destination in the free space
///
/// This is the decision problem: given eps, determine if the Frechet distance is ≤ eps.
fn decision_problem<T: CoordSequence>(traj_p: &T, traj_q: &T, p: usize, q: usize, eps: f64) -> bool
where
    T::Coord: AsCoord,
{
    // Compute free space boundaries
    // LF: (p-1) * q entries — free space of segment [P_i, P_{i+1}] from point Q_j
    let mut lf: Vec<Option<(f64, f64)>> = vec![None; (p - 1) * q];
    for j in 0..q {
        for i in 0..p - 1 {
            lf[i * q + j] = free_line(&traj_q.get(j), &traj_p.get(i), &traj_p.get(i + 1), eps);
        }
    }

    // BF: p * (q-1) entries — free space of segment [Q_j, Q_{j+1}] from point P_i
    let mut bf: Vec<Option<(f64, f64)>> = vec![None; p * (q - 1)];
    for j in 0..q - 1 {
        for i in 0..p {
            bf[i * (q - 1) + j] =
                free_line(&traj_p.get(i), &traj_q.get(j), &traj_q.get(j + 1), eps);
        }
    }

    // Helper closures for indexing
    let lf_get = |i: usize, j: usize| -> Option<(f64, f64)> { lf[i * q + j] };
    let bf_get = |i: usize, j: usize| -> Option<(f64, f64)> { bf[i * (q - 1) + j] };

    // Pre-check: start and end must be in free space
    let start_ok =
        lf_get(0, 0).is_some_and(|(a, _)| a <= 0.0) && bf_get(0, 0).is_some_and(|(a, _)| a <= 0.0);
    let end_ok = lf_get(p - 2, q - 1).is_some_and(|(_, b)| b >= 1.0)
        && bf_get(p - 1, q - 2).is_some_and(|(_, b)| b >= 1.0);

    if !start_ok || !end_ok {
        return false;
    }

    // LR and BR: reachability arrays
    let mut lr = vec![false; (p - 1) * q];
    let mut br = vec![false; p * (q - 1)];

    lr[0] = true; // LR[(0,0)]
    br[0] = true; // BR[(0,0)]

    // First column of LR
    for i in 1..p - 1 {
        let prev_full = lf_get(i - 1, 0).is_some_and(|(a, b)| a == 0.0 && b == 1.0);
        lr[i * q] = lf_get(i, 0).is_some() && prev_full;
    }

    // First row of BR
    #[allow(clippy::needless_range_loop)]
    for j in 1..q - 1 {
        let prev_full = bf_get(0, j - 1).is_some_and(|(a, b)| a == 0.0 && b == 1.0);
        br[j] = bf_get(0, j).is_some() && prev_full;
    }

    // Fill remaining cells
    for i in 0..p - 1 {
        for j in 0..q - 1 {
            let lr_ij = lr[i * q + j];
            let br_ij = br[i * (q - 1) + j];

            if lr_ij || br_ij {
                // LR[(i, j+1)]
                if j + 1 < q && lf_get(i, j + 1).is_some() {
                    lr[i * q + (j + 1)] = true;
                }
                // BR[(i+1, j)]
                if i + 1 < p && bf_get(i + 1, j).is_some() {
                    br[(i + 1) * (q - 1) + j] = true;
                }
            }
        }
    }

    // Check if destination is reachable
    br[(p - 2) * (q - 1) + (q - 2)] || lr[(p - 2) * q + (q - 2)]
}

/// Compute all critical values between two trajectories
///
/// Critical values are the candidate distances at which the Frechet distance
/// could change. The actual Frechet distance is one of these values.
fn compute_critical_values<T: CoordSequence>(traj_p: &T, traj_q: &T, p: usize, q: usize) -> Vec<f64>
where
    T::Coord: AsCoord,
{
    let origin_dist = euclidean_distance(&traj_p.get(0), &traj_q.get(0));
    let end_dist = euclidean_distance(&traj_p.get(p - 1), &traj_q.get(q - 1));
    let end_point = origin_dist.max(end_dist);

    let mut values = vec![end_point];

    for i in 0..p - 1 {
        for j in 0..q - 1 {
            let lij = point_to_segment_distance(&traj_q.get(j), &traj_p.get(i), &traj_p.get(i + 1));
            if lij > end_point {
                values.push(lij);
            }
            let bij = point_to_segment_distance(&traj_p.get(i), &traj_q.get(j), &traj_q.get(j + 1));
            if bij > end_point {
                values.push(bij);
            }
        }
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values.dedup();
    values
}

/// Compute the Frechet distance between two trajectories
///
/// The Frechet distance considers all continuous points along the curve segments,
/// providing an exact solution (unlike Discrete Frechet which only considers vertices).
///
/// # Arguments
///
/// * `traj1` - First trajectory
/// * `traj2` - Second trajectory
///
/// # Returns
///
/// The Frechet distance as `f64`.
/// If either trajectory has fewer than 2 points, returns `f64::MAX`.
///
/// # Note
///
/// This function only works with Euclidean distance (2D Cartesian coordinates).
/// Spherical distance is not supported.
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::frechet::frechet;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
/// let traj2 = vec![[0.0, 0.5], [1.0, 1.5], [2.0, 0.5]];
///
/// let distance = frechet(&traj1, &traj2);
/// println!("Frechet distance: {}", distance);
/// ```
pub fn frechet<T: CoordSequence>(traj1: &T, traj2: &T) -> f64
where
    T::Coord: AsCoord,
{
    let p = traj1.len();
    let q = traj2.len();

    // Need at least 2 points for segments
    if p < 2 || q < 2 {
        return f64::MAX;
    }

    let mut cc = compute_critical_values(traj1, traj2, p, q);
    let mut eps = cc[0];

    while cc.len() != 1 {
        let m_i = cc.len() / 2 - 1;
        eps = cc[m_i];
        if decision_problem(traj1, traj2, p, q, eps) {
            cc = cc[..m_i + 1].to_vec();
        } else {
            cc = cc[m_i + 1..].to_vec();
        }
    }

    eps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frechet_simple() {
        let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];
        let dist = frechet(&traj1, &traj2);
        assert!(dist > 0.0);
        assert!(dist.is_finite());
    }

    #[test]
    fn test_frechet_identical() {
        let traj: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
        let dist = frechet(&traj, &traj);
        assert!(
            dist < 1.5e-8,
            "Identical trajectories should have distance ~0, got {}",
            dist
        );
    }

    #[test]
    fn test_frechet_too_short() {
        let single: Vec<[f64; 2]> = vec![[0.0, 0.0]];
        let traj: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        assert_eq!(frechet(&single, &traj), f64::MAX);
        assert_eq!(frechet(&traj, &single), f64::MAX);

        let empty: Vec<[f64; 2]> = vec![];
        assert_eq!(frechet(&empty, &traj), f64::MAX);
    }

    #[test]
    fn test_frechet_symmetry() {
        let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
        let traj2: Vec<[f64; 2]> = vec![[0.0, 0.5], [1.0, 1.5], [2.0, 0.5]];

        let dist12 = frechet(&traj1, &traj2);
        let dist21 = frechet(&traj2, &traj1);

        assert!(
            (dist12 - dist21).abs() < 1.5e-8,
            "Frechet should be symmetric: {} vs {}",
            dist12,
            dist21
        );
    }

    #[test]
    fn test_frechet_leq_discret_frechet() {
        use crate::distance::discret_frechet::discret_frechet_euclidean;

        let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]];
        let traj2: Vec<[f64; 2]> = vec![[0.5, 0.5], [2.0, 1.5], [3.5, 2.5]];

        let cont_dist = frechet(&traj1, &traj2);
        let disc_dist = discret_frechet_euclidean(&traj1, &traj2, false).distance;

        assert!(
            cont_dist <= disc_dist + 1.5e-8,
            "Continuous Frechet ({}) should be <= Discrete Frechet ({})",
            cont_dist,
            disc_dist
        );
    }

    #[test]
    fn test_frechet_different_lengths() {
        let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [3.0, 1.0]];

        let dist = frechet(&traj1, &traj2);
        assert!(dist > 0.0);
        assert!(dist.is_finite());
        // The distance should be 1.0 (perpendicular distance between parallel lines)
        assert!((dist - 1.0).abs() < 1.5e-8, "Expected ~1.0, got {}", dist);
    }

    #[test]
    fn test_free_line_point_inside() {
        let p = [0.5, 0.5];
        let s1 = [0.0, 0.0];
        let s2 = [1.0, 0.0];
        let result = free_line(&p, &s1, &s2, 1.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_free_line_point_outside() {
        let p = [0.5, 10.0];
        let s1 = [0.0, 0.0];
        let s2 = [1.0, 0.0];
        let result = free_line(&p, &s1, &s2, 0.1);
        assert!(result.is_none());
    }

    #[test]
    fn test_free_line_degenerate_segment() {
        let p = [1.0, 0.0];
        let s = [0.0, 0.0];
        assert!(free_line(&p, &s, &s, 2.0).is_some());
        assert!(free_line(&p, &s, &s, 0.5).is_none());
    }
}
