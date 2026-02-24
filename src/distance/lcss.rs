//! # LCSS (Longest Common Subsequence) Algorithm
//!
//! This module implements the Longest Common Subsequence algorithm for comparing trajectories.
//! LCSS is a similarity measure that finds the longest subsequence common to both trajectories.
//! Two points are considered matching if their distance is less than a given threshold (epsilon).
//!
//! ## Algorithm Description
//!
//! The LCSS distance is calculated as:
//! 1 - (length of longest common subsequence) / min(len(t0), len(t1))
//! where two points are considered matching if their distance is less than `eps` according to the specified distance type.
//!
//! ## Complexity
//!
//! - Time complexity: O(n*m) where n and m are the lengths of the two trajectories
//! - Space complexity:
//!   - O(n*m) when using full matrix (use_full_matrix=true)
//!   - O(min(n,m)) when using optimized 2-row matrix (use_full_matrix=false, default)

use crate::distance::base::DistanceCalculator;

/// LCSS (Longest Common Subsequence) distance using a distance calculator
///
/// The LCSS distance is calculated as 1 - (length of longest common subsequence) / min(len(t0), len(t1))
/// where two points are considered matching if their distance is less than eps according to the specified distance type.
///
/// # Arguments
///
/// * `calculator` - A distance calculator that implements the `DistanceCalculator` trait
/// * `eps` - The distance threshold for considering two points as matching
/// * `use_full_matrix` - If true, compute and return the full DP matrix;
///   if false (default), return None for the matrix to save space
///
/// # Type Parameters
///
/// * `D` - A type that implements the `DistanceCalculator` trait
///
/// # Returns
///
/// Returns a `DpResult` containing the LCSS distance (value between 0 and 1) and optionally the full DP matrix.
/// If either trajectory is empty, returns `DpResult` with 1.0 as distance and `None` as matrix.
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::lcss::lcss;
/// use traj_dist_rs::distance::base::{DistanceCalculator, TrajectoryCalculator};
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 0.1], [1.0, 1.1]];
///
/// let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
/// let result = lcss(&calculator, 0.5, false);
/// println!("LCSS distance: {}", result.distance);
/// ```
pub fn lcss<D: DistanceCalculator>(
    calculator: &D,
    eps: f64,
    use_full_matrix: bool,
) -> crate::distance::DpResult {
    let n0 = calculator.len_seq1();
    let n1 = calculator.len_seq2();

    if n0 == 0 || n1 == 0 {
        return crate::distance::DpResult::new(1.0);
    }

    let min_len = n0.min(n1);

    if use_full_matrix {
        // Create LCS matrix (n0 + 1) x (n1 + 1)
        let mut c = vec![0usize; (n0 + 1) * (n1 + 1)];

        for i in 1..=n0 {
            for j in 1..=n1 {
                let dist = calculator.dis_between(i - 1, j - 1);

                if dist < eps {
                    c[i * (n1 + 1) + j] = c[(i - 1) * (n1 + 1) + (j - 1)] + 1;
                } else {
                    let left = c[i * (n1 + 1) + (j - 1)];
                    let top = c[(i - 1) * (n1 + 1) + j];
                    c[i * (n1 + 1) + j] = left.max(top);
                }
            }
        }

        let lcss_value = c[n0 * (n1 + 1) + n1] as f64;
        let distance = 1.0 - lcss_value / min_len as f64;
        // Convert usize matrix to f64 matrix for consistency
        let matrix: Vec<f64> = c.into_iter().map(|x| x as f64).collect();
        crate::distance::DpResult::with_matrix(distance, matrix)
    } else {
        // Optimized version: use only 2 rows
        let mut prev_row = vec![0usize; n1 + 1];
        let mut curr_row = vec![0usize; n1 + 1];

        for i in 1..=n0 {
            curr_row[0] = 0;
            for j in 1..=n1 {
                let dist = calculator.dis_between(i - 1, j - 1);

                if dist < eps {
                    curr_row[j] = prev_row[j - 1] + 1;
                } else {
                    curr_row[j] = curr_row[j - 1].max(prev_row[j]);
                }
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        let lcss_value = prev_row[n1] as f64;
        let distance = 1.0 - lcss_value / min_len as f64;
        crate::distance::DpResult::new(distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::base::TrajectoryCalculator;
    use crate::distance::distance_type::DistanceType;

    #[test]
    fn test_lcss_euclidean_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let dist = lcss(&calculator, 0.5, false);
        println!("LCSS Euclidean distance: {}", dist);
        assert!(dist.distance >= 0.0 && dist.distance <= 1.0);
    }

    #[test]
    fn test_lcss_euclidean_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result = lcss(&calculator, 1e-6, false);
        println!(
            "LCSS Euclidean distance for identical trajectories: {}",
            result.distance
        );
        assert!(result.distance < 1e-6);
    }

    #[test]
    fn test_lcss_euclidean_eps() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let calc_small = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let calc_large = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);

        // Small eps should match identical points
        let result_small = lcss(&calc_small, 1e-6, false);
        assert!(result_small.distance < 1e-6);

        // Large eps should match all points
        let result_large = lcss(&calc_large, 1000.0, false);
        assert!(result_large.distance < 1e-6);
    }

    #[test]
    fn test_lcss_spherical_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);
        let result = lcss(&calculator, 100000.0, false);
        println!("LCSS Spherical distance: {}", result.distance);
        assert!(result.distance >= 0.0 && result.distance <= 1.0);
    }

    #[test]
    fn test_lcss_spherical_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);
        let result = lcss(&calculator, 1e-6, false);
        println!(
            "LCSS Spherical distance for identical trajectories: {}",
            result.distance
        );
        assert!(result.distance < 1e-6);
    }

    #[test]
    fn test_lcss_empty_trajectory() {
        let t0: Vec<[f64; 2]> = vec![];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let calc_euclid = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result = lcss(&calc_euclid, 1.0, false);
        assert_eq!(result.distance, 1.0);

        let calc_spherical = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);
        let result = lcss(&calc_spherical, 1.0, false);
        assert_eq!(result.distance, 1.0);
    }

    #[test]
    fn test_lcss_with_both_distance_types() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let euclidean_calc = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let spherical_calc = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);

        let euclidean_result = lcss(&euclidean_calc, 100.0, false);
        let spherical_result = lcss(&spherical_calc, 100000.0, false);

        assert!(euclidean_result.distance >= 0.0 && euclidean_result.distance <= 1.0);
        assert!(spherical_result.distance >= 0.0 && spherical_result.distance <= 1.0);
    }

    #[test]
    fn test_lcss_consistency_between_modes() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result_optimized = lcss(&calculator, 1.0, false);
        let result_full = lcss(&calculator, 1.0, true);

        // Both modes should produce the same distance
        assert!((result_optimized.distance - result_full.distance).abs() < 1e-10);

        // Optimized mode should not return matrix
        assert!(result_optimized.matrix.is_none());

        // Full mode should return matrix
        assert!(result_full.matrix.is_some());
    }

    #[test]
    fn test_lcss_with_precomputed_distances() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let distance_matrix =
            crate::distance::utils::precompute_distance_matrix(&t0, &t1, DistanceType::Euclidean);

        let calculator = crate::distance::base::PrecomputedDistanceCalculator::new(
            &distance_matrix,
            t0.len(),
            t1.len(),
        );

        let result = lcss(&calculator, 0.5, false);

        println!(
            "LCSS distance with precomputed distances: {}",
            result.distance
        );

        assert!(result.distance >= 0.0 && result.distance <= 1.0);
    }
}
