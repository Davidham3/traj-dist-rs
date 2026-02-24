//! # DTW (Dynamic Time Warping) Algorithm
//!
//! This module implements the Dynamic Time Warping algorithm for comparing trajectories.
//! DTW finds the optimal alignment between two sequences by dynamically warping the time axis,
//! allowing for similar patterns that are out of phase to be matched.
//!
//! ## Algorithm Description
//!
//! The DTW algorithm works as follows:
//! 1. Construct a cost matrix where each cell (i,j) represents the distance between points i and j
//! 2. Find the optimal warping path that minimizes the cumulative distance
//! 3. The final DTW distance is the total cost along the optimal path
//!
//! ## Complexity
//!
//! - Time complexity: O(n*m) where n and m are the lengths of the two trajectories
//! - Space complexity:
//!   - O(n*m) when using full matrix (use_full_matrix=true)
//!   - O(min(n,m)) when using optimized 2-row matrix (use_full_matrix=false, default)

use crate::distance::base::DistanceCalculator;

/// DTW (Dynamic Time Warping) distance using a distance calculator
///
/// Computes the Dynamic Time Warping distance using the provided distance calculator.
/// DTW finds the optimal alignment between two sequences by dynamically warping the time axis,
/// allowing for similar patterns that are out of phase.
///
/// # Arguments
///
/// * `calculator` - A distance calculator that implements the `DistanceCalculator` trait
/// * `use_full_matrix` - If true, compute and return the full (n0+1) x (n1+1) DP matrix;
///   if false, use optimized 2-row matrix to save space (default recommended)
///
/// # Type Parameters
///
/// * `D` - A type that implements the `DistanceCalculator` trait
///
/// # Returns
///
/// Returns a `DpResult` containing the distance and optionally the full DP matrix.
/// If either trajectory is empty, returns `DpResult` with `f64::MAX` as distance and `None` as matrix.
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::dtw::dtw;
/// use traj_dist_rs::distance::base::{DistanceCalculator, TrajectoryCalculator};
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];
///
/// let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
///
/// // Without matrix
/// let result = dtw(&calculator, false);
/// println!("DTW distance: {}", result.distance);
///
/// // With matrix
/// let result_with_matrix = dtw(&calculator, true);
/// println!("DTW distance: {}", result_with_matrix.distance);
/// if let Some(matrix) = result_with_matrix.matrix {
///     println!("Matrix length: {}", matrix.len());
/// }
/// ```
pub fn dtw<D: DistanceCalculator>(
    calculator: &D,
    use_full_matrix: bool,
) -> crate::distance::DpResult {
    let n0 = calculator.len_seq1();
    let n1 = calculator.len_seq2();

    if n0 == 0 || n1 == 0 {
        return crate::distance::DpResult::new(f64::MAX);
    }

    if use_full_matrix {
        // Create cost matrix (n0 + 1) x (n1 + 1)
        let mut c = vec![f64::INFINITY; (n0 + 1) * (n1 + 1)];
        c[0] = 0.0;

        for i in 1..=n0 {
            for j in 1..=n1 {
                let dist = calculator.dis_between(i - 1, j - 1);

                let min_prev = c[(i - 1) * (n1 + 1) + (j - 1)]
                    .min(c[(i - 1) * (n1 + 1) + j])
                    .min(c[i * (n1 + 1) + (j - 1)]);

                c[i * (n1 + 1) + j] = dist + min_prev;
            }
        }

        crate::distance::DpResult::with_matrix(c[n0 * (n1 + 1) + n1], c)
    } else {
        // Optimized version: use only 2 rows to save space
        let mut prev_row = vec![f64::INFINITY; n1 + 1];
        let mut curr_row = vec![f64::INFINITY; n1 + 1];
        prev_row[0] = 0.0;

        for i in 1..=n0 {
            curr_row[0] = f64::INFINITY;
            for j in 1..=n1 {
                let dist = calculator.dis_between(i - 1, j - 1);

                let min_prev = prev_row[j - 1].min(prev_row[j]).min(curr_row[j - 1]);
                curr_row[j] = dist + min_prev;
            }
            // Swap rows for next iteration
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        crate::distance::DpResult::new(prev_row[n1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::base::TrajectoryCalculator;
    use crate::distance::distance_type::DistanceType;

    #[test]
    fn test_dtw_euclidean_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result = dtw(&calculator, false);
        println!("DTW Euclidean distance: {}", result.distance);
        assert!(result.distance > 0.0);
        assert!(result.matrix.is_none());
    }

    #[test]
    fn test_dtw_euclidean_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result = dtw(&calculator, false);
        println!(
            "DTW Euclidean distance for identical trajectories: {}",
            result.distance
        );
        assert!(result.distance < 1e-6);
    }

    #[test]
    fn test_dtw_spherical_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);
        let result = dtw(&calculator, false);
        println!("DTW Spherical distance: {}", result.distance);
        assert!(result.distance > 0.0);
    }

    #[test]
    fn test_dtw_with_both_distance_types() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let euclidean_calc = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let spherical_calc = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);

        let euclidean_result = dtw(&euclidean_calc, false);
        let spherical_result = dtw(&spherical_calc, false);

        assert!(euclidean_result.distance > 0.0);
        assert!(spherical_result.distance > 0.0);
    }

    #[test]
    fn test_dtw_consistency_between_modes() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result_optimized = dtw(&calculator, false);
        let result_full = dtw(&calculator, true);

        // Both modes should produce the same distance
        assert!((result_optimized.distance - result_full.distance).abs() < 1e-10);

        // Optimized mode should not return matrix
        assert!(result_optimized.matrix.is_none());

        // Full mode should return matrix
        assert!(result_full.matrix.is_some());
        if let Some(matrix) = result_full.matrix {
            // Matrix should be (n0+1) x (n1+1) = 4 x 4 = 16
            assert_eq!(matrix.len(), 16);
        }
    }

    #[test]
    fn test_dtw_empty_trajectories() {
        let t0: Vec<[f64; 2]> = vec![];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result = dtw(&calculator, false);
        assert_eq!(result.distance, f64::MAX);
        assert!(result.matrix.is_none());
    }

    #[test]
    fn test_dtw_matrix_content() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result = dtw(&calculator, true);

        assert!(result.matrix.is_some());
        if let Some(matrix) = result.matrix {
            // Matrix should be (n0+1) x (n1+1) = 3 x 3 = 9
            assert_eq!(matrix.len(), 9);

            // Check that matrix[0] is 0.0 (initialization)
            assert_eq!(matrix[0], 0.0);

            // Check that matrix[4] (the result) equals result.distance
            // For 2x2 trajectories, result is at matrix[2*3 + 2] = matrix[8]
            assert!((matrix[8] - result.distance).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dtw_with_precomputed_distances() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        // Precompute distance matrix using utility function

        let distance_matrix =
            crate::distance::utils::precompute_distance_matrix(&t0, &t1, DistanceType::Euclidean);

        let calculator = crate::distance::base::PrecomputedDistanceCalculator::new(
            &distance_matrix,
            t0.len(),
            t1.len(),
        );

        let result = dtw(&calculator, false);

        println!(
            "DTW distance with precomputed distances: {}",
            result.distance
        );

        assert!(result.distance > 0.0);
    }

    #[test]
    fn test_dtw_with_precomputed_distances_spherical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        // Precompute distance matrix for spherical distances

        let distance_matrix =
            crate::distance::utils::precompute_distance_matrix(&t0, &t1, DistanceType::Spherical);

        let calculator = crate::distance::base::PrecomputedDistanceCalculator::new(
            &distance_matrix,
            t0.len(),
            t1.len(),
        );

        let result = dtw(&calculator, false);

        println!(
            "DTW distance with precomputed spherical distances: {}",
            result.distance
        );

        assert!(result.distance > 0.0);
    }

    #[test]
    fn test_dtw_precomputed_vs_trajectory_calculator() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];

        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

        // Using TrajectoryCalculator

        let traj_calc = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);

        let result_traj = dtw(&traj_calc, false);

        // Using PrecomputedDistanceCalculator

        let distance_matrix =
            crate::distance::utils::precompute_distance_matrix(&t0, &t1, DistanceType::Euclidean);

        let precomp_calc = crate::distance::base::PrecomputedDistanceCalculator::new(
            &distance_matrix,
            t0.len(),
            t1.len(),
        );

        let result_precomp = dtw(&precomp_calc, false);

        // Both should produce the same distance

        assert!((result_traj.distance - result_precomp.distance).abs() < 1e-10);
    }
}
