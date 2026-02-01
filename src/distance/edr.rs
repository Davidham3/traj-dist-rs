//! # EDR (Edit Distance on Real sequence) Algorithm
//!
//! This module implements the Edit Distance on Real sequence algorithm for comparing trajectories.
//! EDR is a distance measure that allows for gaps in the matching and uses a threshold to determine
//! if two points match. It's particularly useful for comparing trajectories with different lengths
//! or with temporal shifts.
//!
//! ## Algorithm Description
//!
//! EDR computes the minimum number of edit operations (insertions, deletions, substitutions)
//! needed to transform one trajectory into another, where two points are considered matching
//! if their distance is less than a given threshold (epsilon).
//!
//! The distance is normalized by the maximum length of the two trajectories.
//!
//! ## Complexity
//!
//! - Time complexity: O(n*m) where n and m are the lengths of the two trajectories
//! - Space complexity:
//!   - O(n*m) when using full matrix (use_full_matrix=true)
//!   - O(min(n,m)) when using optimized 2-row matrix (use_full_matrix=false, default)

use crate::distance::base::DistanceCalculator;

/// EDR (Edit Distance on Real sequence) distance using a distance calculator
///
/// EDR is a distance measure for trajectories that allows for gaps in the matching.
/// It uses a threshold `eps` to determine if two points match.
///
/// Note: This implementation follows the original traj-dist Python implementation,
/// which initializes the first row and column to 0 (not to the index values).
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
/// Returns a `DpResult` containing the EDR distance (value between 0 and 1) and optionally the full DP matrix.
/// If either trajectory is empty, returns `DpResult` with 1.0 as distance and `None` as matrix.
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::edr::edr;
/// use traj_dist_rs::distance::base::{DistanceCalculator, TrajectoryCalculator};
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 0.1], [1.0, 1.1]];
///
/// let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
/// let result = edr(&calculator, 0.5, false);
/// println!("EDR distance: {}", result.distance);
/// ```
pub fn edr<D: DistanceCalculator>(
    calculator: &D,
    eps: f64,
    use_full_matrix: bool,
) -> crate::distance::DpResult {
    let n0 = calculator.len_seq1();
    let n1 = calculator.len_seq2();

    if n0 == 0 || n1 == 0 {
        return crate::distance::DpResult::new(1.0);
    }

    let max_len = n0.max(n1);

    if use_full_matrix {
        let width = n1 + 1;
        // Create cost matrix (n0 + 1) x (n1 + 1)
        // All values are initialized to 0 (following traj-dist implementation)
        let mut c = vec![0.0; (n0 + 1) * width];

        for i in 1..=n0 {
            let row_offset = i * width;
            let prev_row_offset = (i - 1) * width;

            for j in 1..=n1 {
                let dist = calculator.dis_between(i - 1, j - 1);

                // If distance is less than eps, points match (subcost = 0), otherwise subcost = 1
                let subcost = if dist < eps { 0.0 } else { 1.0 };

                // Find minimum of three possible operations:
                // 1. Insert: c[i][j-1] + 1
                // 2. Delete: c[i-1][j] + 1
                // 3. Match/Substitute: c[i-1][j-1] + subcost
                let insert_cost: f64 = c[row_offset + (j - 1)] + 1.0;
                let delete_cost: f64 = c[prev_row_offset + j] + 1.0;
                let match_cost: f64 = c[prev_row_offset + (j - 1)] + subcost;

                c[row_offset + j] = insert_cost.min(delete_cost).min(match_cost);
            }
        }

        // Normalize by the maximum length
        let distance = c[n0 * width + n1] / max_len as f64;
        crate::distance::DpResult::with_matrix(distance, c)
    } else {
        // Optimized version: use only 2 rows
        let mut prev_row = vec![0.0; n1 + 1];
        let mut curr_row = vec![0.0; n1 + 1];

        for i in 1..=n0 {
            curr_row[0] = 0.0;
            for j in 1..=n1 {
                let dist = calculator.dis_between(i - 1, j - 1);

                // If distance is less than eps, points match (subcost = 0), otherwise subcost = 1
                let subcost = if dist < eps { 0.0 } else { 1.0 };

                // Find minimum of three possible operations
                let insert_cost: f64 = curr_row[j - 1] + 1.0;
                let delete_cost: f64 = prev_row[j] + 1.0;
                let match_cost: f64 = prev_row[j - 1] + subcost;

                curr_row[j] = insert_cost.min(delete_cost).min(match_cost);
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        // Normalize by the maximum length
        let distance = prev_row[n1] / max_len as f64;
        crate::distance::DpResult::new(distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::base::TrajectoryCalculator;
    use crate::distance::distance_type::DistanceType;

    #[test]
    fn test_edr_euclidean_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];
        let eps = 0.5;

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result = edr(&calculator, eps, false);
        println!("EDR Euclidean distance: {}", result.distance);
        assert!(result.distance >= 0.0 && result.distance <= 1.0);
    }

    #[test]
    fn test_edr_euclidean_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let eps = 1e-6;

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result = edr(&calculator, eps, false);
        println!(
            "EDR Euclidean distance for identical trajectories: {}",
            result.distance
        );
        assert!(result.distance < 1e-6);
    }

    #[test]
    fn test_edr_euclidean_eps_effect() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.1], [1.0, 1.1]];

        let calc_small = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let calc_large = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);

        // Small eps: points don't match, distance is larger
        let result_small = edr(&calc_small, 0.05, false);

        // Large eps: points match, distance is smaller
        let result_large = edr(&calc_large, 1000.0, false);

        assert!(result_large.distance <= result_small.distance);
    }

    #[test]
    fn test_edr_euclidean_empty() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![];
        let eps = 1.0;

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result = edr(&calculator, eps, false);
        println!(
            "EDR Euclidean distance for empty trajectory: {}",
            result.distance
        );
        assert_eq!(result.distance, 1.0);
    }

    #[test]
    fn test_edr_spherical_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];
        let eps = 100.0;

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);
        let result = edr(&calculator, eps, false);
        println!("EDR Spherical distance: {}", result.distance);
        assert!(result.distance >= 0.0 && result.distance <= 1.0);
    }

    #[test]
    fn test_edr_spherical_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let eps = 1e-6;

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);
        let result = edr(&calculator, eps, false);
        println!(
            "EDR Spherical distance for identical trajectories: {}",
            result.distance
        );
        assert!(result.distance < 1e-6);
    }

    #[test]
    fn test_edr_spherical_eps_effect() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.1], [1.0, 1.1]];

        let calc_small = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);
        let calc_large = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);

        // Small eps: points don't match, distance is larger
        let result_small = edr(&calc_small, 10.0, false);

        // Large eps: points match, distance is smaller
        let result_large = edr(&calc_large, 100000.0, false);

        assert!(result_large.distance <= result_small.distance);
    }

    #[test]
    fn test_edr_with_both_distance_types() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];
        let eps = 100.0;

        let euclidean_calc = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let spherical_calc = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);

        let euclidean_result = edr(&euclidean_calc, eps, false);
        let spherical_result = edr(&spherical_calc, eps, false);

        assert!(euclidean_result.distance >= 0.0 && euclidean_result.distance <= 1.0);
        assert!(spherical_result.distance >= 0.0 && spherical_result.distance <= 1.0);
    }

    #[test]
    fn test_edr_consistency_between_modes() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];
        let eps = 1.0;

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result_optimized = edr(&calculator, eps, false);
        let result_full = edr(&calculator, eps, true);

        // Both modes should produce the same distance
        assert!((result_optimized.distance - result_full.distance).abs() < 1e-10);

        // Optimized mode should not return matrix
        assert!(result_optimized.matrix.is_none());

        // Full mode should return matrix
        assert!(result_full.matrix.is_some());
    }

    #[test]
    fn test_edr_with_precomputed_distances() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let distance_matrix =
            crate::distance::utils::precompute_distance_matrix(&t0, &t1, DistanceType::Euclidean);

        let calculator =
            crate::distance::base::PrecomputedDistanceCalculator::new(&distance_matrix);

        let result = edr(&calculator, 0.5, false);

        println!(
            "EDR distance with precomputed distances: {}",
            result.distance
        );

        assert!(result.distance >= 0.0 && result.distance <= 1.0);
    }
}
