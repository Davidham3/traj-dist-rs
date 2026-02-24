//! # Discret Frechet Distance Algorithm
//!
//! This module implements the Discrete Fréchet distance algorithm for comparing trajectories.
//! The Discrete Fréchet distance is a measure of similarity between two curves that takes
//! into account the location and ordering of the points along the curves.
//!
//! ## Algorithm Description
//!
//! The Discrete Fréchet distance is computed using dynamic programming where:
//! - C[i][j] = max(dist(t0[i-1], t1[j-1]), min(C[i][j-1], C[i-1][j-1], C[i-1][j]))
//! - First row and column (except C[0][0]) are initialized to infinity
//!
//! Intuitively, it represents the minimum length of a leash required for a person
//! and their dog to walk along the two trajectories without backtracking.
//!
//! ## Complexity
//!
//! - Time complexity: O(n*m) where n and m are the lengths of the two trajectories
//! - Space complexity:
//!   - O(n*m) when using full matrix (use_full_matrix=true)
//!   - O(min(n,m)) when using optimized 2-row matrix (use_full_matrix=false, default)

use crate::distance::base::DistanceCalculator;

/// Discret Frechet distance using a distance calculator
///
/// The discret Frechet distance is a measure of similarity between two curves
/// that takes into account the location and ordering of the points along the curves.
///
/// It is computed using dynamic programming where:
/// - C[i][j] = max(dist(t0[i-1], t1[j-1]), min(C[i][j-1], C[i-1][j-1], C[i-1][j]))
/// - First row and column (except C[0][0]) are initialized to infinity
///
/// Note: This algorithm only supports Euclidean distance. The distance calculator
/// must be configured with Euclidean distance type.
///
/// # Arguments
///
/// * `calculator` - A distance calculator that implements the `DistanceCalculator` trait
/// * `use_full_matrix` - If true, compute and return the full DP matrix;
///   if false (default), return None for the matrix to save space
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
/// use traj_dist_rs::distance::discret_frechet::discret_frechet;
/// use traj_dist_rs::distance::base::{DistanceCalculator, TrajectoryCalculator};
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 0.1], [1.0, 1.1]];
///
/// let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
/// let result = discret_frechet(&calculator, false);
/// println!("Discrete Fréchet distance: {}", result.distance);
/// ```
pub fn discret_frechet<D: DistanceCalculator>(
    calculator: &D,
    use_full_matrix: bool,
) -> crate::distance::DpResult {
    let n0 = calculator.len_seq1();
    let n1 = calculator.len_seq2();

    if n0 == 0 || n1 == 0 {
        return crate::distance::DpResult::new(f64::MAX);
    }

    if use_full_matrix {
        let width = n1 + 1;
        // Create cost matrix (n0 + 1) x (n1 + 1)
        let mut c = vec![0.0; (n0 + 1) * width];

        // Initialize first row and column to infinity (except C[0][0])
        for i in 1..=n0 {
            c[i * width] = f64::INFINITY;
        }
        for val in &mut c[1..=n1] {
            *val = f64::INFINITY;
        }

        for i in 1..=n0 {
            let row_offset = i * width;
            let prev_row_offset = (i - 1) * width;

            for j in 1..=n1 {
                let dist = calculator.dis_between(i - 1, j - 1);

                // C[i][j] = max(dist, min(C[i][j-1], C[i-1][j-1], C[i-1][j]))
                let min_prev = c[row_offset + (j - 1)]
                    .min(c[prev_row_offset + (j - 1)])
                    .min(c[prev_row_offset + j]);

                c[row_offset + j] = dist.max(min_prev);
            }
        }

        crate::distance::DpResult::with_matrix(c[n0 * width + n1], c)
    } else {
        // Optimized version: use only 2 rows
        let mut prev_row = vec![f64::INFINITY; n1 + 1];
        let mut curr_row = vec![f64::INFINITY; n1 + 1];
        prev_row[0] = 0.0;

        for i in 1..=n0 {
            curr_row[0] = f64::INFINITY;
            for j in 1..=n1 {
                let dist = calculator.dis_between(i - 1, j - 1);

                // C[i][j] = max(dist, min(C[i][j-1], C[i-1][j-1], C[i-1][j]))
                let min_prev = curr_row[j - 1].min(prev_row[j - 1]).min(prev_row[j]);
                curr_row[j] = dist.max(min_prev);
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        crate::distance::DpResult::new(prev_row[n1])
    }
}

/// Discret Frechet distance between two trajectories using Euclidean geometry
///
/// This is a convenience function that creates a TrajectoryCalculator with Euclidean distance
/// and calls discret_frechet. This function is kept for backward compatibility.
///
/// # Arguments
///
/// * `t0` - The first trajectory to compare
/// * `t1` - The second trajectory to compare
/// * `use_full_matrix` - If true, compute and return the full DP matrix;
///   if false (default), return None for the matrix to save space
///
/// # Type Parameters
///
/// * `T` - A type that implements the `CoordSequence` trait
///
/// # Returns
///
/// Returns a `DpResult` containing the distance and optionally the full DP matrix.
/// If either trajectory is empty, returns `DpResult` with `f64::MAX` as distance and `None` as matrix.
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::discret_frechet::discret_frechet_euclidean;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 0.1], [1.0, 1.1]];
///
/// let result = discret_frechet_euclidean(&traj1, &traj2, false);
/// println!("Discrete Fréchet distance: {}", result.distance);
/// ```
pub fn discret_frechet_euclidean<T: crate::traits::CoordSequence>(
    t0: &T,
    t1: &T,
    use_full_matrix: bool,
) -> crate::distance::DpResult
where
    T::Coord: crate::traits::AsCoord,
{
    use crate::distance::base::TrajectoryCalculator;
    use crate::distance::distance_type::DistanceType;

    let calculator = TrajectoryCalculator::new(t0, t1, DistanceType::Euclidean);
    discret_frechet(&calculator, use_full_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::base::TrajectoryCalculator;
    use crate::distance::distance_type::DistanceType;

    #[test]
    fn test_discret_frechet_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let dist = discret_frechet(&calculator, false);
        println!("Discret Frechet distance: {}", dist);
        assert!(dist.distance > 0.0);
    }

    #[test]
    fn test_discret_frechet_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result = discret_frechet(&calculator, false);
        println!(
            "Discret Frechet distance for identical trajectories: {}",
            result.distance
        );
        assert!(result.distance < 1e-6);
    }

    #[test]
    fn test_discret_frechet_euclidean_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let dist = discret_frechet_euclidean(&t0, &t1, false);
        println!("Discret Frechet Euclidean distance: {}", dist);
        assert!(dist.distance > 0.0);
    }

    #[test]
    fn test_discret_frechet_euclidean_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let result = discret_frechet_euclidean(&t0, &t1, false);
        println!(
            "Discret Frechet Euclidean distance for identical trajectories: {}",
            result.distance
        );
        assert!(result.distance < 1e-6);
    }

    #[test]
    fn test_discret_frechet_euclidean_empty() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![];

        let result = discret_frechet_euclidean(&t0, &t1, false);
        println!(
            "Discret Frechet Euclidean distance for empty trajectory: {}",
            result.distance
        );
        assert_eq!(result.distance, f64::MAX);
    }

    #[test]
    fn test_discret_frechet_euclidean_single_point() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0]];
        let t1: Vec<[f64; 2]> = vec![[1.0, 1.0]];

        let result = discret_frechet_euclidean(&t0, &t1, false);
        println!(
            "Discret Frechet Euclidean distance for single point: {}",
            result.distance
        );

        // Single point trajectories: distance should be the Euclidean distance
        let sqrt_2 = std::f64::consts::SQRT_2;
        assert!((result.distance - sqrt_2).abs() < 1e-6);
    }

    #[test]
    fn test_discret_frechet_consistency_between_modes() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let result_optimized = discret_frechet(&calculator, false);
        let result_full = discret_frechet(&calculator, true);

        // Both modes should produce the same distance
        assert!((result_optimized.distance - result_full.distance).abs() < 1e-10);

        // Optimized mode should not return matrix
        assert!(result_optimized.matrix.is_none());

        // Full mode should return matrix
        assert!(result_full.matrix.is_some());
    }

    #[test]
    fn test_discret_frechet_with_precomputed_distances() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let distance_matrix =
            crate::distance::utils::precompute_distance_matrix(&t0, &t1, DistanceType::Euclidean);

        let calculator = crate::distance::base::PrecomputedDistanceCalculator::new(
            &distance_matrix,
            t0.len(),
            t1.len(),
        );

        let result = discret_frechet(&calculator, false);

        println!(
            "Discret Frechet distance with precomputed distances: {}",
            result.distance
        );

        assert!(result.distance > 0.0);
    }
}
