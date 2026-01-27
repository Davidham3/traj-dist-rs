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
//! The time complexity is O(n*m) and space complexity is O(n*m) where n and m are the lengths of the two trajectories.

use crate::distance::distance_type::DistanceType;
use crate::distance::euclidean::euclidean_distance;
use crate::distance::spherical::great_circle_distance;
use crate::traits::{AsCoord, CoordSequence};

/// EDR (Edit Distance on Real sequence) distance between two trajectories using specified distance type
///
/// EDR is a distance measure for trajectories that allows for gaps in the matching.
/// It uses a threshold `eps` to determine if two points match.
///
/// Note: This implementation follows the original traj-dist Python implementation,
/// which initializes the first row and column to 0 (not to the index values).
///
/// # Arguments
///
/// * `t0` - The first trajectory to compare
/// * `t1` - The second trajectory to compare
/// * `eps` - The distance threshold for considering two points as matching
/// * `dist_type` - The type of distance to use (Euclidean or Spherical)
///
/// # Type Parameters
///
/// * `T` - A type that implements the `CoordSequence` trait
///
/// # Returns
///
/// Returns the EDR distance between the two trajectories (value between 0 and 1),
/// or 1.0 if either trajectory is empty
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::edr::edr;
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 0.1], [1.0, 1.1]];
///
/// let distance = edr(&traj1, &traj2, 0.5, DistanceType::Euclidean);
/// println!("EDR distance: {}", distance);
/// ```
pub fn edr<T: CoordSequence>(t0: &T, t1: &T, eps: f64, dist_type: DistanceType) -> f64
where
    T::Coord: AsCoord,
{
    let n0 = t0.len();
    let n1 = t1.len();

    if n0 == 0 || n1 == 0 {
        return 1.0;
    }

    let width = n1 + 1;

    // Create cost matrix (n0 + 1) x (n1 + 1)
    // All values are initialized to 0 (following traj-dist implementation)
    let mut c = vec![0.0; (n0 + 1) * width];

    for i in 1..=n0 {
        let row_offset = i * width;
        let prev_row_offset = (i - 1) * width;

        for j in 1..=n1 {
            let p0 = t0.get(i - 1);
            let p1 = t1.get(j - 1);
            let dist = match dist_type {
                DistanceType::Euclidean => euclidean_distance(&p0, &p1),
                DistanceType::Spherical => great_circle_distance(&p0, &p1),
            };

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
    c[n0 * width + n1] / (n0.max(n1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edr_euclidean_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];
        let eps = 0.5;

        let dist = edr(&t0, &t1, eps, DistanceType::Euclidean);
        println!("EDR Euclidean distance: {}", dist);
        assert!(dist >= 0.0 && dist <= 1.0);
    }

    #[test]
    fn test_edr_euclidean_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let eps = 1e-6;

        let dist = edr(&t0, &t1, eps, DistanceType::Euclidean);
        println!(
            "EDR Euclidean distance for identical trajectories: {}",
            dist
        );
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_edr_euclidean_eps_effect() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.1], [1.0, 1.1]];

        // Small eps: points don't match, distance is larger
        let dist_small = edr(&t0, &t1, 0.05, DistanceType::Euclidean);

        // Large eps: points match, distance is smaller
        let dist_large = edr(&t0, &t1, 1000.0, DistanceType::Euclidean);

        assert!(dist_large <= dist_small);
    }

    #[test]
    fn test_edr_euclidean_empty() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![];
        let eps = 1.0;

        let dist = edr(&t0, &t1, eps, DistanceType::Euclidean);
        println!("EDR Euclidean distance for empty trajectory: {}", dist);
        assert_eq!(dist, 1.0);
    }

    #[test]
    fn test_edr_spherical_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];
        let eps = 100.0;

        let dist = edr(&t0, &t1, eps, DistanceType::Spherical);
        println!("EDR Spherical distance: {}", dist);
        assert!(dist >= 0.0 && dist <= 1.0);
    }

    #[test]
    fn test_edr_spherical_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let eps = 1e-6;

        let dist = edr(&t0, &t1, eps, DistanceType::Spherical);
        println!(
            "EDR Spherical distance for identical trajectories: {}",
            dist
        );
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_edr_spherical_eps_effect() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.1], [1.0, 1.1]];

        // Small eps: points don't match, distance is larger
        let dist_small = edr(&t0, &t1, 10.0, DistanceType::Spherical);

        // Large eps: points match, distance is smaller
        let dist_large = edr(&t0, &t1, 100000.0, DistanceType::Spherical);

        assert!(dist_large <= dist_small);
    }

    #[test]
    fn test_edr_with_both_distance_types() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];
        let eps = 100.0;

        let euclidean_dist = edr(&t0, &t1, eps, DistanceType::Euclidean);
        let spherical_dist = edr(&t0, &t1, eps, DistanceType::Spherical);

        assert!(euclidean_dist >= 0.0 && euclidean_dist <= 1.0);
        assert!(spherical_dist >= 0.0 && spherical_dist <= 1.0);
    }
}
