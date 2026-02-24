//! # Utility functions for distance calculations
//!
//! This module provides helper functions for distance calculations,
//! including precomputing distance matrices.

use crate::distance::distance_type::DistanceType;
use crate::distance::euclidean::euclidean_distance;
use crate::distance::spherical::great_circle_distance;
use crate::traits::{AsCoord, CoordSequence};

/// Precompute the distance matrix between two trajectories
///
/// Given two trajectories and a distance type, this function computes
/// the distance between every pair of points from the two trajectories.
///
/// # Arguments
///
/// * `t0` - The first trajectory
/// * `t1` - The second trajectory
/// * `dist_type` - The type of distance to use (Euclidean or Spherical)
///
/// # Type Parameters
///
/// * `T` - A type that implements the `CoordSequence` trait
/// * `U` - A type that implements the `CoordSequence` trait
///
/// # Returns
///
/// Returns a 1D vector `Vec<f64>` representing a row-major distance matrix
/// where `matrix[i * n1 + j]` represents the distance between `t0[i]` and `t1[j]`.
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::utils::precompute_distance_matrix;
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];
///
/// let matrix = precompute_distance_matrix(&traj1, &traj2, DistanceType::Euclidean);
/// println!("Distance between traj1[0] and traj2[0]: {}", matrix[0]);
/// println!("Distance between traj1[0] and traj2[1]: {}", matrix[1]);
/// ```
pub fn precompute_distance_matrix<T: CoordSequence, U: CoordSequence>(
    t0: &T,
    t1: &U,
    dist_type: DistanceType,
) -> Vec<f64>
where
    T::Coord: AsCoord,
    U::Coord: AsCoord,
{
    let n0 = t0.len();
    let n1 = t1.len();

    let mut distance_matrix = vec![0.0; n0 * n1];

    for i in 0..n0 {
        for j in 0..n1 {
            let p0 = t0.get(i);
            let p1 = t1.get(j);
            distance_matrix[i * n1 + j] = match dist_type {
                DistanceType::Euclidean => euclidean_distance(&p0, &p1),
                DistanceType::Spherical => great_circle_distance(&p0, &p1),
            };
        }
    }

    distance_matrix
}

/// Extract coordinates from a trajectory into a vector of [f64; 2]
///
/// This helper function is used to convert a trajectory into a format
/// that can be used with PrecomputedDistanceCalculator.
///
/// # Arguments
///
/// * `t` - The trajectory to extract coordinates from
///
/// # Type Parameters
///
/// * `T` - A type that implements the `CoordSequence` trait
///
/// # Returns
///
/// Returns a vector of coordinates as [f64; 2] arrays.
pub fn extract_coords<T: CoordSequence>(t: &T) -> Vec<[f64; 2]>
where
    T::Coord: AsCoord,
{
    let mut coords = Vec::with_capacity(t.len());
    for i in 0..t.len() {
        let coord = t.get(i);
        coords.push([coord.x(), coord.y()]);
    }
    coords
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precompute_euclidean() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let matrix = precompute_distance_matrix(&t0, &t1, DistanceType::Euclidean);
        let n1 = t1.len();

        // Check dimensions (flat array with n0 * n1 elements)
        assert_eq!(matrix.len(), 4);

        // Check some values using row-major indexing
        // Distance between [0,0] and [0,1] should be 1.0
        assert!((matrix[0 * n1 + 0] - 1.0).abs() < 1e-10);
        // Distance between [0,0] and [1,0] should be 1.0
        assert!((matrix[0 * n1 + 1] - 1.0).abs() < 1e-10);
        // Distance between [1,1] and [0,1] should be 1.0
        assert!((matrix[1 * n1 + 0] - 1.0).abs() < 1e-10);
        // Distance between [1,1] and [1,0] should be 1.0
        assert!((matrix[1 * n1 + 1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_precompute_spherical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let matrix = precompute_distance_matrix(&t0, &t1, DistanceType::Spherical);
        let n1 = t1.len();

        // Check dimensions (flat array with n0 * n1 elements)
        assert_eq!(matrix.len(), 4);

        // All distances should be positive
        assert!(matrix[0 * n1 + 0] > 0.0);
        assert!(matrix[0 * n1 + 1] > 0.0);
        assert!(matrix[1 * n1 + 0] > 0.0);
        assert!(matrix[1 * n1 + 1] > 0.0);
    }

    #[test]
    fn test_precompute_empty() {
        let t0: Vec<[f64; 2]> = vec![];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0]];

        let matrix = precompute_distance_matrix(&t0, &t1, DistanceType::Euclidean);

        assert_eq!(matrix.len(), 0);
    }

    #[test]
    fn test_precompute_consistency() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let matrix = precompute_distance_matrix(&t0, &t1, DistanceType::Euclidean);
        let n1 = t1.len();

        // Verify that matrix values match direct distance calculations
        assert!((matrix[0 * n1 + 0] - euclidean_distance(&t0[0], &t1[0])).abs() < 1e-10);
        assert!((matrix[0 * n1 + 1] - euclidean_distance(&t0[0], &t1[1])).abs() < 1e-10);
        assert!((matrix[1 * n1 + 0] - euclidean_distance(&t0[1], &t1[0])).abs() < 1e-10);
        assert!((matrix[1 * n1 + 1] - euclidean_distance(&t0[1], &t1[1])).abs() < 1e-10);
    }
}
