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
/// Returns a 2D vector `Vec<Vec<f64>>` where `matrix[i][j]` represents
/// the distance between `t0[i]` and `t1[j]`.
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
/// println!("Distance between traj1[0] and traj2[0]: {}", matrix[0][0]);
/// ```
pub fn precompute_distance_matrix<T: CoordSequence, U: CoordSequence>(
    t0: &T,
    t1: &U,
    dist_type: DistanceType,
) -> Vec<Vec<f64>>
where
    T::Coord: AsCoord,
    U::Coord: AsCoord,
{
    let n0 = t0.len();
    let n1 = t1.len();

    let mut distance_matrix = vec![vec![0.0; n1]; n0];

    for (i, row) in distance_matrix.iter_mut().enumerate() {
        for (j, dist) in row.iter_mut().enumerate() {
            let p0 = t0.get(i);
            let p1 = t1.get(j);
            *dist = match dist_type {
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

        // Check dimensions
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
        assert_eq!(matrix[1].len(), 2);

        // Check some values
        // Distance between [0,0] and [0,1] should be 1.0
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
        // Distance between [0,0] and [1,0] should be 1.0
        assert!((matrix[0][1] - 1.0).abs() < 1e-10);
        // Distance between [1,1] and [0,1] should be 1.0
        assert!((matrix[1][0] - 1.0).abs() < 1e-10);
        // Distance between [1,1] and [1,0] should be 1.0
        assert!((matrix[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_precompute_spherical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let matrix = precompute_distance_matrix(&t0, &t1, DistanceType::Spherical);

        // Check dimensions
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);

        // All distances should be positive
        assert!(matrix[0][0] > 0.0);
        assert!(matrix[0][1] > 0.0);
        assert!(matrix[1][0] > 0.0);
        assert!(matrix[1][1] > 0.0);
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

        // Verify that matrix values match direct distance calculations
        assert!((matrix[0][0] - euclidean_distance(&t0[0], &t1[0])).abs() < 1e-10);
        assert!((matrix[0][1] - euclidean_distance(&t0[0], &t1[1])).abs() < 1e-10);
        assert!((matrix[1][0] - euclidean_distance(&t0[1], &t1[0])).abs() < 1e-10);
        assert!((matrix[1][1] - euclidean_distance(&t0[1], &t1[1])).abs() < 1e-10);
    }
}
