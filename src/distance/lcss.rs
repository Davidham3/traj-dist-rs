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
//! The time complexity is O(n*m) and space complexity is O(n*m) where n and m are the lengths of the two trajectories.

use crate::distance::distance_type::DistanceType;
use crate::distance::euclidean::euclidean_distance;
use crate::distance::spherical::great_circle_distance;
use crate::traits::{AsCoord, CoordSequence};

/// LCSS (Longest Common Subsequence) distance between two trajectories using specified distance type
///
/// The LCSS distance is calculated as 1 - (length of longest common subsequence) / min(len(t0), len(t1))
/// where two points are considered matching if their distance is less than eps according to the specified distance type.
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
/// Returns the LCSS distance between the two trajectories (value between 0 and 1), 
/// or 1.0 if either trajectory is empty
/// 
/// # Example
/// 
/// ```rust
/// use traj_dist_rs::{lcss, DistanceType};
/// 
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 0.1], [1.0, 1.1]];
/// 
/// let distance = lcss(&traj1, &traj2, 0.5, DistanceType::Euclidean);
/// println!("LCSS distance: {}", distance);
/// ```
pub fn lcss<T: CoordSequence>(t0: &T, t1: &T, eps: f64, dist_type: DistanceType) -> f64
where
    T::Coord: AsCoord,
{
    let n0 = t0.len();
    let n1 = t1.len();

    if n0 == 0 || n1 == 0 {
        return 1.0;
    }

    let min_len = n0.min(n1);

    // Create LCS matrix (n0 + 1) x (n1 + 1)
    let mut c = vec![0usize; (n0 + 1) * (n1 + 1)];

    for i in 1..=n0 {
        for j in 1..=n1 {
            let p0 = t0.get(i - 1);
            let p1 = t1.get(j - 1);

            let dist = match dist_type {
                DistanceType::Euclidean => euclidean_distance(&p0, &p1),
                DistanceType::Spherical => great_circle_distance(&p0, &p1),
            };

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
    1.0 - lcss_value / min_len as f64
}

// Keep the old functions for backward compatibility
/// Computes LCSS distance using Euclidean distance calculation
/// 
/// # Deprecated
/// 
/// This function is deprecated. Use `lcss()` with `DistanceType::Euclidean` instead.
#[deprecated(note = "Use lcss() with DistanceType instead")]
pub fn lcss_euclidean<T: CoordSequence>(t0: &T, t1: &T, eps: f64) -> f64
where
    T::Coord: AsCoord,
{
    lcss(t0, t1, eps, DistanceType::Euclidean)
}

/// Computes LCSS distance using Spherical distance calculation
/// 
/// # Deprecated
/// 
/// This function is deprecated. Use `lcss()` with `DistanceType::Spherical` instead.
#[deprecated(note = "Use lcss() with DistanceType instead")]
pub fn lcss_spherical<T: CoordSequence>(t0: &T, t1: &T, eps: f64) -> f64
where
    T::Coord: AsCoord,
{
    lcss(t0, t1, eps, DistanceType::Spherical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcss_euclidean_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let dist = lcss(&t0, &t1, 0.5, DistanceType::Euclidean);
        println!("LCSS Euclidean distance: {}", dist);
        assert!(dist >= 0.0 && dist <= 1.0);
    }

    #[test]
    fn test_lcss_euclidean_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let dist = lcss(&t0, &t1, 1e-6, DistanceType::Euclidean);
        println!(
            "LCSS Euclidean distance for identical trajectories: {}",
            dist
        );
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_lcss_euclidean_eps() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        // Small eps should match identical points
        let dist_small = lcss(&t0, &t1, 1e-6, DistanceType::Euclidean);
        assert!(dist_small < 1e-6);

        // Large eps should match all points
        let dist_large = lcss(&t0, &t1, 1000.0, DistanceType::Euclidean);
        assert!(dist_large < 1e-6);
    }

    #[test]
    fn test_lcss_spherical_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let dist = lcss(&t0, &t1, 100000.0, DistanceType::Spherical);
        println!("LCSS Spherical distance: {}", dist);
        assert!(dist >= 0.0 && dist <= 1.0);
    }

    #[test]
    fn test_lcss_spherical_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let dist = lcss(&t0, &t1, 1e-6, DistanceType::Spherical);
        println!(
            "LCSS Spherical distance for identical trajectories: {}",
            dist
        );
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_lcss_empty_trajectory() {
        let t0: Vec<[f64; 2]> = vec![];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let dist = lcss(&t0, &t1, 1.0, DistanceType::Euclidean);
        assert_eq!(dist, 1.0);

        let dist = lcss(&t0, &t1, 1.0, DistanceType::Spherical);
        assert_eq!(dist, 1.0);
    }

    #[test]
    fn test_lcss_with_both_distance_types() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let euclidean_dist = lcss(&t0, &t1, 100.0, DistanceType::Euclidean);
        let spherical_dist = lcss(&t0, &t1, 100000.0, DistanceType::Spherical);

        assert!(euclidean_dist >= 0.0 && euclidean_dist <= 1.0);
        assert!(spherical_dist >= 0.0 && spherical_dist <= 1.0);
    }
}
