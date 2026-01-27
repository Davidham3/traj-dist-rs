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
//! The time complexity is O(n*m) and space complexity is O(n*m) where n and m are the lengths of the two trajectories.

use crate::distance::euclidean::euclidean_distance;
use crate::traits::{AsCoord, CoordSequence};

/// Discret Frechet distance between two trajectories using Euclidean geometry
///
/// The discret Frechet distance is a measure of similarity between two curves
/// that takes into account the location and ordering of the points along the curves.
///
/// It is computed using dynamic programming where:
/// - C[i][j] = max(dist(t0[i-1], t1[j-1]), min(C[i][j-1], C[i-1][j-1], C[i-1][j]))
/// - First row and column (except C[0][0]) are initialized to infinity
///
/// # Arguments
///
/// * `t0` - The first trajectory to compare
/// * `t1` - The second trajectory to compare
///
/// # Type Parameters
///
/// * `T` - A type that implements the `CoordSequence` trait
///
/// # Returns
///
/// Returns the Discrete Fréchet distance between the two trajectories,
/// or `f64::MAX` if either trajectory is empty
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::discret_frechet::discret_frechet_euclidean;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 0.1], [1.0, 1.1]];
///
/// let distance = discret_frechet_euclidean(&traj1, &traj2);
/// println!("Discrete Fréchet distance: {}", distance);
/// ```
pub fn discret_frechet_euclidean<T: CoordSequence>(t0: &T, t1: &T) -> f64
where
    T::Coord: AsCoord,
{
    let n0 = t0.len();
    let n1 = t1.len();

    if n0 == 0 || n1 == 0 {
        return f64::MAX;
    }

    let width = n1 + 1;

    // Create cost matrix (n0 + 1) x (n1 + 1)
    let mut c = vec![0.0; (n0 + 1) * width];

    // Initialize first row and column to infinity (except C[0][0])
    for i in 1..=n0 {
        c[i * width] = f64::INFINITY;
    }
    for (_j, val) in c.iter_mut().enumerate().take(n1).skip(1) {
        *val = f64::INFINITY;
    }

    for i in 1..=n0 {
        let row_offset = i * width;
        let prev_row_offset = (i - 1) * width;

        for j in 1..=n1 {
            let p0 = t0.get(i - 1);
            let p1 = t1.get(j - 1);
            let dist = euclidean_distance(&p0, &p1);

            // C[i][j] = max(dist, min(C[i][j-1], C[i-1][j-1], C[i-1][j]))
            let min_prev = c[row_offset + (j - 1)]
                .min(c[prev_row_offset + (j - 1)])
                .min(c[prev_row_offset + j]);

            c[row_offset + j] = dist.max(min_prev);
        }
    }

    c[n0 * width + n1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discret_frechet_euclidean_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let dist = discret_frechet_euclidean(&t0, &t1);
        println!("Discret Frechet Euclidean distance: {}", dist);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_discret_frechet_euclidean_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let dist = discret_frechet_euclidean(&t0, &t1);
        println!(
            "Discret Frechet Euclidean distance for identical trajectories: {}",
            dist
        );
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_discret_frechet_euclidean_empty() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![];

        let dist = discret_frechet_euclidean(&t0, &t1);
        println!(
            "Discret Frechet Euclidean distance for empty trajectory: {}",
            dist
        );
        assert_eq!(dist, f64::MAX);
    }

    #[test]
    fn test_discret_frechet_euclidean_single_point() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0]];
        let t1: Vec<[f64; 2]> = vec![[1.0, 1.0]];

        let dist = discret_frechet_euclidean(&t0, &t1);
        println!(
            "Discret Frechet Euclidean distance for single point: {}",
            dist
        );

        // Single point trajectories: distance should be the Euclidean distance
        let sqrt_2 = std::f64::consts::SQRT_2;
        assert!((dist - sqrt_2).abs() < 1e-6);
    }
}
