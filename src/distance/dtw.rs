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
//! The time complexity is O(n*m) and space complexity is O(n*m) where n and m are the lengths of the two trajectories.

use crate::distance::distance_type::DistanceType;
use crate::distance::euclidean::euclidean_distance;
use crate::distance::spherical::great_circle_distance;
use crate::traits::{AsCoord, CoordSequence};

/// DTW (Dynamic Time Warping) distance between two trajectories using specified distance type
///
/// Computes the Dynamic Time Warping distance between two trajectories using the specified
/// distance type (Euclidean or Spherical). DTW finds the optimal alignment between two
/// sequences by dynamically warping the time axis, allowing for similar patterns that are out of phase.
///
/// # Arguments
///
/// * `t0` - The first trajectory to compare
/// * `t1` - The second trajectory to compare
/// * `dist_type` - The type of distance to use (Euclidean or Spherical)
///
/// # Type Parameters
///
/// * `T` - A type that implements the `CoordSequence` trait
///
/// # Returns
///
/// Returns the DTW distance between the two trajectories, or `f64::MAX` if either trajectory is empty
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::dtw::dtw;
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];
///
/// let distance = dtw(&traj1, &traj2, DistanceType::Euclidean);
/// println!("DTW distance: {}", distance);
/// ```
pub fn dtw<T: CoordSequence>(t0: &T, t1: &T, dist_type: DistanceType) -> f64
where
    T::Coord: AsCoord,
{
    let n0 = t0.len();
    let n1 = t1.len();

    if n0 == 0 || n1 == 0 {
        return f64::MAX;
    }

    // Create cost matrix (n0 + 1) x (n1 + 1)
    let mut c = vec![f64::INFINITY; (n0 + 1) * (n1 + 1)];
    c[0] = 0.0;

    for i in 1..=n0 {
        for j in 1..=n1 {
            let p0 = t0.get(i - 1);
            let p1 = t1.get(j - 1);
            let dist = match dist_type {
                DistanceType::Euclidean => euclidean_distance(&p0, &p1),
                DistanceType::Spherical => great_circle_distance(&p0, &p1),
            };

            let min_prev = c[(i - 1) * (n1 + 1) + (j - 1)]
                .min(c[(i - 1) * (n1 + 1) + j])
                .min(c[i * (n1 + 1) + (j - 1)]);

            c[i * (n1 + 1) + j] = dist + min_prev;
        }
    }

    c[n0 * (n1 + 1) + n1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtw_euclidean_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let dist = dtw(&t0, &t1, DistanceType::Euclidean);
        println!("DTW Euclidean distance: {}", dist);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_dtw_euclidean_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let dist = dtw(&t0, &t1, DistanceType::Euclidean);
        println!(
            "DTW Euclidean distance for identical trajectories: {}",
            dist
        );
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_dtw_spherical_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let dist = dtw(&t0, &t1, DistanceType::Spherical);
        println!("DTW Spherical distance: {}", dist);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_dtw_with_both_distance_types() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let euclidean_dist = dtw(&t0, &t1, DistanceType::Euclidean);
        let spherical_dist = dtw(&t0, &t1, DistanceType::Spherical);

        assert!(euclidean_dist > 0.0);
        assert!(spherical_dist > 0.0);
    }
}
