//! # SSPD (Symmetric Segment-Path Distance) Algorithm
//!
//! This module implements the Symmetric Segment-Path Distance algorithm for comparing trajectories.
//! SSPD measures the distance between two trajectories by computing the average distance from
//! each point in one trajectory to the other trajectory, and then symmetrizing the result.
//!
//! ## Algorithm Description
//!
//! The SSPD algorithm works as follows:
//! 1. For each point in trajectory A, compute its distance to trajectory B
//! 2. For each point in trajectory B, compute its distance to trajectory A
//! 3. Average these distances to get the symmetric distance
//!
//! For Euclidean distance, this is the average of the two directional distances.
//! For Spherical distance, this is the sum of the two directional distances.
//!
//! ## Complexity
//!
//! The time complexity is O(n*m) where n and m are the lengths of the two trajectories.

use crate::distance::distance_type::DistanceType;
use crate::distance::euclidean::point_to_trajectory_simple;
use crate::distance::spherical::point_to_path_simple;
use crate::traits::{AsCoord, CoordSequence};

/// SPD (Symmetric Path Distance) from t1 to t2 using specified distance type
///
/// This is a helper function that computes the one-way distance from trajectory t1 to t2.
/// It is used internally by the main `sspd` function to compute the symmetric distance.
///
/// # Arguments
///
/// * `t1` - The source trajectory
/// * `t2` - The target trajectory  
/// * `dist_type` - The type of distance to use (Euclidean or Spherical)
///
/// # Type Parameters
///
/// * `T` - A type that implements the `CoordSequence` trait
///
/// # Returns
///
/// Returns the one-way distance from t1 to t2, or `f64::MAX` if the inputs are invalid
fn spd<T: CoordSequence>(t1: &T, t2: &T, dist_type: DistanceType) -> f64
where
    T::Coord: AsCoord,
{
    let l_t1 = t1.len();
    let l_t2 = t2.len();

    if (dist_type == DistanceType::Euclidean && (l_t1 == 0 || l_t2 < 2))
        || (dist_type == DistanceType::Spherical && (l_t1 < 2 || l_t2 == 0))
    {
        return f64::MAX;
    }

    let mut sum = 0.0;

    match dist_type {
        DistanceType::Euclidean => {
            for i in 0..l_t1 {
                let point = t1.get(i);
                sum += point_to_trajectory_simple(t2, &point);
            }
        }
        DistanceType::Spherical => {
            for j in 0..l_t2 {
                let mut dist_j0 = f64::MAX;

                for i in 0..(l_t1 - 1) {
                    let p0 = t1.get(i);
                    let p1 = t1.get(i + 1);
                    let p2 = t2.get(j);
                    let d = point_to_path_simple(&p0, &p1, &p2);
                    dist_j0 = dist_j0.min(d);
                }

                sum += dist_j0;
            }
        }
    }

    sum / if dist_type == DistanceType::Euclidean {
        l_t1
    } else {
        l_t2
    } as f64
}

/// SSPD (Symmetric Segmented Path Distance) between two trajectories using specified distance type
///
/// Computes the symmetric segment-path distance between two trajectories using the specified
/// distance type (Euclidean or Spherical). This is the main function for SSPD computation.
///
/// # Arguments
///
/// * `t1` - The first trajectory to compare
/// * `t2` - The second trajectory to compare
/// * `dist_type` - The type of distance to use (Euclidean or Spherical)
///
/// # Type Parameters
///
/// * `T` - A type that implements the `CoordSequence` trait
///
/// # Returns
///
/// Returns the SSPD distance between the two trajectories, or `f64::MAX` if either trajectory is empty
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::sspd::sspd;
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];
///
/// let distance = sspd(&traj1, &traj2, DistanceType::Euclidean);
/// println!("SSPD distance: {}", distance);
/// ```
pub fn sspd<T: CoordSequence>(t1: &T, t2: &T, dist_type: DistanceType) -> f64
where
    T::Coord: AsCoord,
{
    let l_t1 = t1.len();
    let l_t2 = t2.len();

    if l_t1 == 0 || l_t2 == 0 {
        return f64::MAX;
    }

    match dist_type {
        DistanceType::Euclidean => {
            // Compute symmetric distance
            let spd1 = spd(t1, t2, dist_type);
            let spd2 = spd(t2, t1, dist_type);

            (spd1 + spd2) / 2.0
        }
        DistanceType::Spherical => {
            // Compute symmetric distance
            let dist1 = spd(t1, t2, dist_type);
            let dist2 = spd(t2, t1, dist_type);

            dist1 + dist2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sspd_euclidean_simple() {
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let dist = sspd(&t1, &t2, DistanceType::Euclidean);
        println!("SSPD Euclidean distance: {}", dist);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_sspd_euclidean_identical() {
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t2: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let dist = sspd(&t1, &t2, DistanceType::Euclidean);
        println!(
            "SSPD Euclidean distance for identical trajectories: {}",
            dist
        );
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_sspd_spherical_simple() {
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let dist = sspd(&t1, &t2, DistanceType::Spherical);
        println!("SSPD Spherical distance: {}", dist);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_sspd_with_both_distance_types() {
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let euclidean_dist = sspd(&t1, &t2, DistanceType::Euclidean);
        let spherical_dist = sspd(&t1, &t2, DistanceType::Spherical);

        assert!(euclidean_dist > 0.0);
        assert!(spherical_dist > 0.0);
    }
}
