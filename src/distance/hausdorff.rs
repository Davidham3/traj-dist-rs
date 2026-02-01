//! # Hausdorff Distance Algorithm
//!
//! This module implements the Hausdorff distance algorithm for comparing trajectories.
//! The Hausdorff distance measures how far two subsets of a metric space are from each other.
//! It is defined as the maximum of all distances from a point in one set to the closest point in the other set.
//!
//! ## Algorithm Description
//!
//! The Hausdorff distance between two sets A and B is calculated as:
//! H(A, B) = max(h(A, B), h(B, A))
//! where h(A, B) = max(min(d(a, b))) for all a in A, b in B
//!
//! ## Complexity
//!
//! The time complexity is O(n*m) where n and m are the lengths of the two trajectories.

use crate::distance::distance_type::DistanceType;
use crate::distance::euclidean::{euclidean_distance, point_to_trajectory};
use crate::distance::spherical::{great_circle_distance, point_to_path};
use crate::traits::{AsCoord, CoordSequence};

/// Directed Hausdorff distance from trajectory t1 to trajectory t2 using specified distance type
///
/// This is a helper function that computes the one-way Hausdorff distance from trajectory t1 to t2.
/// It is used internally by the main `hausdorff` function to compute the symmetric distance.
///
/// # Arguments
///
/// * `t1` - The source trajectory
/// * `t2` - The target trajectory
/// * `mdist` - Precomputed pairwise distances between points in t1 and t2
/// * `t2_dist` - Precomputed distances between consecutive points in t2
/// * `dist_type` - The type of distance to use (Euclidean or Spherical)
///
/// # Type Parameters
///
/// * `T` - A type that implements the `CoordSequence` trait
///
/// # Returns
///
/// Returns the directed Hausdorff distance from t1 to t2
fn directed_hausdorff<T: CoordSequence>(
    t1: &T,
    t2: &T,
    mdist: &[f64],
    t2_dist: &[f64],
    dist_type: DistanceType,
) -> f64
where
    T::Coord: AsCoord,
{
    let l_t1 = t1.len();
    let l_t2 = t2.len();

    let mut dh: f64 = 0.0;
    for i1 in 0..l_t1 {
        let point = t1.get(i1);
        // Extract distances from point i1 to all points in t2
        let mdist_i1 = &mdist[i1 * l_t2..(i1 + 1) * l_t2];

        let dist = match dist_type {
            DistanceType::Euclidean => point_to_trajectory(&point, t2, mdist_i1, t2_dist, l_t2),
            DistanceType::Spherical => {
                // For spherical, we calculate point-to-path distances for each point in t1 to segments in t2
                let mut min_dist = f64::MAX;
                for j in 0..(l_t2 - 1) {
                    let p0 = t2.get(j);
                    let p1 = t2.get(j + 1);
                    let d_ij = mdist_i1[j];
                    let d_i1j = mdist_i1[j + 1];
                    let seg_len = t2_dist[j];
                    let dist = point_to_path(&p0, &p1, &point, d_ij, d_i1j, seg_len);
                    min_dist = min_dist.min(dist);
                }
                min_dist
            }
        };
        dh = dh.max(dist);
    }
    dh
}

/// Directed Hausdorff distance for spherical geometry from t1 to t2
///
/// This is a specialized helper function for computing the directed Hausdorff distance
/// when using spherical distance calculations.
///
/// # Arguments
///
/// * `t1` - The source trajectory
/// * `t2` - The target trajectory
/// * `mdist` - Precomputed pairwise distances between points in t1 and t2
/// * `t1_dist` - Precomputed distances between consecutive points in t1
///
/// # Type Parameters
///
/// * `T` - A type that implements the `CoordSequence` trait
///
/// # Returns
///
/// Returns the directed Hausdorff distance from t1 to t2 for spherical geometry
fn directed_hausdorff_spherical<T: CoordSequence>(
    t1: &T,
    t2: &T,
    mdist: &[f64],
    t1_dist: &[f64],
) -> f64
where
    T::Coord: AsCoord,
{
    let n0 = t1.len();
    let n1 = t2.len();

    let mut dh: f64 = 0.0;
    for j in 0..n1 {
        let mut dist_j0 = f64::MAX;
        let point_j = t2.get(j);

        for i in 0..(n0 - 1) {
            let point_i = t1.get(i);
            let point_i1 = t1.get(i + 1);

            let d_ij = mdist[i * n1 + j];
            let d_i1j = mdist[(i + 1) * n1 + j];
            let seg_len = t1_dist[i];

            let dist = point_to_path(&point_i, &point_i1, &point_j, d_ij, d_i1j, seg_len);
            dist_j0 = dist_j0.min(dist);
        }
        dh = dh.max(dist_j0);
    }
    dh
}

/// Hausdorff distance between two trajectories using specified distance type
///
/// Computes the Hausdorff distance between two trajectories using the specified
/// distance type (Euclidean or Spherical). The Hausdorff distance is defined as
/// the maximum of all distances from a point in one trajectory to the closest point in the other trajectory.
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
/// Returns the Hausdorff distance between the two trajectories, or `f64::MAX` if either trajectory is empty
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::hausdorff::hausdorff;
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];
///
/// let distance = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
/// println!("Hausdorff distance: {}", distance);
/// ```
pub fn hausdorff<T: CoordSequence>(t1: &T, t2: &T, dist_type: DistanceType) -> f64
where
    T::Coord: AsCoord,
{
    let l_t1 = t1.len();
    let l_t2 = t2.len();

    if l_t1 == 0 || l_t2 == 0 {
        return f64::MAX;
    }

    // Compute pairwise distances
    let mdist = {
        let mut distances = vec![0.0; l_t1 * l_t2];
        for i in 0..l_t1 {
            let coord1 = t1.get(i);
            for j in 0..l_t2 {
                let coord2 = t2.get(j);
                let dist = match dist_type {
                    DistanceType::Euclidean => euclidean_distance(&coord1, &coord2),
                    DistanceType::Spherical => great_circle_distance(&coord1, &coord2),
                };
                distances[i * l_t2 + j] = dist;
            }
        }
        distances
    };

    // Compute distances between consecutive points
    let t1_dist = {
        let mut distances = Vec::with_capacity(l_t1.saturating_sub(1));
        for i in 0..(l_t1 - 1) {
            let dist = match dist_type {
                DistanceType::Euclidean => euclidean_distance(&t1.get(i), &t1.get(i + 1)),
                DistanceType::Spherical => great_circle_distance(&t1.get(i), &t1.get(i + 1)),
            };
            distances.push(dist);
        }
        distances
    };

    let t2_dist = {
        let mut distances = Vec::with_capacity(l_t2.saturating_sub(1));
        for i in 0..(l_t2 - 1) {
            let dist = match dist_type {
                DistanceType::Euclidean => euclidean_distance(&t2.get(i), &t2.get(i + 1)),
                DistanceType::Spherical => great_circle_distance(&t2.get(i), &t2.get(i + 1)),
            };
            distances.push(dist);
        }
        distances
    };

    let dh1 = match dist_type {
        DistanceType::Euclidean => directed_hausdorff(t1, t2, &mdist, &t2_dist, dist_type),
        DistanceType::Spherical => directed_hausdorff_spherical(t1, t2, &mdist, &t1_dist),
    };

    // Transpose the distance matrix for the second direction
    let mdist_transposed = {
        let mut transposed = vec![0.0; l_t1 * l_t2];
        for i in 0..l_t1 {
            for j in 0..l_t2 {
                transposed[j * l_t1 + i] = mdist[i * l_t2 + j];
            }
        }
        transposed
    };

    let dh2 = match dist_type {
        DistanceType::Euclidean => {
            directed_hausdorff(t2, t1, &mdist_transposed, &t1_dist, dist_type)
        }
        DistanceType::Spherical => {
            directed_hausdorff_spherical(t2, t1, &mdist_transposed, &t2_dist)
        }
    };

    dh1.max(dh2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hausdorff_euclidean_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let dist = hausdorff(&t0, &t1, DistanceType::Euclidean);
        println!("Hausdorff Euclidean distance: {}", dist);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_hausdorff_euclidean_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let dist = hausdorff(&t0, &t1, DistanceType::Euclidean);
        println!(
            "Hausdorff Euclidean distance for identical trajectories: {}",
            dist
        );
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_hausdorff_spherical_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let dist = hausdorff(&t0, &t1, DistanceType::Spherical);
        println!("Hausdorff Spherical distance: {}", dist);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_hausdorff_spherical_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

        let dist = hausdorff(&t0, &t1, DistanceType::Spherical);
        println!(
            "Hausdorff Spherical distance for identical trajectories: {}",
            dist
        );
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_hausdorff_with_both_distance_types() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

        let euclidean_dist = hausdorff(&t0, &t1, DistanceType::Euclidean);
        let spherical_dist = hausdorff(&t0, &t1, DistanceType::Spherical);

        assert!(euclidean_dist > 0.0);
        assert!(spherical_dist > 0.0);
    }
}
