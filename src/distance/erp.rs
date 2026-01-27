//! # ERP (Edit distance with Real Penalty) Algorithm
//!
//! This module implements the Edit distance with Real Penalty algorithm for comparing trajectories.
//! ERP is a distance measure that allows for gaps in the matching and uses a reference point `g`
//! to penalize insertions and deletions.
//!
//! ## Algorithm Description
//!
//! ERP computes the minimum cost of transforming one trajectory into another using three operations:
//! - Match: match two points with their distance as cost
//! - Delete: remove a point from trajectory 1 with distance to reference point `g` as cost
//! - Insert: add a point from trajectory 2 with distance to reference point `g` as cost
//!
//! Two implementations are provided:
//! - `erp_standard`: Correct implementation with cumulative initialization
//! - `erp_compat_traj_dist`: Compatible implementation that matches the bug in the original traj-dist library
//!
//! ## Complexity
//!
//! The time complexity is O(n*m) and space complexity is O(n*m) where n and m are the lengths of the two trajectories.

use crate::distance::distance_type::DistanceType;
use crate::distance::euclidean::euclidean_distance;
use crate::distance::spherical::great_circle_distance;
use crate::traits::{AsCoord, CoordSequence};

/// ERP (Edit distance with Real Penalty) distance between two trajectories using specified distance type
///
/// This is the standard ERP implementation that correctly accumulates distances by index.
/// This function uses the correct algorithm where the initialization of the first row and column
/// uses cumulative sums rather than total sums, avoiding the bug present in the original traj-dist library.
///
/// # Arguments
///
/// * `t0` - The first trajectory to compare
/// * `t1` - The second trajectory to compare
/// * `g` - The reference point used for penalty calculations
/// * `dist_type` - The type of distance to use (Euclidean or Spherical)
///
/// # Type Parameters
///
/// * `T` - A type that implements the `CoordSequence` trait with coordinate type [f64; 2]
///
/// # Returns
///
/// Returns the ERP distance between the two trajectories, or `f64::MAX` if either trajectory is empty
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::erp::erp_standard;
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];
/// let reference_point = [0.0, 0.0];
///
/// let distance = erp_standard(&traj1, &traj2, &reference_point, DistanceType::Euclidean);
/// println!("ERP distance (standard): {}", distance);
/// ```
pub fn erp_standard<T: CoordSequence>(t0: &T, t1: &T, g: &T::Coord, dist_type: DistanceType) -> f64
where
    T::Coord: AsCoord,
{
    let n0 = t0.len();
    let n1 = t1.len();

    if n0 == 0 || n1 == 0 {
        return f64::MAX;
    }

    // Create cost matrix (n0 + 1) x (n1 + 1)
    let mut c = vec![0.0; (n0 + 1) * (n1 + 1)];

    // Compute distances from each point to g
    let mut gt0_dist = Vec::with_capacity(n0);
    let mut gt1_dist = Vec::with_capacity(n1);

    for i in 0..n0 {
        let p = t0.get(i);
        let dist = match dist_type {
            DistanceType::Euclidean => euclidean_distance(&p, g),
            DistanceType::Spherical => great_circle_distance(&p, g),
        };
        gt0_dist.push(dist);
    }

    for j in 0..n1 {
        let p = t1.get(j);
        let dist = match dist_type {
            DistanceType::Euclidean => euclidean_distance(&p, g),
            DistanceType::Spherical => great_circle_distance(&p, g),
        };
        gt1_dist.push(dist);
    }

    // Initialize first column and row with cumulative sum (CORRECT implementation)
    let mut sum_t0 = 0.0;
    for i in 1..=n0 {
        sum_t0 += gt0_dist[i - 1];
        c[i * (n1 + 1)] = sum_t0;
    }

    let mut sum_t1 = 0.0;
    #[allow(clippy::needless_range_loop)]
    for j in 1..=n1 {
        sum_t1 += gt1_dist[j - 1];
        c[j] = sum_t1;
    }

    // Fill the cost matrix
    for i in 1..=n0 {
        for j in 1..=n1 {
            let p0 = t0.get(i - 1);
            let p1 = t1.get(j - 1);
            let dist = match dist_type {
                DistanceType::Euclidean => euclidean_distance(&p0, &p1),
                DistanceType::Spherical => great_circle_distance(&p0, &p1),
            };

            let derp0 = c[(i - 1) * (n1 + 1) + j] + gt0_dist[i - 1];
            let derp1 = c[i * (n1 + 1) + (j - 1)] + gt1_dist[j - 1];
            let derp01 = c[(i - 1) * (n1 + 1) + (j - 1)] + dist;

            c[i * (n1 + 1) + j] = derp0.min(derp1).min(derp01);
        }
    }

    c[n0 * (n1 + 1) + n1]
}

/// ERP (Edit distance with Real Penalty) distance between two trajectories using specified distance type
///
/// This is the traj-dist compatible implementation that uses the INCORRECT initialization
/// where all points to g distance sums are used instead of cumulative sums.
/// This matches the bug in traj-dist's Python implementation.
///
/// Use this function when you need to maintain compatibility with the original traj-dist library.
/// For correct ERP calculations, use `erp_standard` instead.
///
/// # Arguments
///
/// * `t0` - The first trajectory to compare
/// * `t1` - The second trajectory to compare
/// * `g` - The reference point used for penalty calculations
/// * `dist_type` - The type of distance to use (Euclidean or Spherical)
///
/// # Type Parameters
///
/// * `T` - A type that implements the `CoordSequence` trait with coordinate type [f64; 2]
///
/// # Returns
///
/// Returns the ERP distance between the two trajectories, or `f64::MAX` if either trajectory is empty
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::erp::erp_compat_traj_dist;
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];
/// let reference_point = [0.0, 0.0];
///
/// let distance = erp_compat_traj_dist(&traj1, &traj2, &reference_point, DistanceType::Euclidean);
/// println!("ERP distance (compat): {}", distance);
/// ```
pub fn erp_compat_traj_dist<T: CoordSequence>(
    t0: &T,
    t1: &T,
    g: &T::Coord,
    dist_type: DistanceType,
) -> f64
where
    T::Coord: AsCoord,
{
    let n0 = t0.len();
    let n1 = t1.len();

    if n0 == 0 || n1 == 0 {
        return f64::MAX;
    }

    // Create cost matrix (n0 + 1) x (n1 + 1)
    let mut c = vec![0.0; (n0 + 1) * (n1 + 1)];

    // Compute distances from each point to g
    let mut gt0_dist = Vec::with_capacity(n0);
    let mut gt1_dist = Vec::with_capacity(n1);

    for i in 0..n0 {
        let p = t0.get(i);
        let dist = match dist_type {
            DistanceType::Euclidean => euclidean_distance(&p, g),
            DistanceType::Spherical => great_circle_distance(&p, g),
        };
        gt0_dist.push(dist);
    }

    for j in 0..n1 {
        let p = t1.get(j);
        let dist = match dist_type {
            DistanceType::Euclidean => euclidean_distance(&p, g),
            DistanceType::Spherical => great_circle_distance(&p, g),
        };
        gt1_dist.push(dist);
    }

    // Compute total sums (INCORRECT implementation - matches traj-dist bug)
    let sum_t0_total: f64 = gt0_dist.iter().sum();
    let sum_t1_total: f64 = gt1_dist.iter().sum();

    // Initialize first column and row with TOTAL sum (INCORRECT - traj-dist bug)
    for i in 1..=n0 {
        c[i * (n1 + 1)] = sum_t0_total;
    }

    #[allow(clippy::needless_range_loop)]
    for j in 1..=n1 {
        c[j] = sum_t1_total;
    }

    // Fill the cost matrix
    for i in 1..=n0 {
        for j in 1..=n1 {
            let p0 = t0.get(i - 1);
            let p1 = t1.get(j - 1);
            let dist = match dist_type {
                DistanceType::Euclidean => euclidean_distance(&p0, &p1),
                DistanceType::Spherical => great_circle_distance(&p0, &p1),
            };

            let derp0 = c[(i - 1) * (n1 + 1) + j] + gt0_dist[i - 1];
            let derp1 = c[i * (n1 + 1) + (j - 1)] + gt1_dist[j - 1];
            let derp01 = c[(i - 1) * (n1 + 1) + (j - 1)] + dist;

            c[i * (n1 + 1) + j] = derp0.min(derp1).min(derp01);
        }
    }

    c[n0 * (n1 + 1) + n1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erp_euclidean_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];
        let g = [0.0, 0.0];

        let dist_compat = erp_compat_traj_dist(&t0, &t1, &g, DistanceType::Euclidean);
        let dist_standard = erp_standard(&t0, &t1, &g, DistanceType::Euclidean);

        println!("ERP Euclidean (compat) distance: {}", dist_compat);
        println!("ERP Euclidean (standard) distance: {}", dist_standard);

        // Note: For some trajectory pairs, the bug may not cause different results
        // The difference is most apparent when trajectories have different lengths
        assert!(dist_compat >= 0.0);
        assert!(dist_standard >= 0.0);
    }

    #[test]
    fn test_erp_euclidean_different_lengths() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0]];
        let g = [0.0, 0.0];

        let dist_compat = erp_compat_traj_dist(&t0, &t1, &g, DistanceType::Euclidean);
        let dist_standard = erp_standard(&t0, &t1, &g, DistanceType::Euclidean);

        println!(
            "ERP Euclidean (compat) distance (different lengths): {}",
            dist_compat
        );
        println!(
            "ERP Euclidean (standard) distance (different lengths): {}",
            dist_standard
        );

        // They should be different due to the bug
        assert!(dist_compat != dist_standard);
    }

    #[test]
    fn test_erp_euclidean_identical() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let g = [0.0, 0.0];

        let dist_compat = erp_compat_traj_dist(&t0, &t1, &g, DistanceType::Euclidean);
        let dist_standard = erp_standard(&t0, &t1, &g, DistanceType::Euclidean);

        println!(
            "ERP Euclidean (compat) distance for identical trajectories: {}",
            dist_compat
        );
        println!(
            "ERP Euclidean (standard) distance for identical trajectories: {}",
            dist_standard
        );

        // For identical trajectories, both should be small
        assert!(dist_compat < 1e-6);
        assert!(dist_standard < 1e-6);
    }

    #[test]
    fn test_erp_spherical_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];
        let g = [0.0, 0.0];

        let dist_compat = erp_compat_traj_dist(&t0, &t1, &g, DistanceType::Spherical);
        let dist_standard = erp_standard(&t0, &t1, &g, DistanceType::Spherical);

        println!("ERP Spherical (compat) distance: {}", dist_compat);
        println!("ERP Spherical (standard) distance: {}", dist_standard);

        // They should be different due to the bug
        assert!(dist_compat != dist_standard);
        assert!(dist_compat > 0.0);
        assert!(dist_standard > 0.0);
    }

    #[test]
    fn test_erp_difference_between_implementations() {
        // Test to verify the difference between the two implementations
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0]];
        let g = [0.0, 0.0];

        let dist_compat_euclidean = erp_compat_traj_dist(&t0, &t1, &g, DistanceType::Euclidean);
        let dist_standard_euclidean = erp_standard(&t0, &t1, &g, DistanceType::Euclidean);
        let dist_compat_spherical = erp_compat_traj_dist(&t0, &t1, &g, DistanceType::Spherical);
        let dist_standard_spherical = erp_standard(&t0, &t1, &g, DistanceType::Spherical);

        println!(
            "ERP Euclidean (compat): {}, (standard): {}",
            dist_compat_euclidean, dist_standard_euclidean
        );
        println!(
            "ERP Spherical (compat): {}, (standard): {}",
            dist_compat_spherical, dist_standard_spherical
        );

        // The compat version (with bug) should give different results
        assert!(dist_compat_euclidean != dist_standard_euclidean);
        assert!(dist_compat_spherical != dist_standard_spherical);
    }
}
