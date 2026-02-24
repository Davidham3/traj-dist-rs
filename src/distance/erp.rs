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
//! - Time complexity: O(n*m) where n and m are the lengths of the two trajectories
//! - Space complexity:
//!   - O(n*m) when using full matrix (use_full_matrix=true)
//!   - O(min(n,m)) when using optimized 2-row matrix (use_full_matrix=false, default)

use crate::distance::base::DistanceCalculator;
use crate::traits::AsCoord;

/// ERP (Edit distance with Real Penalty) distance using a distance calculator
///
/// This is the standard ERP implementation that correctly accumulates distances by index.
/// This function uses the correct algorithm where the initialization of the first row and column
/// uses cumulative sums rather than total sums, avoiding the bug present in the original traj-dist library.
///
/// # Arguments
///
/// * `calculator` - A distance calculator that implements the `DistanceCalculator` trait
/// * `g` - The reference point used for penalty calculations
/// * `use_full_matrix` - If true, use full (n0+1) x (n1+1) DP matrix (for future matrix return support);
///   if false (default), use optimized 2-row matrix to save space
///
/// # Type Parameters
///
/// * `D` - A type that implements the `DistanceCalculator` trait
/// * `C` - A type that implements the `AsCoord` trait (for the reference point g)
///
/// # Returns
///
/// Returns the ERP distance between the two trajectories, or `f64::MAX` if either trajectory is empty
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::erp::erp_standard;
/// use traj_dist_rs::distance::base::{DistanceCalculator, TrajectoryCalculator};
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];
/// let reference_point = [0.0, 0.0];
///
/// let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
/// let distance = erp_standard(&calculator, &reference_point, false);
/// println!("ERP distance (standard): {}", distance);
/// ```
pub fn erp_standard<D: DistanceCalculator, C: AsCoord>(
    calculator: &D,
    g: &C,
    use_full_matrix: bool,
) -> crate::distance::DpResult {
    let n0 = calculator.len_seq1();
    let n1 = calculator.len_seq2();

    if n0 == 0 || n1 == 0 {
        return crate::distance::DpResult::new(f64::MAX);
    }

    // Compute distances from each point to g using the calculator
    let mut gt0_dist = Vec::with_capacity(n0);
    let mut gt1_dist = Vec::with_capacity(n1);

    for i in 0..n0 {
        let dist = calculator.compute_dis_for_extra_point(0, i, Some(g));
        gt0_dist.push(dist);
    }

    for j in 0..n1 {
        let dist = calculator.compute_dis_for_extra_point(1, j, Some(g));
        gt1_dist.push(dist);
    }

    if use_full_matrix {
        // Create cost matrix (n0 + 1) x (n1 + 1)
        let mut c = vec![0.0; (n0 + 1) * (n1 + 1)];

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
                let dist = calculator.dis_between(i - 1, j - 1);

                let derp0 = c[(i - 1) * (n1 + 1) + j] + gt0_dist[i - 1];
                let derp1 = c[i * (n1 + 1) + (j - 1)] + gt1_dist[j - 1];
                let derp01 = c[(i - 1) * (n1 + 1) + (j - 1)] + dist;

                c[i * (n1 + 1) + j] = derp0.min(derp1).min(derp01);
            }
        }

        crate::distance::DpResult::with_matrix(c[n0 * (n1 + 1) + n1], c)
    } else {
        // Optimized version: use only 2 rows
        let mut prev_row = vec![0.0; n1 + 1];
        let mut curr_row = vec![0.0; n1 + 1];

        // Initialize prev_row (row 0) with cumulative sum for the first row
        // prev_row[j] = c[j] = sum_{k=0}^{j-1} gt1_dist[k]
        let mut sum_t1 = 0.0;
        for j in 1..=n1 {
            sum_t1 += gt1_dist[j - 1];
            prev_row[j] = sum_t1;
        }

        // Keep track of cumulative sum for column 0 (sum of gt0_dist[0..i])
        let mut sum_t0 = 0.0;

        for i in 1..=n0 {
            // Set curr_row[0] to cumulative sum of gt0_dist[0..i]
            // This matches the full matrix initialization where c[i*(n1+1)] = sum_{k=0}^{i-1} gt0_dist[k]
            curr_row[0] = sum_t0;

            for j in 1..=n1 {
                let dist = calculator.dis_between(i - 1, j - 1);

                let derp0 = prev_row[j] + gt0_dist[i - 1];
                let derp1 = curr_row[j - 1] + gt1_dist[j - 1];
                let derp01 = prev_row[j - 1] + dist;

                curr_row[j] = derp0.min(derp1).min(derp01);
            }

            // Update cumulative sum for next iteration
            sum_t0 += gt0_dist[i - 1];
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        crate::distance::DpResult::new(prev_row[n1])
    }
}

/// ERP (Edit distance with Real Penalty) distance using a distance calculator (compat version)
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
/// * `calculator` - A distance calculator that implements the `DistanceCalculator` trait
/// * `g` - The reference point used for penalty calculations
/// * `use_full_matrix` - If true, use full (n0+1) x (n1+1) DP matrix (for future matrix return support);
///   if false (default), use optimized 2-row matrix to save space
///
/// # Type Parameters
///
/// * `D` - A type that implements the `DistanceCalculator` trait
/// * `C` - A type that implements the `AsCoord` trait (for the reference point g)
///
/// # Returns
///
/// Returns the ERP distance between the two trajectories, or `f64::MAX` if either trajectory is empty
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::erp::erp_compat_traj_dist;
/// use traj_dist_rs::distance::base::{DistanceCalculator, TrajectoryCalculator};
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
/// let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];
/// let reference_point = [0.0, 0.0];
///
/// let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
/// let distance = erp_compat_traj_dist(&calculator, &reference_point, false);
/// println!("ERP distance (compat): {}", distance);
/// ```
pub fn erp_compat_traj_dist<D: DistanceCalculator, C: AsCoord>(
    calculator: &D,
    g: &C,
    use_full_matrix: bool,
) -> crate::distance::DpResult {
    let n0 = calculator.len_seq1();
    let n1 = calculator.len_seq2();

    if n0 == 0 || n1 == 0 {
        return crate::distance::DpResult::new(f64::MAX);
    }

    // Compute distances from each point to g using the calculator
    let mut gt0_dist = Vec::with_capacity(n0);
    let mut gt1_dist = Vec::with_capacity(n1);

    for i in 0..n0 {
        let dist = calculator.compute_dis_for_extra_point(0, i, Some(g));
        gt0_dist.push(dist);
    }

    for j in 0..n1 {
        let dist = calculator.compute_dis_for_extra_point(1, j, Some(g));
        gt1_dist.push(dist);
    }

    // Compute total sums (INCORRECT implementation - matches traj-dist bug)
    let sum_t0_total: f64 = gt0_dist.iter().sum();
    let sum_t1_total: f64 = gt1_dist.iter().sum();

    if use_full_matrix {
        // Create cost matrix (n0 + 1) x (n1 + 1)
        let mut c = vec![0.0; (n0 + 1) * (n1 + 1)];

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
                let dist = calculator.dis_between(i - 1, j - 1);

                let derp0 = c[(i - 1) * (n1 + 1) + j] + gt0_dist[i - 1];
                let derp1 = c[i * (n1 + 1) + (j - 1)] + gt1_dist[j - 1];
                let derp01 = c[(i - 1) * (n1 + 1) + (j - 1)] + dist;

                c[i * (n1 + 1) + j] = derp0.min(derp1).min(derp01);
            }
        }

        crate::distance::DpResult::with_matrix(c[n0 * (n1 + 1) + n1], c)
    } else {
        // Optimized version: use only 2 rows
        let mut prev_row = vec![0.0; n1 + 1];
        let mut curr_row = vec![0.0; n1 + 1];

        // Initialize prev_row (row 0) with TOTAL sum (INCORRECT - traj-dist bug)
        #[allow(clippy::needless_range_loop)]
        for j in 1..=n1 {
            prev_row[j] = sum_t1_total;
        }

        for i in 1..=n0 {
            // Initialize first column with TOTAL sum (INCORRECT - traj-dist bug)
            curr_row[0] = sum_t0_total;

            for j in 1..=n1 {
                let dist = calculator.dis_between(i - 1, j - 1);

                let derp0 = prev_row[j] + gt0_dist[i - 1];
                let derp1 = curr_row[j - 1] + gt1_dist[j - 1];
                let derp01 = prev_row[j - 1] + dist;

                curr_row[j] = derp0.min(derp1).min(derp01);
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        crate::distance::DpResult::new(prev_row[n1])
    }
}

/// ERP (compat version) using precomputed distance arrays
///
/// This is a convenience function that accepts precomputed distance arrays
/// instead of computing them on the fly.
///
/// # Arguments
///
/// * `calculator` - A distance calculator that implements the `DistanceCalculator` trait
/// * `gt0_dist` - Precomputed distances from points in trajectory 0 to gap point g
/// * `gt1_dist` - Precomputed distances from points in trajectory 1 to gap point g
/// * `use_full_matrix` - If true, use full (n0+1) x (n1+1) DP matrix;
///   if false (default), use optimized 2-row matrix to save space
///
/// # Type Parameters
///
/// * `D` - A type that implements the `DistanceCalculator` trait
///
/// # Returns
///
/// Returns a `DpResult` containing the ERP distance and optionally the full DP matrix.
pub fn erp_compat_traj_dist_with_distances<D: DistanceCalculator>(
    calculator: &D,
    gt0_dist: &[f64],
    gt1_dist: &[f64],
    use_full_matrix: bool,
) -> crate::distance::DpResult {
    let n0 = calculator.len_seq1();
    let n1 = calculator.len_seq2();

    if n0 == 0 || n1 == 0 {
        return crate::distance::DpResult::new(f64::MAX);
    }

    // Compute total sums (INCORRECT implementation - matches traj-dist bug)
    let sum_t0_total: f64 = gt0_dist.iter().sum();
    let sum_t1_total: f64 = gt1_dist.iter().sum();

    if use_full_matrix {
        // Create cost matrix (n0 + 1) x (n1 + 1)
        let mut c = vec![0.0; (n0 + 1) * (n1 + 1)];

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
                let dist = calculator.dis_between(i - 1, j - 1);

                let derp0 = c[(i - 1) * (n1 + 1) + j] + gt0_dist[i - 1];
                let derp1 = c[i * (n1 + 1) + (j - 1)] + gt1_dist[j - 1];
                let derp01 = c[(i - 1) * (n1 + 1) + (j - 1)] + dist;

                c[i * (n1 + 1) + j] = derp0.min(derp1).min(derp01);
            }
        }

        crate::distance::DpResult::with_matrix(c[n0 * (n1 + 1) + n1], c)
    } else {
        // Optimized version: use only 2 rows
        let mut prev_row = vec![0.0; n1 + 1];
        let mut curr_row = vec![0.0; n1 + 1];

        // Initialize prev_row (row 0) with TOTAL sum (INCORRECT - traj-dist bug)
        #[allow(clippy::needless_range_loop)]
        for j in 1..=n1 {
            prev_row[j] = sum_t1_total;
        }

        for i in 1..=n0 {
            // Initialize first column with TOTAL sum (INCORRECT - traj-dist bug)
            curr_row[0] = sum_t0_total;

            for j in 1..=n1 {
                let dist = calculator.dis_between(i - 1, j - 1);

                let derp0 = prev_row[j] + gt0_dist[i - 1];
                let derp1 = curr_row[j - 1] + gt1_dist[j - 1];
                let derp01 = prev_row[j - 1] + dist;

                curr_row[j] = derp0.min(derp1).min(derp01);
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        crate::distance::DpResult::new(prev_row[n1])
    }
}

/// ERP (standard version) using precomputed distance arrays
///
/// This is a convenience function that accepts precomputed distance arrays
/// instead of computing them on the fly.
///
/// # Arguments
///
/// * `calculator` - A distance calculator that implements the `DistanceCalculator` trait
/// * `gt0_dist` - Precomputed distances from points in trajectory 0 to gap point g
/// * `gt1_dist` - Precomputed distances from points in trajectory 1 to gap point g
/// * `use_full_matrix` - If true, use full (n0+1) x (n1+1) DP matrix;
///   if false (default), use optimized 2-row matrix to save space
///
/// # Type Parameters
///
/// * `D` - A type that implements the `DistanceCalculator` trait
///
/// # Returns
///
/// Returns a `DpResult` containing the ERP distance and optionally the full DP matrix.
pub fn erp_standard_with_distances<D: DistanceCalculator>(
    calculator: &D,
    gt0_dist: &[f64],
    gt1_dist: &[f64],
    use_full_matrix: bool,
) -> crate::distance::DpResult {
    let n0 = calculator.len_seq1();
    let n1 = calculator.len_seq2();

    if n0 == 0 || n1 == 0 {
        return crate::distance::DpResult::new(f64::MAX);
    }

    if use_full_matrix {
        // Create cost matrix (n0 + 1) x (n1 + 1)
        let mut c = vec![0.0; (n0 + 1) * (n1 + 1)];

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
                let dist = calculator.dis_between(i - 1, j - 1);

                let derp0 = c[(i - 1) * (n1 + 1) + j] + gt0_dist[i - 1];
                let derp1 = c[i * (n1 + 1) + (j - 1)] + gt1_dist[j - 1];
                let derp01 = c[(i - 1) * (n1 + 1) + (j - 1)] + dist;

                c[i * (n1 + 1) + j] = derp0.min(derp1).min(derp01);
            }
        }

        crate::distance::DpResult::with_matrix(c[n0 * (n1 + 1) + n1], c)
    } else {
        // Optimized version: use only 2 rows
        let mut prev_row = vec![0.0; n1 + 1];
        let mut curr_row = vec![0.0; n1 + 1];

        // Initialize prev_row (row 0) with cumulative sum for the first row
        let mut sum_t1 = 0.0;
        for j in 1..=n1 {
            sum_t1 += gt1_dist[j - 1];
            prev_row[j] = sum_t1;
        }

        // Keep track of cumulative sum for column 0
        let mut sum_t0 = 0.0;

        for i in 1..=n0 {
            // Set curr_row[0] to cumulative sum of gt0_dist[0..i]
            sum_t0 += gt0_dist[i - 1];
            curr_row[0] = sum_t0;

            for j in 1..=n1 {
                let dist = calculator.dis_between(i - 1, j - 1);

                let derp0 = prev_row[j] + gt0_dist[i - 1];
                let derp1 = curr_row[j - 1] + gt1_dist[j - 1];
                let derp01 = prev_row[j - 1] + dist;

                curr_row[j] = derp0.min(derp1).min(derp01);
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        crate::distance::DpResult::new(prev_row[n1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::base::TrajectoryCalculator;
    use crate::distance::distance_type::DistanceType;

    #[test]
    fn test_erp_euclidean_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];
        let g = [0.0, 0.0];

        let calc_compat = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let calc_standard = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);

        let dist_compat = erp_compat_traj_dist(&calc_compat, &g, false);
        let dist_standard = erp_standard(&calc_standard, &g, false);

        println!("ERP Euclidean (compat) distance: {}", dist_compat);
        println!("ERP Euclidean (standard) distance: {}", dist_standard);

        // Note: For some trajectory pairs, the bug may not cause different results
        // The difference is most apparent when trajectories have different lengths
        assert!(dist_compat.distance >= 0.0);
        assert!(dist_standard.distance >= 0.0);
    }

    #[test]
    fn test_erp_euclidean_different_lengths() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0]];
        let g = [0.0, 0.0];

        let calc_compat = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let calc_standard = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);

        let dist_compat = erp_compat_traj_dist(&calc_compat, &g, false);
        let dist_standard = erp_standard(&calc_standard, &g, false);

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

        let calc_compat = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let calc_standard = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);

        let dist_compat = erp_compat_traj_dist(&calc_compat, &g, false);
        let dist_standard = erp_standard(&calc_standard, &g, false);

        println!(
            "ERP Euclidean (compat) distance for identical trajectories: {}",
            dist_compat
        );
        println!(
            "ERP Euclidean (standard) distance for identical trajectories: {}",
            dist_standard
        );

        // For identical trajectories, both should be small
        assert!(dist_compat.distance < 1e-6);
        assert!(dist_standard.distance < 1e-6);
    }

    #[test]
    fn test_erp_spherical_simple() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];
        let g = [0.0, 0.0];

        let calc_compat = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);
        let calc_standard = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);

        let dist_compat = erp_compat_traj_dist(&calc_compat, &g, false);
        let dist_standard = erp_standard(&calc_standard, &g, false);

        println!("ERP Spherical (compat) distance: {}", dist_compat);
        println!("ERP Spherical (standard) distance: {}", dist_standard);

        // They should be different due to the bug
        assert!(dist_compat != dist_standard);
        assert!(dist_compat.distance > 0.0);
        assert!(dist_standard.distance > 0.0);
    }

    #[test]
    fn test_erp_difference_between_implementations() {
        // Test to verify the difference between the two implementations
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0]];
        let g = [0.0, 0.0];

        let calc_compat_euclidean = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let calc_standard_euclidean = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let calc_compat_spherical = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);
        let calc_standard_spherical = TrajectoryCalculator::new(&t0, &t1, DistanceType::Spherical);

        let dist_compat_euclidean = erp_compat_traj_dist(&calc_compat_euclidean, &g, false);
        let dist_standard_euclidean = erp_standard(&calc_standard_euclidean, &g, false);
        let dist_compat_spherical = erp_compat_traj_dist(&calc_compat_spherical, &g, false);
        let dist_standard_spherical = erp_standard(&calc_standard_spherical, &g, false);

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

    #[test]
    fn test_erp_consistency_between_modes_standard() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];
        let g = [0.0, 0.0];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let dist_optimized = erp_standard(&calculator, &g, false);
        let dist_full = erp_standard(&calculator, &g, true);

        // Both modes should produce the same result
        assert!((dist_optimized.distance - dist_full.distance).abs() < 1e-10);
    }

    #[test]
    fn test_erp_consistency_between_modes_compat() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];
        let g = [0.0, 0.0];

        let calculator = TrajectoryCalculator::new(&t0, &t1, DistanceType::Euclidean);
        let dist_optimized = erp_compat_traj_dist(&calculator, &g, false);
        let dist_full = erp_compat_traj_dist(&calculator, &g, true);

        // Both modes should produce the same result
        assert!((dist_optimized.distance - dist_full.distance).abs() < 1e-10);
    }

    #[test]
    fn test_erp_with_precomputed_distances() {
        let t0: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let t1: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];
        let g = [0.0, 0.0];

        let distance_matrix =
            crate::distance::utils::precompute_distance_matrix(&t0, &t1, DistanceType::Euclidean);

        let mut gt0_dist = Vec::with_capacity(t0.len());
        let mut gt1_dist = Vec::with_capacity(t1.len());

        for i in 0..t0.len() {
            let dist = DistanceType::Euclidean.distance(&g, &t0[i]);
            gt0_dist.push(dist);
        }

        for j in 0..t1.len() {
            let dist = DistanceType::Euclidean.distance(&g, &t1[j]);
            gt1_dist.push(dist);
        }

        let calculator = crate::distance::base::PrecomputedDistanceCalculator::with_extra_distances(
            &distance_matrix,
            t0.len(),
            t1.len(),
            Some(&gt0_dist),
            Some(&gt1_dist),
        );

        let result = erp_standard(&calculator, &g, false);

        println!(
            "ERP distance with precomputed distances: {}",
            result.distance
        );

        assert!(result.distance >= 0.0);
    }
}
