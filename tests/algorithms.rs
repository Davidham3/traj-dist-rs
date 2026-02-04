//! Integration tests for all algorithms
//!
//! Tests correctness of all distance algorithms

use traj_dist_rs::distance::base::TrajectoryCalculator;
use traj_dist_rs::distance::discret_frechet::discret_frechet;
use traj_dist_rs::distance::distance_type::DistanceType;
use traj_dist_rs::distance::dtw::dtw;
use traj_dist_rs::distance::edr::edr;
use traj_dist_rs::distance::erp::erp_compat_traj_dist;
use traj_dist_rs::distance::erp::erp_standard;
use traj_dist_rs::distance::hausdorff::hausdorff;
use traj_dist_rs::distance::lcss::lcss;
use traj_dist_rs::distance::sspd::sspd;

mod common;
use common::{assert_valid_distance, assert_valid_dp_result};

/// Test SSPD algorithm
#[test]
fn test_sspd_euclidean() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);
    assert!(
        dist > 0.0,
        "SSPD should give positive distance for different trajectories"
    );
}

/// Test SSPD with spherical distance
#[test]
fn test_sspd_spherical() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let dist = sspd(&traj1, &traj2, DistanceType::Spherical);
    assert_valid_distance(dist);
    assert!(
        dist > 0.0,
        "SSPD should give positive distance for different trajectories"
    );
}

/// Test SSPD symmetry property
#[test]
fn test_sspd_symmetry() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let dist12 = sspd(&traj1, &traj2, DistanceType::Euclidean);
    let dist21 = sspd(&traj2, &traj1, DistanceType::Euclidean);

    // SSPD should be symmetric
    assert!(
        (dist12 - dist21).abs() < 1e-10,
        "SSPD should be symmetric: {} vs {}",
        dist12,
        dist21
    );
}

/// Test DTW algorithm
#[test]
fn test_dtw_euclidean() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);

    assert_valid_dp_result(&result);
    assert!(
        result.distance > 0.0,
        "DTW should give positive distance for different trajectories"
    );
}

/// Test DTW with spherical distance
#[test]
fn test_dtw_spherical() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Spherical);
    let result = dtw(&calculator, false);

    assert_valid_dp_result(&result);
    assert!(
        result.distance > 0.0,
        "DTW should give positive distance for different trajectories"
    );
}

/// Test DTW symmetry property
#[test]
fn test_dtw_symmetry() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let calc1 = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result1 = dtw(&calc1, false);

    let calc2 = TrajectoryCalculator::new(&traj2, &traj1, DistanceType::Euclidean);
    let result2 = dtw(&calc2, false);

    // DTW should be symmetric
    assert!(
        (result1.distance - result2.distance).abs() < 1e-10,
        "DTW should be symmetric: {} vs {}",
        result1.distance,
        result2.distance
    );
}

/// Test Discret Frechet algorithm
#[test]
fn test_discret_frechet_euclidean() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = discret_frechet(&calculator, false);

    assert_valid_dp_result(&result);
    assert!(
        result.distance > 0.0,
        "Discret Frechet should give positive distance for different trajectories"
    );
}

/// Test Discret Frechet symmetry property
#[test]
fn test_discret_frechet_symmetry() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let calc1 = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result1 = discret_frechet(&calc1, false);

    let calc2 = TrajectoryCalculator::new(&traj2, &traj1, DistanceType::Euclidean);
    let result2 = discret_frechet(&calc2, false);

    // Discret Frechet should be symmetric
    assert!(
        (result1.distance - result2.distance).abs() < 1e-10,
        "Discret Frechet should be symmetric: {} vs {}",
        result1.distance,
        result2.distance
    );
}

/// Test Hausdorff algorithm
#[test]
fn test_hausdorff_euclidean() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);

    assert_valid_distance(dist);
    assert!(
        dist > 0.0,
        "Hausdorff should give positive distance for different trajectories"
    );
}

/// Test Hausdorff with spherical distance
#[test]
fn test_hausdorff_spherical() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let dist = hausdorff(&traj1, &traj2, DistanceType::Spherical);

    assert_valid_distance(dist);
    assert!(
        dist > 0.0,
        "Hausdorff should give positive distance for different trajectories"
    );
}

/// Test Hausdorff symmetry property
#[test]
fn test_hausdorff_symmetry() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let dist12 = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    let dist21 = hausdorff(&traj2, &traj1, DistanceType::Euclidean);

    // Hausdorff should be symmetric
    assert!(
        (dist12 - dist21).abs() < 1e-10,
        "Hausdorff should be symmetric: {} vs {}",
        dist12,
        dist21
    );
}

/// Test LCSS algorithm
#[test]
fn test_lcss_euclidean() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]];

    let eps = 0.2;
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = lcss(&calculator, eps, false);

    assert_valid_dp_result(&result);
    // With small epsilon, distance should be positive
    assert!(
        result.distance >= 0.0 && result.distance <= 1.0,
        "LCSS distance should be in [0, 1]"
    );
}

/// Test LCSS with spherical distance
#[test]
fn test_lcss_spherical() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.01], [1.0, 1.01]];

    let eps = 0.1;
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Spherical);
    let result = lcss(&calculator, eps, false);

    assert_valid_dp_result(&result);
    assert!(
        result.distance >= 0.0 && result.distance <= 1.0,
        "LCSS distance should be in [0, 1]"
    );
}

/// Test LCSS symmetry property
#[test]
fn test_lcss_symmetry() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let eps = 0.1;
    let calc1 = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result1 = lcss(&calc1, eps, false);

    let calc2 = TrajectoryCalculator::new(&traj2, &traj1, DistanceType::Euclidean);
    let result2 = lcss(&calc2, eps, false);

    // LCSS should be symmetric
    assert!(
        (result1.distance - result2.distance).abs() < 1e-10,
        "LCSS should be symmetric: {} vs {}",
        result1.distance,
        result2.distance
    );
}

/// Test EDR algorithm
#[test]
fn test_edr_euclidean() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]];

    let eps = 0.2;
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = edr(&calculator, eps, false);

    assert_valid_dp_result(&result);
    // With small epsilon, distance should be small
    assert!(
        result.distance >= 0.0,
        "EDR distance should be non-negative"
    );
}

/// Test EDR with spherical distance
#[test]
fn test_edr_spherical() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.01], [1.0, 1.01]];

    let eps = 0.1;
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Spherical);
    let result = edr(&calculator, eps, false);

    assert_valid_dp_result(&result);
    assert!(
        result.distance >= 0.0,
        "EDR distance should be non-negative"
    );
}

/// Test EDR symmetry property
#[test]
fn test_edr_symmetry() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let eps = 0.1;
    let calc1 = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result1 = edr(&calc1, eps, false);

    let calc2 = TrajectoryCalculator::new(&traj2, &traj1, DistanceType::Euclidean);
    let result2 = edr(&calc2, eps, false);

    // EDR should be symmetric
    assert!(
        (result1.distance - result2.distance).abs() < 1e-10,
        "EDR should be symmetric: {} vs {}",
        result1.distance,
        result2.distance
    );
}

/// Test ERP standard algorithm
#[test]
fn test_erp_standard_euclidean() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    let g = [0.0, 0.0];
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = erp_standard(&calculator, &g, false);

    assert_valid_dp_result(&result);
    assert!(
        result.distance > 0.0,
        "ERP should give positive distance for different trajectories"
    );
}

/// Test ERP with spherical distance
#[test]
fn test_erp_standard_spherical() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let g = [0.0, 0.0];
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Spherical);
    let result = erp_standard(&calculator, &g, false);

    assert_valid_dp_result(&result);
    assert!(
        result.distance > 0.0,
        "ERP should give positive distance for different trajectories"
    );
}

/// Test ERP compat_traj_dist algorithm
#[test]
fn test_erp_compat_traj_dist() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    let g = [0.0, 0.0];
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = erp_compat_traj_dist(&calculator, &g, false);

    assert_valid_dp_result(&result);
    assert!(
        result.distance > 0.0,
        "ERP compat should give positive distance for different trajectories"
    );
}

/// Test algorithm monotonicity (identical trajectories should have smallest distance)
#[test]
fn test_algorithm_monotonicity() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj3: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    // DTW
    let calc_12 = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let dist_12 = dtw(&calc_12, false).distance;

    let calc_13 = TrajectoryCalculator::new(&traj1, &traj3, DistanceType::Euclidean);
    let dist_13 = dtw(&calc_13, false).distance;

    assert!(
        dist_12 < dist_13,
        "Identical trajectories should have smaller distance"
    );

    // Hausdorff
    let dist_12 = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    let dist_13 = hausdorff(&traj1, &traj3, DistanceType::Euclidean);

    assert!(
        dist_12 < dist_13,
        "Identical trajectories should have smaller distance"
    );
}

/// Test all algorithms with same trajectory pair
#[test]
fn test_all_algorithms_same_pair() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    // SSPD
    let dist_sspd = sspd(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist_sspd);

    // DTW
    let calc = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let dist_dtw = dtw(&calc, false).distance;
    assert_valid_distance(dist_dtw);

    // Discret Frechet
    let dist_frechet = discret_frechet(&calc, false).distance;
    assert_valid_distance(dist_frechet);

    // Hausdorff
    let dist_hausdorff = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist_hausdorff);

    // LCSS
    let dist_lcss = lcss(&calc, 0.1, false).distance;
    assert!(dist_lcss >= 0.0 && dist_lcss <= 1.0);

    // EDR
    let dist_edr = edr(&calc, 0.1, false).distance;
    assert_valid_distance(dist_edr);

    // ERP
    let dist_erp = erp_standard(&calc, &[0.0, 0.0], false).distance;
    assert_valid_distance(dist_erp);
}

/// Test algorithms with increasing trajectory lengths
#[test]
fn test_algorithms_increasing_length() {
    let base_traj: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];

    // Create trajectories with increasing lengths
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let traj3: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];

    // SSPD
    let dist_2 = sspd(&base_traj, &traj2, DistanceType::Euclidean);
    let dist_3 = sspd(&base_traj, &traj3, DistanceType::Euclidean);
    assert_valid_distance(dist_2);
    assert_valid_distance(dist_3);

    // DTW
    let calc_2 = TrajectoryCalculator::new(&base_traj, &traj2, DistanceType::Euclidean);
    let dtw_2 = dtw(&calc_2, false).distance;

    let calc_3 = TrajectoryCalculator::new(&base_traj, &traj3, DistanceType::Euclidean);
    let dtw_3 = dtw(&calc_3, false).distance;

    assert_valid_distance(dtw_2);
    assert_valid_distance(dtw_3);

    // Hausdorff
    let hausdorff_2 = hausdorff(&base_traj, &traj2, DistanceType::Euclidean);
    let hausdorff_3 = hausdorff(&base_traj, &traj3, DistanceType::Euclidean);

    assert_valid_distance(hausdorff_2);
    assert_valid_distance(hausdorff_3);
}

/// Test algorithms with very close trajectories
#[test]
fn test_algorithms_very_close_trajectories() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.001, 0.001], [1.001, 1.001], [2.001, 2.001]];

    // SSPD
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    assert!(
        dist < 0.01,
        "Very close trajectories should have small distance"
    );

    // DTW
    let calc = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calc, false);
    assert!(
        result.distance < 0.01,
        "Very close trajectories should have small distance"
    );

    // Hausdorff
    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    assert!(
        dist < 0.01,
        "Very close trajectories should have small distance"
    );
}

/// Test algorithms with orthogonal trajectories
#[test]
fn test_algorithms_orthogonal_trajectories() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]; // Horizontal
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]; // Vertical

    // SSPD
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);

    // DTW
    let calc = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calc, false);
    assert_valid_dp_result(&result);

    // Hausdorff
    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);
}
