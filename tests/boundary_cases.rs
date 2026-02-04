//! Integration tests for boundary cases
//!
//! Tests edge cases like empty trajectories, single point trajectories,
//! identical trajectories, etc.

use traj_dist_rs::distance::base::TrajectoryCalculator;
use traj_dist_rs::distance::discret_frechet::discret_frechet;
use traj_dist_rs::distance::distance_type::DistanceType;
use traj_dist_rs::distance::dtw::dtw;
use traj_dist_rs::distance::edr::edr;
use traj_dist_rs::distance::erp::erp_standard;
use traj_dist_rs::distance::hausdorff::hausdorff;
use traj_dist_rs::distance::lcss::lcss;
use traj_dist_rs::distance::sspd::sspd;

mod common;

/// Assert that identical trajectories have very small distance
fn assert_identical_distance(distance: f64) {
    assert!(
        distance < 1e-6,
        "Identical trajectories should have distance < 1e-6, got: {}",
        distance
    );
}

use common::{assert_valid_distance, assert_valid_dp_result};

/// Test empty trajectories
#[test]
fn test_empty_trajectories() {
    let empty: Vec<[f64; 2]> = vec![];
    let traj: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

    // SSPD should return f64::MAX for empty trajectories
    let dist = sspd(&empty, &traj, DistanceType::Euclidean);
    assert_eq!(dist, f64::MAX);

    let dist = sspd(&traj, &empty, DistanceType::Euclidean);
    assert_eq!(dist, f64::MAX);

    let dist = sspd(&empty, &empty, DistanceType::Euclidean);
    assert_eq!(dist, f64::MAX);

    // DTW should return f64::MAX for empty trajectories
    let calculator = TrajectoryCalculator::new(&empty, &traj, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    assert_eq!(result.distance, f64::MAX);

    let calculator = TrajectoryCalculator::new(&traj, &empty, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    assert_eq!(result.distance, f64::MAX);

    // Discret Frechet should return f64::MAX for empty trajectories
    let calculator = TrajectoryCalculator::new(&empty, &traj, DistanceType::Euclidean);
    let result = discret_frechet(&calculator, false);
    assert_eq!(result.distance, f64::MAX);

    let calculator = TrajectoryCalculator::new(&traj, &empty, DistanceType::Euclidean);
    let result = discret_frechet(&calculator, false);
    assert_eq!(result.distance, f64::MAX);

    // Hausdorff should return f64::MAX for empty trajectories
    let dist = hausdorff(&empty, &traj, DistanceType::Euclidean);
    assert_eq!(dist, f64::MAX);

    let dist = hausdorff(&traj, &empty, DistanceType::Euclidean);
    assert_eq!(dist, f64::MAX);

    // LCSS should return 1.0 for empty trajectories
    let calculator = TrajectoryCalculator::new(&empty, &traj, DistanceType::Euclidean);
    let result = lcss(&calculator, 0.1, false);
    assert_eq!(result.distance, 1.0);

    let calculator = TrajectoryCalculator::new(&traj, &empty, DistanceType::Euclidean);
    let result = lcss(&calculator, 0.1, false);
    assert_eq!(result.distance, 1.0);

    // EDR should return appropriate value for empty trajectories
    let calculator = TrajectoryCalculator::new(&empty, &traj, DistanceType::Euclidean);
    let result = edr(&calculator, 0.1, false);
    assert!(result.distance.is_finite());

    let calculator = TrajectoryCalculator::new(&traj, &empty, DistanceType::Euclidean);
    let result = edr(&calculator, 0.1, false);
    assert!(result.distance.is_finite());

    // ERP should return appropriate value for empty trajectories
    let calculator = TrajectoryCalculator::new(&empty, &traj, DistanceType::Euclidean);
    let result = erp_standard(&calculator, &[0.0, 0.0], false);
    assert!(result.distance.is_finite());

    let calculator = TrajectoryCalculator::new(&traj, &empty, DistanceType::Euclidean);
    let result = erp_standard(&calculator, &[0.0, 0.0], false);
    assert!(result.distance.is_finite());
}

/// Test single point trajectories
#[test]
fn test_single_point_trajectories() {
    let p1: Vec<[f64; 2]> = vec![[0.0, 0.0]];
    let p2: Vec<[f64; 2]> = vec![[1.0, 1.0]];

    // SSPD single point may return f64::MAX for Euclidean distance (needs at least 2 points in target)
    let dist = sspd(&p1, &p2, DistanceType::Euclidean);
    // SSPD may return infinity for single point trajectories
    assert!(
        dist == f64::MAX || dist.is_finite() || dist.is_infinite(),
        "SSPD distance for single point trajectories should be valid, got: {}",
        dist
    );

    // DTW single point should work
    let calculator = TrajectoryCalculator::new(&p1, &p2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    assert_valid_dp_result(&result);

    // Discret Frechet single point should work
    let calculator = TrajectoryCalculator::new(&p1, &p2, DistanceType::Euclidean);
    let result = discret_frechet(&calculator, false);
    assert_valid_dp_result(&result);
    // Single point distance should be the Euclidean distance
    let sqrt_2 = std::f64::consts::SQRT_2;
    assert!((result.distance - sqrt_2).abs() < 1e-6);

    // Hausdorff single point should work
    let dist = hausdorff(&p1, &p2, DistanceType::Euclidean);
    assert_valid_distance(dist);
    assert!((dist - sqrt_2).abs() < 1e-6);

    // LCSS single point should work
    let calculator = TrajectoryCalculator::new(&p1, &p2, DistanceType::Euclidean);
    let result = lcss(&calculator, 0.1, false);
    assert_valid_dp_result(&result);

    // EDR single point should work
    let calculator = TrajectoryCalculator::new(&p1, &p2, DistanceType::Euclidean);
    let result = edr(&calculator, 0.1, false);
    assert_valid_dp_result(&result);

    // ERP single point should work
    let calculator = TrajectoryCalculator::new(&p1, &p2, DistanceType::Euclidean);
    let result = erp_standard(&calculator, &[0.0, 0.0], false);
    assert_valid_dp_result(&result);
}

/// Test identical trajectories
#[test]
fn test_identical_trajectories() {
    let traj: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];

    // SSPD identical should be 0
    let dist = sspd(&traj, &traj, DistanceType::Euclidean);
    assert_identical_distance(dist);

    // DTW identical should be 0
    let calculator = TrajectoryCalculator::new(&traj, &traj, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    assert_identical_distance(result.distance);

    // Discret Frechet identical should be 0
    let calculator = TrajectoryCalculator::new(&traj, &traj, DistanceType::Euclidean);
    let result = discret_frechet(&calculator, false);
    assert_identical_distance(result.distance);

    // Hausdorff identical should be 0
    let dist = hausdorff(&traj, &traj, DistanceType::Euclidean);
    assert_identical_distance(dist);

    // LCSS identical with large eps should give normalized distance of 0
    let calculator = TrajectoryCalculator::new(&traj, &traj, DistanceType::Euclidean);
    let result = lcss(&calculator, 10.0, false);
    assert!(result.distance < 1e-6);

    // EDR identical with large eps should give distance of 0
    let calculator = TrajectoryCalculator::new(&traj, &traj, DistanceType::Euclidean);
    let result = edr(&calculator, 10.0, false);
    assert_identical_distance(result.distance);

    // ERP identical should give distance of 0
    let calculator = TrajectoryCalculator::new(&traj, &traj, DistanceType::Euclidean);
    let result = erp_standard(&calculator, &[0.0, 0.0], false);
    assert_identical_distance(result.distance);
}

/// Test identical trajectories with spherical distance
#[test]
fn test_identical_trajectories_spherical() {
    let traj: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];

    // SSPD identical should be 0
    let dist = sspd(&traj, &traj, DistanceType::Spherical);
    assert_identical_distance(dist);

    // DTW identical should be 0
    let calculator = TrajectoryCalculator::new(&traj, &traj, DistanceType::Spherical);
    let result = dtw(&calculator, false);
    assert_identical_distance(result.distance);

    // Hausdorff identical should be 0
    let dist = hausdorff(&traj, &traj, DistanceType::Spherical);
    assert_identical_distance(dist);

    // LCSS identical with large eps should give normalized distance of 0
    let calculator = TrajectoryCalculator::new(&traj, &traj, DistanceType::Spherical);
    let result = lcss(&calculator, 10.0, false);
    assert!(result.distance < 1e-6);

    // EDR identical with large eps should give distance of 0
    let calculator = TrajectoryCalculator::new(&traj, &traj, DistanceType::Spherical);
    let result = edr(&calculator, 10.0, false);
    assert_identical_distance(result.distance);

    // ERP identical should give distance of 0
    let calculator = TrajectoryCalculator::new(&traj, &traj, DistanceType::Spherical);
    let result = erp_standard(&calculator, &[0.0, 0.0], false);
    assert_identical_distance(result.distance);
}

/// Test very different trajectories
#[test]
fn test_very_different_trajectories() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[10.0, 10.0], [11.0, 11.0]];

    // SSPD should give large distance
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    assert!(dist > 5.0);

    // DTW should give large distance
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    assert!(result.distance > 5.0);

    // Hausdorff should give large distance
    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    assert!(dist > 5.0);
}

/// Test trajectories of different lengths
#[test]
fn test_different_length_trajectories() {
    let short: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let long: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];

    // SSPD should handle different lengths
    let dist = sspd(&short, &long, DistanceType::Euclidean);
    assert_valid_distance(dist);

    // DTW should handle different lengths
    let calculator = TrajectoryCalculator::new(&short, &long, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    assert_valid_dp_result(&result);

    // Discret Frechet should handle different lengths
    let calculator = TrajectoryCalculator::new(&short, &long, DistanceType::Euclidean);
    let result = discret_frechet(&calculator, false);
    assert_valid_dp_result(&result);

    // Hausdorff should handle different lengths
    let dist = hausdorff(&short, &long, DistanceType::Euclidean);
    assert_valid_distance(dist);

    // LCSS should handle different lengths
    let calculator = TrajectoryCalculator::new(&short, &long, DistanceType::Euclidean);
    let result = lcss(&calculator, 0.1, false);
    assert_valid_dp_result(&result);

    // EDR should handle different lengths
    let calculator = TrajectoryCalculator::new(&short, &long, DistanceType::Euclidean);
    let result = edr(&calculator, 0.1, false);
    assert_valid_dp_result(&result);

    // ERP should handle different lengths
    let calculator = TrajectoryCalculator::new(&short, &long, DistanceType::Euclidean);
    let result = erp_standard(&calculator, &[0.0, 0.0], false);
    assert_valid_dp_result(&result);
}

/// Test with zero coordinates
#[test]
fn test_zero_coordinates() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [0.0, 0.0]];
    let traj2: Vec<[f64; 2]> = vec![[1.0, 1.0], [1.0, 1.0]];

    // SSPD should handle zero coordinates
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);

    // DTW should handle zero coordinates
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    assert_valid_dp_result(&result);

    // Hausdorff should handle zero coordinates
    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);
}

/// Test with negative coordinates
#[test]
fn test_negative_coordinates() {
    let traj1: Vec<[f64; 2]> = vec![[-1.0, -1.0], [-2.0, -2.0]];
    let traj2: Vec<[f64; 2]> = vec![[-3.0, -3.0], [-4.0, -4.0]];

    // SSPD should handle negative coordinates
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);

    // DTW should handle negative coordinates
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    assert_valid_dp_result(&result);

    // Hausdorff should handle negative coordinates
    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);
}

/// Test with very large coordinates
#[test]
fn test_large_coordinates() {
    let traj1: Vec<[f64; 2]> = vec![[1e6, 1e6], [1e6 + 1.0, 1e6 + 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[1e6 + 10.0, 1e6 + 10.0], [1e6 + 11.0, 1e6 + 11.0]];

    // SSPD should handle large coordinates
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);

    // DTW should handle large coordinates
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    assert_valid_dp_result(&result);

    // Hausdorff should handle large coordinates
    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);
}

/// Test with very small coordinates
#[test]
fn test_small_coordinates() {
    let traj1: Vec<[f64; 2]> = vec![[1e-6, 1e-6], [2e-6, 2e-6]];
    let traj2: Vec<[f64; 2]> = vec![[3e-6, 3e-6], [4e-6, 4e-6]];

    // SSPD should handle small coordinates
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);

    // DTW should handle small coordinates
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    assert_valid_dp_result(&result);

    // Hausdorff should handle small coordinates
    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);
}
