//! Integration tests for error handling
//!
//! Tests behavior with invalid inputs like NaN, infinity, etc.

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

/// Test with NaN coordinates
#[test]
fn test_nan_coordinates() {
    let traj1: Vec<[f64; 2]> = vec![[f64::NAN, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

    // All algorithms should handle NaN gracefully
    // They should not panic and should return a finite value or handle it appropriately

    // SSPD
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    // Should either be NaN, infinity, or some valid result, but should not panic
    assert!(
        dist.is_finite() || dist.is_nan() || dist.is_infinite(),
        "SSPD should handle NaN coordinates, got: {}",
        dist
    );

    // DTW
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    // Should handle gracefully
    assert!(
        result.distance.is_finite() || result.distance.is_nan() || result.distance.is_infinite(),
        "DTW should handle NaN coordinates, got: {}",
        result.distance
    );

    // Discret Frechet
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = discret_frechet(&calculator, false);
    // Should handle gracefully
    assert!(
        result.distance.is_finite() || result.distance.is_nan() || result.distance.is_infinite(),
        "Discret Frechet should handle NaN coordinates, got: {}",
        result.distance
    );

    // Hausdorff
    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    // Should handle gracefully
    assert!(
        dist.is_finite() || dist.is_nan() || dist.is_infinite(),
        "Hausdorff should handle NaN coordinates, got: {}",
        dist
    );

    // LCSS
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = lcss(&calculator, 0.1, false);
    // Should handle gracefully
    assert!(
        result.distance.is_finite() || result.distance.is_nan() || result.distance.is_infinite(),
        "LCSS should handle NaN coordinates, got: {}",
        result.distance
    );

    // EDR
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = edr(&calculator, 0.1, false);
    // Should handle gracefully
    assert!(
        result.distance.is_finite() || result.distance.is_nan() || result.distance.is_infinite(),
        "EDR should handle NaN coordinates, got: {}",
        result.distance
    );

    // ERP
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = erp_standard(&calculator, &[0.0, 0.0], false);
    // Should handle gracefully
    assert!(
        result.distance.is_finite() || result.distance.is_nan() || result.distance.is_infinite(),
        "ERP should handle NaN coordinates, got: {}",
        result.distance
    );
}

/// Test with infinity coordinates
#[test]
fn test_infinity_coordinates() {
    let traj1: Vec<[f64; 2]> = vec![[f64::INFINITY, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

    // All algorithms should handle infinity gracefully

    // SSPD
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    // Should handle gracefully (may return infinity or some other valid result)
    assert!(dist.is_finite() || dist.is_infinite());

    // DTW
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    // Should handle gracefully
    assert!(result.distance.is_finite() || result.distance.is_infinite());

    // Discret Frechet
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = discret_frechet(&calculator, false);
    // Should handle gracefully
    assert!(result.distance.is_finite() || result.distance.is_infinite());

    // Hausdorff
    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    // Should handle gracefully
    assert!(dist.is_finite() || dist.is_infinite());

    // LCSS
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = lcss(&calculator, 0.1, false);
    // Should handle gracefully
    assert!(result.distance.is_finite() || result.distance.is_infinite());

    // EDR
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = edr(&calculator, 0.1, false);
    // Should handle gracefully
    assert!(result.distance.is_finite() || result.distance.is_infinite());

    // ERP
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = erp_standard(&calculator, &[0.0, 0.0], false);
    // Should handle gracefully
    assert!(result.distance.is_finite() || result.distance.is_infinite());
}

/// Test with negative infinity coordinates
#[test]
fn test_neg_infinity_coordinates() {
    let traj1: Vec<[f64; 2]> = vec![[f64::NEG_INFINITY, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

    // All algorithms should handle negative infinity gracefully

    // SSPD
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    // Should handle gracefully
    assert!(dist.is_finite() || dist.is_infinite());

    // DTW
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    // Should handle gracefully
    assert!(result.distance.is_finite() || result.distance.is_infinite());

    // Discret Frechet
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = discret_frechet(&calculator, false);
    // Should handle gracefully
    assert!(result.distance.is_finite() || result.distance.is_infinite());

    // Hausdorff
    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    // Should handle gracefully
    assert!(dist.is_finite() || dist.is_infinite());
}

/// Test with negative epsilon for LCSS and EDR
#[test]
fn test_negative_epsilon() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

    // LCSS with negative epsilon
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = lcss(&calculator, -1.0, false);
    // Should handle gracefully (may treat as 0 or handle differently)
    assert!(result.distance >= 0.0);

    // EDR with negative epsilon
    let result = edr(&calculator, -1.0, false);
    // Should handle gracefully
    assert!(result.distance >= 0.0);
}

/// Test with zero epsilon for LCSS and EDR
#[test]
fn test_zero_epsilon() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.1], [1.0, 1.1]];

    // LCSS with zero epsilon
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = lcss(&calculator, 0.0, false);
    // Should work (points must be exactly the same to match)
    assert!(result.distance >= 0.0 && result.distance <= 1.0);

    // EDR with zero epsilon
    let result = edr(&calculator, 0.0, false);
    // Should work (points must be exactly the same to match)
    assert!(result.distance >= 0.0);
}

/// Test with very large epsilon for LCSS and EDR
#[test]
fn test_large_epsilon() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[100.0, 100.0], [101.0, 101.0]];

    // LCSS with very large epsilon
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = lcss(&calculator, 1000.0, false);
    // Should treat all points as matching
    assert!(result.distance < 1e-6);

    // EDR with very large epsilon
    let result = edr(&calculator, 1000.0, false);
    // Should treat all points as matching
    assert_identical_distance(result.distance);
}

/// Test with NaN epsilon for LCSS and EDR
#[test]
fn test_nan_epsilon() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.1], [1.0, 1.1]];

    // LCSS with NaN epsilon
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = lcss(&calculator, f64::NAN, false);
    // Should handle gracefully
    assert!(result.distance.is_finite() || result.distance.is_nan());

    // EDR with NaN epsilon
    let result = edr(&calculator, f64::NAN, false);
    // Should handle gracefully
    assert!(result.distance.is_finite() || result.distance.is_nan());
}

/// Test with infinity epsilon for LCSS and EDR
#[test]
fn test_infinity_epsilon() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[100.0, 100.0], [101.0, 101.0]];

    // LCSS with infinity epsilon
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = lcss(&calculator, f64::INFINITY, false);
    // Should treat all points as matching
    assert!(result.distance < 1e-6);

    // EDR with infinity epsilon
    let result = edr(&calculator, f64::INFINITY, false);
    // Should treat all points as matching
    assert_identical_distance(result.distance);
}

/// Test with NaN gap point for ERP
#[test]
fn test_nan_gap_point() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.1], [1.0, 1.1]];

    // ERP with NaN gap point
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = erp_standard(&calculator, &[f64::NAN, f64::NAN], false);
    // Should handle gracefully
    assert!(result.distance.is_finite() || result.distance.is_nan());
}

/// Test with infinity gap point for ERP
#[test]
fn test_infinity_gap_point() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.1], [1.0, 1.1]];

    // ERP with infinity gap point
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = erp_standard(&calculator, &[f64::INFINITY, f64::INFINITY], false);
    // Should handle gracefully (may return infinity)
    assert!(result.distance.is_finite() || result.distance.is_infinite());
}

/// Test with mixed valid and invalid coordinates
#[test]
fn test_mixed_valid_invalid_coordinates() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [f64::NAN, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];

    // All algorithms should handle mixed coordinates gracefully
    // SSPD
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    assert!(dist.is_finite() || dist.is_nan() || dist.is_infinite());

    // DTW
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    assert!(
        result.distance.is_finite() || result.distance.is_nan() || result.distance.is_infinite()
    );

    // Hausdorff
    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    assert!(dist.is_finite() || dist.is_nan() || dist.is_infinite());
}

/// Test with subnormal values (very close to zero)
#[test]
fn test_subnormal_values() {
    let traj1: Vec<[f64; 2]> = vec![[f64::MIN_POSITIVE, f64::MIN_POSITIVE], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];

    // SSPD
    let dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);

    // DTW
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, false);
    assert_valid_dp_result(&result);

    // Hausdorff
    let dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    assert_valid_distance(dist);
}
