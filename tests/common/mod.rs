//! Common utilities for integration tests

/// Assert that a distance is finite and non-negative
pub fn assert_valid_distance(distance: f64) {
    assert!(
        distance.is_finite(),
        "Distance should be finite, got: {}",
        distance
    );
    assert!(
        distance >= 0.0,
        "Distance should be non-negative, got: {}",
        distance
    );
}

/// Assert that a DpResult is valid
pub fn assert_valid_dp_result(result: &traj_dist_rs::distance::DpResult) {
    assert_valid_distance(result.distance);
}
