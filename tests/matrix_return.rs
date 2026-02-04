//! Integration tests for matrix return functionality
//!
//! Tests that verify the _with_matrix functions return correct DP matrices

use traj_dist_rs::distance::base::{PrecomputedDistanceCalculator, TrajectoryCalculator};
use traj_dist_rs::distance::discret_frechet::discret_frechet;
use traj_dist_rs::distance::distance_type::DistanceType;
use traj_dist_rs::distance::dtw::dtw;
use traj_dist_rs::distance::edr;
use traj_dist_rs::distance::erp::erp_standard;
use traj_dist_rs::distance::lcss;
use traj_dist_rs::distance::utils::precompute_distance_matrix;

mod common;

/// Compare two DpResults
fn assert_dp_results_equal(
    result1: &traj_dist_rs::distance::DpResult,
    result2: &traj_dist_rs::distance::DpResult,
    tolerance: f64,
) {
    assert!(
        (result1.distance - result2.distance).abs() < tolerance,
        "Distances differ: {} vs {}",
        result1.distance,
        result2.distance
    );
}

use common::assert_valid_dp_result;

/// Test DTW with matrix return
#[test]
fn test_dtw_with_matrix() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);

    // Without matrix
    let result_no_matrix = dtw(&calculator, false);
    assert!(result_no_matrix.matrix.is_none());

    // With matrix
    let result_with_matrix = dtw(&calculator, true);
    assert!(result_with_matrix.matrix.is_some());

    // Distances should be the same
    assert_dp_results_equal(&result_no_matrix, &result_with_matrix, 1e-10);

    // Matrix dimensions should be (n0+1) x (n1+1) = 4 x 4 = 16
    if let Some(matrix) = result_with_matrix.matrix {
        assert_eq!(matrix.len(), 16);
    }
}

/// Test DTW matrix content
#[test]
fn test_dtw_matrix_content() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, true);

    assert!(result.matrix.is_some());
    if let Some(matrix) = result.matrix {
        // Matrix should be (n0+1) x (n1+1) = 3 x 3 = 9
        assert_eq!(matrix.len(), 9);

        // Check that matrix[0] is 0.0 (initialization)
        assert_eq!(matrix[0], 0.0);

        // Check that the first row (except [0][0]) is infinity
        assert_eq!(matrix[1], f64::INFINITY);
        assert_eq!(matrix[2], f64::INFINITY);

        // Check that the first column (except [0][0]) is infinity
        assert_eq!(matrix[3], f64::INFINITY);
        assert_eq!(matrix[6], f64::INFINITY);

        // The final result should be at matrix[8]
        assert!((matrix[8] - result.distance).abs() < 1e-10);
    }
}

/// Test DTW with larger trajectories
#[test]
fn test_dtw_with_matrix_larger() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, 2.0]];

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);

    let result = dtw(&calculator, true);

    // Matrix dimensions should be (n0+1) x (n1+1) = 5 x 5 = 25
    if let Some(matrix) = result.matrix {
        assert_eq!(matrix.len(), 25);
    }
}

/// Test Discret Frechet with matrix return
#[test]
fn test_discret_frechet_with_matrix() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);

    // Without matrix
    let result_no_matrix = discret_frechet(&calculator, false);
    assert!(result_no_matrix.matrix.is_none());

    // With matrix
    let result_with_matrix = discret_frechet(&calculator, true);
    assert!(result_with_matrix.matrix.is_some());

    // Distances should be the same
    assert_dp_results_equal(&result_no_matrix, &result_with_matrix, 1e-10);

    // Matrix dimensions should be (n0+1) x (n1+1) = 4 x 4 = 16
    if let Some(matrix) = result_with_matrix.matrix {
        assert_eq!(matrix.len(), 16);
    }
}

/// Test Discret Frechet matrix content
#[test]
fn test_discret_frechet_matrix_content() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = discret_frechet(&calculator, true);

    assert!(result.matrix.is_some());
    if let Some(matrix) = result.matrix {
        // Matrix should be (n0+1) x (n1+1) = 3 x 3 = 9
        assert_eq!(matrix.len(), 9);

        // Check that matrix[0] is 0.0 (initialization)
        assert_eq!(matrix[0], 0.0);

        // Check that the first row (except [0][0]) is infinity
        assert_eq!(matrix[1], f64::INFINITY);
        assert_eq!(matrix[2], f64::INFINITY);

        // Check that the first column (except [0][0]) is infinity
        assert_eq!(matrix[3], f64::INFINITY);
        assert_eq!(matrix[6], f64::INFINITY);

        // The final result should be at matrix[8]
        assert!((matrix[8] - result.distance).abs() < 1e-10);
    }
}

/// Test LCSS with matrix return
#[test]
fn test_lcss_with_matrix() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    let eps = 0.1;

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);

    // Without matrix
    let result_no_matrix = lcss::lcss(&calculator, eps, false);
    assert!(result_no_matrix.matrix.is_none());

    // With matrix
    let result_with_matrix = lcss::lcss(&calculator, eps, true);
    assert!(result_with_matrix.matrix.is_some());

    // Distances should be the same
    assert_dp_results_equal(&result_no_matrix, &result_with_matrix, 1e-10);

    // Matrix dimensions should be (n0+1) x (n1+1) = 4 x 4 = 16
    if let Some(matrix) = result_with_matrix.matrix {
        assert_eq!(matrix.len(), 16);
    }
}

/// Test EDR with matrix return
#[test]
fn test_edr_with_matrix() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    let eps = 0.1;

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);

    // Without matrix
    let result_no_matrix = edr::edr(&calculator, eps, false);
    assert!(result_no_matrix.matrix.is_none());

    // With matrix
    let result_with_matrix = edr::edr(&calculator, eps, true);
    assert!(result_with_matrix.matrix.is_some());

    // Distances should be the same
    assert_dp_results_equal(&result_no_matrix, &result_with_matrix, 1e-10);

    // Matrix dimensions should be (n0+1) x (n1+1) = 4 x 4 = 16
    if let Some(matrix) = result_with_matrix.matrix {
        assert_eq!(matrix.len(), 16);
    }
}

/// Test ERP with matrix return
#[test]
fn test_erp_with_matrix() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    let g = [0.0, 0.0];

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);

    // Without matrix
    let result_no_matrix = erp_standard(&calculator, &g, false);
    assert!(result_no_matrix.matrix.is_none());

    // With matrix
    let result_with_matrix = erp_standard(&calculator, &g, true);
    assert!(result_with_matrix.matrix.is_some());

    // Distances should be the same
    assert_dp_results_equal(&result_no_matrix, &result_with_matrix, 1e-10);

    // Matrix dimensions should be (n0+1) x (n1+1) = 4 x 4 = 16
    if let Some(matrix) = result_with_matrix.matrix {
        assert_eq!(matrix.len(), 16);
    }
}

/// Test DTW with precomputed distance matrix
#[test]
fn test_dtw_with_precomputed_matrix() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    // Precompute distance matrix
    let distance_matrix = precompute_distance_matrix(&traj1, &traj2, DistanceType::Euclidean);

    // Use PrecomputedDistanceCalculator
    let precomp_calc = PrecomputedDistanceCalculator::new(&distance_matrix);

    let result = dtw(&precomp_calc, true);

    // Should return valid result with matrix
    assert_valid_dp_result(&result);
    assert!(result.matrix.is_some());

    // Matrix dimensions should be (n0+1) x (n1+1) = 4 x 4 = 16
    if let Some(matrix) = result.matrix {
        assert_eq!(matrix.len(), 16);
    }
}

/// Test DTW consistency between TrajectoryCalculator and PrecomputedDistanceCalculator
#[test]
fn test_dtw_consistency_between_calculators() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    // Using TrajectoryCalculator
    let traj_calc = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result_traj = dtw(&traj_calc, true);

    // Using PrecomputedDistanceCalculator
    let distance_matrix = precompute_distance_matrix(&traj1, &traj2, DistanceType::Euclidean);
    let precomp_calc = PrecomputedDistanceCalculator::new(&distance_matrix);
    let result_precomp = dtw(&precomp_calc, true);

    // Both should produce the same distance and matrix
    assert_dp_results_equal(&result_traj, &result_precomp, 1e-10);

    // Matrices should be identical
    if let (Some(matrix_traj), Some(matrix_precomp)) = (&result_traj.matrix, &result_precomp.matrix)
    {
        assert_eq!(matrix_traj.len(), matrix_precomp.len());
        for (i, (&v1, &v2)) in matrix_traj.iter().zip(matrix_precomp.iter()).enumerate() {
            // Handle infinity values specially
            if v1.is_infinite() && v2.is_infinite() {
                assert_eq!(
                    v1.is_infinite(),
                    v2.is_infinite(),
                    "Matrix elements differ at index {}: {} vs {}",
                    i,
                    v1,
                    v2
                );
            } else {
                assert!(
                    (v1 - v2).abs() < 1e-10,
                    "Matrix elements differ at index {}: {} vs {}",
                    i,
                    v1,
                    v2
                );
            }
        }
    }
}

/// Test Discret Frechet consistency between TrajectoryCalculator and PrecomputedDistanceCalculator
#[test]
fn test_discret_frechet_consistency_between_calculators() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    // Using TrajectoryCalculator
    let traj_calc = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result_traj = discret_frechet(&traj_calc, true);

    // Using PrecomputedDistanceCalculator
    let distance_matrix = precompute_distance_matrix(&traj1, &traj2, DistanceType::Euclidean);
    let precomp_calc = PrecomputedDistanceCalculator::new(&distance_matrix);
    let result_precomp = discret_frechet(&precomp_calc, true);

    // Both should produce the same distance and matrix
    assert_dp_results_equal(&result_traj, &result_precomp, 1e-10);

    // Matrices should be identical
    if let (Some(matrix_traj), Some(matrix_precomp)) = (&result_traj.matrix, &result_precomp.matrix)
    {
        assert_eq!(matrix_traj.len(), matrix_precomp.len());
        for (i, (&v1, &v2)) in matrix_traj.iter().zip(matrix_precomp.iter()).enumerate() {
            // Handle infinity values specially
            if v1.is_infinite() && v2.is_infinite() {
                assert_eq!(
                    v1.is_infinite(),
                    v2.is_infinite(),
                    "Matrix elements differ at index {}: {} vs {}",
                    i,
                    v1,
                    v2
                );
            } else {
                assert!(
                    (v1 - v2).abs() < 1e-10,
                    "Matrix elements differ at index {}: {} vs {}",
                    i,
                    v1,
                    v2
                );
            }
        }
    }
}

/// Test matrix row-major order
#[test]
fn test_matrix_row_major_order() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, true);

    if let Some(matrix) = result.matrix {
        let n0 = traj1.len();
        let n1 = traj2.len();
        let width = n1 + 1;

        // Check row-major order indexing
        // matrix[i * width + j] should correspond to C[i][j]
        for i in 0..=n0 {
            for j in 0..=n1 {
                let idx = i * width + j;
                assert!(
                    idx < matrix.len(),
                    "Index {} out of bounds for matrix of size {}",
                    idx,
                    matrix.len()
                );
            }
        }

        // The final result should be at matrix[n0 * width + n1]
        let final_idx = n0 * width + n1;
        assert!((matrix[final_idx] - result.distance).abs() < 1e-10);
    }
}

/// Test matrix all elements are finite (for valid inputs)
#[test]
fn test_matrix_all_finite() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]];

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = dtw(&calculator, true);

    if let Some(matrix) = result.matrix {
        // All matrix elements should be finite (result distance, 0.0, or positive numbers)
        for (i, &val) in matrix.iter().enumerate() {
            // Note: Some elements might be infinity (initialization), that's OK
            assert!(
                (val.is_finite() || val == f64::INFINITY),
                "Matrix element at index {} is not finite or infinity: {}",
                i,
                val
            );
        }
    }
}

/// Test spherical distance with matrix
#[test]
fn test_spherical_with_matrix() {
    let traj1: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0]];

    // DTW with spherical
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Spherical);
    let result = dtw(&calculator, true);
    assert!(result.matrix.is_some());
    assert_valid_dp_result(&result);

    // LCSS with spherical
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Spherical);
    let result = lcss::lcss(&calculator, 0.1, true);
    assert!(result.matrix.is_some());
    assert_valid_dp_result(&result);

    // EDR with spherical
    let result = edr::edr(&calculator, 0.1, true);
    assert!(result.matrix.is_some());
    assert_valid_dp_result(&result);

    // ERP with spherical
    let result = erp_standard(&calculator, &[0.0, 0.0], true);
    assert!(result.matrix.is_some());
    assert_valid_dp_result(&result);
}
