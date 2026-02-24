//! Multiple Distance Calculations with TrajectoryCalculator
//!
//! This example demonstrates how to efficiently compute multiple distances
//! for the same trajectory pair using TrajectoryCalculator.

use traj_dist_rs::distance::{
    base::TrajectoryCalculator, discret_frechet::discret_frechet, distance_type::DistanceType,
    dtw::dtw, edr::edr, erp::erp_standard, lcss::lcss,
};

// Type alias for trajectory pairs to reduce type complexity
type TrajectoryPair = (Vec<[f64; 2]>, Vec<[f64; 2]>);

fn main() {
    println!("============================================================");
    println!("Multiple Distance Calculations Examples");
    println!("============================================================");

    // Define sample trajectories
    let traj1 = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];

    let traj2 = vec![[0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [3.1, 3.1]];

    let traj3 = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];

    // Example 1: DTW with precomputed distance matrix
    println!("\n1. DTW with TrajectoryCalculator");
    println!("{}", "-".repeat(60));

    // Create trajectory calculator
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);

    // Calculate DTW without matrix (optimized)
    let result = dtw(&calculator, false);
    println!("  DTW distance (no matrix): {:.6}", result.distance);

    // Calculate DTW with full matrix
    let result = dtw(&calculator, true);
    println!("  DTW distance (with matrix): {:.6}", result.distance);
    if let Some(matrix) = &result.matrix {
        let rows = traj1.len() + 1;
        let cols = traj2.len() + 1;
        println!("  Matrix shape: {}x{}", rows, cols);
        println!("  Matrix length: {}", matrix.len());
    }

    // Example 2: Comparing multiple algorithms with same trajectory pair
    println!("\n2. Comparing Multiple Algorithms with Same Trajectory Pair");
    println!("{}", "-".repeat(60));

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);

    // DTW
    let result = dtw(&calculator, false);
    println!("  DTW distance: {:.6}", result.distance);

    // LCSS
    let result = lcss(&calculator, 0.1, false);
    println!("  LCSS distance (epsilon=0.1): {:.6}", result.distance);

    // EDR
    let result = edr(&calculator, 0.1, false);
    println!("  EDR distance (epsilon=0.1): {:.6}", result.distance);

    // ERP
    let gap_point = [0.0, 0.0];
    let result = erp_standard(&calculator, &gap_point, false);
    println!("  ERP distance (gap=[0,0]): {:.6}", result.distance);

    // Discret Frechet
    let result = discret_frechet(&calculator, false);
    println!("  Discret Frechet distance: {:.6}", result.distance);

    // Example 3: Testing different parameter values
    println!("\n3. Testing Different Parameter Values");
    println!("{}", "-".repeat(60));

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);

    // LCSS with different epsilon values
    for eps in [0.05, 0.1, 0.2, 0.5, 1.0] {
        let result = lcss(&calculator, eps, false);
        println!(
            "  LCSS distance (epsilon={:.2}): {:.6}",
            eps, result.distance
        );
    }

    // EDR with different epsilon values
    for eps in [0.05, 0.1, 0.2, 0.5, 1.0] {
        let result = edr(&calculator, eps, false);
        println!(
            "  EDR distance (epsilon={:.2}): {:.6}",
            eps, result.distance
        );
    }

    // Example 4: Precomputed matrix with different distance types
    println!("\n4. Different Distance Types with Same Trajectory Pair");
    println!("{}", "-".repeat(60));

    // Euclidean distance
    let calc_euclidean = TrajectoryCalculator::new(&traj1, &traj3, DistanceType::Euclidean);
    let result_euclidean = dtw(&calc_euclidean, false);
    println!("  Euclidean DTW distance: {:.6}", result_euclidean.distance);

    // Spherical distance
    let calc_spherical = TrajectoryCalculator::new(&traj1, &traj3, DistanceType::Spherical);
    let result_spherical = dtw(&calc_spherical, false);
    println!("  Spherical DTW distance: {:.6}", result_spherical.distance);

    // Example 5: Multiple trajectory pairs
    println!("\n5. Computing Distances for Multiple Trajectory Pairs");
    println!("{}", "-".repeat(60));

    let trajectory_pairs: Vec<TrajectoryPair> = vec![
        (traj1.clone(), traj2.clone()),
        (traj1.clone(), traj3.clone()),
        (traj2.clone(), traj3.clone()),
    ];

    for (i, (t1, t2)) in trajectory_pairs.iter().enumerate() {
        let calculator = TrajectoryCalculator::new(t1, t2, DistanceType::Euclidean);
        let result = dtw(&calculator, false);
        println!("  Pair {} DTW distance: {:.6}", i + 1, result.distance);
    }

    // Example 6: Performance considerations
    println!("\n6. Performance Considerations");
    println!("{}", "-".repeat(60));
    println!("  TrajectoryCalculator is beneficial when:");
    println!("    - Computing multiple distances for the same trajectory pair");
    println!("    - Testing different parameter values (epsilon, etc.)");
    println!("    - Comparing different algorithms on the same data");
    println!();
    println!("  Example use cases:");
    println!("    - Algorithm comparison and benchmarking");
    println!("    - Parameter tuning and optimization");
    println!("    - Computing both distance and full DP matrix");

    println!("\n============================================================");
    println!("All multiple distance calculations completed successfully!");
    println!("============================================================");
}
