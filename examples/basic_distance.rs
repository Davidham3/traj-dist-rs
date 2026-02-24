//! Basic Distance Calculation Examples
//!
//! This example demonstrates how to calculate distances between trajectories
//! using various distance algorithms supported by traj-dist-rs.

use traj_dist_rs::distance::{
    base::TrajectoryCalculator, discret_frechet::discret_frechet, distance_type::DistanceType,
    dtw::dtw, edr::edr, erp::erp_compat_traj_dist, erp::erp_standard, hausdorff::hausdorff,
    lcss::lcss, sspd::sspd,
};

fn main() {
    println!("============================================================");
    println!("Basic Distance Calculation Examples");
    println!("============================================================");

    // Define sample trajectories
    let traj1 = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];

    let traj2 = vec![[0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [3.1, 3.1]];

    let traj3 = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];

    // SSPD (Symmetric Segment-Path Distance)
    println!("\n1. SSPD (Symmetric Segment-Path Distance)");
    println!("{}", "-".repeat(40));
    let dist_euclidean = sspd(&traj1, &traj2, DistanceType::Euclidean);
    let dist_spherical = sspd(&traj1, &traj2, DistanceType::Spherical);
    println!("  Euclidean: {:.6}", dist_euclidean);
    println!("  Spherical: {:.6}", dist_spherical);

    // DTW (Dynamic Time Warping)
    println!("\n2. DTW (Dynamic Time Warping)");
    println!("{}", "-".repeat(40));
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);

    // DTW without matrix
    let result_no_matrix = dtw(&calculator, false);
    println!("  Distance (no matrix): {:.6}", result_no_matrix.distance);

    // DTW with matrix
    let result_with_matrix = dtw(&calculator, true);
    println!(
        "  Distance (with matrix): {:.6}",
        result_with_matrix.distance
    );
    if let Some(matrix) = &result_with_matrix.matrix {
        let rows = traj1.len() + 1;
        let cols = traj2.len() + 1;
        println!("  Matrix shape: {}x{}", rows, cols);
        println!("  Matrix length: {}", matrix.len());
    }

    // Hausdorff Distance
    println!("\n3. Hausdorff Distance");
    println!("{}", "-".repeat(40));
    let dist_euclidean = hausdorff(&traj1, &traj3, DistanceType::Euclidean);
    let dist_spherical = hausdorff(&traj1, &traj3, DistanceType::Spherical);
    println!("  Euclidean: {:.6}", dist_euclidean);
    println!("  Spherical: {:.6}", dist_spherical);

    // LCSS (Longest Common Subsequence)
    println!("\n4. LCSS (Longest Common Subsequence)");
    println!("{}", "-".repeat(40));
    // LCSS with epsilon parameter (similarity threshold)
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = lcss(&calculator, 0.1, true);
    println!("  Distance (epsilon=0.1): {:.6}", result.distance);
    if let Some(matrix) = &result.matrix {
        let rows = traj1.len() + 1;
        let cols = traj2.len() + 1;
        println!("  Matrix shape: {}x{}", rows, cols);
        println!("  Matrix length: {}", matrix.len());
    }

    // LCSS with different epsilon
    let calculator = TrajectoryCalculator::new(&traj1, &traj3, DistanceType::Euclidean);
    let result = lcss(&calculator, 0.5, false);
    println!("  Distance (epsilon=0.5): {:.6}", result.distance);

    // EDR (Edit Distance on Real sequence)
    println!("\n5. EDR (Edit Distance on Real sequence)");
    println!("{}", "-".repeat(40));
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = edr(&calculator, 0.1, true);
    println!("  Distance (epsilon=0.1): {:.6}", result.distance);
    if let Some(matrix) = &result.matrix {
        let rows = traj1.len() + 1;
        let cols = traj2.len() + 1;
        println!("  Matrix shape: {}x{}", rows, cols);
        println!("  Matrix length: {}", matrix.len());
    }

    // Discret Frechet Distance
    println!("\n6. Discret Frechet Distance");
    println!("{}", "-".repeat(40));
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result = discret_frechet(&calculator, true);
    println!("  Distance: {:.6}", result.distance);
    if let Some(matrix) = &result.matrix {
        let rows = traj1.len() + 1;
        let cols = traj2.len() + 1;
        println!("  Matrix shape: {}x{}", rows, cols);
        println!("  Matrix length: {}", matrix.len());
    }

    // ERP (Edit distance with Real Penalty)
    println!("\n7. ERP (Edit distance with Real Penalty)");
    println!("{}", "-".repeat(40));
    // Standard ERP implementation
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let gap_point = [0.0, 0.0]; // Gap point for ERP
    let result_standard = erp_standard(&calculator, &gap_point, true);
    println!("  Standard ERP distance: {:.6}", result_standard.distance);
    if let Some(matrix) = &result_standard.matrix {
        let rows = traj1.len() + 1;
        let cols = traj2.len() + 1;
        println!("  Standard ERP matrix shape: {}x{}", rows, cols);
        println!("  Standard ERP matrix length: {}", matrix.len());
    }

    // ERP compatible with traj-dist
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
    let result_compat = erp_compat_traj_dist(&calculator, &gap_point, false);
    println!("  Compatible ERP distance: {:.6}", result_compat.distance);

    println!("\n============================================================");
    println!("All distance calculations completed successfully!");
    println!("============================================================");
}
