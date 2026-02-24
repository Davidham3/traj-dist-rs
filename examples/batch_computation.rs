//! Batch Computation Examples
//!
//! This example demonstrates how to use batch computation functions (pdist and cdist)
//! to efficiently calculate distances between multiple trajectories.

use traj_dist_rs::distance::{
    batch::{DistanceAlgorithm, Metric, cdist, pdist},
    distance_type::DistanceType,
};

fn main() {
    println!("============================================================");
    println!("Batch Computation Examples");
    println!("============================================================");

    // Create a set of sample trajectories
    let num_trajectories = 10;
    let trajectories: Vec<Vec<[f64; 2]>> = (0..num_trajectories)
        .map(|_| {
            (0..20)
                .map(|_| [rand::random::<f64>() * 10.0, rand::random::<f64>() * 10.0])
                .collect()
        })
        .collect();

    println!(
        "\nCreated {} trajectories with 20 points each",
        num_trajectories
    );

    // Method 1: Using pdist (pairwise distances in compressed format)
    println!("\n1. pdist - Pairwise Distance Matrix (Compressed)");
    println!("{}", "-".repeat(60));

    // Calculate pairwise distances using pdist
    let metric = Metric::new(DistanceAlgorithm::SSPD, DistanceType::Euclidean);
    match pdist(&trajectories, &metric, false) {
        Ok(compressed_distances) => {
            println!(
                "  Number of pairwise distances: {}",
                compressed_distances.len()
            );
            println!(
                "  Expected: {}",
                num_trajectories * (num_trajectories - 1) / 2
            );
            println!("  First 5 distances:");
            for (i, &dist) in compressed_distances.iter().take(5).enumerate() {
                println!("    [{}]: {:.6}", i, dist);
            }

            let min_dist = compressed_distances
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let max_dist = compressed_distances
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            println!("  Min distance: {:.6}", min_dist);
            println!("  Max distance: {:.6}", max_dist);
        }
        Err(e) => {
            println!("  Error: {}", e);
        }
    }

    // Method 2: Using pdist with different algorithms
    println!("\n2. pdist with Different Algorithms");
    println!("{}", "-".repeat(60));

    let algorithms = vec![
        (DistanceAlgorithm::SSPD, "SSPD"),
        (DistanceAlgorithm::DTW, "DTW"),
        (DistanceAlgorithm::Hausdorff, "Hausdorff"),
        (DistanceAlgorithm::LCSS { eps: 0.1 }, "LCSS"),
        (DistanceAlgorithm::EDR { eps: 0.1 }, "EDR"),
        (DistanceAlgorithm::ERP { g: [0.0, 0.0] }, "ERP"),
        (DistanceAlgorithm::DiscretFrechet, "Discret Frechet"),
    ];

    for (algorithm, name) in algorithms {
        let metric = Metric::new(algorithm, DistanceType::Euclidean);
        match pdist(&trajectories, &metric, false) {
            Ok(distances) => {
                let mean = distances.iter().sum::<f64>() / distances.len() as f64;
                println!(
                    "  {:20}: {} distances, mean={:.6}",
                    name,
                    distances.len(),
                    mean
                );
            }
            Err(e) => {
                println!("  {:20}: Error - {}", name, e);
            }
        }
    }

    // Method 3: Using cdist (full distance matrix)
    println!("\n3. cdist - Full Distance Matrix");
    println!("{}", "-".repeat(60));

    // Create two sets of trajectories
    let traj_set_a: Vec<_> = trajectories.iter().take(5).cloned().collect();
    let traj_set_b: Vec<_> = trajectories.iter().skip(5).cloned().collect();

    // Calculate distance matrix using cdist
    let metric = Metric::new(DistanceAlgorithm::SSPD, DistanceType::Euclidean);
    match cdist(&traj_set_a, &traj_set_b, &metric, false) {
        Ok(distance_matrix) => {
            let rows = traj_set_a.len();
            let cols = traj_set_b.len();
            println!("  Distance matrix shape: {}x{}", rows, cols);
            println!("  Expected: {}x{}", rows, cols);
            println!("  Full matrix:");
            for row in 0..rows {
                print!("    [");
                for col in 0..cols {
                    let idx = row * cols + col;
                    if col > 0 {
                        print!(", ");
                    }
                    print!("{:.6}", distance_matrix[idx]);
                }
                println!("]");
            }
        }
        Err(e) => {
            println!("  Error: {}", e);
        }
    }

    // Method 4: cdist with spherical distance
    println!("\n4. cdist with Spherical Distance");
    println!("{}", "-".repeat(60));

    // Create trajectories with geographic coordinates (latitude, longitude)
    let geo_trajectories = vec![
        vec![
            [40.7128, -74.0060],
            [40.7306, -73.9352],
            [40.6413, -73.7781],
        ], // New York
        vec![
            [34.0522, -118.2437],
            [34.0522, -118.2437],
            [33.9425, -118.4081],
        ], // Los Angeles
        vec![[51.5074, -0.1278], [51.4700, -0.4543], [51.5055, -0.2799]], // London
    ];

    let metric = Metric::new(DistanceAlgorithm::SSPD, DistanceType::Spherical);
    match cdist(&geo_trajectories, &geo_trajectories, &metric, false) {
        Ok(distance_matrix) => {
            let rows = geo_trajectories.len();
            let cols = geo_trajectories.len();
            println!("  Distance matrix (Haversine, in kilometers):");
            for row in 0..rows {
                print!("    [");
                for col in 0..cols {
                    let idx = row * cols + col;
                    if col > 0 {
                        print!(", ");
                    }
                    print!("{:.3}", distance_matrix[idx] / 1000.0); // Convert to kilometers
                }
                println!("]");
            }
        }
        Err(e) => {
            println!("  Error: {}", e);
        }
    }

    // Method 5: pdist with parallel processing
    println!("\n5. pdist with Parallel Processing");
    println!("{}", "-".repeat(60));

    let metric = Metric::new(DistanceAlgorithm::SSPD, DistanceType::Euclidean);

    // Sequential
    match pdist(&trajectories, &metric, false) {
        Ok(distances) => {
            println!("  Sequential: {} distances computed", distances.len());
        }
        Err(e) => {
            println!("  Sequential error: {}", e);
        }
    }

    // Parallel
    match pdist(&trajectories, &metric, true) {
        Ok(distances) => {
            println!("  Parallel: {} distances computed", distances.len());
        }
        Err(e) => {
            println!("  Parallel error: {}", e);
        }
    }

    // Method 6: cdist with parallel processing
    println!("\n6. cdist with Parallel Processing");
    println!("{}", "-".repeat(60));

    let metric = Metric::new(DistanceAlgorithm::DTW, DistanceType::Euclidean);

    // Sequential
    match cdist(&traj_set_a, &traj_set_b, &metric, false) {
        Ok(_matrix) => {
            let rows = traj_set_a.len();
            let cols = traj_set_b.len();
            println!("  Sequential: {}x{} matrix computed", rows, cols);
        }
        Err(e) => {
            println!("  Sequential error: {}", e);
        }
    }

    // Parallel
    match cdist(&traj_set_a, &traj_set_b, &metric, true) {
        Ok(_matrix) => {
            let rows = traj_set_a.len();
            let cols = traj_set_b.len();
            println!("  Parallel: {}x{} matrix computed", rows, cols);
        }
        Err(e) => {
            println!("  Parallel error: {}", e);
        }
    }

    println!("\n============================================================");
    println!("All batch computations completed successfully!");
    println!("============================================================");
}

// Note: This example requires the `rand` crate for random number generation.
// Add this to your Cargo.toml:
// [dependencies]
// rand = "0.8"
// traj-dist-rs = { path = "..", features = ["parallel"] }
