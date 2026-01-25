# traj-dist-rs User Guide (Rust)

This guide provides comprehensive usage examples for the traj-dist-rs library, a high-performance trajectory distance library implemented in Rust.

## Installation

To use traj-dist-rs in your Rust project, add it to your `Cargo.toml`:

```toml
[dependencies]
traj-dist-rs = "0.1.0"
```

## Basic Concepts

The library provides implementations of several trajectory distance algorithms:

- **SSPD**: Symmetric Segment-Path Distance
- **DTW**: Dynamic Time Warping
- **Hausdorff**: Hausdorff Distance
- **LCSS**: Longest Common Subsequence
- **EDR**: Edit Distance on Real sequence
- **ERP**: Edit distance with Real Penalty
- **Discret Frechet**: Discrete Fréchet Distance

Each algorithm supports both Euclidean (Cartesian) and Spherical (Haversine) distance calculations.

## Basic Usage Example

Let's start with a simple example using two trajectories:

```rust
use traj_dist_rs::{sspd, dtw, hausdorff, DistanceType};

fn main() {
    // Define two trajectories as vectors of [x, y] coordinates
    let traj1 = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2 = vec![[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]];

    // Calculate different trajectory distances using Euclidean distance
    let sspd_dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    let dtw_dist = dtw(&traj1, &traj2, DistanceType::Euclidean);
    let hausdorff_dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);

    println!("SSPD distance: {}", sspd_dist);
    println!("DTW distance: {}", dtw_dist);
    println!("Hausdorff distance: {}", hausdorff_dist);
}
```

## Using Spherical Distance for Geographic Coordinates

For geographic coordinates (latitude/longitude), use spherical distance:

```rust
use traj_dist_rs::{sspd, dtw, hausdorff, DistanceType};

fn main() {
    // Geographic coordinates as [latitude, longitude]
    // New York City trajectory
    let nyc_traj1 = vec![[40.7128, -74.0060], [40.7589, -73.9851], [40.7831, -73.9712]];
    // Similar NYC trajectory with slight variations
    let nyc_traj2 = vec![[40.7228, -74.0160], [40.7689, -73.9951], [40.7931, -73.9812]];

    // Calculate distances using spherical distance (Haversine formula)
    let sspd_dist = sspd(&nyc_traj1, &nyc_traj2, DistanceType::Spherical);
    let dtw_dist = dtw(&nyc_traj1, &nyc_traj2, DistanceType::Spherical);
    let hausdorff_dist = hausdorff(&nyc_traj1, &nyc_traj2, DistanceType::Spherical);

    println!("SSPD spherical distance: {}", sspd_dist);
    println!("DTW spherical distance: {}", dtw_dist);
    println!("Hausdorff spherical distance: {}", hausdorff_dist);
}
```

## Algorithms with Parameters

Some algorithms require additional parameters:

```rust
use traj_dist_rs::{lcss, edr, erp_compat_with_dist_type, DistanceType};

fn main() {
    let traj1 = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let traj2 = vec![[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]];
    
    // LCSS with epsilon parameter - threshold for point matching
    let lcss_dist = lcss(&traj1, &traj2, 1.0, DistanceType::Euclidean);
    
    // EDR with epsilon parameter - threshold for point matching
    let edr_dist = edr(&traj1, &traj2, 1.0, DistanceType::Euclidean);
    
    // ERP with gap point - reference point for penalties
    let gap_point = [0.0, 0.0];
    let erp_dist = erp_compat_with_dist_type(&traj1, &traj2, gap_point, DistanceType::Euclidean);

    println!("LCSS distance (eps=1.0): {}", lcss_dist);
    println!("EDR distance (eps=1.0): {}", edr_dist);
    println!("ERP distance (g=[0,0]): {}", erp_dist);
}
```

## Comparing All Available Algorithms

Let's compare all available algorithms on the same trajectories:

```rust
use traj_dist_rs::{
    sspd, dtw, hausdorff, discret_frechet_euclidean, 
    lcss, edr, erp_compat_with_dist_type, erp_standard_with_dist_type, 
    DistanceType
};

fn main() {
    let traj1 = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let traj2 = vec![[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]];

    println!("Comparing All Algorithms:");
    println!("Trajectory 1: {:?}", traj1);
    println!("Trajectory 2: {:?}", traj2);

    // Standard algorithms without parameters
    let sspd_dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    let dtw_dist = dtw(&traj1, &traj2, DistanceType::Euclidean);
    let hausdorff_dist = hausdorff(&traj1, &traj2, DistanceType::Euclidean);
    let discret_frechet_dist = discret_frechet_euclidean(&traj1, &traj2);

    // Algorithms with parameters
    let lcss_dist = lcss(&traj1, &traj2, 2.0, DistanceType::Euclidean);
    let edr_dist = edr(&traj1, &traj2, 2.0, DistanceType::Euclidean);
    
    let gap_point = [0.0, 0.0];
    let erp_compat_dist = erp_compat_with_dist_type(&traj1, &traj2, gap_point, DistanceType::Euclidean);
    let erp_standard_dist = erp_standard_with_dist_type(&traj1, &traj2, gap_point, DistanceType::Euclidean);

    println!("SSPD: {}", sspd_dist);
    println!("DTW: {}", dtw_dist);
    println!("Hausdorff: {}", hausdorff_dist);
    println!("Discret Frechet: {}", discret_frechet_dist);
    println!("LCSS: {}", lcss_dist);
    println!("EDR: {}", edr_dist);
    println!("ERP (compat): {}", erp_compat_dist);
    println!("ERP (standard): {}", erp_standard_dist);
}
```

## Working with Different Trajectory Types

The library is generic and works with any type that implements the `CoordSequence` trait:

```rust
use traj_dist_rs::{sspd, dtw, DistanceType, CoordSequence, AsCoord};

// Example with a custom coordinate sequence implementation
struct CustomTrajectory {
    points: Vec<[f64; 2]>,
}

impl CoordSequence for CustomTrajectory {
    type Coord = [f64; 2];

    fn len(&self) -> usize {
        self.points.len()
    }

    fn get(&self, i: usize) -> Self::Coord {
        self.points[i]
    }
}

impl AsCoord for [f64; 2] {
    fn x(&self) -> f64 {
        self[0]
    }

    fn y(&self) -> f64 {
        self[1]
    }
}

fn main() {
    let traj1 = CustomTrajectory {
        points: vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    };
    
    let traj2 = CustomTrajectory {
        points: vec![[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]],
    };

    let sspd_dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    let dtw_dist = dtw(&traj1, &traj2, DistanceType::Euclidean);

    println!("SSPD distance with custom type: {}", sspd_dist);
    println!("DTW distance with custom type: {}", dtw_dist);
}
```

## Performance Considerations

The library is designed for high performance. Here's an example of how to measure performance:

```rust
use traj_dist_rs::{sspd, dtw, DistanceType};
use std::time::Instant;

fn create_random_trajectory(length: usize) -> Vec<[f64; 2]> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..length).map(|_| [rng.gen_range(0.0..10.0), rng.gen_range(0.0..10.0)]).collect()
}

fn main() {
    let traj1 = create_random_trajectory(100);
    let traj2 = create_random_trajectory(100);

    // Measure SSPD performance
    let start = Instant::now();
    let sspd_dist = sspd(&traj1, &traj2, DistanceType::Euclidean);
    let sspd_time = start.elapsed();
    
    // Measure DTW performance
    let start = Instant::now();
    let dtw_dist = dtw(&traj1, &traj2, DistanceType::Euclidean);
    let dtw_time = start.elapsed();

    println!("SSPD distance: {}, time: {:?}", sspd_dist, sspd_time);
    println!("DTW distance: {}, time: {:?}", dtw_dist, dtw_time);
}
```

## Error Handling

The library handles invalid inputs gracefully:

```rust
use traj_dist_rs::{sspd, dtw, DistanceType};

fn main() {
    // Empty trajectories return f64::MAX
    let empty_traj: Vec<[f64; 2]> = vec![];
    let normal_traj = vec![[0.0, 0.0], [1.0, 1.0]];
    
    let sspd_dist = sspd(&empty_traj, &normal_traj, DistanceType::Euclidean);
    let dtw_dist = dtw(&empty_traj, &normal_traj, DistanceType::Euclidean);
    
    println!("SSPD with empty trajectory: {}", sspd_dist); // Should be f64::MAX
    println!("DTW with empty trajectory: {}", dtw_dist);   // Should be f64::MAX
}
```

## Working with Different Distance Types

You can dynamically parse distance types from strings:

```rust
use traj_dist_rs::{DistanceType, sspd};

fn main() {
    let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
    let traj2 = vec![[0.1, 0.1], [1.1, 1.1]];
    
    // Parse distance type from string
    let dist_type_str = "euclidean";
    let dist_type = DistanceType::parse_distance_type(dist_type_str).unwrap();
    
    let distance = sspd(&traj1, &traj2, dist_type);
    println!("Distance with parsed type: {}", distance);
}
```

## Summary

This guide covered:

1. Basic installation and usage
2. Different trajectory distance algorithms
3. Euclidean vs. Spherical distance calculations
4. Algorithms with parameters (LCSS, EDR, ERP)
5. Working with custom coordinate sequence types
6. Performance considerations
7. Error handling
8. Dynamic distance type selection

The `traj-dist-rs` library provides high-performance trajectory distance calculations with a clean, generic API that works well with various trajectory representations. The algorithms are implemented with performance in mind, making it suitable for large-scale trajectory analysis applications.