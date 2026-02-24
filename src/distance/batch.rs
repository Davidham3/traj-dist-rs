//! # Batch Computation Module
//!
//! This module provides batch computation functions for computing distances
//! between multiple trajectories efficiently.
//!
//! ## Functions
//!
//! - `pdist`: Compute pairwise distances between trajectories in a list
//! - `cdist`: Compute distances between two collections of trajectories
//!
//! ## Features
//!
//! - Support for all distance algorithms via `Metric` configuration
//! - Parallel processing with Rayon (optional)
//! - Zero-copy access to NumPy arrays (via Python bindings)
//! - Efficient memory layout
//!
//! ## Architecture
//!
//! This module uses the `Metric` struct which encapsulates the distance algorithm
//! and its parameters. To add a new algorithm:
//! 1. Add a new variant to the `DistanceAlgorithm` enum
/// 2. Add a matching branch in `Metric::distance()` method
/// 3. Add a factory method in the Python binding (`PyMetric`)
use crate::distance::base::TrajectoryCalculator;
pub use crate::distance::distance_type::DistanceType;
use crate::err::TrajDistError;
use crate::traits::CoordSequence;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Distance algorithm with parameters
///
/// This enum combines the algorithm selection with its required parameters,
/// providing type safety and preventing invalid parameter combinations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceAlgorithm {
    /// Symmetric Segment-Path Distance (no parameters)
    SSPD,
    /// Dynamic Time Warping (no parameters)
    DTW,
    /// Hausdorff Distance (no parameters)
    Hausdorff,
    /// Longest Common Subsequence with epsilon threshold
    LCSS { eps: f64 },
    /// Edit Distance on Real sequence with epsilon threshold
    EDR { eps: f64 },
    /// Edit distance with Real Penalty and gap point
    ERP { g: [f64; 2] },
    /// Discrete Frechet Distance (no parameters)
    DiscretFrechet,
}

/// Metric configuration for distance calculations
///
/// This struct wraps a `DistanceAlgorithm` and `DistanceType` to provide
/// distance calculation for batch computation.
///
/// Note: This is different from `TrajectoryCalculator` in `base.rs`, which
/// is used for single trajectory pair computations with precomputed matrices.
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::batch::{Metric, DistanceAlgorithm, DistanceType};
///
/// let metric = Metric::new(
///     DistanceAlgorithm::LCSS { eps: 5.0 },
///     DistanceType::Euclidean
/// );
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Metric {
    algorithm: DistanceAlgorithm,
    distance_type: DistanceType,
}

impl Metric {
    /// Create a new metric
    ///
    /// # Arguments
    /// - `algorithm`: The distance algorithm to use
    /// - `distance_type`: The distance type (Euclidean or Spherical)
    pub fn new(algorithm: DistanceAlgorithm, distance_type: DistanceType) -> Self {
        Self {
            algorithm,
            distance_type,
        }
    }

    /// Get the algorithm
    pub fn algorithm(&self) -> DistanceAlgorithm {
        self.algorithm
    }

    /// Get the distance type
    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    /// Compute the distance between two trajectories
    ///
    /// # Arguments
    /// - `traj1`: First trajectory
    /// - `traj2`: Second trajectory
    ///
    /// # Returns
    /// The distance value as `f64`
    pub fn distance<T: CoordSequence>(&self, traj1: &T, traj2: &T) -> f64 {
        match self.algorithm {
            DistanceAlgorithm::SSPD => {
                crate::distance::sspd::sspd(traj1, traj2, self.distance_type)
            }
            DistanceAlgorithm::DTW => {
                let calculator = TrajectoryCalculator::new(traj1, traj2, self.distance_type);
                crate::distance::dtw::dtw(&calculator, false).distance
            }
            DistanceAlgorithm::Hausdorff => {
                crate::distance::hausdorff::hausdorff(traj1, traj2, self.distance_type)
            }
            DistanceAlgorithm::LCSS { eps } => {
                let calculator = TrajectoryCalculator::new(traj1, traj2, self.distance_type);
                crate::distance::lcss::lcss(&calculator, eps, false).distance
            }
            DistanceAlgorithm::EDR { eps } => {
                let calculator = TrajectoryCalculator::new(traj1, traj2, self.distance_type);
                crate::distance::edr::edr(&calculator, eps, false).distance
            }
            DistanceAlgorithm::ERP { g } => {
                let calculator = TrajectoryCalculator::new(traj1, traj2, self.distance_type);
                crate::distance::erp::erp_standard(&calculator, &g, false).distance
            }
            DistanceAlgorithm::DiscretFrechet => {
                let calculator = TrajectoryCalculator::new(traj1, traj2, self.distance_type);
                crate::distance::discret_frechet::discret_frechet(&calculator, false).distance
            }
        }
    }
}

/// Compute pairwise distances between trajectories
///
/// Returns a compressed distance matrix (1D array) containing
/// distances for all unique pairs (i, j) where i < j.
///
/// # Symmetry Assumption
///
/// This function assumes that the distance metric is **symmetric**, i.e.,
/// `distance(A, B) == distance(B, A)`. All standard distance algorithms
/// in traj-dist-rs (SSPD, DTW, Hausdorff, LCSS, EDR, ERP, Discret Frechet)
/// satisfy this property.
///
/// **Important**: If your distance metric is **asymmetric**, use `cdist` instead
/// to compute the full distance matrix. Using `pdist` with asymmetric distances
/// will only compute half of the distances and may produce incorrect results.
///
/// # Arguments
/// * `trajectories` - Slice of trajectories
/// * `metric` - Metric configuration for distance calculation
/// * `parallel` - Whether to use parallel processing (default: true)
///
/// # Returns
/// * Vector of distances in row-major order of upper triangle
///
/// # Output Format
///
/// For `n` trajectories, the result is a 1D array of length `n * (n - 1) / 2`.
/// The distances are ordered as `d(0,1), d(0,2), ..., d(0,n-1), d(1,2), d(1,3), ..., d(n-2,n-1)`.
///
/// # Performance Notes
///
/// For large datasets (e.g., 10,000 trajectories), the parallel implementation
/// creates an intermediate index vector containing all unique pairs. This can
/// consume significant memory (~50 million tuples ≈ 400MB). For very large
/// datasets, consider processing in smaller batches.
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::batch::{pdist, Metric, DistanceAlgorithm, DistanceType};
///
/// let trajectories = vec![
///     vec![[0.0, 0.0], [1.0, 1.0]],
///     vec![[0.0, 1.0], [1.0, 0.0]],
///     vec![[0.5, 0.5], [1.5, 1.5]],
/// ];
///
/// let metric = Metric::new(
///     DistanceAlgorithm::SSPD,
///     DistanceType::Euclidean
/// );
/// let distances = pdist(&trajectories, &metric, true).unwrap();
/// ```
pub fn pdist<T>(
    trajectories: &[T],
    metric: &Metric,
    parallel: bool,
) -> Result<Vec<f64>, TrajDistError>
where
    T: CoordSequence + Sync,
{
    let n = trajectories.len();
    if n < 2 {
        return Err(TrajDistError::InvalidParams(
            "pdist requires at least 2 trajectories".to_string(),
        ));
    }

    // Use Rayon's global thread pool for parallel processing
    #[cfg(feature = "parallel")]
    let distances = if parallel {
        compute_pdist_parallel(trajectories, metric)
    } else {
        compute_pdist_sequential(trajectories, metric)
    };

    #[cfg(not(feature = "parallel"))]
    {
        let _ = parallel; // Suppress unused variable warning when parallel feature is disabled
    }

    #[cfg(not(feature = "parallel"))]
    let distances = compute_pdist_sequential(trajectories, metric);

    Ok(distances)
}

/// Compute pairwise distances sequentially
fn compute_pdist_sequential<T: CoordSequence>(trajectories: &[T], metric: &Metric) -> Vec<f64> {
    let n = trajectories.len();
    let mut distances = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = metric.distance(&trajectories[i], &trajectories[j]);
            distances.push(dist);
        }
    }

    distances
}

/// Compute pairwise distances in parallel
///
/// Uses Rayon's global thread pool for efficient parallel execution.
/// Creates an index vector of all unique pairs, then computes distances
/// in parallel.
#[cfg(feature = "parallel")]
fn compute_pdist_parallel<T: CoordSequence + Sync>(
    trajectories: &[T],
    metric: &Metric,
) -> Vec<f64> {
    let n = trajectories.len();

    // Create index pairs for all unique pairs (i, j) where i < j
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    // Compute distances in parallel using Rayon's global thread pool
    pairs
        .into_par_iter()
        .map(|(i, j)| metric.distance(&trajectories[i], &trajectories[j]))
        .collect()
}

/// Compute distances between two trajectory collections
///
/// Returns a full distance matrix (2D array flattened to 1D).
///
/// # When to Use `cdist` vs `pdist`
///
/// - **Use `cdist`** when:
///   - Computing distances between two different trajectory collections
///   - Your distance metric is **asymmetric** (distance(A, B) != distance(B, A))
///   - You need the full distance matrix for both directions
///
/// - **Use `pdist`** when:
///   - Computing distances within a single trajectory collection
///   - Your distance metric is **symmetric** (distance(A, B) == distance(B, A))
///   - You want to save memory by using the compressed distance matrix format
///
/// # Arguments
/// * `trajectories_a` - First collection of trajectories
/// * `trajectories_b` - Second collection of trajectories
/// * `metric` - Metric configuration for distance calculation
/// * `parallel` - Whether to use parallel processing (default: true)
///
/// # Returns
/// * Flattened 2D array of distances with shape (n_a, n_b)
///
/// # Output Format
///
/// For `n_a` trajectories in the first collection and `n_b` trajectories in the second,
/// the result is a 1D array of length `n_a * n_b`. The distance at index `i * n_b + j`
/// represents the distance from `trajectories_a[i]` to `trajectories_b[j]`.
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::batch::{cdist, Metric, DistanceAlgorithm, DistanceType};
///
/// let trajectories_a = vec![
///     vec![[0.0, 0.0], [1.0, 1.0]],
///     vec![[0.0, 1.0], [1.0, 0.0]],
/// ];
/// let trajectories_b = vec![
///     vec![[0.5, 0.5], [1.5, 1.5]],
/// ];
///
/// let metric = Metric::new(
///     DistanceAlgorithm::SSPD,
///     DistanceType::Euclidean
/// );
/// let distances = cdist(&trajectories_a, &trajectories_b, &metric, true).unwrap();
/// ```
pub fn cdist<T>(
    trajectories_a: &[T],
    trajectories_b: &[T],
    metric: &Metric,
    parallel: bool,
) -> Result<Vec<f64>, TrajDistError>
where
    T: CoordSequence + Sync,
{
    let n_a = trajectories_a.len();
    let n_b = trajectories_b.len();

    if n_a == 0 {
        return Err(TrajDistError::InvalidParams(
            "cdist requires at least 1 trajectory in the first collection".to_string(),
        ));
    }

    if n_b == 0 {
        return Err(TrajDistError::InvalidParams(
            "cdist requires at least 1 trajectory in the second collection".to_string(),
        ));
    }

    let mut distances = vec![0.0; n_a * n_b];

    // Use Rayon's global thread pool for parallel processing
    #[cfg(feature = "parallel")]
    {
        if parallel {
            compute_cdist_parallel(trajectories_a, trajectories_b, &mut distances, metric);
        } else {
            compute_cdist_sequential(trajectories_a, trajectories_b, &mut distances, metric);
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = parallel; // Suppress unused variable warning when parallel feature is disabled
        compute_cdist_sequential(trajectories_a, trajectories_b, &mut distances, metric);
    }

    Ok(distances)
}

/// Compute cdist distances sequentially
fn compute_cdist_sequential<T: CoordSequence>(
    trajectories_a: &[T],
    trajectories_b: &[T],
    distances: &mut [f64],
    metric: &Metric,
) {
    let n_b = trajectories_b.len();

    for (i, traj_a) in trajectories_a.iter().enumerate() {
        for (j, traj_b) in trajectories_b.iter().enumerate() {
            let idx = i * n_b + j;
            distances[idx] = metric.distance(traj_a, traj_b);
        }
    }
}

/// Compute cdist distances in parallel
///
/// Uses Rayon's global thread pool and `par_chunks_mut` for efficient
/// in-place matrix filling.
#[cfg(feature = "parallel")]
fn compute_cdist_parallel<T: CoordSequence + Sync>(
    trajectories_a: &[T],
    trajectories_b: &[T],
    distances: &mut [f64],
    metric: &Metric,
) {
    let n_b = trajectories_b.len();

    distances
        .par_chunks_mut(n_b)
        .enumerate()
        .for_each(|(i, row)| {
            for (j, dist) in row.iter_mut().enumerate() {
                *dist = metric.distance(&trajectories_a[i], &trajectories_b[j]);
            }
        });
}
