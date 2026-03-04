//! # traj-dist-rs
//!
//! `traj-dist-rs` is a high-performance trajectory distance and similarity measurement library implemented in Rust.
//! It provides efficient implementations of various trajectory distance algorithms with both
//! Rust native APIs and Python bindings.
//!
//! ## Overview
//!
//! This crate implements multiple trajectory distance algorithms including:
//!
//! - SSPD (Symmetric Segment-Path Distance)
//! - DTW (Dynamic Time Warping)
//! - Hausdorff Distance
//! - LCSS (Longest Common Subsequence)
//! - EDR (Edit Distance on Real sequence)
//! - ERP (Edit distance with Real Penalty)
//! - Discrete Frechet Distance
//! - Frechet Distance (planned)
//! - SOWD (One-Way Distance) (planned)
//!
//! Trajectory similarity is often measured via trajectory distances such as DTW, LCSS, EDR, ERP, Hausdorff, (Discrete) Fréchet, and SSPD.
//!
//! ## Features
//!
//! - High-performance Rust implementations
//! - Support for both Euclidean and Spherical(Haversine / great-circle) distance calculations
//! - Python bindings via PyO3
//! - Compatible with the original `traj-dist` library
//! - Support for various trajectory distance algorithms
//! - Batch computation functions (pdist, cdist) with parallel processing
//! - Zero-copy NumPy array support for Python bindings
//!
//! ## Similarity vs Distance
//!
//! This library computes trajectory **distances**, which can be easily converted to **similarity scores**:
//!
//! - **Similarity = 1 / (1 + distance)**: Commonly used for similarity scoring
//! - **Similarity = exp(-distance / σ)**: Gaussian kernel similarity, where σ controls the sensitivity
//! - **Normalized LCSS/EDR**: These algorithms can return normalized scores in [0, 1] range
//!
//! For example, if DTW distance is 2.0:
//! ```rust
//! let distance = 2.0;
//! let similarity = 1.0 / (1.0 + distance);  // similarity = 0.333
//! ```
//!
//! ## Use Cases and Applications
//!
//! This library is useful for various trajectory analysis tasks:
//!
//! - **Trajectory similarity search**: Find similar trajectories from a database via kNN and pdist/cdist
//! - **Nearest neighbor retrieval**: Query by example trajectory
//! - **Trajectory clustering**: Group similar trajectories together
//! - **Map-matching preprocessing**: Match GPS traces to road networks
//! - **Anomaly detection**: Identify unusual trajectory patterns
//! - **Route pattern mining**: Discover common movement patterns
//! - **Mobility data analysis**: Analyze GPS traces and movement data
//! - **Time series similarity**: Treat trajectories as 2D time series
//!
//! ## Glossary and Terminology
//!
//! - **Trajectory similarity / distance / dissimilarity**: Related concepts for measuring how alike two trajectories are
//! - **Sequence alignment**: Matching elements between two sequences (e.g., DTW, LCSS)
//! - **Warping / matching**: Non-linear alignment between sequences (DTW concept)
//! - **Spatiotemporal trajectories / GPS traces**: Trajectories with both spatial (location) and temporal (time) information
//!
//! ## Usage
//!
//! ```rust
//! use traj_dist_rs::distance::sspd::sspd;
//! use traj_dist_rs::distance::distance_type::DistanceType;
//!
//! let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
//! let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];
//!
//! let distance = sspd(&traj1, &traj2, DistanceType::Euclidean);
//! println!("SSPD distance: {}", distance);
//!
//! // Convert to similarity score
//! let similarity = 1.0 / (1.0 + distance);
//! println!("SSPD similarity: {}", similarity);
//! ```

pub mod distance;
pub mod err;
pub mod traits;

#[cfg(feature = "python-binding")]
pub mod binding;

#[cfg(feature = "python-binding")]
use pyo3::prelude::*;

/// Trajectory distance calculation library
#[cfg(feature = "python-binding")]
#[pymodule]
fn _lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register the DpResult class
    m.add_class::<crate::binding::PyDpResult>()?;

    // Register the Metric class for batch computation
    m.add_class::<crate::binding::batch::PyMetric>()?;

    // Register batch computation functions
    m.add_function(wrap_pyfunction!(crate::binding::batch::pdist, m)?)?;
    m.add_function(wrap_pyfunction!(crate::binding::batch::cdist, m)?)?;

    // Register distance functions
    m.add_function(wrap_pyfunction!(crate::binding::distance::sspd::sspd, m)?)?;
    m.add_function(wrap_pyfunction!(crate::binding::distance::dtw::dtw, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::distance::dtw::dtw_with_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::distance::hausdorff::hausdorff,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::binding::distance::lcss::lcss, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::distance::lcss::lcss_with_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::binding::distance::edr::edr, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::distance::edr::edr_with_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::distance::discret_frechet::discret_frechet,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::distance::discret_frechet::discret_frechet_with_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::distance::erp::erp_compat_traj_dist,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::distance::erp::erp_compat_traj_dist_with_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::distance::erp::erp_standard,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::distance::erp::erp_standard_with_matrix,
        m
    )?)?;

    // Register helper function for pickle deserialization
    m.add_function(wrap_pyfunction!(
        crate::binding::__dp_result_from_pickle,
        m
    )?)?;

    Ok(())
}
