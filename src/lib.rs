//! # traj-dist-rs
//!
//! `traj-dist-rs` is a high-performance trajectory distance calculation library implemented in Rust.
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
//! - Discret Frechet Distance
//! - Frechet Distance (planned)
//! - SOWD (One-Way Distance) (planned)
//!
//! ## Features
//!
//! - High-performance Rust implementations
//! - Support for both Euclidean and Spherical distance calculations
//! - Python bindings via PyO3
//! - Compatible with the original `traj-dist` library
//! - Support for various trajectory distance algorithms
//!
//! ## Usage
//!
//! ```rust
//! use traj_dist_rs::{sspd, DistanceType};
//!
//! let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
//! let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];
//!
//! let distance = sspd(&traj1, &traj2, DistanceType::Euclidean);
//! println!("SSPD distance: {}", distance);
//! ```

pub mod distance;
pub mod traits;
pub use traits::*;

// Re-export distance functions for debugging
#[allow(deprecated)]
pub use distance::hausdorff::hausdorff_euclidean;
#[allow(deprecated)]
pub use distance::hausdorff::hausdorff_spherical;

#[cfg(feature = "python-binding")]
pub mod binding;

#[cfg(feature = "python-binding")]
use pyo3::prelude::*;

/// Trajectory distance calculation library
#[cfg(feature = "python-binding")]
#[pymodule]
fn _lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    use crate::binding::distance::discret_frechet::discret_frechet;
    use crate::binding::distance::dtw::dtw;
    use crate::binding::distance::edr::edr;
    use crate::binding::distance::erp::{erp_compat_traj_dist, erp_standard};
    use crate::binding::distance::hausdorff::hausdorff;
    use crate::binding::distance::lcss::lcss;
    use crate::binding::distance::sspd::sspd;
    m.add_function(wrap_pyfunction!(sspd, m)?)?;
    m.add_function(wrap_pyfunction!(dtw, m)?)?;
    m.add_function(wrap_pyfunction!(hausdorff, m)?)?;
    m.add_function(wrap_pyfunction!(lcss, m)?)?;
    m.add_function(wrap_pyfunction!(edr, m)?)?;
    m.add_function(wrap_pyfunction!(discret_frechet, m)?)?;
    m.add_function(wrap_pyfunction!(erp_compat_traj_dist, m)?)?;
    m.add_function(wrap_pyfunction!(erp_standard, m)?)?;
    Ok(())
}
