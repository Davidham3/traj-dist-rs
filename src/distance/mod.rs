//! # Distance Algorithms Module
//!
//! This module contains implementations of various trajectory distance algorithms.
//! Each algorithm is implemented in its own submodule and provides both Euclidean
//! and Spherical distance calculations where applicable.
//!
//! ## Available Algorithms
//!
//! - `discret_frechet`: Discrete Fréchet distance calculation
//! - `dtw`: Dynamic Time Warping distance calculation
//! - `hausdorff`: Hausdorff distance calculation
//! - `lcss`: Longest Common Subsequence distance calculation
//! - `edr`: Edit Distance on Real sequence calculation
//! - `erp`: Edit distance with Real Penalty calculation
//! - `sspd`: Symmetric Segment-Path Distance calculation
//!
//! ## Distance Types
//!
//! The algorithms support two types of distance calculations:
//!
//! - Euclidean: Standard 2D Cartesian distance
//! - Spherical: Great circle distance on Earth (Haversine formula)

/// Result type for dynamic programming based distance algorithms
///
/// Contains the computed distance and optionally the full DP matrix.
/// The matrix is flattened as a 1D vector in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct DpResult {
    /// The computed distance value
    pub distance: f64,
    /// The full DP matrix (flattened in row-major order), or None if not computed
    pub matrix: Option<Vec<f64>>,
}

impl std::fmt::Display for DpResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.matrix {
            Some(_) => write!(f, "{}", self.distance),
            None => write!(f, "{}", self.distance),
        }
    }
}

impl DpResult {
    /// Creates a new DpResult with distance but no matrix
    pub fn new(distance: f64) -> Self {
        Self {
            distance,
            matrix: None,
        }
    }

    /// Creates a new DpResult with distance and matrix
    pub fn with_matrix(distance: f64, matrix: Vec<f64>) -> Self {
        Self {
            distance,
            matrix: Some(matrix),
        }
    }
}

pub mod base;
pub mod discret_frechet;
pub mod distance_type;
pub mod dtw;
pub mod edr;
pub mod erp;
pub mod euclidean;
pub mod hausdorff;
pub mod lcss;
pub mod spherical;
pub mod sspd;
pub mod utils;
