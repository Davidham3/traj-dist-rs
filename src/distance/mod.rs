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
