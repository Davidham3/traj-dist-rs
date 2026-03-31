//! # Spherical Distance Module
//!
//! This module provides spherical (great-circle) distance calculations for geographic coordinates.
//!
//! ## Functions
//!
//! - `great_circle_distance`: Distance between two geographic points using Haversine formula
//! - `point_to_path`: Minimum distance from a point to a great circle path
//! - `point_to_path_simple`: Simplified version that computes distances internally
//!
//! ## Formula
//!
//! Uses the Haversine formula to calculate distance on a sphere:
//! ```text
//! a = sin²(Δlat/2) + cos(lat₁) × cos(lat₂) × sin²(Δlon/2)
//! c = 2 × atan2(√a, √(1-a))
//! d = R × c
//! ```
//!
//! Where R is Earth's radius (6,378,137 meters).
//!
//! ## Usage
//!
//! ```rust
//! use traj_dist_rs::distance::spherical::great_circle_distance;
//!
//! // New York to London
//! let nyc = [-74.006, 40.7128];
//! let london = [-0.1278, 51.5074];
//! let dist = great_circle_distance(&nyc, &london);
//! // Approximately 5570 km
//! ```
//!
//! ## Coordinate Format
//!
//! Coordinates should be in `[longitude, latitude]` format in degrees.
//! - Longitude: -180 to 180 (positive east, negative west)
//! - Latitude: -90 to 90 (positive north, negative south)
//!
//! ## Performance Optimization
//!
//! For algorithms that compute many distances between points in two trajectories,
//! use `SphericalTrajectoryCache` to precompute intermediate values (lat/lon in radians
//! and cos(lat)). This can reduce trigonometric computations by 50-70%.

use crate::traits::{AsCoord, CoordSequence};

const RAD: f64 = std::f64::consts::PI / 180.0;
const R: f64 = 6378137.0; // Earth radius in meters

/// Cache for spherical distance calculations on a single trajectory
///
/// Precomputes longitude/latitude in radians and cos(latitude) for each point,
/// eliminating redundant trigonometric calculations in algorithms like
/// DTW, LCSS, EDR, ERP, Hausdorff, and SSPD.
///
/// # Example
///
/// ```rust
/// use traj_dist_rs::distance::spherical::SphericalTrajectoryCache;
///
/// let trajectory = vec![[0.0, 40.0], [1.0, 41.0], [2.0, 42.0]];
/// let cache = SphericalTrajectoryCache::from_trajectory(&trajectory);
/// ```
pub struct SphericalTrajectoryCache {
    /// Longitude in radians
    lng_rad: Vec<f64>,
    /// Latitude in radians
    lat_rad: Vec<f64>,
    /// cos(latitude in radians)
    cos_lat: Vec<f64>,
}

impl SphericalTrajectoryCache {
    /// Create a new cache from a trajectory
    ///
    /// Precomputes all intermediate values needed for Haversine calculations.
    pub fn from_trajectory<T: CoordSequence>(traj: &T) -> Self
    where
        T::Coord: AsCoord,
    {
        let n = traj.len();
        let mut lng_rad = Vec::with_capacity(n);
        let mut lat_rad = Vec::with_capacity(n);
        let mut cos_lat = Vec::with_capacity(n);

        for i in 0..n {
            let p = traj.get(i);
            let lat_r = RAD * p.y();
            lng_rad.push(RAD * p.x());
            lat_rad.push(lat_r);
            cos_lat.push(lat_r.cos());
        }

        Self {
            lng_rad,
            lat_rad,
            cos_lat,
        }
    }

    /// Get the length of the cached trajectory
    pub fn len(&self) -> usize {
        self.lat_rad.len()
    }

    /// Check if the cached trajectory is empty
    pub fn is_empty(&self) -> bool {
        self.lat_rad.is_empty()
    }
}

/// Compute the great circle distance between two points using precomputed caches
///
/// Uses the Haversine formula with precomputed values for optimal performance.
/// This is significantly faster than `great_circle_distance` when computing
/// many distances between points in the same trajectories.
#[inline]
pub fn great_circle_distance_cached(
    cache1: &SphericalTrajectoryCache,
    i: usize,
    cache2: &SphericalTrajectoryCache,
    j: usize,
) -> f64 {
    let lat1 = cache1.lat_rad[i];
    let lon1 = cache1.lng_rad[i];
    let lat2 = cache2.lat_rad[j];
    let lon2 = cache2.lng_rad[j];

    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let sin_dlat_half = (dlat * 0.5).sin();
    let sin_dlon_half = (dlon * 0.5).sin();
    let a = sin_dlat_half * sin_dlat_half
        + cache1.cos_lat[i] * cache2.cos_lat[j] * sin_dlon_half * sin_dlon_half;
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    R * c
}

/// Compute the great circle distance between two points
///
/// Uses the Haversine formula to calculate distance on a sphere.
pub fn great_circle_distance<C: AsCoord, D: AsCoord>(p1: &C, p2: &D) -> f64 {
    let lat1: f64 = p1.y();
    let lon1 = p1.x();
    let lat2: f64 = p2.y();
    let lon2 = p2.x();

    let dlat = RAD * (lat2 - lat1);
    let dlon = RAD * (lon2 - lon1);
    let sin_dlat_half = (dlat * 0.5).sin();
    let sin_dlon_half = (dlon * 0.5).sin();
    let cos_lat1 = (RAD * lat1).cos();
    let cos_lat2 = (RAD * lat2).cos();
    let a = sin_dlat_half * sin_dlat_half + cos_lat1 * cos_lat2 * sin_dlon_half * sin_dlon_half;
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    R * c
}

/// Compute the initial bearing from point 1 to point 2
fn initial_bearing<C: AsCoord>(p1: &C, p2: &C) -> f64 {
    let lat1 = p1.y();
    let lon1 = p1.x();
    let lat2 = p2.y();
    let lon2 = p2.x();

    let dlon = RAD * (lon2 - lon1);
    let y = dlon.sin() * (RAD * lat2).cos();
    let x = (RAD * lat1).cos() * (RAD * lat2).sin()
        - (RAD * lat1).sin() * (RAD * lat2).cos() * dlon.cos();
    y.atan2(x)
}

/// Cross-track distance from point 3 to the great circle path from point 1 to point 2
fn cross_track_distance<C: AsCoord>(p1: &C, p2: &C, p3: &C, d13: f64) -> f64 {
    let theta13 = initial_bearing(p1, p3);
    let theta12 = initial_bearing(p1, p2);

    ((d13 / R).sin() * (theta13 - theta12).sin()).asin() * R
}

/// Along-track distance from the start point to the closest point on the path
fn along_track_distance(crt: f64, d13: f64) -> f64 {
    ((d13 / R).cos() / (crt / R).cos()).acos() * R
}

/// Point to path distance between point 3 and path from point 1 to point 2
pub fn point_to_path<C: AsCoord>(p1: &C, p2: &C, p3: &C, d13: f64, d23: f64, d12: f64) -> f64 {
    let crt = cross_track_distance(p1, p2, p3, d13);
    let d1p = along_track_distance(crt, d13);
    let d2p = along_track_distance(crt, d23);

    if d1p > d12 || d2p > d12 {
        d13.min(d23)
    } else {
        crt.abs()
    }
}

/// Point to path distance between point 3 and path from point 1 to point 2 (computes distances internally)
pub fn point_to_path_simple<C: AsCoord>(p1: &C, p2: &C, p3: &C) -> f64 {
    let d13 = great_circle_distance(p1, p3);
    let d23 = great_circle_distance(p2, p3);
    let d12 = great_circle_distance(p1, p2);
    point_to_path(p1, p2, p3, d13, d23, d12)
}

// ============================================================================
// Cached versions for SSPD and Hausdorff algorithms
// ============================================================================

/// Compute the initial bearing from point 1 to point 2 using cached values
#[inline]
fn initial_bearing_cached(
    cache1: &SphericalTrajectoryCache,
    i: usize,
    cache2: &SphericalTrajectoryCache,
    j: usize,
) -> f64 {
    let lat1 = cache1.lat_rad[i];
    let lon1 = cache1.lng_rad[i];
    let lat2 = cache2.lat_rad[j];
    let lon2 = cache2.lng_rad[j];

    let dlon = lon2 - lon1;
    let y = dlon.sin() * cache2.cos_lat[j];
    let x = cache1.cos_lat[i] * lat2.sin() - lat1.sin() * cache2.cos_lat[j] * dlon.cos();
    y.atan2(x)
}

/// Cross-track distance using cached values
#[inline]
fn cross_track_distance_cached(
    cache1: &SphericalTrajectoryCache,
    i: usize,
    cache2: &SphericalTrajectoryCache,
    j: usize,
    cache3: &SphericalTrajectoryCache,
    k: usize,
    d13: f64,
) -> f64 {
    let theta13 = initial_bearing_cached(cache1, i, cache3, k);
    let theta12 = initial_bearing_cached(cache1, i, cache2, j);

    ((d13 / R).sin() * (theta13 - theta12).sin()).asin() * R
}

/// Point to path distance using cached values
///
/// Computes the distance from point k in cache3 to the path from point i to point j
/// in cache1-cache2. The distances d_ik, d_jk, and d_ij must be precomputed.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn point_to_path_cached(
    cache1: &SphericalTrajectoryCache,
    i: usize,
    cache2: &SphericalTrajectoryCache,
    j: usize,
    cache3: &SphericalTrajectoryCache,
    k: usize,
    d_ik: f64,
    d_jk: f64,
    d_ij: f64,
) -> f64 {
    let crt = cross_track_distance_cached(cache1, i, cache2, j, cache3, k, d_ik);
    let d1p = along_track_distance(crt, d_ik);
    let d2p = along_track_distance(crt, d_jk);

    if d1p > d_ij || d2p > d_ij {
        d_ik.min(d_jk)
    } else {
        crt.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_great_circle_distance() {
        // Distance from New York to London
        let p1: [f64; 2] = [-74.006, 40.7128];
        let p2: [f64; 2] = [-0.1278, 51.5074];
        let dist = great_circle_distance(&p1, &p2);
        // Expected distance is approximately 5570 km
        assert!((dist - 5570000.0).abs() < 10000.0);
    }

    #[test]
    fn test_point_to_path() {
        // Point to path distance test
        let p1: [f64; 2] = [0.0, 0.0];
        let p2: [f64; 2] = [10.0, 0.0];
        let p3: [f64; 2] = [5.0, 1.0];

        let dist = point_to_path_simple(&p1, &p2, &p3);
        // The point (5, 1) should be approximately 111 km from the path (0,0) to (10,0)
        assert!((dist - 111195.0).abs() < 1000.0);
    }

    #[test]
    fn test_spherical_cache_basic() {
        let traj: Vec<[f64; 2]> = vec![[0.0, 40.0], [1.0, 41.0], [2.0, 42.0]];
        let cache = SphericalTrajectoryCache::from_trajectory(&traj);

        assert_eq!(cache.len(), 3);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cached_vs_uncached_distance() {
        let traj1: Vec<[f64; 2]> = vec![[-74.006, 40.7128], [-73.9857, 40.7484]];
        let traj2: Vec<[f64; 2]> = vec![[-0.1278, 51.5074], [-0.1676, 51.5552]];

        let cache1 = SphericalTrajectoryCache::from_trajectory(&traj1);
        let cache2 = SphericalTrajectoryCache::from_trajectory(&traj2);

        // Compare cached vs uncached distances
        for i in 0..traj1.len() {
            for j in 0..traj2.len() {
                let dist_uncached = great_circle_distance(&traj1[i], &traj2[j]);
                let dist_cached = great_circle_distance_cached(&cache1, i, &cache2, j);
                assert!(
                    (dist_uncached - dist_cached).abs() < 1e-6,
                    "Mismatch at ({}, {}): uncached={}, cached={}",
                    i,
                    j,
                    dist_uncached,
                    dist_cached
                );
            }
        }
    }

    #[test]
    fn test_empty_trajectory_cache() {
        let traj: Vec<[f64; 2]> = vec![];
        let cache = SphericalTrajectoryCache::from_trajectory(&traj);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
