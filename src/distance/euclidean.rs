//! # Euclidean Distance Module
//!
//! This module provides Euclidean (Cartesian) distance calculations for 2D coordinates.
//!
//! ## Functions
//!
//! - `euclidean_distance`: Distance between two points
//! - `euclidean_distance_traj`: Pairwise distances between all points in two trajectories
//! - `point_to_segment`: Minimum distance from a point to a line segment
//! - `point_to_trajectory`: Minimum distance from a point to any segment of a trajectory
//!
//! ## Formula
//!
//! Euclidean distance between points (x₁, y₁) and (x₂, y₂):
//! ```text
//! d = √[(x₂-x₁)² + (y₂-y₁)²]
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use traj_dist_rs::distance::euclidean::euclidean_distance;
//!
//! let p1 = [0.0, 0.0];
//! let p2 = [3.0, 4.0];
//! let dist = euclidean_distance(&p1, &p2);
//! assert_eq!(dist, 5.0);
//! ```

use crate::traits::{AsCoord, CoordSequence};

/// Euclidean distance between two points
///
/// Uses the standard Euclidean distance formula: √[(x₂-x₁)² + (y₂-y₁)²]
pub fn euclidean_distance<C: AsCoord, D: AsCoord>(p1: &C, p2: &D) -> f64 {
    let dx = p1.x() - p2.x();
    let dy = p1.y() - p2.y();
    (dx * dx + dy * dy).sqrt()
}

/// Compute pairwise Euclidean distances between two trajectories
pub fn euclidean_distance_traj<T: CoordSequence>(t1: &T, t2: &T) -> Vec<f64>
where
    T::Coord: AsCoord,
{
    let l_t1 = t1.len();
    let l_t2 = t2.len();
    let mut mdist = vec![0.0; l_t1 * l_t2];

    for i in 0..l_t1 {
        let coord1 = t1.get(i);
        for j in 0..l_t2 {
            let coord2 = t2.get(j);
            mdist[i * l_t2 + j] = euclidean_distance(&coord1, &coord2);
        }
    }

    mdist
}

/// Point to segment distance
pub fn point_to_segment<C: AsCoord>(
    point: &C,
    seg_start: &C,
    seg_end: &C,
    d_start: f64,
    d_end: f64,
    seg_len: f64,
) -> f64 {
    if seg_len == 0.0 {
        return d_start;
    }

    // Project point onto the line segment
    let dx = seg_end.x() - seg_start.x();
    let dy = seg_end.y() - seg_start.y();

    let t =
        ((point.x() - seg_start.x()) * dx + (point.y() - seg_start.y()) * dy) / (seg_len * seg_len);

    if t <= 0.00001 || t >= 1.0 {
        // closest point does not fall within the line segment, take the shorter distance to an endpoint
        d_start.min(d_end)
    } else {
        // Intersecting point is on the line, use the formula
        let proj_x = seg_start.x() + t * dx;
        let proj_y = seg_start.y() + t * dy;
        let dx2 = point.x() - proj_x;
        let dy2 = point.y() - proj_y;
        (dx2 * dx2 + dy2 * dy2).sqrt()
    }
}

/// Project a point onto a line segment and return the projected point coordinates
///
/// This function projects a point onto a line segment defined by two endpoints.
/// If the projection falls outside the segment, it returns the nearest endpoint.
///
/// # Arguments
///
/// * `point` - The point to project
/// * `seg_start` - The start point of the segment
/// * `seg_end` - The end point of the segment
///
/// # Returns
///
/// A tuple `(x, y)` representing the projected point coordinates
///
/// # Examples
///
/// ```rust
/// use traj_dist_rs::distance::euclidean::project_point_to_segment;
///
/// let point = [0.0, 1.0];
/// let seg_start = [0.0, 0.0];
/// let seg_end = [2.0, 0.0];
///
/// let projected = project_point_to_segment(&point, &seg_start, &seg_end);
/// assert_eq!(projected, (0.0, 0.0)); // Projects onto the segment at x=0
/// ```
pub fn project_point_to_segment<C: AsCoord, D: AsCoord, E: AsCoord>(
    point: &C,
    seg_start: &D,
    seg_end: &E,
) -> (f64, f64) {
    let dx = seg_end.x() - seg_start.x();
    let dy = seg_end.y() - seg_start.y();

    let l2 = dx * dx + dy * dy;

    if l2 == 0.0 {
        // Segment is degenerate (zero length), return the projected point itself
        // This matches the Python _line_map behavior: when l2==0, return p (the point being projected)
        return (point.x(), point.y());
    }

    // Compute projection parameter t
    let t = ((point.x() - seg_start.x()) * dx + (point.y() - seg_start.y()) * dy) / l2;

    if t < 0.0 {
        // Projection falls before the segment, return start point
        (seg_start.x(), seg_start.y())
    } else if t > 1.0 {
        // Projection falls after the segment, return end point
        (seg_end.x(), seg_end.y())
    } else {
        // Projection falls within the segment, compute projected point
        let proj_x = seg_start.x() + t * dx;
        let proj_y = seg_start.y() + t * dy;
        (proj_x, proj_y)
    }
}

/// Point to trajectory distance (minimum distance from point to any segment of trajectory)
pub fn point_to_trajectory<T: CoordSequence>(
    point: &T::Coord,
    trajectory: &T,
    mdist_p: &[f64],
    t_dist: &[f64],
    l_t: usize,
) -> f64
where
    T::Coord: AsCoord,
{
    if l_t == 0 {
        return f64::MAX;
    }

    if l_t == 1 {
        // Single point trajectory: distance is the distance to that point
        return mdist_p[0];
    }

    let mut min_dist = f64::MAX;

    for i in 0..(l_t - 1) {
        let seg_start = trajectory.get(i);
        let seg_end = trajectory.get(i + 1);

        // Use pre-calculated distances
        let d_start = mdist_p[i];
        let d_end = mdist_p[i + 1];
        let seg_len = t_dist[i];

        let d = point_to_segment(point, &seg_start, &seg_end, d_start, d_end, seg_len);
        min_dist = min_dist.min(d);
    }

    min_dist
}

/// Point to trajectory distance (minimum distance from point to any segment of trajectory)
pub fn point_to_trajectory_simple<T: CoordSequence>(trajectory: &T, point: &T::Coord) -> f64
where
    T::Coord: AsCoord,
{
    let l_t = trajectory.len();

    if l_t < 2 {
        return f64::MAX;
    }

    let mut min_dist = f64::MAX;

    for i in 0..(l_t - 1) {
        let p1 = trajectory.get(i);
        let p2 = trajectory.get(i + 1);
        let d_start = euclidean_distance(point, &p1);
        let d_end = euclidean_distance(point, &p2);
        let seg_len = euclidean_distance(&p1, &p2);

        let d = point_to_segment(point, &p1, &p2, d_start, d_end, seg_len);
        min_dist = min_dist.min(d);
    }

    min_dist
}

/// Point to segment distance (simplified version without precomputed parameters)
///
/// Returns the minimum Euclidean distance from a point to a line segment.
/// This is a convenience function that computes all intermediate values internally,
/// unlike `point_to_segment` which requires precomputed distances and segment length.
///
/// Note: Uses threshold `t <= 0.00001` (matching the original traj-dist implementation)
/// to handle floating-point edge cases near segment endpoints.
#[inline]
pub fn point_to_segment_distance<C: AsCoord, D: AsCoord, E: AsCoord>(
    point: &C,
    seg_start: &D,
    seg_end: &E,
) -> f64 {
    let s1x = seg_start.x();
    let s1y = seg_start.y();
    let s2x = seg_end.x();
    let s2y = seg_end.y();

    if s1x == s2x && s1y == s2y {
        return euclidean_distance(point, seg_start);
    }

    let dx = s2x - s1x;
    let dy = s2y - s1y;
    let seg_len_sq = dx * dx + dy * dy;

    let u = ((point.x() - s1x) * dx + (point.y() - s1y) * dy) / seg_len_sq;

    if !(0.00001..=1.0).contains(&u) {
        // Closest point does not fall within the segment
        euclidean_distance(point, seg_start).min(euclidean_distance(point, seg_end))
    } else {
        let proj_x = s1x + u * dx;
        let proj_y = s1y + u * dy;
        let dpx = point.x() - proj_x;
        let dpy = point.y() - proj_y;
        (dpx * dpx + dpy * dpy).sqrt()
    }
}

/// Find intersections between a circle and a line
///
/// Returns the two intersection points of the circle centered at (px, py) with
/// radius `eps` and the line passing through (s1x, s1y) and (s2x, s2y).
///
/// Assumes the intersection exists. Returns two identical points for tangent lines.
#[inline]
pub fn circle_line_intersection(
    px: f64,
    py: f64,
    s1x: f64,
    s1y: f64,
    s2x: f64,
    s2y: f64,
    eps: f64,
) -> [(f64, f64); 2] {
    if s2x == s1x {
        // Vertical line
        let rac = ((eps * eps) - (s1x - px) * (s1x - px)).sqrt();
        [(s1x, py + rac), (s1x, py - rac)]
    } else {
        let m = (s2y - s1y) / (s2x - s1x);
        let c = s2y - m * s2x;
        let a_coeff = m * m + 1.0;
        let b_coeff = 2.0 * (m * c - m * py - px);
        let c_coeff = py * py - eps * eps + px * px - 2.0 * c * py + c * c;
        let delta = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;

        if delta <= 0.0 {
            let x = -b_coeff / (2.0 * a_coeff);
            let y = m * x + c;
            [(x, y), (x, y)]
        } else {
            let sdelta = delta.sqrt();
            let x1 = (-b_coeff + sdelta) / (2.0 * a_coeff);
            let y1 = m * x1 + c;
            let x2 = (-b_coeff - sdelta) / (2.0 * a_coeff);
            let y2 = m * x2 + c;
            [(x1, y1), (x2, y2)]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let p1: [f64; 2] = [0.0, 0.0];
        let p2: [f64; 2] = [3.0, 4.0];
        let dist = euclidean_distance(&p1, &p2);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_point_to_segment() {
        let point: [f64; 2] = [0.0, 1.0];
        let seg_start: [f64; 2] = [0.0, 0.0];
        let seg_end: [f64; 2] = [2.0, 0.0];
        let d_start = 1.0;
        let d_end = 5.0;
        let seg_len = 2.0;

        let dist = point_to_segment(&point, &seg_start, &seg_end, d_start, d_end, seg_len);
        assert!((dist - 1.0).abs() < 1e-6);
    }
}
