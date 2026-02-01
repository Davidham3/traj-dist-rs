use crate::traits::{AsCoord, CoordSequence};

/// Euclidean distance between two points
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
