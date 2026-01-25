use crate::traits::{AsCoord, CoordSequence};

const RAD: f64 = std::f64::consts::PI / 180.0;
const R: f64 = 6378137.0; // Earth radius in meters

/// Compute the great circle distance between two points
pub fn great_circle_distance<C: AsCoord>(p1: &C, p2: &C) -> f64 {
    let lat1 = p1.y();
    let lon1 = p1.x();
    let lat2 = p2.y();
    let lon2 = p2.x();

    let dlat = RAD * (lat2 - lat1);
    let dlon = RAD * (lon2 - lon1);
    let a = (dlat / 2.0).sin().powi(2)
        + (RAD * lat1).cos() * (RAD * lat2).cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    R * c
}

/// Compute pairwise great circle distances between two trajectories
pub fn great_circle_distance_traj<T: CoordSequence>(t1: &T, t2: &T) -> Vec<f64>
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
            mdist[i * l_t2 + j] = great_circle_distance(&coord1, &coord2);
        }
    }

    mdist
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
}
