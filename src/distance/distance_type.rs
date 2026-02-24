use crate::{
    distance::{euclidean::euclidean_distance, spherical::great_circle_distance},
    traits::AsCoord,
};

/// Enum representing the type of distance to use for trajectory calculations
///
/// This enum specifies whether to use Euclidean (Cartesian) or Spherical
/// (Great circle/Haversine) distance calculations in trajectory algorithms.
///
/// ## Examples
///
/// ```rust
/// use traj_dist_rs::distance::distance_type::DistanceType;
/// use std::str::FromStr;
///
/// let euclidean = DistanceType::Euclidean;
/// let spherical = DistanceType::Spherical;
///
/// // Parse from string (case-insensitive)
/// let parsed = DistanceType::from_str("euclidean").unwrap();
/// assert_eq!(parsed, DistanceType::Euclidean);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, strum_macros::EnumString, strum_macros::Display)]
#[strum(serialize_all = "snake_case")]
pub enum DistanceType {
    /// Euclidean distance (2D Cartesian space)
    ///
    /// Uses standard Euclidean distance formula: √[(x₂-x₁)² + (y₂-y₁)²]
    /// Suitable for coordinates in a Cartesian plane.
    Euclidean,
    /// Spherical distance (Great circle distance on Earth)
    ///
    /// Uses Haversine formula to calculate distance on a sphere.
    /// Suitable for geographic coordinates (latitude/longitude).
    Spherical,
}

impl DistanceType {
    pub fn distance<C: AsCoord, D: AsCoord>(&self, p1: &C, p2: &D) -> f64 {
        match self {
            DistanceType::Euclidean => euclidean_distance(p1, p2),
            DistanceType::Spherical => great_circle_distance(p1, p2),
        }
    }
}
