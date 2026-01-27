/// Enum representing the type of distance to use for trajectory calculations
///
/// This enum specifies whether to use Euclidean (Cartesian) or Spherical
/// (Great circle/Haversine) distance calculations in trajectory algorithms.
///
/// ## Examples
///
/// ```rust
/// use traj_dist_rs::distance::distance_type::DistanceType;
///
/// let euclidean = DistanceType::Euclidean;
/// let spherical = DistanceType::Spherical;
///
/// // Parse from string
/// let parsed = DistanceType::parse_distance_type("euclidean").unwrap();
/// assert_eq!(parsed, DistanceType::Euclidean);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Parse distance type from string
    ///
    /// Converts a string representation to a `DistanceType` variant.
    /// The comparison is case-insensitive.
    ///
    /// # Arguments
    ///
    /// * `s` - A string slice that should be either "euclidean" or "spherical"
    ///
    /// # Returns
    ///
    /// * `Ok(DistanceType)` - If the string matches a valid variant
    /// * `Err(String)` - If the string doesn't match any variant
    ///
    /// # Example
    ///
    /// ```rust
    /// use traj_dist_rs::distance::distance_type::DistanceType;
    ///
    /// assert_eq!(DistanceType::parse_distance_type("euclidean").unwrap(), DistanceType::Euclidean);
    /// assert_eq!(DistanceType::parse_distance_type("SPHERICAL").unwrap(), DistanceType::Spherical);
    /// assert!(DistanceType::parse_distance_type("invalid").is_err());
    /// ```
    pub fn parse_distance_type(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "euclidean" => Ok(DistanceType::Euclidean),
            "spherical" => Ok(DistanceType::Spherical),
            _ => Err(format!(
                "Invalid distance type '{}'. Expected 'euclidean' or 'spherical'",
                s
            )),
        }
    }
}
