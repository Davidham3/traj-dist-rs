/// Trait for coordinate representation
/// 
/// This trait defines the interface for accessing x and y coordinates
/// of a point in 2D space. It is used by trajectory algorithms to
/// uniformly access coordinate values regardless of the underlying
/// coordinate representation.
/// 
/// ## Example
/// 
/// ```rust
/// use traj_dist_rs::AsCoord;
/// 
/// let point = [1.0, 2.0];
/// assert_eq!(point.x(), 1.0);
/// assert_eq!(point.y(), 2.0);
/// ```
pub trait AsCoord {
    /// Get the x-coordinate (longitude or easting)
    /// 
    /// Returns the x-coordinate value of the point.
    fn x(&self) -> f64;

    /// Get the y-coordinate (latitude or northing)
    /// 
    /// Returns the y-coordinate value of the point.
    fn y(&self) -> f64;
}

/// Trait for coordinate sequence representation
/// 
/// This trait defines the interface for accessing sequences of coordinates,
/// such as trajectories or paths. It provides methods to get the length
/// of the sequence and access individual coordinates by index.
/// 
/// ## Example
/// 
/// ```rust
/// use traj_dist_rs::{CoordSequence, AsCoord};
/// 
/// let trajectory = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
/// assert_eq!(trajectory.len(), 3);
/// assert_eq!(trajectory.get(1).x(), 1.0);
/// assert_eq!(trajectory.get(1).y(), 1.0);
/// ```
pub trait CoordSequence {
    /// The type of coordinate in this sequence
    type Coord: AsCoord;

    /// Get the number of coordinates in the sequence
    /// 
    /// Returns the total number of coordinate points in the sequence.
    fn len(&self) -> usize;

    /// Check if the sequence is empty
    /// 
    /// Returns `true` if the sequence contains no coordinates, `false` otherwise.
    /// This is implemented as `self.len() == 0` by default.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the i-th coordinate in the sequence
    /// 
    /// Returns a copy of the coordinate at the specified index.
    /// 
    /// # Panics
    /// 
    /// This method may panic if the index is out of bounds.
    fn get(&self, i: usize) -> Self::Coord;
}

/// Implementation of `AsCoord` for 2-element array of f64
/// 
/// This implementation allows `[f64; 2]` arrays to be used as coordinates.
/// The first element is treated as x-coordinate and the second as y-coordinate.
impl AsCoord for [f64; 2] {
    fn x(&self) -> f64 {
        self[0]
    }

    fn y(&self) -> f64 {
        self[1]
    }
}

/// Implementation of `CoordSequence` for vector of 2-element f64 arrays
/// 
/// This implementation allows `Vec<[f64; 2]>` to be used as a sequence of coordinates,
/// which is a common representation for trajectories.
impl CoordSequence for Vec<[f64; 2]> {
    type Coord = [f64; 2];

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn get(&self, i: usize) -> Self::Coord {
        self.as_slice()[i]
    }
}
