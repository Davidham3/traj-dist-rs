use crate::traits::CoordSequence;
use pyo3::prelude::*;
use std::convert::TryFrom;

/// Enum representing a trajectory that can be converted from Python objects
/// 
/// This enum provides a way to represent trajectory data that originates from
/// Python, allowing for efficient processing in Rust while maintaining
/// compatibility with Python's data structures.
#[derive(Debug, Clone)]
pub enum PyTrajectory {
    /// A trajectory represented as a vector of 2D coordinate pairs
    /// 
    /// Each coordinate pair is represented as [longitude, latitude] or [x, y].
    List(Vec<[f64; 2]>),
}

impl CoordSequence for PyTrajectory {
    type Coord = [f64; 2];

    fn len(&self) -> usize {
        match self {
            PyTrajectory::List(v) => v.len(),
        }
    }

    fn get(&self, i: usize) -> Self::Coord {
        match self {
            PyTrajectory::List(v) => v[i],
        }
    }
}

/// Try to convert a Python object to PyTrajectory
/// 
/// This implementation allows converting Python lists of coordinate pairs into
/// the PyTrajectory enum, which can then be processed using the CoordSequence
/// trait. The Python object is expected to be a list of 2-element lists/tuples
/// representing coordinates.
/// 
/// # Example
/// 
/// In Python:
/// ```python
/// trajectory = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
/// # This can be converted to PyTrajectory
/// ```
impl<'a> TryFrom<&Bound<'a, PyAny>> for PyTrajectory {
    type Error = PyErr;

    fn try_from(obj: &Bound<'a, PyAny>) -> Result<Self, Self::Error> {
        // Try to extract as a list of lists
        let py_list: Vec<Vec<f64>> = obj.extract().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected a list of [longitude, latitude] pairs",
            )
        })?;

        // Convert to Vec<[f64; 2]>
        let mut trajectory = Vec::with_capacity(py_list.len());
        for (i, point) in py_list.iter().enumerate() {
            if point.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Point at index {} should have exactly 2 elements [longitude, latitude], found {}",
                    i,
                    point.len()
                )));
            }
            trajectory.push([point[0], point[1]]);
        }

        Ok(PyTrajectory::List(trajectory))
    }
}
