use numpy::PyReadonlyArray2;
use pyo3::{
    Bound, PyAny,
    types::{PyAnyMethods, PyList},
};

use crate::traits::CoordSequence;
use crate::{err::TrajDistError, traits::AsCoord};

use super::numpy::TrajectoryRef;

/// Holds a reference to the underlying Rust slice and the point index, achieving true zero-copy access
#[derive(Debug, Clone, Copy)]
pub struct PointRef<'a> {
    data: &'a [f64],
    idx: usize,
}

impl<'a> PointRef<'a> {
    pub fn new(data: &'a [f64], idx: usize) -> Self {
        Self { data, idx }
    }
}

impl<'a> AsCoord for PointRef<'a> {
    #[inline(always)]
    fn x(&self) -> f64 {
        self.data[self.idx * 2]
    }

    #[inline(always)]
    fn y(&self) -> f64 {
        self.data[self.idx * 2 + 1]
    }
}

#[derive(Debug)]
pub enum PyTrajectoryType<'a> {
    /// Use TrajectoryRef to achieve zero-copy access to NumPy arrays
    Numpy(TrajectoryRef<'a>),

    /// Data copied from Python List
    Owned(Vec<[f64; 2]>),
}

impl<'a> TryFrom<&Bound<'a, PyAny>> for PyTrajectoryType<'a> {
    type Error = TrajDistError;

    fn try_from(seq: &Bound<'a, PyAny>) -> Result<Self, Self::Error> {
        // Try to convert to numpy array first
        if let Ok(readonly_array) = seq.extract::<PyReadonlyArray2<'a, f64>>() {
            // Use TrajectoryRef to wrap for zero-copy access
            let traj_ref = TrajectoryRef::new(readonly_array)?;
            Ok(Self::Numpy(traj_ref))
        }
        // Otherwise try to convert to Python List
        else if let Ok(loc_list) = seq.downcast::<PyList>() {
            let owned_vec = loc_list.extract::<Vec<[f64; 2]>>()
                .map_err(|_| TrajDistError::DataConvertionError(
                    "Failed to extract List[List[float]] into Vec<[f64; 2]>. Check list structure and element types.".to_string()
                ))?;
            Ok(Self::Owned(owned_vec))
        } else {
            Err(TrajDistError::DataConvertionError(format!(
                "Input must be a 2D numpy.ndarray of float64 or a List[List[float]], not a {:?}",
                seq.get_type()
            )))
        }
    }
}

impl<'a> CoordSequence for PyTrajectoryType<'a> {
    type Coord = PointRef<'a>;

    fn len(&self) -> usize {
        match self {
            PyTrajectoryType::Numpy(traj_ref) => traj_ref.len(),
            PyTrajectoryType::Owned(vec) => vec.len(),
        }
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> Self::Coord {
        match self {
            PyTrajectoryType::Numpy(traj_ref) => traj_ref.get(idx),
            PyTrajectoryType::Owned(vec) => {
                // Get slice reference directly from Vec
                let data = unsafe {
                    std::slice::from_raw_parts(vec.as_ptr() as *const f64, vec.len() * 2)
                };
                PointRef::new(data, idx)
            }
        }
    }
}
