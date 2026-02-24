use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use std::marker::PhantomData;

use crate::binding::sequence::types::PointRef;
use crate::err::TrajDistError;
use crate::traits::CoordSequence;

/// Holds the underlying data pointer of NumPy array, achieving zero-copy access to the entire trajectory
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryRef<'a> {
    data_ptr: *const f64,
    len: usize,
    _phantom: PhantomData<&'a f64>,
}

// SAFETY: TrajectoryRef only contains a pointer to immutable NumPy array data
// that is guaranteed to be valid for the lifetime 'a. The pointer is never
// dereferenced mutably, so it's safe to share between threads.
unsafe impl<'a> Send for TrajectoryRef<'a> {}
unsafe impl<'a> Sync for TrajectoryRef<'a> {}

impl<'a> TrajectoryRef<'a> {
    pub fn new(array: PyReadonlyArray2<'a, f64>) -> Result<Self, TrajDistError> {
        let shape = array.shape();
        if shape.len() != 2 || shape[1] != 2 {
            return Err(TrajDistError::DataConvertionError(format!(
                "Numpy array must have a shape of (N, 2), but got {:?}",
                shape
            )));
        }
        // Check if array is contiguous
        let view = array.as_array();
        if view.is_standard_layout() {
            Ok(Self {
                data_ptr: view.as_ptr(),
                len: shape[0],
                _phantom: PhantomData,
            })
        } else {
            Err(TrajDistError::DataConvertionError(
                "Numpy array must be contiguous (C-order)".to_string(),
            ))
        }
    }

    /// Get underlying data slice (created using pointer)
    #[inline(always)]
    fn get_data(&self) -> &'a [f64] {
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.len * 2) }
    }
}

impl<'a> CoordSequence for TrajectoryRef<'a> {
    type Coord = PointRef<'a>;

    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> Self::Coord {
        PointRef::new(self.get_data(), idx)
    }
}
