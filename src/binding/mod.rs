#[cfg(feature = "python-binding")]
pub mod distance;

#[cfg(feature = "python-binding")]
pub mod sequence;

#[cfg(feature = "python-binding")]
pub mod err;

#[cfg(feature = "python-binding")]
use numpy::PyArray1;

#[cfg(feature = "python-binding")]
use pyo3::prelude::*;

#[cfg(feature = "python-binding")]
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Python wrapper for the Rust DpResult struct
///
/// This class wraps the Rust DpResult and provides Python-friendly access
/// to the distance and optional matrix.
#[cfg(feature = "python-binding")]
#[gen_stub_pyclass]
#[pyclass(name = "DpResult")]
pub struct PyDpResult {
    /// The inner Rust DpResult
    pub inner: crate::distance::DpResult,
}

#[cfg(feature = "python-binding")]
#[gen_stub_pymethods]
#[pymethods]
impl PyDpResult {
    /// Get the distance value
    #[getter]
    fn distance(&self) -> f64 {
        self.inner.distance
    }

    /// Get the matrix (or None if not computed)
    ///
    /// Returns a numpy array if use_full_matrix was True, otherwise None
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> Option<Py<PyArray1<f64>>> {
        self.inner.matrix.as_ref().map(|m| {
            let array = PyArray1::from_vec(py, m.clone());
            array.unbind()
        })
    }

    /// String representation
    fn __repr__(&self) -> String {
        match &self.inner.matrix {
            Some(_) => format!(
                "DpResult(distance={}, matrix=<numpy array>)",
                self.inner.distance
            ),
            None => format!("DpResult(distance={}, matrix=None)", self.inner.distance),
        }
    }

    /// String representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
}
