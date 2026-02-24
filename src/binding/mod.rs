#[cfg(feature = "python-binding")]
pub mod batch;

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
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

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

    /// Pickle serialization support using __reduce__
    ///
    /// Uses bincode to serialize the entire DpResult::inner as bytes for better performance.
    /// Returns a tuple (callable, args) that pickle can use to reconstruct the object.
    fn __reduce__(&self, py: Python) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        use pyo3::prelude::*;
        use pyo3::types::{PyBytes, PyTuple};

        // Import the module and get the helper function
        let module = py.import("traj_dist_rs")?;
        let helper_func = module.getattr("__dp_result_from_pickle")?;

        // Serialize the entire DpResult using bincode
        let serialized =
            bincode::encode_to_vec(&self.inner, bincode::config::standard()).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e))
            })?;

        // Create args tuple containing the serialized bytes
        let bytes_py = PyBytes::new(py, &serialized);
        let args_tuple = PyTuple::new(py, [bytes_py.as_any()])?;

        // Return (helper_func, args, state) where state is None
        Ok((helper_func.unbind(), args_tuple.unbind().into(), py.None()))
    }
}

/// Helper function to create DpResult from pickle data
///
/// Deserializes the DpResult from bincode-encoded bytes.
#[cfg(feature = "python-binding")]
#[gen_stub_pyfunction]
#[pyfunction]
pub fn __dp_result_from_pickle(
    #[gen_stub(override_type(type_repr = "bytes"))] data: &[u8],
) -> PyResult<PyDpResult> {
    bincode::decode_from_slice(data, bincode::config::standard())
        .map(|(dp_result, _)| PyDpResult { inner: dp_result })
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Deserialization failed: {}", e))
        })
}
