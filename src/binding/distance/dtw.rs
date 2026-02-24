use crate::binding::PyDpResult;
use crate::binding::sequence::types::PyTrajectoryType;
use crate::distance::base::TrajectoryCalculator;
use crate::distance::distance_type::DistanceType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};
use std::str::FromStr;

/// Compute the DTW (Dynamic Time Warping) distance between two trajectories
///
/// # Arguments
/// * `t1` - First trajectory (list of [longitude, latitude] pairs)
/// * `t2` - Second trajectory (list of [longitude, latitude] pairs)
/// * `dist_type` - Distance type: "euclidean" or "spherical"
/// * `use_full_matrix` - If true, compute and return the full DP matrix;
///                        if false (default), return None for the matrix to save space
///
/// # Returns
/// * A `DpResult` object with two properties:
///   - `distance`: DTW distance as f64
///   - `matrix`: numpy array of shape (n0+1, n1+1) if use_full_matrix=True, else None
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0]]
/// t2 = [[0.0, 1.0], [1.0, 0.0]]
///
/// # Without matrix
/// result = traj_dist_rs.dtw(t1, t2, "euclidean")
/// print(result.distance)  # Distance value
/// print(result.matrix)  # None
///
/// # With matrix
/// result = traj_dist_rs.dtw(t1, t2, "euclidean", True)
/// print(result.distance)  # Distance value
/// print(result.matrix)  # numpy array
/// ```
#[gen_stub_pyfunction]
#[pyfunction(signature = (t1, t2, dist_type, use_full_matrix=false))]
pub fn dtw<'py>(
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]] | numpy.ndarray", imports=("typing", "numpy")))]
    t1: &Bound<'py, PyAny>,
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]] | numpy.ndarray", imports=("typing", "numpy")))]
    t2: &Bound<'py, PyAny>,
    dist_type: String,
    use_full_matrix: bool,
) -> PyResult<PyDpResult> {
    // Convert Python objects to PyTrajectoryType
    let traj1 = PyTrajectoryType::try_from(t1)?;
    let traj2 = PyTrajectoryType::try_from(t2)?;

    // Parse distance type using FromStr from strum
    let distance_type = DistanceType::from_str(&dist_type).map_err(|_| {
        PyValueError::new_err(format!(
            "Invalid distance type '{}'. Expected 'euclidean' or 'spherical'",
            dist_type
        ))
    })?;

    // Create a distance calculator
    let calculator: TrajectoryCalculator<PyTrajectoryType, PyTrajectoryType> =
        TrajectoryCalculator::new(&traj1, &traj2, distance_type);

    // Compute DTW using the calculator
    let result = crate::distance::dtw::dtw(&calculator, use_full_matrix);

    Ok(PyDpResult { inner: result })
}

/// Compute the DTW (Dynamic Time Warping) distance using a precomputed distance matrix
///
/// This function allows you to use a precomputed distance matrix instead of computing
/// distances between trajectory points on the fly. This can be useful when:
/// - You need to compute multiple distance measures on the same trajectory pair
/// - You want to reuse the same distance matrix across different computations
/// - You have custom distance calculations that don't fit the standard distance types
///
/// # Arguments
/// * `distance_matrix` - A 2D numpy array where matrix[i][j] is the distance between
///                       point i of trajectory 1 and point j of trajectory 2
/// * `use_full_matrix` - If true, compute and return the full DP matrix;
///                        if false (default), return None for the matrix to save space
///
/// # Returns
/// * A `DpResult` object with two properties:
///   - `distance`: DTW distance as f64
///   - `matrix`: numpy array of shape (n0+1, n1+1) if use_full_matrix=True, else None
///
/// # Examples
/// ```python
/// import traj_dist_rs
/// import numpy as np
///
/// t1 = [[0.0, 0.0], [1.0, 1.0]]
/// t2 = [[0.0, 1.0], [1.0, 0.0]]
///
/// # Precompute distance matrix
/// dist_matrix = np.array([
///     [1.0, 1.0],  # distance from t1[0] to t2[0], t2[1]
///     [1.0, 1.0],  # distance from t1[1] to t2[0], t2[1]
/// ])
///
/// # Without matrix
/// result = traj_dist_rs.dtw_with_matrix(dist_matrix)
/// print(result.distance)  # Distance value
/// print(result.matrix)  # None
///
/// # With matrix
/// result = traj_dist_rs.dtw_with_matrix(dist_matrix, True)
/// print(result.distance)  # Distance value
/// print(result.matrix)  # numpy array
/// ```
#[gen_stub_pyfunction]
#[pyfunction(signature = (distance_matrix, use_full_matrix=false))]
pub fn dtw_with_matrix<'py>(
    #[gen_stub(override_type(type_repr="numpy.ndarray", imports=("numpy",)))]
    distance_matrix: &Bound<'py, PyAny>,
    use_full_matrix: bool,
) -> PyResult<PyDpResult> {
    use numpy::{PyArrayMethods, PyUntypedArrayMethods};

    // Convert Python object to NumPy array
    let array = distance_matrix
        .downcast::<numpy::PyArray2<f64>>()
        .map_err(|_| {
            PyValueError::new_err("distance_matrix must be a 2D numpy array of float64 values")
        })?;

    let readonly_array = array.readonly();
    let shape = readonly_array.shape();

    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "distance_matrix must be a 2D array, got shape {:?}",
            shape
        )));
    }

    let n0 = shape[0];
    let n1 = shape[1];

    // Check if array is contiguous (C-order) for zero-copy
    let view = readonly_array.as_array();
    if !view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "distance_matrix must be contiguous (C-order)",
        ));
    }

    // Zero-copy: get raw pointer directly
    let data_ptr = view.as_ptr();
    let distance_matrix_slice = unsafe { std::slice::from_raw_parts(data_ptr, n0 * n1) };

    // Create a precomputed distance calculator with flat array (zero-copy)
    let calculator =
        crate::distance::base::PrecomputedDistanceCalculator::new(distance_matrix_slice, n0, n1);

    // Compute DTW using the calculator
    let result = crate::distance::dtw::dtw(&calculator, use_full_matrix);

    Ok(PyDpResult { inner: result })
}

define_stub_info_gatherer!(stub_info);
