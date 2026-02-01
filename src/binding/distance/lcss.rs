use crate::binding::PyDpResult;
use crate::binding::sequence::types::PyTrajectoryType;
use crate::distance::base::TrajectoryCalculator;
use crate::distance::distance_type::DistanceType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};

/// Compute the LCSS (Longest Common Subsequence) distance between two trajectories
///
/// The LCSS distance is calculated as 1 - (length of longest common subsequence) / min(len(t0), len(t1))
/// where two points are considered matching if their distance is less than eps.
///
/// # Arguments
/// * `t1` - First trajectory (list of [longitude, latitude] pairs)
/// * `t2` - Second trajectory (list of [longitude, latitude] pairs)
/// * `dist_type` - Distance type: "euclidean" or "spherical"
/// * `eps` - Epsilon threshold for matching points
/// * `use_full_matrix` - If true, compute and return the full DP matrix;
///                        if false (default), return None for the matrix to save space
///
/// # Returns
/// * A `DpResult` object with two properties:
///   - `distance`: LCSS distance as f64
///   - `matrix`: numpy array of shape (n0+1, n1+1) if use_full_matrix=True, else None
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0]]
/// t2 = [[0.0, 1.0], [1.0, 0.0]]
/// result = traj_dist_rs.lcss(t1, t2, "euclidean", eps=0.5)
/// print(result.distance)  # Distance value
/// print(result.matrix)  # None
///
/// result = traj_dist_rs.lcss(t1, t2, "euclidean", eps=0.5, use_full_matrix=True)
/// print(result.matrix)  # numpy array
/// ```
#[gen_stub_pyfunction]
#[pyfunction(signature = (t1, t2, dist_type, eps, use_full_matrix=false))]
pub fn lcss(
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]]", imports=("typing")))]
    t1: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]]", imports=("typing")))]
    t2: &Bound<'_, PyAny>,
    dist_type: String,
    eps: f64,
    use_full_matrix: bool,
) -> PyResult<PyDpResult> {
    // Convert Python objects to PyTrajectoryType
    let traj1 = PyTrajectoryType::try_from(t1)?;
    let traj2 = PyTrajectoryType::try_from(t2)?;

    // Compute distance based on type
    let distance_type = match dist_type.to_lowercase().as_str() {
        "euclidean" => DistanceType::Euclidean,
        "spherical" => DistanceType::Spherical,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid distance type '{}'. Expected 'euclidean' or 'spherical'",
                dist_type
            )));
        }
    };

    let calculator = TrajectoryCalculator::new(&traj1, &traj2, distance_type);
    let result = crate::distance::lcss::lcss(&calculator, eps, use_full_matrix);

    Ok(PyDpResult { inner: result })
}

/// Compute the LCSS (Longest Common Subsequence) distance using a precomputed distance matrix
///
/// This function allows you to use a precomputed distance matrix instead of computing
/// distances between trajectory points on the fly.
///
/// # Arguments
/// * `distance_matrix` - A 2D numpy array where matrix[i][j] is the distance between
///                       point i of trajectory 1 and point j of trajectory 2
/// * `eps` - Epsilon threshold for matching points
/// * `use_full_matrix` - If true, compute and return the full DP matrix;
///                        if false (default), return None for the matrix to save space
///
/// # Returns
/// * A `DpResult` object with two properties:
///   - `distance`: LCSS distance as f64
///   - `matrix`: numpy array of shape (n0+1, n1+1) if use_full_matrix=True, else None
///
/// # Examples
/// ```python
/// import traj_dist_rs
/// import numpy as np
///
/// dist_matrix = np.array([
///     [1.0, 1.0],
///     [1.0, 1.0],
/// ])
///
/// # Without matrix
/// result = traj_dist_rs.lcss_with_matrix(dist_matrix, eps=0.5)
/// print(result.distance)  # Distance value
/// print(result.matrix)  # None
///
/// # With matrix
/// result = traj_dist_rs.lcss_with_matrix(dist_matrix, eps=0.5, use_full_matrix=True)
/// print(result.matrix)  # numpy array
/// ```
#[gen_stub_pyfunction]
#[pyfunction(signature = (distance_matrix, eps, use_full_matrix=false))]
pub fn lcss_with_matrix<'py>(
    #[gen_stub(override_type(type_repr="numpy.ndarray", imports=("numpy",)))]
    distance_matrix: &Bound<'py, PyAny>,
    eps: f64,
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

    // Convert to Vec<Vec<f64>>
    let matrix_view = readonly_array.as_array();
    let mut distance_matrix_vec: Vec<Vec<f64>> = Vec::with_capacity(n0);
    for row in matrix_view.rows() {
        distance_matrix_vec.push(row.to_vec());
    }

    // Create a precomputed distance calculator
    let calculator =
        crate::distance::base::PrecomputedDistanceCalculator::new(&distance_matrix_vec);

    // Compute LCSS using the calculator
    let result = crate::distance::lcss::lcss(&calculator, eps, use_full_matrix);

    Ok(PyDpResult { inner: result })
}

define_stub_info_gatherer!(stub_info);
