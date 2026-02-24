use crate::binding::PyDpResult;
use crate::binding::sequence::types::{PointRef, PyTrajectoryType};
use crate::distance::base::TrajectoryCalculator;
use crate::distance::distance_type::DistanceType;
use crate::traits::{AsCoord, CoordSequence};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};
use std::str::FromStr;

/// Compute the ERP (Edit distance with Real Penalty) distance between two trajectories
///
/// This is the **traj-dist compatible** implementation that matches the (buggy) implementation
/// in traj-dist. This version should be used when compatibility with traj-dist is required.
///
/// ERP is a distance measure for trajectories that uses a gap point `g` as a penalty
/// for insertions and deletions. The distance is computed using dynamic programming.
///
/// **Note**: This implementation has a bug in the DP matrix initialization where it uses
/// the total sum of distances to g instead of cumulative sums. This matches the bug in
/// traj-dist's Python implementation.
///
/// # Arguments
/// * `t1` - First trajectory (list of [longitude, latitude] pairs)
/// * `t2` - Second trajectory (list of [longitude, latitude] pairs)
/// * `dist_type` - Distance type: "euclidean" or "spherical"
/// * `g` - Gap point for penalty (list of [longitude, latitude] or None for centroid)
/// * `use_full_matrix` - If true, compute and return the full DP matrix;
///                        if false (default), return None for the matrix to save space
///
/// # Returns
/// * A `DpResult` object with two properties:
///   - `distance`: ERP distance as f64
///   - `matrix`: numpy array of shape (n0+1, n1+1) if use_full_matrix=True, else None
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0]]
/// t2 = [[0.0, 1.0], [1.0, 0.0]]
/// result = traj_dist_rs.erp_compat_traj_dist(t1, t2, "euclidean", g=[0.0, 0.0])
/// print(result.distance)  # Distance value
/// print(result.matrix)  # None
///
/// result = traj_dist_rs.erp_compat_traj_dist(t1, t2, "euclidean", g=[0.0, 0.0], use_full_matrix=True)
/// print(result.matrix)  # numpy array
/// ```
#[gen_stub_pyfunction]
#[pyfunction(signature = (t1, t2, dist_type, g, use_full_matrix=false))]
pub fn erp_compat_traj_dist(
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]] | numpy.ndarray", imports=("typing", "numpy")))]
    t1: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]] | numpy.ndarray", imports=("typing", "numpy")))]
    t2: &Bound<'_, PyAny>,
    dist_type: String,
    #[gen_stub(
        override_type(
            type_repr="typing.Optional[typing.List[float]]",
            imports=("typing")
        )
    )]
    g: Option<Vec<f64>>,
    use_full_matrix: bool,
) -> PyResult<PyDpResult> {
    // Convert Python objects to PyTrajectoryType
    let traj1 = PyTrajectoryType::try_from(t1)?;
    let traj2 = PyTrajectoryType::try_from(t2)?;

    // Compute centroid if g is None
    let gap_point = if let Some(g_vec) = g {
        if g_vec.len() != 2 {
            return Err(PyValueError::new_err(
                "Gap point g must have exactly 2 coordinates",
            ));
        }
        [g_vec[0], g_vec[1]]
    } else {
        // Compute centroid of both trajectories
        let n1 = traj1.len();
        let n2 = traj2.len();
        if n1 == 0 || n2 == 0 {
            return Err(PyValueError::new_err(
                "Cannot compute centroid for empty trajectory",
            ));
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        for i in 0..n1 {
            let p = traj1.get(i);
            sum_x += p.x();
            sum_y += p.y();
        }

        for j in 0..n2 {
            let p = traj2.get(j);
            sum_x += p.x();
            sum_y += p.y();
        }

        let total = (n1 + n2) as f64;
        [sum_x / total, sum_y / total]
    };

    // Parse distance type using FromStr from strum
    let distance_type = DistanceType::from_str(&dist_type).map_err(|_| {
        PyValueError::new_err(format!(
            "Invalid distance type '{}'. Expected 'euclidean' or 'spherical'",
            dist_type
        ))
    })?;

    let gap_point = PointRef::new(&gap_point, 0);
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, distance_type);
    let result =
        crate::distance::erp::erp_compat_traj_dist(&calculator, &gap_point, use_full_matrix);

    Ok(PyDpResult { inner: result })
}

/// Compute the ERP (Edit distance with Real Penalty) distance between two trajectories
///
/// This is the **standard** ERP implementation with correct cumulative initialization.
/// This version should be used for new applications where correctness is more important
/// than compatibility with traj-dist.
///
/// ERP is a distance measure for trajectories that uses a gap point `g` as a penalty
/// for insertions and deletions. The distance is computed using dynamic programming.
///
/// **Note**: This implementation correctly accumulates distances by index in the DP matrix
/// initialization, unlike the buggy implementation in traj-dist.
///
/// # Arguments
/// * `t1` - First trajectory (list of [longitude, latitude] pairs)
/// * `t2` - Second trajectory (list of [longitude, latitude] pairs)
/// * `dist_type` - Distance type: "euclidean" or "spherical"
/// * `g` - Gap point for penalty (list of [longitude, latitude] or None for centroid)
/// * `use_full_matrix` - If true, compute and return the full DP matrix;
///                        if false (default), return None for the matrix to save space
///
/// # Returns
/// * A `DpResult` object with two properties:
///   - `distance`: ERP distance as f64
///   - `matrix`: numpy array of shape (n0+1, n1+1) if use_full_matrix=True, else None
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0]]
/// t2 = [[0.0, 1.0], [1.0, 0.0]]
/// result = traj_dist_rs.erp_standard(t1, t2, "euclidean", g=[0.0, 0.0])
/// print(result.distance)  # Distance value
/// print(result.matrix)  # None
///
/// result = traj_dist_rs.erp_standard(t1, t2, "euclidean", g=[0.0, 0.0], use_full_matrix=True)
/// print(result.matrix)  # numpy array
/// ```
#[gen_stub_pyfunction]
#[pyfunction(signature = (t1, t2, dist_type, g, use_full_matrix=false))]
pub fn erp_standard(
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]] | numpy.ndarray", imports=("typing", "numpy")))]
    t1: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]] | numpy.ndarray", imports=("typing", "numpy")))]
    t2: &Bound<'_, PyAny>,
    dist_type: String,
    #[gen_stub(
        override_type(
            type_repr="typing.Optional[typing.List[float]]",
            imports=("typing")
        )
    )]
    g: Option<Vec<f64>>,
    use_full_matrix: bool,
) -> PyResult<PyDpResult> {
    // Convert Python objects to PyTrajectoryType
    let traj1 = PyTrajectoryType::try_from(t1)?;
    let traj2 = PyTrajectoryType::try_from(t2)?;

    // Compute centroid if g is None
    let gap_point = if let Some(g_vec) = g {
        if g_vec.len() != 2 {
            return Err(PyValueError::new_err(
                "Gap point g must have exactly 2 coordinates",
            ));
        }
        [g_vec[0], g_vec[1]]
    } else {
        // Compute centroid of both trajectories
        let n1 = traj1.len();
        let n2 = traj2.len();
        if n1 == 0 || n2 == 0 {
            return Err(PyValueError::new_err(
                "Cannot compute centroid for empty trajectory",
            ));
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        for i in 0..n1 {
            let p = traj1.get(i);
            sum_x += p.x();
            sum_y += p.y();
        }

        for j in 0..n2 {
            let p = traj2.get(j);
            sum_x += p.x();
            sum_y += p.y();
        }

        let total = (n1 + n2) as f64;
        [sum_x / total, sum_y / total]
    };

    // Parse distance type using FromStr from strum
    let distance_type = DistanceType::from_str(&dist_type).map_err(|_| {
        PyValueError::new_err(format!(
            "Invalid distance type '{}'. Expected 'euclidean' or 'spherical'",
            dist_type
        ))
    })?;

    let gap_point = PointRef::new(&gap_point, 0);
    let calculator = TrajectoryCalculator::new(&traj1, &traj2, distance_type);
    let result = crate::distance::erp::erp_standard(&calculator, &gap_point, use_full_matrix);

    Ok(PyDpResult { inner: result })
}

/// Compute the ERP (Edit distance with Real Penalty) distance using a precomputed distance matrix
///
/// This is the **traj-dist compatible** implementation that matches the (buggy) implementation
/// in traj-dist. This version should be used when compatibility with traj-dist is required.
///
/// This function allows you to use a precomputed distance matrix and extra distance arrays
/// instead of computing distances on the fly.
///
/// # Arguments
/// * `distance_matrix` - A 2D numpy array where matrix[i][j] is the distance between
///                       point i of trajectory 1 and point j of trajectory 2
/// * `seq0_gap_dists` - A 1D numpy array where seq0_gap_dists[i] is the distance between
///                      point i of trajectory 1 and the gap point g
/// * `seq1_gap_dists` - A 1D numpy array where seq1_gap_dists[j] is the distance between
///                      point j of trajectory 2 and the gap point g
/// * `use_full_matrix` - If true, compute and return the full DP matrix;
///                        if false (default), return None for the matrix to save space
///
/// # Returns
/// * A `DpResult` object with two properties:
///   - `distance`: ERP distance as f64
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
/// seq0_gap_dists = np.array([1.0, 1.0])
/// seq1_gap_dists = np.array([1.0, 1.0])
///
/// # Without matrix
/// result = traj_dist_rs.erp_compat_traj_dist_with_matrix(dist_matrix, seq0_gap_dists, seq1_gap_dists)
/// print(result.distance)  # Distance value
/// print(result.matrix)  # None
///
/// # With matrix
/// result = traj_dist_rs.erp_compat_traj_dist_with_matrix(dist_matrix, seq0_gap_dists, seq1_gap_dists, use_full_matrix=True)
/// print(result.matrix)  # numpy array
/// ```
#[gen_stub_pyfunction]
#[pyfunction(signature = (distance_matrix, seq0_gap_dists, seq1_gap_dists, use_full_matrix=false))]
pub fn erp_compat_traj_dist_with_matrix<'py>(
    #[gen_stub(override_type(type_repr="numpy.ndarray", imports=("numpy",)))]
    distance_matrix: &Bound<'py, PyAny>,
    #[gen_stub(override_type(type_repr="numpy.ndarray", imports=("numpy",)))]
    seq0_gap_dists: &Bound<'py, PyAny>,
    #[gen_stub(override_type(type_repr="numpy.ndarray", imports=("numpy",)))]
    seq1_gap_dists: &Bound<'py, PyAny>,
    use_full_matrix: bool,
) -> PyResult<PyDpResult> {
    use numpy::{PyArrayMethods, PyUntypedArrayMethods};

    // Convert distance_matrix to flat array (zero-copy)
    let dist_array = distance_matrix
        .downcast::<numpy::PyArray2<f64>>()
        .map_err(|_| {
            PyValueError::new_err("distance_matrix must be a 2D numpy array of float64 values")
        })?;

    let dist_readonly = dist_array.readonly();
    let dist_shape = dist_readonly.shape();

    if dist_shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "distance_matrix must be a 2D array, got shape {:?}",
            dist_shape
        )));
    }

    let n0 = dist_shape[0];
    let n1 = dist_shape[1];

    // Check if array is contiguous (C-order) for zero-copy
    let dist_view = dist_readonly.as_array();
    if !dist_view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "distance_matrix must be contiguous (C-order)",
        ));
    }

    // Zero-copy: get raw pointer directly
    let dist_data_ptr = dist_view.as_ptr();
    let distance_matrix_slice = unsafe { std::slice::from_raw_parts(dist_data_ptr, n0 * n1) };

    // Convert seq0_gap_dists to slice (zero-copy)
    let seq0_array = seq0_gap_dists
        .downcast::<numpy::PyArray1<f64>>()
        .map_err(|_| {
            PyValueError::new_err("seq0_gap_dists must be a 1D numpy array of float64 values")
        })?;

    let seq0_readonly = seq0_array.readonly();
    let seq0_slice = seq0_readonly.as_slice()?;

    // Convert seq1_gap_dists to slice (zero-copy)
    let seq1_array = seq1_gap_dists
        .downcast::<numpy::PyArray1<f64>>()
        .map_err(|_| {
            PyValueError::new_err("seq1_gap_dists must be a 1D numpy array of float64 values")
        })?;

    let seq1_readonly = seq1_array.readonly();
    let seq1_slice = seq1_readonly.as_slice()?;

    // Create a precomputed distance calculator with extra distances (zero-copy)
    let calculator = crate::distance::base::PrecomputedDistanceCalculator::with_extra_distances(
        distance_matrix_slice,
        n0,
        n1,
        Some(seq0_slice),
        Some(seq1_slice),
    );

    // Compute ERP using the calculator
    let result = crate::distance::erp::erp_compat_traj_dist_with_distances(
        &calculator,
        seq0_slice,
        seq1_slice,
        use_full_matrix,
    );

    Ok(PyDpResult { inner: result })
}

/// Compute the ERP (Edit distance with Real Penalty) distance using a precomputed distance matrix
///
/// This is the **standard** ERP implementation with correct cumulative initialization.
/// This version should be used for new applications where correctness is more important
/// than compatibility with traj-dist.
///
/// This function allows you to use a precomputed distance matrix and extra distance arrays
/// instead of computing distances on the fly.
///
/// # Arguments
/// * `distance_matrix` - A 2D numpy array where matrix[i][j] is the distance between
///                       point i of trajectory 1 and point j of trajectory 2
/// * `seq0_gap_dists` - A 1D numpy array where seq0_gap_dists[i] is the distance between
///                      point i of trajectory 1 and the gap point g
/// * `seq1_gap_dists` - A 1D numpy array where seq1_gap_dists[j] is the distance between
///                      point j of trajectory 2 and the gap point g
/// * `use_full_matrix` - If true, compute and return the full DP matrix;
///                        if false (default), return None for the matrix to save space
///
/// # Returns
/// * A `DpResult` object with two properties:
///   - `distance`: ERP distance as f64
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
/// seq0_gap_dists = np.array([1.0, 1.0])
/// seq1_gap_dists = np.array([1.0, 1.0])
///
/// # Without matrix
/// result = traj_dist_rs.erp_standard_with_matrix(dist_matrix, seq0_gap_dists, seq1_gap_dists)
/// print(result.distance)  # Distance value
/// print(result.matrix)  # None
///
/// # With matrix
/// result = traj_dist_rs.erp_standard_with_matrix(dist_matrix, seq0_gap_dists, seq1_gap_dists, use_full_matrix=True)
/// print(result.matrix)  # numpy array
/// ```
#[gen_stub_pyfunction]
#[pyfunction(signature = (distance_matrix, seq0_gap_dists, seq1_gap_dists, use_full_matrix=false))]
pub fn erp_standard_with_matrix<'py>(
    #[gen_stub(override_type(type_repr="numpy.ndarray", imports=("numpy",)))]
    distance_matrix: &Bound<'py, PyAny>,
    #[gen_stub(override_type(type_repr="numpy.ndarray", imports=("numpy",)))]
    seq0_gap_dists: &Bound<'py, PyAny>,
    #[gen_stub(override_type(type_repr="numpy.ndarray", imports=("numpy",)))]
    seq1_gap_dists: &Bound<'py, PyAny>,
    use_full_matrix: bool,
) -> PyResult<PyDpResult> {
    use numpy::{PyArrayMethods, PyUntypedArrayMethods};

    // Convert distance_matrix to flat array (zero-copy)
    let dist_array = distance_matrix
        .downcast::<numpy::PyArray2<f64>>()
        .map_err(|_| {
            PyValueError::new_err("distance_matrix must be a 2D numpy array of float64 values")
        })?;

    let dist_readonly = dist_array.readonly();
    let dist_shape = dist_readonly.shape();

    if dist_shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "distance_matrix must be a 2D array, got shape {:?}",
            dist_shape
        )));
    }

    let n0 = dist_shape[0];
    let n1 = dist_shape[1];

    // Check if array is contiguous (C-order) for zero-copy
    let dist_view = dist_readonly.as_array();
    if !dist_view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "distance_matrix must be contiguous (C-order)",
        ));
    }

    // Zero-copy: get raw pointer directly
    let dist_data_ptr = dist_view.as_ptr();
    let distance_matrix_slice = unsafe { std::slice::from_raw_parts(dist_data_ptr, n0 * n1) };

    // Convert seq0_gap_dists to slice (zero-copy)
    let seq0_array = seq0_gap_dists
        .downcast::<numpy::PyArray1<f64>>()
        .map_err(|_| {
            PyValueError::new_err("seq0_gap_dists must be a 1D numpy array of float64 values")
        })?;

    let seq0_readonly = seq0_array.readonly();
    let seq0_slice = seq0_readonly.as_slice()?;

    // Convert seq1_gap_dists to slice (zero-copy)
    let seq1_array = seq1_gap_dists
        .downcast::<numpy::PyArray1<f64>>()
        .map_err(|_| {
            PyValueError::new_err("seq1_gap_dists must be a 1D numpy array of float64 values")
        })?;

    let seq1_readonly = seq1_array.readonly();
    let seq1_slice = seq1_readonly.as_slice()?;

    // Create a precomputed distance calculator with extra distances (zero-copy)
    let calculator = crate::distance::base::PrecomputedDistanceCalculator::with_extra_distances(
        distance_matrix_slice,
        n0,
        n1,
        Some(seq0_slice),
        Some(seq1_slice),
    );

    // Compute ERP using the calculator
    let result = crate::distance::erp::erp_standard_with_distances(
        &calculator,
        seq0_slice,
        seq1_slice,
        use_full_matrix,
    );

    Ok(PyDpResult { inner: result })
}

define_stub_info_gatherer!(stub_info);
