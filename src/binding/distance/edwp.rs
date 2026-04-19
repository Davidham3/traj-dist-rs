use crate::binding::PyDpResult;
use crate::binding::sequence::types::PyTrajectoryType;
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};

/// Compute the EDwP (Edit Distance with Projections) distance between two trajectories
///
/// EDwP is designed for trajectories with inconsistent sampling rates. It uses
/// point-to-segment projections to handle different sampling densities.
///
/// # Arguments
/// * `t1` - First trajectory (list of [x, y] pairs)
/// * `t2` - Second trajectory (list of [x, y] pairs)
/// * `use_full_matrix` - If true, compute and return the full DP matrix;
///                        if false (default), return None for the matrix to save space
///
/// # Returns
/// * A `DpResult` object with two properties:
///   - `distance`: EDwP distance as f64
///   - `matrix`: numpy array of shape (n0, n1) if use_full_matrix=True, else None
///
/// # Notes
/// - EDwP only supports Euclidean distance (not spherical distance)
/// - Passing "spherical" as distance type will result in an error
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
/// t2 = [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]
///
/// # Without matrix
/// result = traj_dist_rs.edwp(t1, t2, False)
/// print(result.distance)  # Distance value
/// print(result.matrix)  # None
///
/// # With matrix
/// result = traj_dist_rs.edwp(t1, t2, True)
/// print(result.distance)  # Distance value
/// print(result.matrix)  # numpy array
/// ```
#[gen_stub_pyfunction]
#[pyfunction(signature = (t1, t2, use_full_matrix=false))]
pub fn edwp<'py>(
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]] | numpy.ndarray", imports=("typing", "numpy")))]
    t1: &Bound<'py, PyAny>,
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]] | numpy.ndarray", imports=("typing", "numpy")))]
    t2: &Bound<'py, PyAny>,
    use_full_matrix: bool,
) -> PyResult<PyDpResult> {
    // Convert Python objects to PyTrajectoryType
    let traj1 = PyTrajectoryType::try_from(t1)?;
    let traj2 = PyTrajectoryType::try_from(t2)?;

    // Compute EDwP (only supports Euclidean distance)
    let result = crate::distance::edwp::edwp(&traj1, &traj2, use_full_matrix);

    Ok(PyDpResult { inner: result })
}

define_stub_info_gatherer!(stub_info);
