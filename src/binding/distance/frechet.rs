use crate::binding::sequence::types::PyTrajectoryType;
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};

/// Compute the Frechet distance between two trajectories
///
/// The Frechet distance considers all continuous points along the curve segments,
/// providing an exact solution (unlike Discrete Frechet which only considers vertices).
///
/// Intuitively, it represents the minimum leash length required for a person
/// and their dog to walk along the two curves without backtracking.
///
/// # Arguments
/// * `t1` - First trajectory (list of [x, y] pairs or numpy array)
/// * `t2` - Second trajectory (list of [x, y] pairs or numpy array)
///
/// # Returns
/// * The Frechet distance as a float
///
/// # Notes
/// - Frechet distance only supports Euclidean distance (not spherical distance)
/// - Trajectories with fewer than 2 points return float('inf')
/// - Result is always <= the Discrete Frechet distance for the same trajectories
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
/// t2 = [[0.0, 0.5], [1.0, 1.5], [2.0, 0.5]]
///
/// distance = traj_dist_rs.frechet(t1, t2)
/// print(f"Frechet distance: {distance}")
/// ```
#[gen_stub_pyfunction]
#[pyfunction]
pub fn frechet(
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]] | numpy.ndarray", imports=("typing", "numpy")))]
    t1: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]] | numpy.ndarray", imports=("typing", "numpy")))]
    t2: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let traj1 = PyTrajectoryType::try_from(t1)?;
    let traj2 = PyTrajectoryType::try_from(t2)?;

    Ok(crate::distance::frechet::frechet(&traj1, &traj2))
}

define_stub_info_gatherer!(stub_info);
