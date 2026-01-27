use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};
use std::convert::TryFrom;

use crate::binding::sequence::types::PyTrajectoryType;

/// Compute the Discret Frechet distance between two trajectories
///
/// The discret Frechet distance is a measure of similarity between two curves
/// that takes into account the location and ordering of the points along the curves.
///
/// # Arguments
/// * `t1` - First trajectory (list of [longitude, latitude] pairs)
/// * `t2` - Second trajectory (list of [longitude, latitude] pairs)
/// * `dist_type` - Distance type: "euclidean" (only Euclidean is supported for Discret Frechet)
///
/// # Returns
/// * Discret Frechet distance as f64
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0]]
/// t2 = [[0.0, 1.0], [1.0, 0.0]]
/// dist = traj_dist_rs.discret_frechet(t1, t2, "euclidean")
/// ```
#[gen_stub_pyfunction]
#[pyfunction]
pub fn discret_frechet(
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]]", imports=("typing")))]
    t1: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]]", imports=("typing")))]
    t2: &Bound<'_, PyAny>,
    dist_type: String,
) -> PyResult<f64> {
    // Convert Python objects to PyTrajectoryType
    let traj1 = PyTrajectoryType::try_from(t1)?;
    let traj2 = PyTrajectoryType::try_from(t2)?;

    // Compute distance based on type (PyTrajectoryType implements CoordSequence)
    let distance = match dist_type.to_lowercase().as_str() {
        "euclidean" => crate::distance::discret_frechet::discret_frechet_euclidean(&traj1, &traj2),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid distance type '{}'. Only 'euclidean' is supported for discret_frechet",
                dist_type
            )));
        }
    };

    Ok(distance)
}

define_stub_info_gatherer!(stub_info);
