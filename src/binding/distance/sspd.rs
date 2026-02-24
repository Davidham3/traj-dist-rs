use crate::binding::sequence::types::PyTrajectoryType;
use crate::distance::distance_type::DistanceType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};
use std::str::FromStr;

/// Compute the SSPD distance between two trajectories
///
/// # Arguments
/// * `t1` - First trajectory (list of [longitude, latitude] pairs)
/// * `t2` - Second trajectory (list of [longitude, latitude] pairs)
/// * `dist_type` - Distance type: "euclidean" or "spherical"
///
/// # Returns
/// * SSPD distance as f64
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0]]
/// t2 = [[0.0, 1.0], [1.0, 0.0]]
/// dist = traj_dist_rs.sspd(t1, t2, "euclidean")
/// ```
#[gen_stub_pyfunction]
#[pyfunction]
pub fn sspd(
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]] | numpy.ndarray", imports=("typing", "numpy")))]
    t1: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]] | numpy.ndarray", imports=("typing", "numpy")))]
    t2: &Bound<'_, PyAny>,
    dist_type: String,
) -> PyResult<f64> {
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

    // SSPD distance requires direct access to trajectory coordinates
    // to compute point-to-segment distances, so we use the CoordSequence version
    let distance = crate::distance::sspd::sspd(&traj1, &traj2, distance_type);

    Ok(distance)
}

define_stub_info_gatherer!(stub_info);
