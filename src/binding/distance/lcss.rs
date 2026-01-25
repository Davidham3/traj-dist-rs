use crate::binding::trajectory::PyTrajectory;
use crate::distance::distance_type::DistanceType;
use crate::distance::lcss::lcss as internal_lcss;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};
use std::convert::TryFrom;

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
///
/// # Returns
/// * LCSS distance as f64
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0]]
/// t2 = [[0.0, 1.0], [1.0, 0.0]]
/// dist = traj_dist_rs.lcss(t1, t2, "euclidean", eps=0.5)
/// ```
#[gen_stub_pyfunction]
#[pyfunction]
pub fn lcss(
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]]", imports=("typing")))]
    t1: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]]", imports=("typing")))]
    t2: &Bound<'_, PyAny>,
    dist_type: String,
    eps: f64,
) -> PyResult<f64> {
    // Convert Python objects to PyTrajectory
    let traj1 = PyTrajectory::try_from(t1)?;
    let traj2 = PyTrajectory::try_from(t2)?;

    // Compute distance based on type (PyTrajectory implements CoordSequence)
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

    let distance = internal_lcss(&traj1, &traj2, eps, distance_type);

    Ok(distance)
}

define_stub_info_gatherer!(stub_info);
