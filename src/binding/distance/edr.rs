use crate::binding::sequence::types::PyTrajectoryType;
use crate::distance::distance_type::DistanceType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};

/// Compute the EDR (Edit Distance on Real sequence) distance between two trajectories
///
/// EDR is a distance measure for trajectories that allows for gaps in the matching.
/// It uses a threshold `eps` to determine if two points match.
/// The distance is normalized by the maximum length of the two trajectories.
///
/// # Arguments
/// * `t1` - First trajectory (list of [longitude, latitude] pairs)
/// * `t2` - Second trajectory (list of [longitude, latitude] pairs)
/// * `dist_type` - Distance type: "euclidean" or "spherical"
/// * `eps` - Epsilon threshold for matching points
///
/// # Returns
/// * EDR distance as f64 (normalized to [0, 1])
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0]]
/// t2 = [[0.0, 1.0], [1.0, 0.0]]
/// dist = traj_dist_rs.edr(t1, t2, "euclidean", eps=0.5)
/// ```
#[gen_stub_pyfunction]
#[pyfunction]
pub fn edr(
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]]", imports=("typing")))]
    t1: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]]", imports=("typing")))]
    t2: &Bound<'_, PyAny>,
    dist_type: String,
    eps: f64,
) -> PyResult<f64> {
    // Convert Python objects to PyTrajectoryType
    let traj1 = PyTrajectoryType::try_from(t1)?;
    let traj2 = PyTrajectoryType::try_from(t2)?;

    // Compute distance based on type (PyTrajectoryType implements CoordSequence)
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

    let distance = crate::distance::edr::edr(&traj1, &traj2, eps, distance_type);

    Ok(distance)
}

define_stub_info_gatherer!(stub_info);
