use crate::binding::sequence::types::PyTrajectoryType;
use crate::distance::distance_type::DistanceType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};

/// Compute the Hausdorff distance between two trajectories
///
/// # Arguments
/// * `t1` - First trajectory (list of [longitude, latitude] pairs)
/// * `t2` - Second trajectory (list of [longitude, latitude] pairs)
/// * `dist_type` - Distance type: "euclidean" or "spherical"
///
/// # Returns
/// * Hausdorff distance as f64
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0]]
/// t2 = [[0.0, 1.0], [1.0, 0.0]]
/// dist = traj_dist_rs.hausdorff(t1, t2, "euclidean")
/// ```
#[gen_stub_pyfunction]
#[pyfunction]
pub fn hausdorff(
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

    // Hausdorff distance requires direct access to trajectory coordinates
    // to compute point-to-segment distances, so we use the CoordSequence version
    let distance = crate::distance::hausdorff::hausdorff(&traj1, &traj2, distance_type);

    Ok(distance)
}

define_stub_info_gatherer!(stub_info);
