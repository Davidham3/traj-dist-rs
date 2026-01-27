use crate::binding::sequence::types::{PointRef, PyTrajectoryType};
use crate::distance::distance_type::DistanceType;
use crate::traits::{AsCoord, CoordSequence};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};

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
///
/// # Returns
/// * ERP distance as f64
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0]]
/// t2 = [[0.0, 1.0], [1.0, 0.0]]
/// dist = traj_dist_rs.erp_compat_traj_dist(t1, t2, "euclidean", g=[0.0, 0.0])
/// ```
#[gen_stub_pyfunction]
#[pyfunction]
pub fn erp_compat_traj_dist(
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]]", imports=("typing")))]
    t1: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]]", imports=("typing")))]
    t2: &Bound<'_, PyAny>,
    dist_type: String,
    #[gen_stub(
        override_type(
            type_repr="typing.Optional[typing.List[float]]",
            imports=("typing")
        )
    )]
    g: Option<Vec<f64>>,
) -> PyResult<f64> {
    // Convert Python objects to PyTrajectoryType
    let traj1 = PyTrajectoryType::try_from(t1)?;
    let traj2 = PyTrajectoryType::try_from(t2)?;

    // TODO: 这里不需要这么复杂的处理方式

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

    let gap_point = PointRef::new(&gap_point, 0);

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

    let distance =
        crate::distance::erp::erp_compat_traj_dist(&traj1, &traj2, &gap_point, distance_type);

    Ok(distance)
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
///
/// # Returns
/// * ERP distance as f64
///
/// # Examples
/// ```python
/// import traj_dist_rs
///
/// t1 = [[0.0, 0.0], [1.0, 1.0]]
/// t2 = [[0.0, 1.0], [1.0, 0.0]]
/// dist = traj_dist_rs.erp_standard(t1, t2, "euclidean", g=[0.0, 0.0])
/// ```
#[gen_stub_pyfunction]
#[pyfunction]
pub fn erp_standard(
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]]", imports=("typing")))]
    t1: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr="typing.List[typing.List[float]]", imports=("typing")))]
    t2: &Bound<'_, PyAny>,
    dist_type: String,
    #[gen_stub(
        override_type(
            type_repr="typing.Optional[typing.List[float]]",
            imports=("typing")
        )
    )]
    g: Option<Vec<f64>>,
) -> PyResult<f64> {
    // Convert Python objects to PyTrajectoryType
    let traj1 = PyTrajectoryType::try_from(t1)?;
    let traj2 = PyTrajectoryType::try_from(t2)?;

    // TODO: 处理g点不需要这么复杂
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

    let gap_point = PointRef::new(&gap_point, 0);

    let distance = crate::distance::erp::erp_standard(&traj1, &traj2, &gap_point, distance_type);

    Ok(distance)
}

define_stub_info_gatherer!(stub_info);
