//! Python bindings for batch computation functions (New elegant design)

use crate::binding::sequence::types::PyTrajectoryType;
use crate::distance::batch::{DistanceAlgorithm, Metric};
use crate::distance::distance_type::DistanceType;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_stub_gen::{
    define_stub_info_gatherer, derive::gen_stub_pyclass, derive::gen_stub_pyfunction,
};
use pyo3_stub_gen_derive::gen_stub_pymethods;
use std::str::FromStr;

// ===================================================================================
// 1. Define a PyClass that holds the complete configuration, but hide its internal construction
// ===================================================================================

/// A metric configuration object for distance calculations.
///
/// This class encapsulates the distance algorithm, its parameters, and the
/// underlying point-to-point distance type (e.g., Euclidean or Spherical).
///
/// Do not instantiate this class directly. Use the provided static factory
/// methods instead (e.g., `Metric.sspd()`, `Metric.lcss(...)`).
#[cfg(feature = "python-binding")]
#[gen_stub_pyclass]
#[pyclass(name = "Metric")]
pub struct PyMetric {
    // Internally holds a pure Rust Metric
    // This field is invisible to Python, forcing users to use factory methods
    inner: Metric,
}

#[cfg(feature = "python-binding")]
#[gen_stub_pymethods]
#[pymethods]
impl PyMetric {
    // Hide default constructor, preventing users from calling Metric()
    #[new]
    #[pyo3(signature = ())]
    fn new() -> PyResult<Self> {
        Err(PyValueError::new_err(
            "Metric objects cannot be created directly. Use a factory method like Metric.sspd() or Metric.lcss().",
        ))
    }

    // ================== Factory Methods ==================

    /// Symmetric Segment-Path Distance.
    #[staticmethod]
    #[pyo3(signature = (type_d = "euclidean"))]
    fn sspd(type_d: &str) -> PyResult<Self> {
        let calculator = build_calculator(DistanceAlgorithm::SSPD, type_d)?;
        Ok(Self { inner: calculator })
    }

    /// Dynamic Time Warping.
    #[staticmethod]
    #[pyo3(signature = (type_d = "euclidean"))]
    fn dtw(type_d: &str) -> PyResult<Self> {
        let calculator = build_calculator(DistanceAlgorithm::DTW, type_d)?;
        Ok(Self { inner: calculator })
    }

    /// Hausdorff Distance.
    #[staticmethod]
    #[pyo3(signature = (type_d = "euclidean"))]
    fn hausdorff(type_d: &str) -> PyResult<Self> {
        let calculator = build_calculator(DistanceAlgorithm::Hausdorff, type_d)?;
        Ok(Self { inner: calculator })
    }

    /// Discrete Frechet Distance.
    #[staticmethod]
    #[pyo3(signature = (type_d = "euclidean"))]
    fn discret_frechet(type_d: &str) -> PyResult<Self> {
        let calculator = build_calculator(DistanceAlgorithm::DiscretFrechet, type_d)?;
        Ok(Self { inner: calculator })
    }

    /// Longest Common Subsequence.
    #[staticmethod]
    #[pyo3(signature = (eps, type_d = "euclidean"))]
    fn lcss(eps: f64, type_d: &str) -> PyResult<Self> {
        let calculator = build_calculator(DistanceAlgorithm::LCSS { eps }, type_d)?;
        Ok(Self { inner: calculator })
    }

    /// Edit Distance on Real sequence.
    #[staticmethod]
    #[pyo3(signature = (eps, type_d = "euclidean"))]
    fn edr(eps: f64, type_d: &str) -> PyResult<Self> {
        let calculator = build_calculator(DistanceAlgorithm::EDR { eps }, type_d)?;
        Ok(Self { inner: calculator })
    }

    /// Edit distance with Real Penalty.
    #[staticmethod]
    #[pyo3(signature = (g, type_d = "euclidean"))]
    fn erp(g: Vec<f64>, type_d: &str) -> PyResult<Self> {
        if g.len() < 2 {
            return Err(PyValueError::new_err(
                "ERP 'g' parameter must have at least 2 elements.",
            ));
        }
        let algorithm = DistanceAlgorithm::ERP { g: [g[0], g[1]] };
        let calculator = build_calculator(algorithm, type_d)?;
        Ok(Self { inner: calculator })
    }
}

// Helper function to keep factory method code clean
fn build_calculator(algorithm: DistanceAlgorithm, type_d: &str) -> PyResult<Metric> {
    let distance_type = DistanceType::from_str(type_d).map_err(|_| {
        PyValueError::new_err(format!(
            "Invalid distance type '{}'. Expected 'euclidean' or 'spherical'.",
            type_d
        ))
    })?;
    Ok(Metric::new(algorithm, distance_type))
}

// ===================================================================================
// 2. Update pdist and cdist signatures
// ===================================================================================

/// Compute pairwise distances between trajectories
///
/// This function computes the distances between all unique pairs of trajectories
/// in the input list. The result is a compressed distance matrix (1D array)
/// containing distances for all pairs (i, j) where i < j.
///
/// # Symmetry Assumption
///
/// This function assumes that the distance metric is **symmetric**, i.e.,
/// `distance(A, B) == distance(B, A)`. All standard distance algorithms
/// in traj-dist-rs (SSPD, DTW, Hausdorff, LCSS, EDR, ERP, Discret Frechet)
/// satisfy this property.
///
/// **Important**: If your distance metric is **asymmetric**, use `cdist` instead
/// to compute the full distance matrix. Using `pdist` with asymmetric distances
/// will only compute half of the distances and may produce incorrect results.
///
/// # Arguments
/// * `trajectories` - List of trajectories, where each trajectory is a 2D numpy array
///                    or list of [x, y] pairs
/// * `metric` - Metric configuration object (e.g., `Metric.sspd()`, `Metric.lcss(eps=5.0)`)
/// * `parallel` - Whether to use parallel processing (default: True)
///
/// # Returns
/// * `distances` - 1D numpy array containing distances for all unique pairs
///
/// # Output Format
///
/// For `n` trajectories, the result is a 1D array of length `n * (n - 1) / 2`.
/// The distances are ordered as `d(0,1), d(0,2), ..., d(0,n-1), d(1,2), d(1,3), ..., d(n-2,n-1)`.
///
/// # Examples
/// ```python
/// import traj_dist_rs
/// import numpy as np
///
/// # Create metric configuration
/// metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
///
/// # Using numpy arrays (zero-copy)
/// trajectories = [
///     np.array([[0.0, 0.0], [1.0, 1.0]]),
///     np.array([[0.0, 1.0], [1.0, 0.0]]),
///     np.array([[0.5, 0.5], [1.5, 1.5]])
/// ]
/// distances = traj_dist_rs.pdist(trajectories, metric=metric)
///
/// # Using lists (will be copied)
/// trajectories = [
///     [[0.0, 0.0], [1.0, 1.0]],
///     [[0.0, 1.0], [1.0, 0.0]],
///     [[0.5, 0.5], [1.5, 1.5]]
/// ]
/// distances = traj_dist_rs.pdist(trajectories, metric=metric)
///
/// # Using LCSS with epsilon parameter
/// metric_lcss = traj_dist_rs.Metric.lcss(eps=5.0, type_d="euclidean")
/// distances = traj_dist_rs.pdist(trajectories, metric=metric_lcss)
/// ```
#[cfg(feature = "python-binding")]
#[gen_stub_pyfunction]
#[pyfunction(signature = (trajectories, metric, parallel=true))]
pub fn pdist<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr="typing.Sequence[typing.List[typing.List[float]] | numpy.ndarray]", imports=("typing", "numpy")))]
    trajectories: &Bound<'py, PyList>,
    metric: &PyMetric,
    parallel: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    if trajectories.len() < 2 {
        return Err(PyValueError::new_err(
            "pdist requires at least 2 trajectories",
        ));
    }

    // Convert each trajectory to PyTrajectoryType
    let trajectories: Vec<PyTrajectoryType> = trajectories
        .iter()
        .map(|t| {
            PyTrajectoryType::try_from(&t)
                .map_err(|e| PyValueError::new_err(format!("Failed to convert trajectory: {}", e)))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Directly use the inner Metric inside PyMetric
    let distances = crate::distance::batch::pdist(&trajectories, &metric.inner, parallel)
        .map_err(|e| PyValueError::new_err(format!("Failed to compute distances: {}", e)))?;

    let array = PyArray1::from_vec(py, distances);
    Ok(array.unbind())
}

/// Compute distances between two trajectory collections
///
/// This function computes the distances between all pairs of trajectories
/// from two collections. The result is a full distance matrix (2D array)
/// with shape (n_a, n_b).
///
/// # When to Use `cdist` vs `pdist`
///
/// - **Use `cdist`** when:
///   - Computing distances between two different trajectory collections
///   - Your distance metric is **asymmetric** (distance(A, B) != distance(B, A))
///   - You need the full distance matrix for both directions
///
/// - **Use `pdist`** when:
///   - Computing distances within a single trajectory collection
///   - Your distance metric is **symmetric** (distance(A, B) == distance(B, A))
///   - You want to save memory by using the compressed distance matrix format
///
/// # Arguments
/// * `trajectories_a` - First collection of trajectories, where each trajectory is a
///                      2D numpy array or list of [x, y] pairs
/// * `trajectories_b` - Second collection of trajectories
/// * `metric` - Metric configuration object (e.g., `Metric.sspd()`, `Metric.lcss(eps=5.0)`)
/// * `parallel` - Whether to use parallel processing (default: True)
///
/// # Returns
/// * `distances` - 2D numpy array with shape (len(trajectories_a), len(trajectories_b))
///
/// # Output Format
///
/// For `n_a` trajectories in the first collection and `n_b` trajectories in the second,
/// the result is a 2D array with shape `(n_a, n_b)`. The distance at index `[i, j]`
/// represents the distance from `trajectories_a[i]` to `trajectories_b[j]`.
///
/// # Examples
/// ```python
/// import traj_dist_rs
/// import numpy as np
///
/// # Create metric configuration
/// metric = traj_dist_rs.Metric.sspd(type_d="euclidean")
///
/// # Using numpy arrays (zero-copy)
/// trajectories_a = [
///     np.array([[0.0, 0.0], [1.0, 1.0]]),
///     np.array([[0.0, 1.0], [1.0, 0.0]])
/// ]
/// trajectories_b = [
///     np.array([[0.5, 0.5], [1.5, 1.5]]),
///     np.array([[0.5, 1.5], [1.5, 0.5]])
/// ]
/// distances = traj_dist_rs.cdist(trajectories_a, trajectories_b, metric=metric)
/// print(distances.shape)  # (2, 2)
///
/// # Using lists (will be copied)
/// trajectories_a = [
///     [[0.0, 0.0], [1.0, 1.0]],
///     [[0.0, 1.0], [1.0, 0.0]]
/// ]
/// trajectories_b = [
///     [[0.5, 0.5], [1.5, 1.5]],
///     [[0.5, 1.5], [1.5, 0.5]]
/// ]
/// distances = traj_dist_rs.cdist(trajectories_a, trajectories_b, metric=metric)
///
/// # Using ERP with gap point parameter
/// metric_erp = traj_dist_rs.Metric.erp(g=[0.0, 1.0], type_d="euclidean")
/// distances = traj_dist_rs.cdist(trajectories_a, trajectories_b, metric=metric_erp)
/// ```
#[cfg(feature = "python-binding")]
#[gen_stub_pyfunction]
#[pyfunction(signature = (trajectories_a, trajectories_b, metric, parallel=true))]
pub fn cdist<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr="typing.Sequence[typing.List[typing.List[float]] | numpy.ndarray]", imports=("typing", "numpy")))]
    trajectories_a: &Bound<'py, PyList>,
    #[gen_stub(override_type(type_repr="typing.Sequence[typing.List[typing.List[float]] | numpy.ndarray]", imports=("typing", "numpy")))]
    trajectories_b: &Bound<'py, PyList>,
    metric: &PyMetric,
    parallel: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    if trajectories_a.is_empty() {
        return Err(PyValueError::new_err(
            "cdist requires at least 1 trajectory in the first collection",
        ));
    }

    if trajectories_b.is_empty() {
        return Err(PyValueError::new_err(
            "cdist requires at least 1 trajectory in the second collection",
        ));
    }

    // Convert each trajectory to PyTrajectoryType
    let trajectories_a: Vec<PyTrajectoryType> = trajectories_a
        .iter()
        .map(|t| {
            PyTrajectoryType::try_from(&t).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to convert trajectory in first collection: {}",
                    e
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let trajectories_b: Vec<PyTrajectoryType> = trajectories_b
        .iter()
        .map(|t| {
            PyTrajectoryType::try_from(&t).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to convert trajectory in second collection: {}",
                    e
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Directly use the inner Metric inside PyMetric
    let distances =
        crate::distance::batch::cdist(&trajectories_a, &trajectories_b, &metric.inner, parallel)
            .map_err(|e| PyValueError::new_err(format!("Failed to compute distances: {}", e)))?;

    let n_b = trajectories_b.len();
    // Convert Vec<f64> to Vec<Vec<f64>> for PyArray2::from_vec2
    let distances_2d: Vec<Vec<f64>> = distances.chunks(n_b).map(|row| row.to_vec()).collect();
    let array = PyArray2::from_vec2(py, &distances_2d)
        .map_err(|e| PyValueError::new_err(format!("Failed to create 2D array: {}", e)))?;

    Ok(array.unbind())
}

define_stub_info_gatherer!(stub_info);
