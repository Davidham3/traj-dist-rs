//! Edit Distance with Projections (EDwP) implementation
//!
//! EDwP is designed for trajectories with inconsistent sampling rates.
//! It uses point-to-segment projections to handle different sampling densities.
//!
//! Reference: https://www.researchgate.net/publication/228636468_Edit_Distance_with_Projections

use crate::distance::DpResult;
use crate::distance::euclidean::{euclidean_distance, project_point_to_segment};
use crate::traits::{AsCoord, CoordSequence};

/// Compute EDwP distance between two trajectories
///
/// # Arguments
///
/// * `traj1` - First trajectory
/// * `traj2` - Second trajectory
/// * `use_full_matrix` - If true, return the full DP matrix; if false, use rolling array optimization
///
/// # Returns
///
/// A `DpResult` containing the distance and optionally the full DP matrix
///
/// # Examples
///
/// ```rust
/// use traj_dist_rs::distance::edwp::edwp;
///
/// let traj1 = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
/// let traj2 = vec![[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]];
///
/// let result = edwp(&traj1, &traj2, false);
/// println!("EDwP distance: {}", result.distance);
/// ```
pub fn edwp<T: CoordSequence>(traj1: &T, traj2: &T, use_full_matrix: bool) -> DpResult
where
    T::Coord: AsCoord,
{
    let t1_len = traj1.len();
    let t2_len = traj2.len();

    // Handle edge cases
    if t1_len == 0 || t2_len == 0 {
        return DpResult {
            distance: f64::MAX,
            matrix: None,
        };
    }

    if use_full_matrix {
        edwp_full_matrix(traj1, traj2)
    } else {
        edwp_rolling_array(traj1, traj2)
    }
}

/// EDwP with full matrix computation
///
/// This version computes and stores the full DP matrix, which is useful for
/// debugging and analysis but uses O(n*m) memory.
fn edwp_full_matrix<T: CoordSequence>(traj1: &T, traj2: &T) -> DpResult
where
    T::Coord: AsCoord,
{
    let t1_len = traj1.len();
    let t2_len = traj2.len();

    // Compute edge lengths
    let mut t1_edge_length = Vec::with_capacity(t1_len.saturating_sub(1));
    for i in 0..t1_len.saturating_sub(1) {
        let p1 = traj1.get(i);
        let p2 = traj1.get(i + 1);
        t1_edge_length.push(euclidean_distance(&p1, &p2));
    }

    let mut t2_edge_length = Vec::with_capacity(t2_len.saturating_sub(1));
    for i in 0..t2_len.saturating_sub(1) {
        let p1 = traj2.get(i);
        let p2 = traj2.get(i + 1);
        t2_edge_length.push(euclidean_distance(&p1, &p2));
    }

    let total_length: f64 = t1_edge_length.iter().sum::<f64>() + t2_edge_length.iter().sum::<f64>();

    // Initialize DP matrix and auxiliary matrices
    let mut value = vec![vec![0.0; t2_len]; t1_len];
    let mut delta = vec![vec![0.0; t2_len]; t1_len];
    let mut col_edits = vec![vec![(0.0, 0.0); t2_len]; t1_len];
    let mut row_edits = vec![vec![(0.0, 0.0); t2_len]; t1_len];

    // Initialize first row and column
    value[0][1..].fill(f64::MAX);
    for row in value.iter_mut().take(t1_len).skip(1) {
        row[0] = f64::MAX;
    }

    // Fill DP matrix
    for i in 1..t1_len {
        for j in 1..t2_len {
            let mut row_delta = f64::MAX;
            let mut col_delta = f64::MAX;
            let mut row_spatial_score = f64::MAX;
            let mut col_spatial_score = f64::MAX;
            let mut t1_insert: Option<(f64, f64)> = None;
            let mut t2_insert: Option<(f64, f64)> = None;

            // Row operation (insert from traj2)
            if i > 1 {
                let t1_edit = row_edits[i - 1][j];
                let t2_edit = col_edits[i - 1][j];
                let t1_edit_arr = [t1_edit.0, t1_edit.1];
                let t2_edit_arr = [t2_edit.0, t2_edit.1];
                let prev_point_edge = euclidean_distance(&traj1.get(i - 1), &t1_edit_arr);

                // Project point onto segment (equivalent to _line_map in Python)
                // _line_map(p1=t2_edit, p2=t2[j], p=t1[i-1])
                let projected =
                    project_point_to_segment(&traj1.get(i - 1), &t2_edit_arr, &traj2.get(j));
                let t2_insert_arr = [projected.0, projected.1];
                t2_insert = Some(projected);

                let row_edit_distance = euclidean_distance(&traj1.get(i - 1), &t2_insert_arr);
                let row_edit_edge = euclidean_distance(&t2_edit_arr, &t2_insert_arr);

                let row_converge1 = (row_edit_edge + prev_point_edge) / total_length;
                let row_converge2 = (euclidean_distance(&traj2.get(j), &t2_insert_arr)
                    + t1_edge_length[i - 1])
                    / total_length;

                row_delta = value[i - 1][j] - delta[i - 1][j]
                    + (row_edit_distance + euclidean_distance(&t1_edit_arr, &t2_edit_arr))
                        * row_converge1;
                row_spatial_score = row_delta
                    + (row_edit_distance + euclidean_distance(&traj2.get(j), &traj1.get(i)))
                        * row_converge2;
            }

            // Column operation (insert from traj1)
            if j > 1 {
                let t1_edit = row_edits[i][j - 1];
                let t2_edit = col_edits[i][j - 1];
                let t1_edit_arr = [t1_edit.0, t1_edit.1];
                let t2_edit_arr = [t2_edit.0, t2_edit.1];

                let prev_point_edge = euclidean_distance(&traj2.get(j - 1), &t2_edit_arr);

                // Project point onto segment (equivalent to _line_map in Python)
                // _line_map(p1=t1_edit, p2=t1[i], p=t2[j-1])
                let projected =
                    project_point_to_segment(&traj2.get(j - 1), &t1_edit_arr, &traj1.get(i));
                let t1_insert_arr = [projected.0, projected.1];
                t1_insert = Some(projected);

                let col_edit_distance = euclidean_distance(&traj2.get(j - 1), &t1_insert_arr);
                let col_edit_edge = euclidean_distance(&t1_edit_arr, &t1_insert_arr);

                let col_converge1 = (col_edit_edge + prev_point_edge) / total_length;
                let col_converge2 = (euclidean_distance(&traj1.get(i), &t1_insert_arr)
                    + t2_edge_length[j - 1])
                    / total_length;

                col_delta = value[i][j - 1] - delta[i][j - 1]
                    + (col_edit_distance + euclidean_distance(&t1_edit_arr, &t2_edit_arr))
                        * col_converge1;
                col_spatial_score = col_delta
                    + (col_edit_distance + euclidean_distance(&traj1.get(i), &traj2.get(j)))
                        * col_converge2;
            }

            // Diagonal operation (match points)
            let diag_coverage = (t1_edge_length[i - 1] + t2_edge_length[j - 1]) / total_length;
            let sub_score = (euclidean_distance(&traj2.get(j), &traj1.get(i))
                + euclidean_distance(&traj2.get(j - 1), &traj1.get(i - 1)))
                * diag_coverage;
            let diag_score = value[i - 1][j - 1] + sub_score;

            // Choose minimum operation
            if diag_score <= col_spatial_score && diag_score <= row_spatial_score {
                value[i][j] = diag_score;
                delta[i][j] = diag_score - value[i - 1][j - 1];
                col_edits[i][j] = (traj2.get(j - 1).x(), traj2.get(j - 1).y());
                row_edits[i][j] = (traj1.get(i - 1).x(), traj1.get(i - 1).y());
            } else if col_spatial_score < row_spatial_score
                || (col_spatial_score == row_spatial_score && t2_len > t1_len)
            {
                value[i][j] = col_spatial_score;
                delta[i][j] = col_spatial_score - col_delta;
                col_edits[i][j] = (traj2.get(j - 1).x(), traj2.get(j - 1).y());
                row_edits[i][j] = t1_insert.unwrap_or((traj1.get(i).x(), traj1.get(i).y()));
            } else {
                value[i][j] = row_spatial_score;
                delta[i][j] = row_spatial_score - row_delta;
                col_edits[i][j] = t2_insert.unwrap_or((traj2.get(j).x(), traj2.get(j).y()));
                row_edits[i][j] = (traj1.get(i - 1).x(), traj1.get(i - 1).y());
            }
        }
    }

    // Get final distance before consuming value
    let final_distance = value[t1_len - 1][t2_len - 1];

    // Flatten matrix for return (row-major order)
    let matrix_flat: Vec<f64> = value.into_iter().flatten().collect();

    DpResult {
        distance: final_distance,
        matrix: Some(matrix_flat),
    }
}

/// EDwP with rolling array optimization
///
/// This version uses O(min(n,m)) memory by only keeping the previous row
/// of the DP matrix. This is the recommended version for production use.
fn edwp_rolling_array<T: CoordSequence>(traj1: &T, traj2: &T) -> DpResult
where
    T::Coord: AsCoord,
{
    let t1_len = traj1.len();
    let t2_len = traj2.len();

    // Compute edge lengths
    let mut t1_edge_length = Vec::with_capacity(t1_len.saturating_sub(1));
    for i in 0..t1_len.saturating_sub(1) {
        let p1 = traj1.get(i);
        let p2 = traj1.get(i + 1);
        t1_edge_length.push(euclidean_distance(&p1, &p2));
    }

    let mut t2_edge_length = Vec::with_capacity(t2_len.saturating_sub(1));
    for i in 0..t2_len.saturating_sub(1) {
        let p1 = traj2.get(i);
        let p2 = traj2.get(i + 1);
        t2_edge_length.push(euclidean_distance(&p1, &p2));
    }

    let total_length: f64 = t1_edge_length.iter().sum::<f64>() + t2_edge_length.iter().sum::<f64>();

    // Initialize DP arrays
    let mut prev_value = vec![f64::MAX; t2_len];
    let mut prev_delta = vec![0.0; t2_len];
    let mut prev_col_edits = vec![(0.0, 0.0); t2_len];
    let mut prev_row_edits = vec![(0.0, 0.0); t2_len];

    let mut curr_value = vec![f64::MAX; t2_len];
    let mut curr_delta = vec![0.0; t2_len];
    let mut curr_col_edits = vec![(0.0, 0.0); t2_len];
    let mut curr_row_edits = vec![(0.0, 0.0); t2_len];

    // Initialize first row (value[0][0] = 0.0, value[0][1:] = f64::MAX)
    prev_value[0] = 0.0;

    // Fill DP matrix using rolling arrays
    for i in 1..t1_len {
        curr_value[0] = f64::MAX;

        for j in 1..t2_len {
            let mut row_delta = f64::MAX;
            let mut col_delta = f64::MAX;
            let mut row_spatial_score = f64::MAX;
            let mut col_spatial_score = f64::MAX;
            let mut t1_insert: Option<(f64, f64)> = None;
            let mut t2_insert: Option<(f64, f64)> = None;

            // Row operation (insert from traj2)
            if i > 1 {
                let t1_edit = prev_row_edits[j];
                let t2_edit = prev_col_edits[j];
                let t1_edit_arr = [t1_edit.0, t1_edit.1];
                let t2_edit_arr = [t2_edit.0, t2_edit.1];
                let prev_point_edge = euclidean_distance(&traj1.get(i - 1), &t1_edit_arr);

                // Project point onto segment (equivalent to _line_map in Python)
                // _line_map(p1=t2_edit, p2=t2[j], p=t1[i-1])
                let projected =
                    project_point_to_segment(&traj1.get(i - 1), &t2_edit_arr, &traj2.get(j));
                let t2_insert_arr = [projected.0, projected.1];
                t2_insert = Some(projected);

                let row_edit_distance = euclidean_distance(&traj1.get(i - 1), &t2_insert_arr);
                let row_edit_edge = euclidean_distance(&t2_edit_arr, &t2_insert_arr);

                let row_converge1 = (row_edit_edge + prev_point_edge) / total_length;
                let row_converge2 = (euclidean_distance(&traj2.get(j), &t2_insert_arr)
                    + t1_edge_length[i - 1])
                    / total_length;

                row_delta = prev_value[j] - prev_delta[j]
                    + (row_edit_distance + euclidean_distance(&t1_edit_arr, &t2_edit_arr))
                        * row_converge1;
                row_spatial_score = row_delta
                    + (row_edit_distance + euclidean_distance(&traj2.get(j), &traj1.get(i)))
                        * row_converge2;
            }

            // Column operation (insert from traj1)
            if j > 1 {
                let t1_edit = curr_row_edits[j - 1];
                let t2_edit = curr_col_edits[j - 1];
                let t1_edit_arr = [t1_edit.0, t1_edit.1];
                let t2_edit_arr = [t2_edit.0, t2_edit.1];

                let prev_point_edge = euclidean_distance(&traj2.get(j - 1), &t2_edit_arr);

                // Project point onto segment (equivalent to _line_map in Python)
                // _line_map(p1=t1_edit, p2=t1[i], p=t2[j-1])
                let projected =
                    project_point_to_segment(&traj2.get(j - 1), &t1_edit_arr, &traj1.get(i));
                let t1_insert_arr = [projected.0, projected.1];
                t1_insert = Some(projected);

                let col_edit_distance = euclidean_distance(&traj2.get(j - 1), &t1_insert_arr);
                let col_edit_edge = euclidean_distance(&t1_edit_arr, &t1_insert_arr);

                let col_converge1 = (col_edit_edge + prev_point_edge) / total_length;
                let col_converge2 = (euclidean_distance(&traj1.get(i), &t1_insert_arr)
                    + t2_edge_length[j - 1])
                    / total_length;

                col_delta = curr_value[j - 1] - curr_delta[j - 1]
                    + (col_edit_distance + euclidean_distance(&t1_edit_arr, &t2_edit_arr))
                        * col_converge1;
                col_spatial_score = col_delta
                    + (col_edit_distance + euclidean_distance(&traj1.get(i), &traj2.get(j)))
                        * col_converge2;
            }

            // Diagonal operation (match points)
            let diag_coverage = (t1_edge_length[i - 1] + t2_edge_length[j - 1]) / total_length;
            let sub_score = (euclidean_distance(&traj2.get(j), &traj1.get(i))
                + euclidean_distance(&traj2.get(j - 1), &traj1.get(i - 1)))
                * diag_coverage;
            let diag_score = prev_value[j - 1] + sub_score;

            // Choose minimum operation
            if diag_score <= col_spatial_score && diag_score <= row_spatial_score {
                curr_value[j] = diag_score;
                curr_delta[j] = diag_score - prev_value[j - 1];
                curr_col_edits[j] = (traj2.get(j - 1).x(), traj2.get(j - 1).y());
                curr_row_edits[j] = (traj1.get(i - 1).x(), traj1.get(i - 1).y());
            } else if col_spatial_score < row_spatial_score
                || (col_spatial_score == row_spatial_score && t2_len > t1_len)
            {
                curr_value[j] = col_spatial_score;
                curr_delta[j] = col_spatial_score - col_delta;
                curr_col_edits[j] = (traj2.get(j - 1).x(), traj2.get(j - 1).y());
                curr_row_edits[j] = t1_insert.unwrap_or((traj1.get(i).x(), traj1.get(i).y()));
            } else {
                curr_value[j] = row_spatial_score;
                curr_delta[j] = row_spatial_score - row_delta;
                curr_col_edits[j] = t2_insert.unwrap_or((traj2.get(j).x(), traj2.get(j).y()));
                curr_row_edits[j] = (traj1.get(i - 1).x(), traj1.get(i - 1).y());
            }
        }

        // Swap current and previous rows
        std::mem::swap(&mut prev_value, &mut curr_value);
        std::mem::swap(&mut prev_delta, &mut curr_delta);
        std::mem::swap(&mut prev_col_edits, &mut curr_col_edits);
        std::mem::swap(&mut prev_row_edits, &mut curr_row_edits);
    }

    DpResult {
        distance: prev_value[t2_len - 1],
        matrix: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edwp_simple() {
        let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
        let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];

        let result = edwp(&traj1, &traj2, false);
        assert!(result.distance > 0.0);
        assert!(result.distance.is_finite());
    }

    #[test]
    fn test_edwp_identical() {
        let traj1 = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];

        let result = edwp(&traj1, &traj1, false);
        assert!(result.distance < 1e-6);
    }

    #[test]
    fn test_edwp_empty() {
        let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
        let traj2: Vec<[f64; 2]> = vec![];

        let result = edwp(&traj1, &traj2, false);
        assert_eq!(result.distance, f64::MAX);
    }

    #[test]
    fn test_edwp_single_point() {
        let traj1 = vec![[0.0, 0.0]];
        let traj2 = vec![[1.0, 1.0]];

        let result = edwp(&traj1, &traj2, false);
        // Single point trajectories should have a finite distance
        assert!(result.distance.is_finite());
    }

    #[test]
    fn test_edwp_with_matrix_consistency() {
        let traj1 = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let traj2 = vec![[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]];

        let result_full = edwp(&traj1, &traj2, true);
        let result_rolling = edwp(&traj1, &traj2, false);

        // Both should produce the same distance
        assert!(
            (result_full.distance - result_rolling.distance).abs() < 1e-6,
            "Distance mismatch: full={}, rolling={}",
            result_full.distance,
            result_rolling.distance
        );

        // Full matrix should return a matrix
        assert!(result_full.matrix.is_some());
        assert!(result_full.matrix.as_ref().unwrap().len() == traj1.len() * traj2.len());

        // Rolling array should not return a matrix
        assert!(result_rolling.matrix.is_none());
    }

    #[test]
    fn test_edwp_matrix_dimensions() {
        let traj1 = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let traj2 = vec![[0.1, 0.1], [1.1, 1.1]];

        let result = edwp(&traj1, &traj2, true);

        assert!(result.matrix.is_some());
        let matrix = result.matrix.as_ref().unwrap();
        assert_eq!(matrix.len(), traj1.len() * traj2.len());
    }

    #[test]
    fn test_edwp_symmetry() {
        let traj1 = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let traj2 = vec![[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]];

        let result1 = edwp(&traj1, &traj2, false);
        let result2 = edwp(&traj2, &traj1, false);

        // EDwP should be symmetric
        assert!(
            (result1.distance - result2.distance).abs() < 1e-6,
            "Symmetry check failed: d(T1,T2)={}, d(T2,T1)={}",
            result1.distance,
            result2.distance
        );
    }

    #[test]
    fn test_edwp_matrix_consistency_diverse() {
        // Test with multiple diverse trajectory pairs
        let test_cases = vec![
            (vec![[0.0, 0.0], [1.0, 1.0]], vec![[0.1, 0.1], [1.1, 1.1]]),
            (
                vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                vec![[0.1, 0.1], [1.1, 1.1]],
            ),
            (
                vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                vec![[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]],
            ),
        ];

        for (traj1, traj2) in test_cases {
            let result_full = edwp(&traj1, &traj2, true);
            let result_rolling = edwp(&traj1, &traj2, false);

            assert!(
                (result_full.distance - result_rolling.distance).abs() < 1e-6,
                "Distance mismatch for traj1.len()={}, traj2.len()={}: full={}, rolling={}",
                traj1.len(),
                traj2.len(),
                result_full.distance,
                result_rolling.distance
            );
        }
    }
}
