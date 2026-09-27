//! EKF-SLAM state vector / covariance layout helpers.
//!
//! Ported from `SLAM_on_cpu.py`'s `EKF_SLAM` class:
//!   - `self.pose_size = len(self.believe)` == 6 (line ~163)
//!   - `self.lm_size = 3` (line ~164)
//!   - `search_correspond_LM_ID` (line ~248): when landmarks with small
//!     Mahalanobis distance (but not the closest match) are deleted, the closest
//!     match's own id is remapped to its new position among the *kept* landmarks
//!     (line ~314-317): `kept_lm_ids = np.where(~delete_candidates_mask)[0]`,
//!     `new_minid_candidate = np.where(kept_lm_ids == minid_candidate)[0][0]`.
//!     If the id being remapped was itself deleted, the Python code takes a
//!     different path entirely (returns `self.nLM`, meaning "not found / new
//!     landmark"); [`remap_after_delete`] instead reports that case as `None`, and
//!     callers should treat `None` however their own "not found" sentinel works.
//!   - `lm_delete` (line ~324) and the mahalanobis-based bulk delete both do
//!     `np.delete(believe, indices)` and `np.delete(cov, indices, axis=0/1)`,
//!     i.e. dropping the corresponding 3-wide row/column block from both the state
//!     vector and the (square, row-major) covariance matrix. [`delete_landmarks`]
//!     is the batched equivalent.

use core::ops::Range;

/// Size of the pose block at the head of the state vector (x, y, z, roll, pitch, yaw).
pub const POSE_SIZE: usize = 6;

/// Size of a single landmark block in the state vector (x, y, z).
pub const LM_SIZE: usize = 3;

/// Total state vector length for `n_lm` landmarks.
pub fn state_len(n_lm: usize) -> usize {
    POSE_SIZE + LM_SIZE * n_lm
}

/// The half-open index range of landmark `id`'s 3 components within the state
/// vector, or `None` if `id >= n_lm`.
///
/// Mirrors `EKF_SLAM.get_LM_Pos_from_state`:
/// `believe[pose_size + lm_size*id : pose_size + lm_size*(id+1)]`.
pub fn lm_range(id: usize, n_lm: usize) -> Option<Range<usize>> {
    if id >= n_lm {
        return None;
    }
    let start = POSE_SIZE + LM_SIZE * id;
    Some(start..start + LM_SIZE)
}

/// Remap a pre-deletion landmark id (`minid`) to its index among the landmarks
/// that survive a deletion pass, given a `deleted_mask` of length `n_lm` (`true`
/// = that landmark index is being deleted).
///
/// Returns `None` when:
///   - `minid >= deleted_mask.len()` (out of range), or
///   - `deleted_mask[minid]` is `true` (the id itself was deleted, so it has no
///     surviving position -- matches the Python `else: return self.nLM` branch's
///     intent of "not the same landmark anymore", even though the Python code's
///     actual sentinel value differs; see module docs).
///
/// Otherwise, returns `minid` minus the number of deleted landmarks with an
/// index strictly less than `minid` -- i.e. its position within the kept subset,
/// preserving order (same as `np.where(kept_lm_ids == minid_candidate)[0][0]`
/// with `kept_lm_ids` sorted ascending).
pub fn remap_after_delete(minid: usize, deleted_mask: &[bool]) -> Option<usize> {
    if minid >= deleted_mask.len() || deleted_mask[minid] {
        return None;
    }
    let deleted_before = deleted_mask[..minid].iter().filter(|&&d| d).count();
    Some(minid - deleted_before)
}

/// Errors for [`delete_landmarks`].
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DeleteLandmarksError {
    /// `mask.len() != n_lm`.
    MaskLenMismatch,
    /// `state.len() != POSE_SIZE + LM_SIZE * n_lm`.
    StateLenMismatch,
    /// `cov.len() != (POSE_SIZE + LM_SIZE * n_lm)^2`.
    CovLenMismatch,
}

/// Delete the landmarks flagged in `mask` (length `n_lm`, `true` = delete) from
/// both `state` (length `POSE_SIZE + LM_SIZE * n_lm`) and `cov` (a row-major
/// `(POSE_SIZE + LM_SIZE * n_lm)`-square matrix), compacting both in place.
///
/// Equivalent to the Python pattern (see module docs):
/// ```python
/// believe = np.delete(believe, global_indices_to_delete)
/// cov = np.delete(cov, global_indices_to_delete, axis=0)
/// cov = np.delete(cov, global_indices_to_delete, axis=1)
/// ```
/// The pose block (first `POSE_SIZE` rows/cols) is always kept.
///
/// On success, returns the new landmark count (`n_lm` minus the number of `true`
/// entries in `mask`) and leaves `state`/`cov` resized and compacted, preserving
/// the relative order of surviving landmarks and the pose block.
pub fn delete_landmarks(
    state: &mut Vec<f64>,
    cov: &mut Vec<f64>,
    n_lm: usize,
    mask: &[bool],
) -> Result<usize, DeleteLandmarksError> {
    if mask.len() != n_lm {
        return Err(DeleteLandmarksError::MaskLenMismatch);
    }
    let old_total = state_len(n_lm);
    if state.len() != old_total {
        return Err(DeleteLandmarksError::StateLenMismatch);
    }
    if cov.len() != old_total * old_total {
        return Err(DeleteLandmarksError::CovLenMismatch);
    }

    // Build the list of kept global indices: the whole pose block, then each
    // surviving landmark's 3-wide block, in original order.
    let mut kept: Vec<usize> = (0..POSE_SIZE).collect();
    for id in 0..n_lm {
        if !mask[id] {
            let start = POSE_SIZE + LM_SIZE * id;
            kept.extend(start..start + LM_SIZE);
        }
    }
    let new_total = kept.len();

    let mut new_state = Vec::with_capacity(new_total);
    for &gi in &kept {
        new_state.push(state[gi]);
    }

    let mut new_cov = Vec::with_capacity(new_total * new_total);
    for &row in &kept {
        for &col in &kept {
            new_cov.push(cov[row * old_total + col]);
        }
    }

    *state = new_state;
    *cov = new_cov;

    let new_n_lm = n_lm - mask.iter().filter(|&&d| d).count();
    Ok(new_n_lm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lm_range_basic() {
        assert_eq!(lm_range(0, 2), Some(6..9));
        assert_eq!(lm_range(1, 2), Some(9..12));
        assert_eq!(lm_range(2, 2), None); // out of range
        assert_eq!(lm_range(0, 0), None);
    }

    #[test]
    fn remap_after_delete_basic() {
        // 4 landmarks, delete #1 and #3. Kept: 0, 2 -> new positions 0, 1.
        let mask = [false, true, false, true];
        assert_eq!(remap_after_delete(0, &mask), Some(0));
        assert_eq!(remap_after_delete(2, &mask), Some(1));
        assert_eq!(remap_after_delete(1, &mask), None); // deleted itself
        assert_eq!(remap_after_delete(3, &mask), None); // deleted itself
        assert_eq!(remap_after_delete(4, &mask), None); // out of range
    }

    #[test]
    fn remap_after_delete_no_deletions() {
        let mask = [false, false, false];
        assert_eq!(remap_after_delete(0, &mask), Some(0));
        assert_eq!(remap_after_delete(2, &mask), Some(2));
    }

    #[test]
    fn delete_landmarks_compacts_state_and_cov() {
        let n_lm = 3;
        let total = state_len(n_lm); // 6 + 9 = 15
        let mut state: Vec<f64> = (0..total).map(|i| i as f64).collect();
        let mut cov: Vec<f64> = (0..total * total).map(|i| i as f64).collect();

        // Delete landmark #1 (middle one), keep #0 and #2.
        let mask = [false, true, false];
        let new_n = delete_landmarks(&mut state, &mut cov, n_lm, &mask).unwrap();

        assert_eq!(new_n, 2);
        let new_total = state_len(new_n);
        assert_eq!(state.len(), new_total);
        assert_eq!(cov.len(), new_total * new_total);

        // Pose block preserved.
        assert_eq!(&state[0..6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        // Landmark #0 (indices 6..9) preserved, landmark #2 (indices 12..15) now at 9..12.
        assert_eq!(&state[6..9], &[6.0, 7.0, 8.0]);
        assert_eq!(&state[9..12], &[12.0, 13.0, 14.0]);

        // Spot-check covariance: new_cov[row][col] should equal old cov[kept[row]][kept[col]].
        let kept = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14];
        for (nr, &or) in kept.iter().enumerate() {
            for (nc, &oc) in kept.iter().enumerate() {
                assert_eq!(cov[nr * new_total + nc], (or * total + oc) as f64);
            }
        }
    }

    #[test]
    fn delete_landmarks_len_mismatch_errors() {
        let mut state = vec![0.0; 6];
        let mut cov = vec![0.0; 36];
        let mask = [false, true]; // n_lm says 0 but mask has 2 entries
        assert_eq!(
            delete_landmarks(&mut state, &mut cov, 0, &mask),
            Err(DeleteLandmarksError::MaskLenMismatch)
        );

        let mut state2 = vec![0.0; 5]; // wrong length for n_lm=0
        let mut cov2 = vec![0.0; 36];
        assert_eq!(
            delete_landmarks(&mut state2, &mut cov2, 0, &[]),
            Err(DeleteLandmarksError::StateLenMismatch)
        );

        let mut state3 = vec![0.0; 6];
        let mut cov3 = vec![0.0; 10]; // wrong length
        assert_eq!(
            delete_landmarks(&mut state3, &mut cov3, 0, &[]),
            Err(DeleteLandmarksError::CovLenMismatch)
        );
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    const MAX_N_LM: usize = 4;

    #[kani::proof]
    #[kani::unwind(6)]
    fn lm_range_is_within_state_bounds() {
        let n_lm: usize = kani::any();
        kani::assume(n_lm <= MAX_N_LM);
        let id: usize = kani::any();
        kani::assume(id <= MAX_N_LM);

        if let Some(r) = lm_range(id, n_lm) {
            assert!(r.start >= POSE_SIZE);
            assert!(r.end <= state_len(n_lm));
            assert_eq!(r.end - r.start, LM_SIZE);
        } else {
            assert!(id >= n_lm);
        }
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn remap_after_delete_points_to_same_landmark_and_is_in_bounds() {
        let n_lm: usize = kani::any();
        kani::assume(n_lm <= MAX_N_LM);

        let mut mask = [false; MAX_N_LM];
        let mut deleted_count = 0usize;
        for i in 0..MAX_N_LM {
            if i < n_lm {
                mask[i] = kani::any();
                if mask[i] {
                    deleted_count += 1;
                }
            }
        }
        let mask_slice = &mask[..n_lm];

        let minid: usize = kani::any();
        kani::assume(minid <= MAX_N_LM);

        let result = remap_after_delete(minid, mask_slice);

        match result {
            None => {
                // Either out of range, or the landmark itself was deleted.
                assert!(minid >= n_lm || mask_slice[minid]);
            }
            Some(new_id) => {
                assert!(minid < n_lm);
                assert!(!mask_slice[minid]);
                let new_count = n_lm - deleted_count;
                assert!(new_id < new_count);
                // Reconstruct: new_id must equal minid minus deletions before it.
                let deleted_before = mask_slice[..minid].iter().filter(|&&d| d).count();
                assert_eq!(new_id, minid - deleted_before);
            }
        }
    }

    // `delete_landmarks` internally loops over `kept` (up to POSE_SIZE + LM_SIZE *
    // n_lm elements) and, for the covariance matrix, over `kept x kept` (up to
    // that squared). Kani's `#[kani::unwind(k)]` is a single global bound that
    // must exceed the largest loop trip count in the harness, so proving this at
    // MAX_N_LM = 4 (an 18x18 = 324-iteration inner loop) needs an unwind bound
    // that made this harness intractable in the time available. It is instead
    // bounded at a smaller `DELETE_MAX_N_LM` (see README.md "制限" section).
    const DELETE_MAX_N_LM: usize = 2;

    #[kani::proof]
    #[kani::unwind(150)]
    fn delete_landmarks_preserves_sizes_and_kept_entries() {
        let n_lm: usize = kani::any();
        kani::assume(n_lm <= DELETE_MAX_N_LM);

        let mut mask = [false; DELETE_MAX_N_LM];
        for i in 0..n_lm {
            mask[i] = kani::any();
        }
        let mask_slice = &mask[..n_lm];

        let total = state_len(n_lm);
        let mut state: Vec<f64> = Vec::with_capacity(total);
        for i in 0..total {
            state.push(i as f64);
        }
        let mut cov: Vec<f64> = Vec::with_capacity(total * total);
        for i in 0..total * total {
            cov.push(i as f64);
        }

        let result = delete_landmarks(&mut state, &mut cov, n_lm, mask_slice);
        assert!(result.is_ok());
        let new_n_lm = result.unwrap();

        let expected_new_total = state_len(new_n_lm);
        assert_eq!(state.len(), expected_new_total);
        assert_eq!(cov.len(), expected_new_total * expected_new_total);

        // Pose block must be preserved exactly (first POSE_SIZE entries, values 0..POSE_SIZE).
        let mut i = 0;
        while i < POSE_SIZE {
            assert_eq!(state[i], i as f64);
            i += 1;
        }
    }
}
