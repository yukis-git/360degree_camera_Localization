/-
  SlamIndex.lean

  Lean 4 (core library only, no Mathlib) verification of state-vector
  indexing invariants used by the EKF-SLAM implementation in
  `SLAM_on_cpu.py`.

  See `README.md` in this directory for a plain-language summary of
  what each theorem says about the Python (and future Rust) code.
-/

namespace SlamIndex

/- ============================================================
   PART 1. State-vector layout.

   The believe/cov state vector is laid out as
     [ pose (6 entries) | landmark 0 (3) | landmark 1 (3) | ... | landmark (n-1) (3) ]
   so landmark `id` occupies the half-open slice
     [6 + 3*id, 6 + 3*id + 3).
   This matches `EKF_SLAM.get_LM_Pos_from_state` (SLAM_on_cpu.py:215-216)
   and the various `lm_start = self.pose_size + minid * self.lm_size`
   computations (e.g. SLAM_on_cpu.py:233, 241, 373).
   ============================================================ -/

/-- Half-open interval of indices `[a, b)`, as membership on `Nat`. -/
def InSlice (a b i : Nat) : Prop := a ≤ i ∧ i < b

/-- The slice occupied by landmark `id` inside a state vector holding `n` landmarks. -/
def lmSlice (id i : Nat) : Prop := InSlice (6 + 3 * id) (6 + 3 * id + 3) i

/-- The slice occupied by the 6-entry pose block. -/
def poseSlice (i : Nat) : Prop := InSlice 0 6 i

/-- The whole state vector for `n` landmarks has length `6 + 3*n`. -/
def stateLen (n : Nat) : Nat := 6 + 3 * n

/-- 1a. Every landmark slice for a valid id lies inside the full state vector. -/
theorem lmSlice_subset_state {n id i : Nat} (hid : id < n) (hi : lmSlice id i) :
    InSlice 0 (stateLen n) i := by
  unfold lmSlice InSlice stateLen at *
  omega

/-- 1b. Landmark slices for distinct ids never overlap. -/
theorem lmSlice_disjoint {id1 id2 i : Nat} (hne : id1 ≠ id2)
    (h1 : lmSlice id1 i) (h2 : lmSlice id2 i) : False := by
  unfold lmSlice InSlice at h1 h2
  omega

/-- 1c. The pose slice never overlaps any landmark slice. -/
theorem pose_disjoint_lm {id i : Nat} (h1 : poseSlice i) (h2 : lmSlice id i) : False := by
  unfold poseSlice at h1
  unfold lmSlice at h2
  unfold InSlice at h1 h2
  omega

/-- Sanity check: the landmark slice for `id` is exactly 3 wide, so it really
    matches `self.believe[pose_size + lm_size*id : pose_size + lm_size*(id+1)]`. -/
theorem lmSlice_width (id : Nat) : ∀ i, lmSlice id i ↔ (6 + 3 * id ≤ i ∧ i < 6 + 3 * id + 3) := by
  intro i; unfold lmSlice InSlice; exact Iff.rfl

/- ============================================================
   PART 2. Deletion / remap invariant.

   `search_correspond_LM_ID` (SLAM_on_cpu.py:248-319) deletes a batch of
   "duplicate" landmarks in one shot and then has to translate the
   surviving candidate id `minid_candidate` into the *new* numbering:

       kept_lm_ids = np.where(~delete_candidates_mask)[0]
       new_minid_candidate = np.where(kept_lm_ids == minid_candidate)[0][0]

   We model `delete_candidates_mask` as a boolean predicate `m : Nat → Bool`
   (`m i = true` means "landmark i is deleted") and show that the position
   of a kept index `minid` inside the kept list is exactly the count of
   *earlier* kept indices, and that this count is a valid index into the
   kept list. This is exactly what `np.where(kept_lm_ids == minid)[0][0]`
   computes.
   ============================================================ -/

/-- Number of *kept* (i.e. `m i = false`) indices among `0, 1, ..., k-1`. -/
def cnt (m : Nat → Bool) : Nat → Nat
  | 0 => 0
  | k + 1 => cnt m k + (if m k then 0 else 1)

/-- The list of kept indices among `0, 1, ..., n-1`, in increasing order. -/
def keptAux (m : Nat → Bool) : Nat → List Nat
  | 0 => []
  | k + 1 => if m k then keptAux m k else keptAux m k ++ [k]

theorem cnt_le (m : Nat → Bool) (k : Nat) : cnt m k ≤ k := by
  induction k with
  | zero => simp [cnt]
  | succ k ih =>
    unfold cnt
    split <;> omega

theorem length_keptAux (m : Nat → Bool) (n : Nat) : (keptAux m n).length = cnt m n := by
  induction n with
  | zero => simp [keptAux, cnt]
  | succ n ih =>
    unfold keptAux cnt
    split
    · simpa using ih
    · simp [ih]

/-- `cnt m` is monotone: kept-counts of longer prefixes are at least as large. -/
theorem cnt_step (m : Nat → Bool) (k : Nat) : cnt m k ≤ cnt m (k + 1) := by
  show cnt m k ≤ cnt m k + (if m k then 0 else 1)
  split <;> omega

theorem cnt_mono (m : Nat → Bool) : ∀ k n, k ≤ n → cnt m k ≤ cnt m n := by
  intro k n h
  induction n with
  | zero => have : k = 0 := Nat.le_zero.mp h; simp [this]
  | succ n ih =>
    by_cases hk : k ≤ n
    · exact Nat.le_trans (ih hk) (cnt_step m n)
    · have : k = n + 1 := by omega
      simp [this]

/-- If `k` is kept, its position in `keptAux m n` (for any `n > k`) is `cnt m k`,
    and that position actually stores `k`. This is the key remap lemma:
    `cnt m k` plays the role of `new_minid_candidate` /
    `np.where(kept_lm_ids == minid)[0][0]`. -/
theorem getElem?_keptAux_cnt (m : Nat → Bool) (k : Nat) (hk : m k = false) :
    ∀ n, k < n → (keptAux m n)[cnt m k]? = some k := by
  intro n
  induction n with
  | zero => intro h; omega
  | succ n ih =>
    intro hn
    by_cases hkn : k < n
    · -- k < n: use the induction hypothesis, then show the extra step at `n`
      -- doesn't disturb position `cnt m k` (it lies strictly before it, since
      -- `cnt m k ≤ cnt m n ≤ (keptAux m n).length`).
      have ihn := ih hkn
      unfold keptAux
      split
      · exact ihn
      · rw [List.getElem?_append_left]
        · exact ihn
        · exact (List.getElem?_eq_some.mp ihn).1
    · -- k ≥ n and k < n + 1, so k = n
      have hkeq : k = n := by omega
      subst hkeq
      unfold keptAux
      rw [if_neg (by simpa using hk)]
      rw [List.getElem?_append_right (Nat.le_of_eq (length_keptAux m k))]
      have heq0 : cnt m k - (keptAux m k).length = 0 := by
        rw [length_keptAux]; omega
      rw [heq0]
      simp

/-- The position `cnt m minid` is a valid index into `keptAux m n` whenever
    `minid` is kept and `minid < n`, i.e. it lies within the surviving state
    vector — matching that `np.where(...)[0][0]` always finds something when
    `minid_candidate ∈ kept_lm_ids`. -/
theorem cnt_lt_length_keptAux (m : Nat → Bool) (n minid : Nat)
    (hmin : minid < n) (hk : m minid = false) :
    cnt m minid < (keptAux m n).length := by
  have h := getElem?_keptAux_cnt m minid hk n hmin
  exact (List.getElem?_eq_some.mp h).1

/-- New state-vector length after deleting every landmark `i < n` with `m i = true`:
    `6 + 3 * (number kept)`, matching `len(self.believe)` after the batched
    `np.delete` in `search_correspond_LM_ID` (SLAM_on_cpu.py:308-313), or the
    single-landmark case in `lm_delete` (SLAM_on_cpu.py:324-334) where exactly
    one landmark is removed. -/
theorem new_state_len (m : Nat → Bool) (n : Nat) :
    stateLen (keptAux m n).length = 6 + 3 * cnt m n := by
  rw [length_keptAux]; rfl

/-- Wiring the abstract `Nat → Bool` mask to the `List Bool` shape described in
    the task: a length-`n` mask list, read with a "delete" default so that
    out-of-range lookups don't matter. -/
def maskFn (mask : List Bool) : Nat → Bool := fun i => mask.getD i true

/-- Concrete corollary of `getElem?_keptAux_cnt` / `cnt_lt_length_keptAux` phrased
    directly in terms of a `List Bool` mask of length `n`, `remap`, and `kept`,
    exactly as in the task statement. -/
theorem mask_remap (mask : List Bool) (n minid : Nat) (_hlen : mask.length = n)
    (hmin : minid < n) (hk : mask.getD minid true = false) :
    let remap := cnt (maskFn mask) minid
    let kept := keptAux (maskFn mask) n
    remap < kept.length ∧ kept[remap]? = some minid := by
  have hk' : maskFn mask minid = false := hk
  exact ⟨cnt_lt_length_keptAux (maskFn mask) n minid hmin hk',
         getElem?_keptAux_cnt (maskFn mask) minid hk' n hmin⟩

/- ============================================================
   PART 3. Equirectangular pixel-column branch partition.

   `Observation._trans_func_vectorized` (SLAM_on_cpu.py:103-120) branches on
     cond1 : 0 <= x <  1.5*H     (p_val + picv = 0.5*H + H = 1.5*H)
     cond2 : 1.5*H <= x <= 2*H
   To stay in `Nat` we double everything (`2*x` vs `3*H`) since `1.5*H` is
   not an integer in general.
   ============================================================ -/

/-- Branch 1 of `_trans_func_vectorized`: `x < 1.5*H`, written as `2*x < 3*H`. -/
def Branch1 (H x : Nat) : Prop := 2 * x < 3 * H

/-- Branch 2 of `_trans_func_vectorized`: `1.5*H <= x <= 2*H`, written with `2*x`. -/
def Branch2 (H x : Nat) : Prop := 3 * H ≤ 2 * x ∧ x ≤ 2 * H

/-- Every pixel column `x ≤ 2*H` falls into at least one of the two branches,
    and never into both: the code's `if`/`elif` covers `[0, 2H]` exactly once. -/
theorem branch_partition (H x : Nat) (hx : x ≤ 2 * H) :
    (Branch1 H x ∨ Branch2 H x) ∧ ¬ (Branch1 H x ∧ Branch2 H x) := by
  unfold Branch1 Branch2
  omega

/- ============================================================
   PART 4 (bonus). Appending a new landmark.

   `observation_update`'s "new landmark" branch (SLAM_on_cpu.py:354-362)
   appends exactly `lm_size = 3` new entries to `self.believe`. We show the
   new landmark's slice is exactly the 3 freshly appended indices.
   ============================================================ -/

/-- After growing from `n` to `n+1` landmarks, the new landmark (id = n) occupies
    exactly the slice `[stateLen n, stateLen n + 3)`, i.e. precisely the tail
    that got appended by `np.concatenate((self.believe, self.calc_LM_Pos(z, R_mat)))`. -/
theorem new_landmark_slice_is_tail (n i : Nat) :
    lmSlice n i ↔ InSlice (stateLen n) (stateLen n + 3) i := by
  unfold lmSlice InSlice stateLen
  omega

end SlamIndex
