"""Property 5: SLAM_on_cpu.py landmark-deletion index bookkeeping.

Source: SLAM_on_cpu.py `search_correspond_LM_ID` (lines 248-319), plus
pose_size/lm_size (lines 163-164: pose_size=6, lm_size=3).

Relevant excerpt (paraphrased with line numbers):
  288: minid_candidate = argmin(mahalanobis_distances)
  291: delete_candidates_mask = (mahalanobis_distances < Duplication)
  294: delete_candidates_mask[minid_candidate] = False   # never delete argmin
  297: indices_to_delete_lm = where(delete_candidates_mask)[0]
  313: self.nLM -= len(indices_to_delete_lm)
  314: kept_lm_ids = where(~delete_candidates_mask)[0]
  315-317: if minid_candidate in kept_lm_ids:
               new_minid_candidate = where(kept_lm_ids == minid_candidate)[0][0]
               minid_candidate = new_minid_candidate
           else: return self.nLM   # (dead branch: minid is always kept,
                                    # since line 294 excludes it from the mask)

We model this with Z3 Bools for an nLM-length delete mask and a symbolic
integer `minid` (0 <= minid < nLM, mask[minid] == False, matching line 294),
with nLM bounded (<= 8, as explicitly permitted by the task) so that all
sums below are finite, concrete-indexed expressions -- fully decidable
linear arithmetic over Booleans/Ints (no quantifiers needed).

We prove, for ALL such mask/minid combinations:
  (1) new_minid == minid - #{deleted indices < minid}
      (this is the array.where(~mask) remapping used at 314-317)
  (2) 0 <= new_minid < nLM_new  where nLM_new = nLM - #deleted
  (3) the landmark's state slice [6+3*new_minid, 6+3*new_minid+3) lies
      within the post-deletion state vector length 6+3*nLM_new
      (pose_size=6, lm_size=3, lines 163-164, 260, 300-302, 325-326).
"""
import sys
from z3 import Bool, Int, Sum, If, And, Not, Implies
from common import prove

POSE_SIZE = 6
LM_SIZE = 3


def build_model(nLM):
    mask = [Bool(f"mask_{i}") for i in range(nLM)]
    minid = Int("minid")

    in_range = And(minid >= 0, minid < nLM)
    minid_not_deleted = Not(_index(mask, minid, nLM))  # emulate mask[minid]==False

    deleted_total = Sum([If(mask[i], 1, 0) for i in range(nLM)])
    nLM_new = nLM - deleted_total

    deleted_before_minid = Sum([If(And(i < minid, mask[i]), 1, 0) for i in range(nLM)])
    rank_among_kept = Sum([If(And(i < minid, Not(mask[i])), 1, 0) for i in range(nLM)])

    return mask, minid, in_range, minid_not_deleted, deleted_total, nLM_new, deleted_before_minid, rank_among_kept


def _index(mask, minid, nLM):
    """mask[minid] where minid is a symbolic Int in [0,nLM): built as a
    chain of If(minid==i, mask[i], ...) -- equivalent to array indexing."""
    expr = mask[nLM - 1]
    for i in range(nLM - 2, -1, -1):
        expr = If(minid == i, mask[i], expr)
    return expr


def main():
    nLM = 8  # bounded, as explicitly permitted by the task description
    ok = True

    mask, minid, in_range, minid_not_deleted, deleted_total, nLM_new, \
        deleted_before_minid, rank_among_kept = build_model(nLM)

    assumptions = [in_range, minid_not_deleted]

    # (1) remapped index == minid - #{deleted indices < minid}
    ok = prove(
        f"new_minid == minid - #deleted_before_minid  (nLM={nLM})",
        Implies(And(*assumptions), rank_among_kept == minid - deleted_before_minid),
    ) and ok

    # (2) remapped index is a valid index into the shrunk landmark array
    ok = prove(
        f"0 <= new_minid < nLM_new  (nLM={nLM})",
        Implies(And(*assumptions), And(rank_among_kept >= 0, rank_among_kept < nLM_new)),
    ) and ok

    # (3) state-vector slice bounds: [6+3*new_minid, 6+3*new_minid+3)
    #     must lie within [0, 6+3*nLM_new)  (SLAM_on_cpu.py:163-164,
    #     216, 233-234, 300-302, 325-326 all use this 6+3*id addressing).
    slice_start = POSE_SIZE + LM_SIZE * rank_among_kept
    slice_end = slice_start + LM_SIZE
    state_len_new = POSE_SIZE + LM_SIZE * nLM_new
    ok = prove(
        f"state slice [6+3*new_minid, 6+3*new_minid+3) within [0, 6+3*nLM_new)  (nLM={nLM})",
        Implies(And(*assumptions), And(slice_start >= 0, slice_end <= state_len_new)),
    ) and ok

    print("ALL PROVED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
