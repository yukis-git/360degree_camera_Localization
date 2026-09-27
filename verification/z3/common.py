"""Small shared helpers for the Z3 proof scripts in this directory.

Not a source module of the main project -- verification-only code.
"""
from z3 import Solver, Not, And, unsat, sat, Tactic, Or


def _solver():
    # qfnra-nlsat handles the nonlinear (degree >= 2) real-arithmetic goals
    # that show up throughout these proofs (products of sin/cos surrogates,
    # quaternion components, etc.) far more reliably than the default tactic.
    try:
        return Tactic('qfnra-nlsat').solver()
    except Exception:
        return Solver()


def prove_equal(name, lhs, rhs, assumptions=None):
    """Prove lhs == rhs for all values of the free variables subject to
    `assumptions`. Prints PROVED or COUNTEREXAMPLE and returns True/False."""
    s = _solver()
    if assumptions:
        s.add(And(*assumptions) if isinstance(assumptions, (list, tuple)) else assumptions)
    s.add(Not(lhs == rhs))
    r = s.check()
    if r == unsat:
        print(f"PROVED: {name}")
        return True
    elif r == sat:
        print(f"COUNTEREXAMPLE: {name}: {s.model()}")
        return False
    else:
        print(f"UNKNOWN: {name} (solver returned unknown)")
        return False


def prove(name, claim, assumptions=None):
    """Prove that `claim` (a boolean Z3 expression) holds for all values of
    the free variables subject to `assumptions`."""
    s = _solver()
    if assumptions:
        s.add(And(*assumptions) if isinstance(assumptions, (list, tuple)) else assumptions)
    s.add(Not(claim))
    r = s.check()
    if r == unsat:
        print(f"PROVED: {name}")
        return True
    elif r == sat:
        print(f"COUNTEREXAMPLE: {name}: {s.model()}")
        return False
    else:
        print(f"UNKNOWN: {name} (solver returned unknown)")
        return False


def mat_equal(name, A, B, assumptions=None):
    """Prove two equal-shaped matrices (list-of-lists of Z3 expressions) are
    entrywise equal for all free variables. Reports each entry."""
    ok = True
    for i in range(len(A)):
        for j in range(len(A[0])):
            e = prove_equal(f"{name}[{i}][{j}]", A[i][j], B[i][j], assumptions)
            ok = ok and e
    return ok
