"""Property 2: Euler-angle rotation-matrix Jacobians.

R = Rz(psi) Ry(theta) Rx(phi). Ground truth derivatives are obtained with
sympy by differentiating R w.r.t. the actual angle (phi/theta/psi) using
real sin/cos functions, THEN rewriting cos(angle)/sin(angle) in terms of
free (s,c) symbol pairs (each constrained by s^2+c^2=1) so the result can
be handed to Z3 as a plain polynomial identity.

Two families are checked against that ground truth:
  (a) EKF_Euler.py:  euler_to_R (16-26), dR_dphi (29-38), dR_dtheta (41-50),
      dR_dpsi (53-62). Angle order: c1,s1=cos/sin(psi), c2,s2=cos/sin(theta),
      c3,s3=cos/sin(phi).
  (b) SLAM_on_cpu.py: Rotate_mat (12-15), P_roll_Rotate (18-21),
      P_pitch_Rotate (24-27), P_yaw_Rotate (30-33), argument order
      (Sr,Cr,Sp,Cp,Sy,Cy). SLAM_on_gpu.py (15-33) carries identical
      formulas (cupy instead of numpy) and is checked the same way.

Both families are algebraically the SAME matrix R=Rz*Ry*Rx, with
(Sr,Cr)=(s3,c3)=roll/phi, (Sp,Cp)=(s2,c2)=pitch/theta, (Sy,Cy)=(s1,c1)=yaw/psi.

Usage:
  python3 euler_jacobian.py            # check against the CORRECT reference
                                        # (SLAM.P_roll_Rotate uses the fixed
                                        # formula here)
  python3 euler_jacobian.py --source   # check the verbatim SOURCE
                                        # transcription instead -- surfaces
                                        # the known P_roll_Rotate[1][2] bug
"""
import sys
import sympy as sp
from z3 import Reals
from common import prove_equal


# ---------------------------------------------------------------------
# Ground truth via real sympy differentiation (ADR: not a substitution
# trick -- entries that do not depend on an angle correctly differentiate
# to 0, which a naive (s,c)->(c,-s) textual substitution would get wrong).
# ---------------------------------------------------------------------
_phi, _theta, _psi = sp.symbols('phi theta psi', real=True)


def _R_sym():
    c1, s1 = sp.cos(_psi), sp.sin(_psi)
    c2, s2 = sp.cos(_theta), sp.sin(_theta)
    c3, s3 = sp.cos(_phi), sp.sin(_phi)
    return sp.Matrix([
        [c1*c2, c1*s2*s3 - s1*c3, c1*s2*c3 + s1*s3],
        [s1*c2, s1*s2*s3 + c1*c3, s1*s2*c3 - c1*s3],
        [-s2, c2*s3, c2*c3],
    ])


def _resymbol(mat, c1n, s1n, c2n, s2n, c3n, s3n):
    """Rewrite cos/sin(psi/theta/phi) as the given symbol names."""
    subs = {sp.cos(_psi): c1n, sp.sin(_psi): s1n,
            sp.cos(_theta): c2n, sp.sin(_theta): s2n,
            sp.cos(_phi): c3n, sp.sin(_phi): s3n}
    return sp.simplify(mat.subs(subs))


def sympy_ground_truth(which, c1n, s1n, c2n, s2n, c3n, s3n):
    R = _R_sym()
    var = {'psi': _psi, 'theta': _theta, 'phi': _phi}[which]
    dR = sp.diff(R, var)
    return _resymbol(dR, c1n, s1n, c2n, s2n, c3n, s3n)


def sympy_mat_to_pylist(mat):
    return [[mat[i, j] for j in range(3)] for i in range(3)]


def sympy_to_z3(expr, var_map):
    """Minimal, exact sympy -> z3 expression converter for the polynomial
    (integer-coefficient) expressions that appear here."""
    from z3 import RealVal
    expr = sp.expand(expr)
    if expr.is_Symbol:
        return var_map[str(expr)]
    if expr.is_Integer:
        return RealVal(int(expr))
    if expr.is_Rational:
        return RealVal(expr.p) / RealVal(expr.q)
    if expr.is_Add:
        terms = [sympy_to_z3(a, var_map) for a in expr.args]
        r = terms[0]
        for t in terms[1:]:
            r = r + t
        return r
    if expr.is_Mul:
        factors = [sympy_to_z3(a, var_map) for a in expr.args]
        r = factors[0]
        for f in factors[1:]:
            r = r * f
        return r
    if expr.is_Pow:
        base = sympy_to_z3(expr.args[0], var_map)
        exp = int(expr.args[1])
        r = base
        for _ in range(exp - 1):
            r = r * base
        return r
    raise ValueError(f"cannot convert sympy expr: {expr} ({type(expr)})")


def ground_truth_z3(which, z3names, symnames):
    """which in {'phi','theta','psi'} (family a) mapped through symnames to
    the z3 variable-name tuple used by the caller."""
    c1n, s1n, c2n, s2n, c3n, s3n = symnames
    dR = sympy_ground_truth(which, c1n, s1n, c2n, s2n, c3n, s3n)
    var_map = dict(zip(symnames, z3names))
    return [[sympy_to_z3(dR[i, j], var_map) for j in range(3)] for i in range(3)]


# ---------- Family (a): EKF_Euler.py verbatim transcriptions ----------
def src_dR_dphi(c1, s1, c2, s2, c3, s3):
    # EKF_Euler.py:34-38
    return [
        [0, c1*s2*c3 + s1*s3, -c1*s2*s3 + s1*c3],
        [0, s1*s2*c3 - c1*s3, -s1*s2*s3 - c1*c3],
        [0, c2*c3, -c2*s3],
    ]


def src_dR_dtheta(c1, s1, c2, s2, c3, s3):
    # EKF_Euler.py:46-50
    return [
        [-c1*s2, c1*c2*s3, c1*c2*c3],
        [-s1*s2, s1*c2*s3, s1*c2*c3],
        [-c2, -s2*s3, -s2*c3],
    ]


def src_dR_dpsi(c1, s1, c2, s2, c3, s3):
    # EKF_Euler.py:58-62
    return [
        [-s1*c2, -s1*s2*s3 - c1*c3, -s1*s2*c3 + c1*s3],
        [c1*c2, c1*s2*s3 - s1*c3, c1*s2*c3 + s1*s3],
        [0, 0, 0],
    ]


# ---------- Family (b): SLAM_on_cpu.py / SLAM_on_gpu.py verbatim ----------
def src_P_roll_Rotate(Sr, Cr, Sp, Cp, Sy, Cy):
    # SLAM_on_cpu.py:18-21 == SLAM_on_gpu.py:20-23
    # NOTE: entry [1][2] below is "-Sy*Sp*Cr-Cy*Cr" in the source, where the
    # correct derivative (see ground truth) has "-Sy*Sp*Sr-Cy*Cr" -- this
    # is the known bug (see README.md).
    return [[0, Cy*Sp*Cr + Sy*Sr, -Cy*Sp*Sr + Sy*Cr],
            [0, Sy*Sp*Cr - Cy*Sr, -Sy*Sp*Cr - Cy*Cr],
            [0, Cp*Cr, -Cp*Sr]]


def src_P_pitch_Rotate(Sr, Cr, Sp, Cp, Sy, Cy):
    # SLAM_on_cpu.py:24-27 == SLAM_on_gpu.py:25-28
    return [[-Cy*Sp, Cy*Cp*Sr, Cy*Cp*Cr],
            [-Sy*Sp, Sy*Cp*Sr, Sy*Cp*Cr],
            [-Cp, -Sp*Sr, -Sp*Cr]]


def src_P_yaw_Rotate(Sr, Cr, Sp, Cp, Sy, Cy):
    # SLAM_on_cpu.py:30-33 == SLAM_on_gpu.py:30-33
    return [[-Sy*Cp, -Sy*Sp*Sr - Cy*Cr, -Sy*Sp*Cr + Cy*Sr],
            [Cy*Cp, Cy*Sp*Sr - Sy*Cr, Cy*Sp*Cr + Sy*Sr],
            [0, 0, 0]]


def ref_P_roll_Rotate(Sr, Cr, Sp, Cp, Sy, Cy):
    # CORRECTED version of src_P_roll_Rotate: Sr instead of Cr at [1][2].
    return [[0, Cy*Sp*Cr + Sy*Sr, -Cy*Sp*Sr + Sy*Cr],
            [0, Sy*Sp*Cr - Cy*Sr, -Sy*Sp*Sr - Cy*Cr],
            [0, Cp*Cr, -Cp*Sr]]


def check_family_a():
    c1, s1, c2, s2, c3, s3 = Reals('c1 s1 c2 s2 c3 s3')
    assumptions = [c1**2 + s1**2 == 1, c2**2 + s2**2 == 1, c3**2 + s3**2 == 1]
    ok = True
    symnames = ('c1', 's1', 'c2', 's2', 'c3', 's3')
    z3names = (c1, s1, c2, s2, c3, s3)
    cases = [
        ('dR_dphi', src_dR_dphi, ground_truth_z3('phi', z3names, symnames)),
        ('dR_dtheta', src_dR_dtheta, ground_truth_z3('theta', z3names, symnames)),
        ('dR_dpsi', src_dR_dpsi, ground_truth_z3('psi', z3names, symnames)),
    ]
    for name, src_fn, truth_mat in cases:
        src_mat = src_fn(c1, s1, c2, s2, c3, s3)
        for i in range(3):
            for j in range(3):
                e = prove_equal(f"EKF_Euler.{name}[{i}][{j}] == d(R)/d{name.split('_')[1]}",
                                 src_mat[i][j], truth_mat[i][j], assumptions)
                ok = ok and e
    return ok


def check_family_b(source_mode):
    Sr, Cr, Sp, Cp, Sy, Cy = Reals('Sr Cr Sp Cp Sy Cy')
    assumptions = [Sr**2 + Cr**2 == 1, Sp**2 + Cp**2 == 1, Sy**2 + Cy**2 == 1]
    ok = True
    # Family (b) uses the SAME R with (c1,s1,c2,s2,c3,s3) <-> (Cy,Sy,Cp,Sp,Cr,Sr)
    symnames = ('Cy', 'Sy', 'Cp', 'Sp', 'Cr', 'Sr')
    z3names = (Cy, Sy, Cp, Sp, Cr, Sr)
    truth = {
        'P_roll_Rotate': ground_truth_z3('phi', z3names, symnames),
        'P_pitch_Rotate': ground_truth_z3('theta', z3names, symnames),
        'P_yaw_Rotate': ground_truth_z3('psi', z3names, symnames),
    }
    cases = [
        ('P_roll_Rotate', src_P_roll_Rotate, ref_P_roll_Rotate),
        ('P_pitch_Rotate', src_P_pitch_Rotate, src_P_pitch_Rotate),
        ('P_yaw_Rotate', src_P_yaw_Rotate, src_P_yaw_Rotate),
    ]
    for name, src_fn, ref_fn in cases:
        chosen_fn = src_fn if source_mode else ref_fn
        mode_tag = "SOURCE" if source_mode else "REFERENCE"
        mat = chosen_fn(Sr, Cr, Sp, Cp, Sy, Cy)
        truth_mat = truth[name]
        for i in range(3):
            for j in range(3):
                e = prove_equal(f"SLAM.{name}[{i}][{j}] ({mode_tag}) == d(R)/d{name.split('_')[1]}",
                                 mat[i][j], truth_mat[i][j], assumptions)
                ok = ok and e
    return ok


def main():
    source_mode = '--source' in sys.argv
    print(f"mode: {'source transcription' if source_mode else 'reference (correct) formulas'}")
    ok_a = check_family_a()
    ok_b = check_family_b(source_mode)
    ok = ok_a and ok_b
    print("ALL PROVED" if ok else "SOME FAILED (see COUNTEREXAMPLE lines above)")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
