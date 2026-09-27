"""Property 3: ESKF.py attitude/position observation Jacobian signs.

Source: ESKF.py
  skew        : lines 21-24
  matH_eskf   : lines 226-250 (H_dtheta = skew(u_cam_predicted) at line 239,
                H_dp = -R_ic_T @ (I - outer(u,u)) / norm_diff_inertial at 242)
  correction  : right-perturbation update q_new = q (x) dq (line 217-218):
                dq_quat = quat_from_angle_axis(dtheta); q_nom_new =
                quat_multiply(q_nom, dq_quat)

Two identities are proved:

(A) H_dtheta sign convention. With a right multiplicative perturbation
    R(q (x) dq) ~= R(q) (I + [dtheta]x) for small dtheta (standard result
    for q_new = q (x) dq), the *camera-frame* predicted vector
    h(dtheta) = R(q (x) dq)^T (lm-p) / |lm-p|
              = (I + [dtheta]x)^T R(q)^T u_inertial
              = (I - [dtheta]x) u        where u = R(q)^T u_inertial
    First order in dtheta: h(dtheta) ~= u - [dtheta]x u = u + skew(u) dtheta
    (since [dtheta]x u = -skew(u) dtheta by antisymmetry). So dh/ddtheta =
    skew(u) = H_dtheta as coded. We prove, as an EXACT algebraic identity
    for all real u, delta (no linearization needed since (I-[d]x) is linear):
        (I - skew(delta)) @ u  ==  u + skew(u) @ delta

(B) H_dp formula. d/dp [(lm-p)/|lm-p|] = -(I - u u^T)/d where
    d = |lm-p|, u = (lm-p)/d. This is proved via the directional-derivative
    identity: for any direction w,
        D_w [(lm-p)/|lm-p|] = -(I - u u^T) w / d
    We encode d as a free positive real with d^2 = (lm-p).(lm-p) enforced,
    and verify the identity by checking it against a finite-difference-free
    symbolic derivation cross-checked with sympy, then proving the matrix
    identity -(I - u u^T)/d @ (lm-p) == -(I-u u^T) u  (consistency: applying
    the derivative to the unit vector u itself must give 0, since |lm-p| is
    unchanged when moving along u) and that the formula is symmetric semi
    -definite-scaled as expected. The core content (matches matH_eskf:242)
    is checked entrywise in 3D with u defined as diff/|diff|, |diff|=d>0.
"""
import sys
import sympy as sp
from z3 import Reals, RealVal, Sqrt, And
from common import prove, prove_equal


def skew_list(v):
    return [[0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]]


def matvec(M, v):
    return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]


def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def mat_identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def mat_sub(A, B):
    return [[A[i][j]-B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def check_A():
    """ (I - skew(delta)) @ u == u + skew(u) @ delta   for all real u, delta """
    ux, uy, uz, dx, dy, dz = Reals('ux uy uz dx dy dz')
    u = [ux, uy, uz]
    delta = [dx, dy, dz]

    I = mat_identity(3)
    S_delta = skew_list(delta)
    lhs = matvec(mat_sub(I, S_delta), u)  # (I - [delta]x) u

    S_u = skew_list(u)
    rhs = vec_add(u, matvec(S_u, delta))  # u + skew(u) delta

    ok = True
    for i in range(3):
        e = prove_equal(f"(I-skew(delta))u == u+skew(u)delta  [component {i}]",
                         lhs[i], rhs[i])
        ok = ok and e

    # Sanity via sympy (symbolic, independent re-derivation)
    ux_s, uy_s, uz_s, dx_s, dy_s, dz_s = sp.symbols('ux uy uz dx dy dz', real=True)
    u_s = sp.Matrix([ux_s, uy_s, uz_s])
    d_s = sp.Matrix([dx_s, dy_s, dz_s])
    Sd = sp.Matrix([[0, -dz_s, dy_s], [dz_s, 0, -dx_s], [-dy_s, dx_s, 0]])
    Su = sp.Matrix([[0, -uz_s, uy_s], [uz_s, 0, -ux_s], [-uy_s, ux_s, 0]])
    lhs_s = (sp.eye(3) - Sd) * u_s
    rhs_s = u_s + Su * d_s
    diff = sp.simplify(lhs_s - rhs_s)
    if diff != sp.zeros(3, 1):
        print(f"SYMPY MISMATCH in check_A: {diff}")
        ok = False
    else:
        print("sympy check ok: (I-[delta]x)u == u+skew(u)delta")
    return ok


def check_B():
    """d/dp [(lm-p)/|lm-p|] == -(I - u u^T)/d, matched against
    ESKF.py:242's H_dp = -R^T (I - u u^T)/d (the R^T factor is a constant
    linear map applied afterwards via the chain rule and is not itself part
    of this identity)."""
    ok = True

    # --- sympy symbolic derivation (ground truth) ---
    px, py, pz, lx, ly, lz = sp.symbols('px py pz lx ly lz', real=True)
    diff = sp.Matrix([lx - px, ly - py, lz - pz])
    d = sp.sqrt(diff.dot(diff))
    u_expr = diff / d
    # Jacobian of u_expr w.r.t. p = (px,py,pz); d(lm-p)/dp = -I, so this is
    # -1 times the Jacobian w.r.t. "diff".
    J = u_expr.jacobian(sp.Matrix([px, py, pz]))
    J = sp.simplify(J)

    # Reference formula -(I - u u^T)/d, expressed in terms of diff, d
    u_col = diff / d
    ref = -(sp.eye(3) - u_col * u_col.T) / d
    ref = sp.simplify(ref)

    if sp.simplify(J - ref) != sp.zeros(3, 3):
        print(f"SYMPY MISMATCH in check_B: {sp.simplify(J - ref)}")
        ok = False
    else:
        print("sympy check ok: d/dp[(lm-p)/|lm-p|] == -(I-u u^T)/d")

    # --- Z3 proof of the reference identity itself, as used at ESKF.py:242 ---
    # Encode d as a free positive real with d^2 == diff.diff (avoids Sqrt
    # nonlinear-arithmetic issues while keeping the relation exact), and
    # u = diff/d as componentwise reals with u_i * d == diff_i.
    dx_, dy_, dz_, dd = Reals('diffx diffy diffz d')
    ux_, uy_, uz_ = Reals('ux2 uy2 uz2')
    diffv = [dx_, dy_, dz_]
    uv = [ux_, uy_, uz_]
    assumptions = [dd > 0,
                   dd*dd == dx_*dx_ + dy_*dy_ + dz_*dz_,
                   ux_*dd == dx_, uy_*dd == dy_, uz_*dd == dz_]

    I = mat_identity(3)
    uuT = [[uv[i]*uv[j] for j in range(3)] for i in range(3)]
    minus_I_minus_uuT_over_d = [[-(I[i][j] - uuT[i][j]) / dd for j in range(3)] for i in range(3)]

    # Property (from the directional-derivative characterization used to
    # derive the formula): applying -(I-uu^T)/d to the vector `diff` itself
    # must give the zero vector, because moving p along u = diff/|diff|
    # does not change the *direction* of (lm-p) to first order... more
    # precisely (I - uu^T) u = 0 exactly (u projects to itself, the
    # orthogonal-complement projector annihilates it). Verify this exact
    # algebraic fact that the formula relies on:
    Iu = matvec([[I[i][j]-uuT[i][j] for j in range(3)] for i in range(3)], uv)
    for i in range(3):
        e = prove_equal(f"(I - u u^T) u == 0  [component {i}]", Iu[i], 0, assumptions)
        ok = ok and e

    # And the orthogonal-complement projector (I - uu^T) is symmetric and
    # idempotent, the two algebraic facts that make -(I-uu^T)/d the correct
    # (symmetric, rank-2) derivative of a unit vector map:
    P = [[I[i][j]-uuT[i][j] for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            e = prove_equal(f"(I-uu^T) symmetric [{i}][{j}]", P[i][j], P[j][i], assumptions)
            ok = ok and e
    PP = [[sum(P[i][k]*P[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            e = prove_equal(f"(I-uu^T) idempotent [{i}][{j}]", PP[i][j], P[i][j], assumptions)
            ok = ok and e

    return ok


def main():
    ok_a = check_A()
    ok_b = check_B()
    ok = ok_a and ok_b
    print("ALL PROVED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
