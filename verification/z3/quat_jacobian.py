"""Property 1: quaternion rotation-matrix Jacobians (ExKalmanFilter.py).

Source: ExKalmanFilter.py
  matR   : lines 11-14
  PRPww  : lines 16-19
  PRPwx  : lines 21-24
  PRPwy  : lines 26-29
  PRPwz  : lines 31-34

matR(ww,wx,wy,wz) =
  [[1-2(wy^2+wz^2),   2(wx*wy-ww*wz),   2(ww*wy+wx*wz)],
   [2(ww*wz+wx*wy),   1-2(wx^2+wz^2),   2(wy*wz-ww*wx)],
   [2(wx*wz-ww*wy),   2(ww*wx+wy*wz),   1-2(wx^2+wy^2)]]

We first use sympy to symbolically differentiate matR w.r.t. each of
ww, wx, wy, wz, then transcribe PRPww/PRPwx/PRPwy/PRPwz verbatim from the
source and use Z3 to prove, entrywise and for ALL real ww,wx,wy,wz
(no unit-quaternion constraint needed -- this is a plain polynomial
identity), that PRPxx == d(matR)/dxx.

We additionally prove matR is orthogonal (R^T R = I) whenever
ww^2+wx^2+wy^2+wz^2 = 1.
"""
import sys
import sympy as sp
from z3 import Reals, RealVal
from common import prove_equal, mat_equal, prove


def sympy_matR():
    ww, wx, wy, wz = sp.symbols('ww wx wy wz', real=True)
    R = sp.Matrix([
        [1 - 2*(wy**2 + wz**2), 2*(wx*wy - ww*wz), 2*(ww*wy + wx*wz)],
        [2*(ww*wz + wx*wy), 1 - 2*(wx**2 + wz**2), 2*(wy*wz - ww*wx)],
        [2*(wx*wz - ww*wy), 2*(ww*wx + wy*wz), 1 - 2*(wx**2 + wy**2)],
    ])
    return R, (ww, wx, wy, wz)


def sympy_derivative_matrix(var_index):
    R, syms = sympy_matR()
    return sp.simplify(R.diff(syms[var_index]))


def z3_matR(ww, wx, wy, wz):
    return [
        [1 - 2*(wy**2 + wz**2), 2*(wx*wy - ww*wz), 2*(ww*wy + wx*wz)],
        [2*(ww*wz + wx*wy), 1 - 2*(wx**2 + wz**2), 2*(wy*wz - ww*wx)],
        [2*(wx*wz - ww*wy), 2*(ww*wx + wy*wz), 1 - 2*(wx**2 + wy**2)],
    ]


# Verbatim transcriptions of PRPww/PRPwx/PRPwy/PRPwz (ExKalmanFilter.py:16-34)
def z3_PRPww(ww, wx, wy, wz):
    return [[2*0.0, 2*(-wz), 2*wy],
            [2*wz, 2*0.0, 2*(-wx)],
            [2*(-wy), 2*wx, 2*0.0]]


def z3_PRPwx(ww, wx, wy, wz):
    return [[2*0.0, 2*wy, 2*wz],
            [2*wy, 2*(-2*wx), 2*(-ww)],
            [2*wz, 2*ww, 2*(-2*wx)]]


def z3_PRPwy(ww, wx, wy, wz):
    return [[2*(-2*wy), 2*wx, 2*ww],
            [2*wx, 2*0.0, 2*wz],
            [2*(-ww), 2*wz, 2*(-2*wy)]]


def z3_PRPwz(ww, wx, wy, wz):
    return [[2*(-2*wz), 2*(-ww), 2*wx],
            [2*ww, 2*(-2*wz), 2*wy],
            [2*wx, 2*wy, 2*0.0]]


def hardcoded_derivative(var_index, ww, wx, wy, wz):
    """Hand-derived (and sympy-cross-checked, see main()) d(matR)/d(var)."""
    if var_index == 0:  # d/d ww
        return [[0, -2*wz, 2*wy], [2*wz, 0, -2*wx], [-2*wy, 2*wx, 0]]
    if var_index == 1:  # d/d wx
        return [[0, 2*wy, 2*wz], [2*wy, -4*wx, -2*ww], [2*wz, 2*ww, -4*wx]]
    if var_index == 2:  # d/d wy
        return [[-4*wy, 2*wx, 2*ww], [2*wx, 0, 2*wz], [-2*ww, 2*wz, -4*wy]]
    if var_index == 3:  # d/d wz
        return [[-4*wz, -2*ww, 2*wx], [2*ww, -4*wz, 2*wy], [2*wx, 2*wy, 0]]


def main():
    ok = True

    # --- sympy sanity check: hardcoded derivative matches sympy.diff ---
    for idx, name in enumerate(['ww', 'wx', 'wy', 'wz']):
        R, syms = sympy_matR()
        sym_d = sp.simplify(R.diff(syms[idx]))
        hc = sp.Matrix(hardcoded_derivative(idx, *syms))
        diff = sp.simplify(sym_d - hc)
        if diff != sp.zeros(3, 3):
            print(f"SYMPY MISMATCH for d/d{name}: {diff}")
            ok = False
        else:
            print(f"sympy check ok: hardcoded d(matR)/d{name} matches sympy.diff")

    # --- Z3 proofs: PRPxx (transcribed from source) == d(matR)/dxx ---
    ww, wx, wy, wz = Reals('ww wx wy wz')
    PRP_funcs = [z3_PRPww, z3_PRPwx, z3_PRPwy, z3_PRPwz]
    names = ['ww', 'wx', 'wy', 'wz']
    for idx, (fn, name) in enumerate(zip(PRP_funcs, names)):
        source_mat = fn(ww, wx, wy, wz)
        deriv_mat = hardcoded_derivative(idx, ww, wx, wy, wz)
        ok = mat_equal(f"PRP{name} == d(matR)/d{name}", source_mat, deriv_mat) and ok

    # --- Orthogonality: R^T R = I given unit quaternion ---
    R = z3_matR(ww, wx, wy, wz)
    unit = (ww**2 + wx**2 + wy**2 + wz**2 == 1)
    RT_R = [[sum(R[k][i]*R[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    I = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            e = prove_equal(f"(R^T R)[{i}][{j}] == I[{i}][{j}] | unit quat",
                             RT_R[i][j], I[i][j], assumptions=[unit])
            ok = ok and e

    print("ALL PROVED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
