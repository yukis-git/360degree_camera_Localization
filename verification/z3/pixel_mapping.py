"""Property 4: equirectangular pixel -> (phi,theta) mapping.

Source:
  SLAM_on_cpu.py `_trans_func_vectorized` (lines 103-120), called with
  picv = H (frame height); pixel x in [0, 2H], y in [0, H]. p_val=q_val=H/2,
  pi_div_picv = pi/H.
    branch1 (0 <= x < 1.5H): phi_theta = -pi/H * (x,y) + pi/H * (H/2, H/2)
    branch2 (1.5H <= x <= 2H): phi_theta = -pi/H*(x,y) + pi/H*(1.5H, H/2)
                                            + (pi, 0)

  ExKalmanFilter.py `calc_obs` (lines 132-149), with W = self.W = 1280
  (resized frame width, W = 2H):
    y = -2*pi/W * means + (pi/2, pi/2)
    i.e. phi = pi/2 - 2*pi/W * x_pix,  theta = pi/2 - 2*pi/W * y_pix

`pi` is treated as a free real symbol with pi > 0 (the mapping is LINEAR in
pi so no numeric approximation is needed for the range/agreement proofs);
for the numeric range claims [-pi,pi]/[-pi/2,pi/2] pi is bounded to
3.14159 < pi < 3.1416 as instructed.
"""
import sys
from z3 import Reals, Real, And, Or, Not, Implies
from common import prove


def trans_func_branch1(x, y, H, pi):
    pi_div_H = pi / H
    phi = -pi_div_H * x + pi_div_H * (H / 2)
    theta = -pi_div_H * y + pi_div_H * (H / 2)
    return phi, theta


def trans_func_branch2(x, y, H, pi):
    pi_div_H = pi / H
    phi = -pi_div_H * x + pi_div_H * (H / 2 + H) + pi
    theta = -pi_div_H * y + pi_div_H * (H / 2)
    return phi, theta


def aruco_calc_obs(xpix, ypix, W, pi):
    # ExKalmanFilter.py:142 -- phi = pi/2 - 2pi/W * xpix, theta likewise for y
    phi = -2 * pi / W * xpix + pi / 2
    theta = -2 * pi / W * ypix + pi / 2
    return phi, theta


def main():
    ok = True
    H, x, y, pi = Reals('H x y pi')
    H_pos = H > 0
    pi_bounds = And(pi > 3.14159, pi < 3.1416)
    pi_symbolic = pi > 0

    # (1) Range of phi, theta over each branch, with numeric pi bounds.
    x_in_branch1 = And(x >= 0, x < 1.5 * H)
    x_in_branch2 = And(x >= 1.5 * H, x <= 2 * H)
    y_in_range = And(y >= 0, y <= H)

    phi1, theta1 = trans_func_branch1(x, y, H, pi)
    ok = prove("branch1: phi in [-pi,pi]",
               Implies(And(H_pos, pi_bounds, x_in_branch1, y_in_range),
                       And(phi1 >= -pi, phi1 <= pi))) and ok
    ok = prove("branch1: theta in [-pi/2,pi/2]",
               Implies(And(H_pos, pi_bounds, x_in_branch1, y_in_range),
                       And(theta1 >= -pi/2, theta1 <= pi/2))) and ok

    phi2, theta2 = trans_func_branch2(x, y, H, pi)
    ok = prove("branch2: phi in [-pi,pi]",
               Implies(And(H_pos, pi_bounds, x_in_branch2, y_in_range),
                       And(phi2 >= -pi, phi2 <= pi))) and ok
    ok = prove("branch2: theta in [-pi/2,pi/2]",
               Implies(And(H_pos, pi_bounds, x_in_branch2, y_in_range),
                       And(theta2 >= -pi/2, theta2 <= pi/2))) and ok

    # (2) Branches agree modulo 2*pi at x = 1.5H (the shared boundary).
    #     branch1's limit as x->1.5H (exclusive) should equal branch2's
    #     value at x=1.5H, modulo 2*pi.
    x_boundary = 1.5 * H
    phi1_b, theta1_b = trans_func_branch1(x_boundary, y, H, pi)
    phi2_b, theta2_b = trans_func_branch2(x_boundary, y, H, pi)
    ok = prove("branches agree mod 2pi at x=1.5H: phi2-phi1 == 2*pi",
               Implies(And(H_pos, pi_symbolic),
                       phi2_b - phi1_b == 2 * pi)) and ok
    ok = prove("branches agree at x=1.5H: theta identical (not just mod 2pi)",
               Implies(H_pos, theta2_b == theta1_b)) and ok

    # (3) Masks partition [0, 2H]: every x in [0,2H] is in exactly one of
    #     cond1_mask, cond2_mask (SLAM_on_cpu.py:114,117), no gap, no overlap.
    x_dom = And(x >= 0, x <= 2 * H)
    cond1 = And(x >= 0, x < 1.5 * H)
    cond2 = And(x >= 1.5 * H, x <= 2 * H)
    ok = prove("masks cover [0,2H] (no pixel left unmapped)",
               Implies(And(H_pos, x_dom), Or(cond1, cond2))) and ok
    ok = prove("masks are disjoint (no pixel counted twice)",
               Implies(H_pos, Not(And(cond1, cond2)))) and ok

    # (4) Equality with ArUco mapping (ExKalmanFilter.py:142), W = 2H,
    #     and pixel coords scaled by W/(2H) = 1 (both already use the same
    #     pixel domain once W=2H, since ArUco's `means` is in the resized
    #     frame's own pixel coordinates of width W).
    W = 2 * H
    phi_a1, theta_a1 = aruco_calc_obs(x, y, W, pi)
    # branch1 region: phi/theta should match exactly (both linear forms
    # pi/2 - (pi/H)*coord)
    ok = prove("ArUco phi == branch1 phi (exact, same domain)",
               Implies(And(H_pos, x_in_branch1),
                       phi_a1 == phi1)) and ok
    ok = prove("ArUco theta == branch1/branch2 theta (exact, always)",
               Implies(H_pos, theta_a1 == theta1)) and ok
    # branch2 region: phi should match modulo 2*pi
    phi_a2, _ = aruco_calc_obs(x, y, W, pi)
    ok = prove("ArUco phi == branch2 phi (mod 2*pi)",
               Implies(And(H_pos, x_in_branch2, pi_symbolic),
                       Or(phi_a2 == phi2, phi_a2 == phi2 - 2*pi, phi_a2 == phi2 + 2*pi))) and ok

    print("ALL PROVED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
