"""
Finite-difference checks of every analytic observation Jacobian against its
observation function, over ~50 random (seeded) cases each.

Central differences, eps=1e-6. Landmarks are kept >= 0.5 m from the camera
and Euler pitch is kept away from +-pi/2 to avoid gimbal-lock degeneracies.
"""
import numpy as np
import pytest
from types import SimpleNamespace

import ExKalmanFilter as EKFq
import EKF_Euler as EKFe
import EKF_Euler_pv as EKFepv
import ESKF as ESKFmod
import SLAM_on_cpu as SLAM

N_CASES = 50
EPS = 1e-6
MIN_DIST = 0.5


def numerical_jacobian(f, x0, eps=EPS):
    """Central-difference Jacobian of vector function f at x0."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    f0 = np.asarray(f(x0), dtype=float)
    m = f0.size
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        f_plus = np.asarray(f(x0 + dx), dtype=float)
        f_minus = np.asarray(f(x0 - dx), dtype=float)
        J[:, i] = (f_plus - f_minus) / (2 * eps)
    return J


def random_unit_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def random_landmark(rng, p, lo=1.0, hi=4.0):
    while True:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        r = rng.uniform(lo, hi)
        lm = p + direction * r
        if np.linalg.norm(lm - p) >= MIN_DIST:
            return lm


# ---------------------------------------------------------------------------
# ExKalmanFilter (quaternion EKF)
# ---------------------------------------------------------------------------

def test_exkf_matH_matches_numerical_obs_func():
    rng = np.random.default_rng(0)
    dummy_map = SimpleNamespace(map=np.zeros((1, 3)))
    ekf = EKFq.ExKalmanFilter(map=dummy_map)

    failures = []
    for case in range(N_CASES):
        q = random_unit_quat(rng)
        p = rng.uniform(-2, 2, size=3)
        v = rng.uniform(-1, 1, size=3)
        pose = np.concatenate([q, p, v])
        lm = random_landmark(rng, p)

        ekf.pose = pose.copy()
        H_analytic = ekf.matH(lm)

        def f(x, lm=lm):
            ekf.pose = x
            return ekf.obs_func(lm)

        H_num = numerical_jacobian(f, pose)
        ekf.pose = pose

        if not np.allclose(H_analytic, H_num, atol=1e-4, rtol=1e-3):
            failures.append(case)

    assert not failures, f"ExKalmanFilter.matH mismatched numerical jac in cases {failures}"


# ---------------------------------------------------------------------------
# EKF_Euler / EKF_Euler_pv
# ---------------------------------------------------------------------------

def _random_euler_pose(rng, extra_dims=0):
    phi = rng.uniform(-np.pi, np.pi)
    psi = rng.uniform(-np.pi, np.pi)
    theta = rng.uniform(-1.3, 1.3)  # keep away from +-pi/2 (~1.5708)
    p = rng.uniform(-2, 2, size=3)
    extra = rng.uniform(-1, 1, size=extra_dims) if extra_dims else np.array([])
    return np.concatenate([[phi, theta, psi], p, extra])


@pytest.mark.parametrize("mod,dim", [(EKFe, 6), (EKFepv, 9)])
def test_euler_ekf_matH_matches_numerical_obs_func(mod, dim):
    rng = np.random.default_rng(1 if dim == 6 else 2)
    dummy_map = SimpleNamespace(map=np.zeros((1, 3)))
    ekf = mod.ExKalmanFilter(map=dummy_map)

    failures = []
    for case in range(N_CASES):
        extra_dims = dim - 6
        pose = _random_euler_pose(rng, extra_dims=extra_dims)
        p = pose[3:6]
        lm = random_landmark(rng, p)

        ekf.pose = pose.copy()
        H_analytic = ekf.matH(lm)

        def f(x, lm=lm):
            ekf.pose = x
            return ekf.obs_func(lm)

        H_num = numerical_jacobian(f, pose)
        ekf.pose = pose

        if not np.allclose(H_analytic, H_num, atol=1e-4, rtol=1e-3):
            failures.append(case)

    assert not failures, f"{mod.__name__}.ExKalmanFilter.matH mismatched numerical jac in cases {failures}"


# ---------------------------------------------------------------------------
# ESKF: error-state Jacobian vs numerical derivative of the error injection
# ---------------------------------------------------------------------------

def _inject_error_state(pose, delta):
    """pose is the 10-dim nominal [q(4), p(3), v(3)]; delta is the 9-dim
    error state [dtheta(3), dp(3), dv(3)]. Returns the injected 10-dim pose."""
    q = pose[0:4].copy()
    p = pose[4:7].copy()
    v = pose[7:10].copy()

    dtheta = np.array(delta[0:3], dtype=float, copy=True)
    dp = delta[3:6]
    dv = delta[6:9]

    dq = ESKFmod.quat_from_angle_axis(dtheta)  # pass a copy; function mutates its arg
    q_new = ESKFmod.quat_multiply(q, dq)

    return np.concatenate([q_new, p + dp, v + dv])


def _obs_func_from_pose(pose, lm):
    q = pose[0:4]
    p = pose[4:7]
    diff = lm - p
    norm = np.linalg.norm(diff)
    if norm == 0:
        return np.zeros(3)
    return ESKFmod.matR(*q).T @ (diff / norm)


def test_eskf_matH_eskf_matches_numerical_error_injection():
    rng = np.random.default_rng(3)
    dummy_map = SimpleNamespace(map=np.zeros((1, 3)))
    eskf = ESKFmod.ESKalmanFilter(map_instance=dummy_map)

    failures = []
    for case in range(N_CASES):
        q = random_unit_quat(rng)
        p = rng.uniform(-2, 2, size=3)
        v = rng.uniform(-1, 1, size=3)
        pose0 = np.concatenate([q, p, v])
        lm = random_landmark(rng, p)

        eskf.pose = pose0.copy()
        H_analytic = eskf.matH_eskf(lm)

        def f(delta, pose0=pose0, lm=lm):
            injected = _inject_error_state(pose0, delta)
            return _obs_func_from_pose(injected, lm)

        H_num = numerical_jacobian(f, np.zeros(9))

        if not np.allclose(H_analytic, H_num, atol=1e-4, rtol=1e-3):
            failures.append(case)

    assert not failures, f"ESKF.matH_eskf mismatched numerical error-injection jac in cases {failures}"


# ---------------------------------------------------------------------------
# SLAM_on_cpu: _matH_vectorized (pose + landmark) vs numerical derivative of
# Observation.observation_function_vectorized
# ---------------------------------------------------------------------------

def _random_slam_cam_pose(rng):
    x, y, z = rng.uniform(-2, 2, size=3)
    roll = rng.uniform(-np.pi, np.pi)
    yaw = rng.uniform(-np.pi, np.pi)
    pitch = rng.uniform(-1.3, 1.3)  # away from +-pi/2
    return np.array([x, y, z, roll, pitch, yaw])


def _slam_rot_mats(cam_pose):
    roll, pitch, yaw = cam_pose[3:6]
    Sr, Cr = np.sin(roll), np.cos(roll)
    Sp, Cp = np.sin(pitch), np.cos(pitch)
    Sy, Cy = np.sin(yaw), np.cos(yaw)
    R = SLAM.Rotate_mat(Sr, Cr, Sp, Cp, Sy, Cy)
    Proll = SLAM.P_roll_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)
    Ppitch = SLAM.P_pitch_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)
    Pyaw = SLAM.P_yaw_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)
    return R, Proll, Ppitch, Pyaw


def test_slam_matH_vectorized_pose_and_landmark_jacobians():
    rng = np.random.default_rng(4)

    pose_col_failures = []  # (case, column_index)
    lm_failures = []

    for case in range(N_CASES):
        cam_pose = _random_slam_cam_pose(rng)
        lm = random_landmark(rng, cam_pose[:3])

        R, Proll, Ppitch, Pyaw = _slam_rot_mats(cam_pose)

        pose_jac_batch, lm_jac_batch = SLAM.EKF_SLAM._matH_vectorized(
            SLAM.EKF_SLAM.__new__(SLAM.EKF_SLAM),  # avoid __init__ needing picv arg-less call
            cam_pose[:3], lm[np.newaxis, :], R.T, Proll.T, Ppitch.T, Pyaw.T,
        )
        H_pose_analytic = pose_jac_batch[0]  # (3,6)
        H_lm_analytic = lm_jac_batch[0]      # (3,3)

        def f_pose(x, lm=lm):
            return SLAM.Observation.observation_function_vectorized(x, lm[np.newaxis, :]).squeeze()

        def f_lm(x, cam_pose=cam_pose):
            return SLAM.Observation.observation_function_vectorized(cam_pose, x[np.newaxis, :]).squeeze()

        H_pose_num = numerical_jacobian(f_pose, cam_pose)
        H_lm_num = numerical_jacobian(f_lm, lm)

        for col in range(6):
            if not np.allclose(H_pose_analytic[:, col], H_pose_num[:, col], atol=1e-4, rtol=1e-3):
                pose_col_failures.append((case, col))

        if not np.allclose(H_lm_analytic, H_lm_num, atol=1e-4, rtol=1e-3):
            lm_failures.append(case)

    # Columns: 0=x,1=y,2=z,3=roll,4=pitch,5=yaw
    roll_failures = [c for c, col in pose_col_failures if col == 3]
    other_failures = [(c, col) for c, col in pose_col_failures if col != 3]

    print(f"pose column failures (expect roll/col=3 to fail due to P_roll_Rotate bug): "
          f"roll={len(roll_failures)}/{N_CASES}, other={other_failures}")
    print(f"landmark jacobian failures: {lm_failures}")

    # Non-roll pose columns and the landmark jacobian should match the numerical
    # derivative exactly regardless of the known P_roll_Rotate bug.
    assert not other_failures, f"Unexpected pose-column Jacobian mismatches: {other_failures}"
    assert not lm_failures, f"Landmark Jacobian mismatches numerical derivative: {lm_failures}"

    # Historically P_roll_Rotate had a sign bug at row1/col2; SLAM_on_cpu.py
    # now carries a "修正" (fixed) comment there and this passes. Kept as a
    # regression guard.
    assert not roll_failures, (
        f"SLAM_on_cpu._matH_vectorized roll column mismatched numerical derivative in "
        f"{len(roll_failures)}/{N_CASES} cases -- see P_roll_Rotate (SLAM_on_cpu.py) for a "
        f"suspected sign error at row index 1, col index 2."
    )


# ---------------------------------------------------------------------------
# Direct rotation-derivative checks: dR_dphi/dtheta/dpsi (EKF_Euler) and
# P_roll/pitch/yaw_Rotate (SLAM_on_cpu) vs numerical derivative of the
# rotation functions.
# ---------------------------------------------------------------------------

def _numerical_matrix_derivative(rot_fn, args, idx, eps=EPS):
    args_plus = list(args)
    args_minus = list(args)
    args_plus[idx] += eps
    args_minus[idx] -= eps
    return (rot_fn(*args_plus) - rot_fn(*args_minus)) / (2 * eps)


def test_euler_dR_derivatives_match_numerical():
    rng = np.random.default_rng(5)
    failures = {"phi": [], "theta": [], "psi": []}
    for case in range(N_CASES):
        phi = rng.uniform(-np.pi, np.pi)
        psi = rng.uniform(-np.pi, np.pi)
        theta = rng.uniform(-1.3, 1.3)

        dR_dphi_num = _numerical_matrix_derivative(EKFe.euler_to_R, (phi, theta, psi), 0)
        dR_dtheta_num = _numerical_matrix_derivative(EKFe.euler_to_R, (phi, theta, psi), 1)
        dR_dpsi_num = _numerical_matrix_derivative(EKFe.euler_to_R, (phi, theta, psi), 2)

        if not np.allclose(EKFe.dR_dphi(phi, theta, psi), dR_dphi_num, atol=1e-4, rtol=1e-3):
            failures["phi"].append(case)
        if not np.allclose(EKFe.dR_dtheta(phi, theta, psi), dR_dtheta_num, atol=1e-4, rtol=1e-3):
            failures["theta"].append(case)
        if not np.allclose(EKFe.dR_dpsi(phi, theta, psi), dR_dpsi_num, atol=1e-4, rtol=1e-3):
            failures["psi"].append(case)

    assert not any(failures.values()), f"EKF_Euler dR_d* mismatches: {failures}"


def test_slam_P_rotate_derivatives_match_numerical():
    rng = np.random.default_rng(6)
    failures = {"roll": [], "pitch": [], "yaw": []}

    def rot_of_roll(roll, pitch, yaw):
        return SLAM.Rotate_mat(np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch), np.sin(yaw), np.cos(yaw))

    for case in range(N_CASES):
        roll = rng.uniform(-np.pi, np.pi)
        yaw = rng.uniform(-np.pi, np.pi)
        pitch = rng.uniform(-1.3, 1.3)

        Sr, Cr = np.sin(roll), np.cos(roll)
        Sp, Cp = np.sin(pitch), np.cos(pitch)
        Sy, Cy = np.sin(yaw), np.cos(yaw)

        dR_droll_num = _numerical_matrix_derivative(rot_of_roll, (roll, pitch, yaw), 0)
        dR_dpitch_num = _numerical_matrix_derivative(rot_of_roll, (roll, pitch, yaw), 1)
        dR_dyaw_num = _numerical_matrix_derivative(rot_of_roll, (roll, pitch, yaw), 2)

        if not np.allclose(SLAM.P_roll_Rotate(Sr, Cr, Sp, Cp, Sy, Cy), dR_droll_num, atol=1e-4, rtol=1e-3):
            failures["roll"].append(case)
        if not np.allclose(SLAM.P_pitch_Rotate(Sr, Cr, Sp, Cp, Sy, Cy), dR_dpitch_num, atol=1e-4, rtol=1e-3):
            failures["pitch"].append(case)
        if not np.allclose(SLAM.P_yaw_Rotate(Sr, Cr, Sp, Cp, Sy, Cy), dR_dyaw_num, atol=1e-4, rtol=1e-3):
            failures["yaw"].append(case)

    print(f"P_roll_Rotate failures (regression guard for a since-fixed row1,col2 sign bug): "
          f"{len(failures['roll'])}/{N_CASES}")
    assert not failures["pitch"], f"P_pitch_Rotate mismatches numerical derivative: {failures['pitch']}"
    assert not failures["yaw"], f"P_yaw_Rotate mismatches numerical derivative: {failures['yaw']}"
    assert not failures["roll"], (
        f"SLAM_on_cpu.P_roll_Rotate mismatched numerical d(Rotate_mat)/d(roll) in "
        f"{len(failures['roll'])}/{N_CASES} cases -- check row index 1, col index 2."
    )
