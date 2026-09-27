"""
Synthetic end-to-end tests: a static camera pose observing a known map of
landmarks with noiseless bearing observations, run through 200 EKF_update
steps. Also covers Obsevation_AR.calc_obs and SLAM_on_cpu's Observation /
EKF_SLAM smoke behaviour.
"""
import numpy as np
import pytest
from types import SimpleNamespace

import ExKalmanFilter as EKFq
import EKF_Euler as EKFe
import EKF_Euler_pv as EKFepv
import ESKF as ESKFmod
import SLAM_on_cpu as SLAM

DT = 1 / 30
N_STEPS = 200

LANDMARKS = np.array([
    [3.0, 0.0, 0.0],
    [0.0, 3.0, 0.0],
    [0.0, 0.0, 3.0],
    [-3.0, 0.0, 1.0],
    [0.0, -3.0, 1.0],
    [2.0, 2.0, 2.0],
    [-2.0, -2.0, -1.0],
    [2.0, -2.0, 1.5],
])

TRUE_P = np.array([0.5, -0.3, 0.1])


def _check_cov(P, atol=1e-8):
    sym_err = np.max(np.abs(P - P.T))
    eigvals = np.linalg.eigvalsh((P + P.T) / 2)
    return sym_err, eigvals.min()


# ---------------------------------------------------------------------------
# ExKalmanFilter (quaternion) synthetic convergence
# ---------------------------------------------------------------------------

def test_exkf_synthetic_converges():
    rvec = np.array([0.05, -0.03, 0.07])
    true_q = ESKFmod.quat_from_angle_axis(rvec.copy())
    R_true = ESKFmod.matR(*true_q)

    dummy_map = SimpleNamespace(map=LANDMARKS)
    ekf = EKFq.ExKalmanFilter(map=dummy_map)

    worst_sym, worst_min_eig = 0.0, np.inf
    for _ in range(N_STEPS):
        rows = []
        for idx, lm in enumerate(LANDMARKS):
            diff = lm - TRUE_P
            u = R_true.T @ (diff / np.linalg.norm(diff))
            rows.append([u[0], u[1], u[2], idx])
        ekf.EKF_update(DT, np.array(rows))

        sym_err, min_eig = _check_cov(ekf.P)
        worst_sym = max(worst_sym, sym_err)
        worst_min_eig = min(worst_min_eig, min_eig)

    pos_err = np.linalg.norm(ekf.pose[4:7] - TRUE_P)
    q_est = ekf.pose[:4]
    dot = min(1.0, abs(np.dot(q_est, true_q)))
    angle_err = 2 * np.arccos(dot)

    print(f"ExKalmanFilter: pos_err={pos_err:.4g} angle_err={angle_err:.4g} "
          f"worst_sym_err={worst_sym:.3g} worst_min_eig={worst_min_eig:.3g}")

    assert pos_err < 0.05, f"position error too large: {pos_err}"
    assert angle_err < 0.05, f"attitude error too large: {angle_err}"
    assert worst_sym < 1e-8, f"P not symmetric within tolerance: {worst_sym}"
    assert worst_min_eig > -1e-9, f"P not PSD: min eigenvalue {worst_min_eig}"


# ---------------------------------------------------------------------------
# EKF_Euler / EKF_Euler_pv synthetic convergence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod,has_v", [(EKFe, False), (EKFepv, True)])
def test_euler_ekf_synthetic_converges(mod, has_v):
    true_rpy = np.array([0.08, -0.05, 0.1])  # phi, theta, psi
    R_true = mod.euler_to_R(*true_rpy)

    dummy_map = SimpleNamespace(map=LANDMARKS)
    ekf = mod.ExKalmanFilter(map=dummy_map)

    worst_sym, worst_min_eig = 0.0, np.inf
    for _ in range(N_STEPS):
        rows = []
        for idx, lm in enumerate(LANDMARKS):
            diff = lm - TRUE_P
            u = R_true.T @ (diff / np.linalg.norm(diff))
            rows.append([u[0], u[1], u[2], idx])
        ekf.EKF_update(DT, np.array(rows))

        sym_err, min_eig = _check_cov(ekf.P)
        worst_sym = max(worst_sym, sym_err)
        worst_min_eig = min(worst_min_eig, min_eig)

    pos_err = np.linalg.norm(ekf.pose[3:6] - TRUE_P)
    att_err = np.max(np.abs(np.arctan2(np.sin(ekf.pose[0:3] - true_rpy), np.cos(ekf.pose[0:3] - true_rpy))))

    print(f"{mod.__name__}: pos_err={pos_err:.4g} att_err={att_err:.4g} "
          f"worst_sym_err={worst_sym:.3g} worst_min_eig={worst_min_eig:.3g}")

    assert pos_err < 0.05, f"position error too large: {pos_err}"
    assert att_err < 0.05, f"attitude error too large: {att_err}"
    assert worst_sym < 1e-8, f"P not symmetric within tolerance: {worst_sym}"
    assert worst_min_eig > -1e-9, f"P not PSD: min eigenvalue {worst_min_eig}"


# ---------------------------------------------------------------------------
# ESKF synthetic convergence
# ---------------------------------------------------------------------------

def test_eskf_synthetic_converges():
    rvec = np.array([0.05, -0.03, 0.07])
    true_q = ESKFmod.quat_from_angle_axis(rvec.copy())
    R_true = ESKFmod.matR(*true_q)

    dummy_map = SimpleNamespace(map=LANDMARKS)
    eskf = ESKFmod.ESKalmanFilter(map_instance=dummy_map)

    worst_sym, worst_min_eig = 0.0, np.inf
    for _ in range(N_STEPS):
        rows = []
        for idx, lm in enumerate(LANDMARKS):
            diff = lm - TRUE_P
            u = R_true.T @ (diff / np.linalg.norm(diff))
            rows.append([u[0], u[1], u[2], idx])
        eskf.EKF_update(DT, np.array(rows))

        sym_err, min_eig = _check_cov(eskf.P)
        worst_sym = max(worst_sym, sym_err)
        worst_min_eig = min(worst_min_eig, min_eig)

    pos_err = np.linalg.norm(eskf.pose[4:7] - TRUE_P)
    q_est = eskf.pose[:4]
    dot = min(1.0, abs(np.dot(q_est, true_q)))
    angle_err = 2 * np.arccos(dot)

    print(f"ESKF: pos_err={pos_err:.4g} angle_err={angle_err:.4g} "
          f"worst_sym_err={worst_sym:.3g} worst_min_eig={worst_min_eig:.3g}")

    assert pos_err < 0.05, f"position error too large: {pos_err}"
    assert angle_err < 0.05, f"attitude error too large: {angle_err}"
    assert worst_sym < 1e-8, f"P not symmetric within tolerance: {worst_sym}"
    assert worst_min_eig > -1e-9, f"P not PSD: min eigenvalue {worst_min_eig}"


# ---------------------------------------------------------------------------
# Obsevation_AR.calc_obs
# ---------------------------------------------------------------------------

def test_calc_obs_pixel_center_maps_to_expected_bearing():
    obs = EKFq.Obsevation_AR()
    W = obs.W  # 1280
    x, y = 320.0, 160.0  # pixel center of a synthetic marker

    # A single square marker whose 4 corners average to (x, y).
    corners = [np.array([[[x - 20, y - 20], [x + 20, y - 20],
                          [x + 20, y + 20], [x - 20, y + 20]]], dtype=np.float32)]
    ids = np.array([[5]])

    result = obs.calc_obs(corners, ids)
    assert result.shape == (1, 4)

    vec = result[0, :3]
    expected_azimuth = np.pi * 0.5 - 2 * np.pi * x / W
    expected_elevation = np.pi * 0.5 - 2 * np.pi * y / W
    expected_vec = np.array([
        np.cos(expected_elevation) * np.cos(expected_azimuth),
        np.cos(expected_elevation) * np.sin(expected_azimuth),
        np.sin(expected_elevation),
    ])

    np.testing.assert_allclose(vec, expected_vec, atol=1e-6)
    assert np.isclose(np.linalg.norm(vec), 1.0)
    assert result[0, 3] == 5


def test_calc_obs_no_ids_returns_empty():
    obs = EKFq.Obsevation_AR()
    result = obs.calc_obs(corners=[], ids=None)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


# ---------------------------------------------------------------------------
# SLAM_on_cpu Observation: _trans_func_vectorized + data()
# ---------------------------------------------------------------------------

def test_slam_observation_data_unit_norm_and_ranges():
    picv = 640
    obsv = SLAM.Observation()

    # corners spanning both branches of _trans_func_vectorized's piecewise map
    corners = np.array([
        [100.0, 300.0],
        [500.0, 200.0],
        [900.0, 350.0],  # >= picv*0.5 + picv branch (cond2)
        [1279.0, 100.0],
    ])

    result = obsv.data(picv, corners)
    assert result.shape == (4, 3)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_slam_observation_data_empty_input_does_not_crash():
    picv = 640
    obsv = SLAM.Observation()

    # Mirrors what Observation.preprocessing() produces when FAST finds no
    # keypoints: np.array([point.pt for point in []]) -> shape (0,), not (0, 2).
    empty_coords = np.array([])

    result = obsv.data(picv, empty_coords)
    assert result.shape == (0, 3)


# ---------------------------------------------------------------------------
# SLAM_on_cpu EKF_SLAM smoke test
# ---------------------------------------------------------------------------

def test_ekf_slam_smoke_runs_without_exception():
    picv = 640
    est = SLAM.EKF_SLAM(picv=picv)
    rng = np.random.default_rng(42)

    for _ in range(5):
        est.motion_update(1 / 30)

        bearings = rng.normal(size=(4, 3))
        bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)

        est.observation_update(bearings)

        assert np.allclose(est.cov, est.cov.T, atol=1e-6), "cov not symmetric"
        expected_len = est.pose_size + est.lm_size * est.nLM
        assert len(est.believe) == expected_len
        assert est.cov.shape == (expected_len, expected_len)
