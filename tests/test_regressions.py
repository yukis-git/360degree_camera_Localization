"""今回修正したバグの回帰テスト"""
import numpy as np

import ESKF
import SLAM_on_cpu


def test_quat_from_angle_axis_does_not_mutate_input():
    v = np.array([0.1, -0.2, 0.3])
    before = v.copy()
    ESKF.quat_from_angle_axis(v)
    assert np.array_equal(v, before)


def test_slam_observation_noise_is_isotropic_small():
    e = SLAM_on_cpu.EKF_SLAM(picv=2880)
    Q = e.matQ()
    assert np.allclose(Q, np.eye(3) * e.direction_dev**2)


def test_slam_sparse_update_matches_dense_formula():
    """疎な S / K / P 更新が密な H を使った教科書式と一致する"""
    rng = np.random.default_rng(3)
    e = SLAM_on_cpu.EKF_SLAM(picv=2880)
    lms = rng.uniform(-5, 5, (12, 3))
    for _ in range(3):
        e.motion_update(1 / 30)
        d = lms - e.believe[:3]
        e.observation_update(d / np.linalg.norm(d, axis=1)[:, None])
    assert e.nLM > 0
    z = lms[0] / np.linalg.norm(lms[0])
    r, p, y = e.believe[3:6]
    args = (np.sin(r), np.cos(r), np.sin(p), np.cos(p), np.sin(y), np.cos(y))
    mats = [f(*args) for f in (SLAM_on_cpu.Rotate_mat, SLAM_on_cpu.P_roll_Rotate,
                               SLAM_on_cpu.P_pitch_Rotate, SLAM_on_cpu.P_yaw_Rotate)]
    lmid = 0
    lm = e.get_LM_Pos_from_state(lmid)
    _, S, H = e.calc_innovation(lm, z, lmid, *mats)
    assert np.allclose(S, e.matQ() + H @ e.cov @ H.T, atol=1e-12)
