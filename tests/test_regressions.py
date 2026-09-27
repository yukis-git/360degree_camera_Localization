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


class _Map:
    def __init__(self, lms):
        self.map = lms


def test_eskf_tracks_rotating_camera():
    """旧 Q (1e-6/フレーム) では 0.5 rad/s のヨー回転に 14° 以上遅れていた"""
    rng = np.random.default_rng(0)
    m = _Map(rng.uniform(-4, 4, (8, 3)))
    f = ESKF.ESKalmanFilter(m)
    dt, worst = 1 / 30, 0.0
    for k in range(300):
        yaw = 0.5 * k * dt
        q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        d = m.map / np.linalg.norm(m.map, axis=1)[:, None]
        u = (ESKF.matR(*q).T @ d.T).T
        f.EKF_update(dt, np.hstack([u, np.arange(8)[:, None]]))
        if k >= 30:
            worst = max(worst, 2 * np.arccos(min(1.0, abs(f.pose[:4] @ q))))
    assert np.degrees(worst) < 1.0


def _flip_bits(d, rng, n=8):
    b = np.unpackbits(d)
    b[rng.choice(256, n, replace=False)] ^= 1
    return np.packbits(b)


def _run_slam(use_desc, n_true=60, frames=120):
    rng = np.random.default_rng(0)
    lms = rng.uniform(-6, 6, (n_true, 3))
    lms[:, 2] = rng.uniform(-0.5, 0.5, n_true)
    D = rng.integers(0, 256, (n_true, 32), dtype=np.uint8)
    e = SLAM_on_cpu.EKF_SLAM(picv=2880)
    r = np.random.default_rng(1)
    for k in range(frames):
        p = np.array([0.01 * k, 0, 0])
        ids = r.choice(n_true, 15, replace=False)
        d = lms[ids] - p
        z = d / np.linalg.norm(d, axis=1)[:, None] + r.normal(0, 1e-3, (15, 3))
        desc = np.array([_flip_bits(D[i], r) for i in ids])
        e.motion_update(1 / 30)
        e.observation_update(z, desc if use_desc else None)
        # 状態ベクトルとランドマーク付随情報は常に同期している
        assert len(e.believe) == 6 + 3 * e.nLM == 6 + 3 * len(e.lm_desc)
        assert len(e.lm_hits) == len(e.lm_born) == e.nLM
        assert np.allclose(e.cov, e.cov.T)
    return e


def test_slam_orb_descriptors_prevent_duplicate_landmarks():
    with_desc = _run_slam(True)
    geometric = _run_slam(False)
    assert with_desc.nLM <= 1.2 * 60
    assert with_desc.nLM < geometric.nLM


def test_slam_motion_noise_only_on_pose_block():
    e = _run_slam(True, frames=5)
    before = e.cov.copy()
    e.motion_update(1 / 30)
    diff = e.cov - before
    assert np.allclose(diff[6:, :], 0) and np.allclose(diff[:, 6:], 0)
    assert np.all(np.diag(diff)[:6] > 0)


def test_slam_rarely_seen_landmarks_are_pruned():
    e = SLAM_on_cpu.EKF_SLAM(picv=2880, prune_age=3, prune_min_hits=2)
    rng = np.random.default_rng(2)
    z = rng.normal(size=(5, 3))
    z /= np.linalg.norm(z, axis=1)[:, None]
    desc = rng.integers(0, 256, (5, 32), dtype=np.uint8)
    e.observation_update(z, desc)
    assert e.nLM == 5
    for _ in range(3):  # 以後一度も観測されない
        e.observation_update(np.zeros((0, 3)), np.zeros((0, 32), dtype=np.uint8))
    assert e.nLM == 0 and len(e.lm_desc) == 0
