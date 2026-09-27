# 360degree_camera_Localization

## 使い方

推奨の自己位置推定スクリプトは **`ESKF.py`** (四元数の誤差状態カルマンフィルタ) です。
`EKF_Euler.py` / `EKF_Euler_pv.py` (オイラー角 EKF) と `ExKalmanFilter.py` (四元数 EKF) はレガシー・比較用として残しています。オイラー角表現はピッチ角 ±90° でジンバルロックが発生するのに対し、`ESKF.py` の四元数表現ではその問題がなく、かつ誤差状態を正規化した後の共分散を接空間へ正しく射影する (ESKF リセット) ため、姿勢推定の数値的な安定性が高くなっています。

各スクリプトは `--video` (入力動画) / `--map` (ランドマークマップ CSV) を引数で指定できます。引数を省略すると、これまでどおりのハードコードされたデフォルトパスで動作します。

```bash
# ESKF (推奨) をそのまま実行 (画面表示)
python3 ESKF.py --video path/to/video.mp4 --map path/to/map.csv

# 画面表示せず動画として保存
python3 ESKF.py --video path/to/video.mp4 --map path/to/map.csv --save output.mp4

# EKF-SLAM (CPU版): ORB 特徴点 (1フレーム最大 --max-features 個) を記述子で対応付け
python3 SLAM_on_cpu.py --video path/to/video.mp4 --threshold 100 --max-features 100 --save
python3 SLAM_on_cpu.py --video path/to/video.mp4 --no-save
```

`--help` で各スクリプトの引数一覧を確認できます (例: `python3 ESKF.py --help`)。

ESKF のプロセスノイズは連続時間の強度で与え、フレーム間隔 dt を掛けて使います
(`ESKalmanFilter(map, q_theta=0.1, q_pos=1e-4, q_vel=0.1)`)。カメラの回転が速い場合は `q_theta` を大きくしてください。
GPU 版 `SLAM_on_gpu.py` は旧来の FAST 特徴点のままで、ORB 対応付けとランドマークの間引きは CPU 版のみです。

## 検証・テスト

| 種類 | 場所 | 実行方法 |
|---|---|---|
| 数値テスト (pytest) | `tests/` | `python3 -m pytest tests -q` |
| Z3 証明 (ヤコビアン・画素写像・添字) | `verification/z3/` | `python3 verification/z3/run_all.py` |
| Lean 証明 (SLAM 状態レイアウト・削除後の再採番) | `verification/lean/` | `lean verification/lean/SlamIndex.lean` |
| Rust コア + Kani | `rust/omni_loc_core/` | `cargo test` / `cargo kani --harness <名前>` |

依存: `pip install numpy scipy matplotlib opencv-contrib-python-headless z3-solver sympy pytest hypothesis`
