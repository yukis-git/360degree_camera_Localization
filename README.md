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

# EKF-SLAM (CPU版) を実行 (しきい値・保存有無も指定可能)
python3 SLAM_on_cpu.py --video path/to/video.mp4 --threshold 100 --save
python3 SLAM_on_cpu.py --video path/to/video.mp4 --no-save
```

`--help` で各スクリプトの引数一覧を確認できます (例: `python3 ESKF.py --help`)。

## 検証・テスト

| 種類 | 場所 | 実行方法 |
|---|---|---|
| 数値テスト (pytest) | `tests/` | `python3 -m pytest tests -q` |
| Z3 証明 (ヤコビアン・画素写像・添字) | `verification/z3/` | `python3 verification/z3/run_all.py` |
| Lean 証明 (SLAM 状態レイアウト・削除後の再採番) | `verification/lean/` | `lean verification/lean/SlamIndex.lean` |
| Rust コア + Kani | `rust/omni_loc_core/` | `cargo test` / `cargo kani --harness <名前>` |

依存: `pip install numpy scipy matplotlib opencv-contrib-python-headless z3-solver sympy pytest hypothesis`
