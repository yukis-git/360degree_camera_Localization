# 360degree_camera_Localization

## 検証・テスト

| 種類 | 場所 | 実行方法 |
|---|---|---|
| 数値テスト (pytest) | `tests/` | `python3 -m pytest tests -q` |
| Z3 証明 (ヤコビアン・画素写像・添字) | `verification/z3/` | `python3 verification/z3/run_all.py` |
| Lean 証明 (SLAM 状態レイアウト・削除後の再採番) | `verification/lean/` | `lean verification/lean/SlamIndex.lean` |
| Rust コア + Kani | `rust/omni_loc_core/` | `cargo test` / `cargo kani --harness <名前>` |

依存: `pip install numpy scipy matplotlib opencv-contrib-python-headless z3-solver sympy pytest hypothesis`
