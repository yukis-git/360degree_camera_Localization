# omni_loc_core

360度カメラ EKF-SLAM の Python プロトタイプ（`SLAM_on_cpu.py` / `ExKalmanFilter.py` /
`ESKF.py`）から、将来の Rust 移植でそのまま再利用できる**依存ゼロの純粋関数コア**を
切り出したクレートです。既存の Python ファイルは一切変更していません。

## モジュール構成

- `src/pixel.rs` — 正距円筒画像のピクセル座標 → (方位角, 仰角) → 単位方向ベクトル。
  `SLAM_on_cpu.py` の `Observation._trans_func_vectorized` / `data()`（画像高さ `H`,
  幅 `2H`、`x = 1.5H` での分岐でアジマスを `[-π, π]` に収める）を移植。
  無効入力（`x∉[0,2H]`、`y∉[0,H]`、`h` が小さすぎる/非有限）は `None` を返す
  （Python 版は `np.zeros_like` で無条件に `0.0` を返してしまう箇所がある）。
- `src/state_layout.rs` — EKF-SLAM の状態ベクトル/共分散行列レイアウト。
  `POSE_SIZE=6`, `LM_SIZE=3`、`lm_range`、`remap_after_delete`
  （`SLAM_on_cpu.py::search_correspond_LM_ID` のランドマーク再採番と同じ意味論）、
  `delete_landmarks`（`np.delete` によるランドマーク削除と同じ操作）。
- `src/quat.rs` — クォータニオン `(w,x,y,z)` の積・正規化・角軸表現からの変換・
  回転行列変換。`ESKF.py` の `quat_multiply` / `Quaternion_Normalization` /
  `quat_from_angle_axis` / `matR` を移植。

## Python コードから見つかったバグ・注意点

1. **`ESKF.py:35-42` `quat_from_angle_axis` の破壊的変更バグ。**
   `q /= angle`（38行目）は引数の numpy 配列を **呼び出し元の変数ごと** 書き換える
   （numpy 配列は参照渡しのため）。`quat.rs::from_angle_axis` は角軸ベクトルを
   `[f64; 3]`（値渡し・`Copy`）で受け取るため、この副作用が構造的に発生しない。
   `from_angle_axis_does_not_mutate_caller_value` テストで確認済み。
2. **`ExKalmanFilter.py::calc_obs`（`W=1280`）の "simplified model" は範囲外の値を返しうる。**
   `SLAM_on_cpu.py` と同じ係数 `2π/W == π/H` を使うが、`x ≥ 1.5H` の分岐を実装して
   いないため、`x` が大きい領域でアジマスが `-π` を下回る（コード内コメントにも
   "this mapping might need refinement" とある）。`pixel.rs` のテスト
   `python_calc_obs_style_formula_can_exceed_minus_pi` で数値的に確認。
3. **Kani で発見: `pixel_to_angles_*` の高さ `h` の下限。**
   `h` を任意の正の小さい値まで許すと、`π / h` が `+inf` にオーバーフローし、
   `inf * 0.0`（`x = 0` は有効なピクセル座標）が `NaN` になってしまい、
   `[-π, π]` / `[-π/2, π/2]` という戻り値の範囲保証が壊れる。これは Kani の
   `pixel_to_angles_f64_in_range_or_none` 証明が実際に反例を見つけて判明した。
   対策として `MIN_HEIGHT`（実用上、画像の高さは常に 1 ピクセル以上）を導入し、
   それを下回る `h` は `None` を返すようにした。Python 側は `picv` が常に実画像の
   高さなのでこの問題には触れない。

## 実行方法

```sh
cd rust/omni_loc_core
cargo test          # 通常のユニットテスト
cargo kani          # 全 Kani 証明ハーネスを実行（時間がかかる。個別実行推奨）
cargo kani --harness <ハーネス名>   # 個別のハーネスだけ実行
```

## Kani 証明ハーネス一覧

| ハーネス | モジュール | 検証内容 |
|---|---|---|
| `pixel_to_angles_f64_in_range_or_none` | pixel.rs | 有効な入力ならアジマス∈[-π,π]・仰角∈[-π/2,π/2]、パニックなし |
| `pixel_to_angles_f32_in_range_or_none` | pixel.rs | 同上（f32） |
| `invalid_inputs_yield_none` | pixel.rs | 範囲外ピクセル/`h`不正 → `None` |
| `non_finite_yields_none` | pixel.rs | NaN/Inf 入力 → `None` |
| `normalize_never_nan_for_bounded_input` | quat.rs | 正規化結果が常に有限（NaN/Infにならない） |
| `mul_no_panic_and_finite` | quat.rs | 四元数積がパニックせず有限値を返す |
| `from_angle_axis_guard_no_panic` | quat.rs | ゼロ判定・除算ガード部分がパニックしない（sin/cos 部分は対象外） |
| `lm_range_is_within_state_bounds` | state_layout.rs | `lm_range` が常に状態ベクトル範囲内 |
| `remap_after_delete_points_to_same_landmark_and_is_in_bounds` | state_layout.rs | 再採番後のIDが同じランドマークを指し、範囲内 |
| `delete_landmarks_preserves_sizes_and_kept_entries` | state_layout.rs | 削除後の状態/共分散のサイズ整合性、姿勢ブロックの保存 |

各ハーネスの成否は `cargo kani --harness <名前>` 実行時の標準出力の末尾
`VERIFICATION:- SUCCESSFUL` / `FAILED` で確認できます。

## 制限事項（三角関数と Kani）

CBMC/Kani は `sin`/`cos` のような超越関数をビット精度でモデル化できません
(未サポートとしてリンクエラーになるか、扱いが不安定です)。そのため:

- `pixel.rs` は `pixel_to_angles_*`（三角関数なし、比較・四則演算のみ）と
  `angles_to_bearing_*`（`sin`/`cos` を使う）に分離し、**前者のみ Kani で証明**、
  後者は通常のユニットテスト（`bearing_is_unit_length` など）でカバーしている。
- `quat.rs::from_angle_axis` も同様に、ゼロ判定・除算ガード部分のみ Kani で検証し、
  半角の `sin`/`cos` 部分はユニットテスト（`from_angle_axis_half_pi_about_x` など）
  でのみ検証している。

## 制限事項（浮動小数点シンボリック実行のコスト）

CBMC の IEEE-754 浮動小数点のビット精度エンコードは、シンボリックな `f64`/`f32`
変数を複数含む式（特に `sqrt` や大きなループ）に対して非常に重くなります:

- `quat.rs` の `normalize_never_nan_for_bounded_input` / `from_angle_axis_guard_no_panic`
  は、4引数すべてが完全にシンボリックな `f64` で `sqrt` を含むため、当初の
  `±1e6` という広い範囲では現実的な時間で完了しませんでした。数値範囲を
  `±10.0`（`SQRT_BOUND`）に絞ることで、探索対象のビット表現の複雑さは変わらない
  もののソルバーの実行時間が現実的になりました。境界値（ゼロ、`1e-12` 近傍）は
  この範囲に含まれるため、意味のある境界条件はカバーされています。
- `state_layout.rs` の `delete_landmarks_preserves_sizes_and_kept_entries` は、
  内部で `kept x kept`（最大 `(6+3*n_lm)^2` 回）のネストしたループを持つため、
  `#[kani::unwind(k)]` の `k` をそのループ回数以上に設定する必要があります。
  `n_lm ≤ 4` のフルレンジ（最大 18×18=324 回）では実行時間内に完了しなかったため、
  このハーネスだけ `n_lm ≤ 2`（`DELETE_MAX_N_LM`、最大 12×12=144 回、
  `unwind(150)`）に絞っています。他の2つの `state_layout.rs` ハーネス
  （`lm_range_is_within_state_bounds` / `remap_after_delete_points_to_same_landmark_and_is_in_bounds`）
  は配列操作のみで浮動小数点を含まないため `n_lm ≤ 4` のフルレンジで検証済みです。

いずれも「証明できないこと」ではなく「この環境・時間内で完了する範囲に絞った」
制限です。境界を広げたい場合は `SQRT_BOUND` / `DELETE_MAX_N_LM` の定数を大きくして
再実行してください（実行時間は指数的に増加する可能性があります）。
