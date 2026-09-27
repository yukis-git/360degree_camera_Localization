# Z3 による検証スクリプト

`360degree_camera_Localization` の EKF/ESKF/EKF-SLAM 実装から数式を書き写し
(該当ファイル:行番号をコメントに明記)、Z3 (z3-solver) と sympy で
「PROVED」または反例を出力する検証スクリプト群です。数式はソースから転記しており、
ソースを import しません(ソースを変更したら転記も更新してください)。

実行方法:
```
python3 verification/z3/run_all.py   # 全スクリプトを実行、失敗があれば非ゼロ終了
python3 verification/z3/<name>.py    # 個別実行
```

## 結果一覧

| # | スクリプト | 対象 | 結果 |
|---|---|---|---|
| 1 | `quat_jacobian.py` | `ExKalmanFilter.py` の `matR`/`PRPww`/`PRPwx`/`PRPwy`/`PRPwz` | **全て PROVED**。sympy の微分と一致し、Z3 でも全成分が恒等的に等しいことを証明。単位四元数 (ww²+wx²+wy²+wz²=1) の下で `matR` が直交行列 (RᵀR=I) であることも PROVED。 |
| 2 | `euler_jacobian.py` | (a) `EKF_Euler.py` の `euler_to_R`/`dR_dphi`/`dR_dtheta`/`dR_dpsi`、(b) `SLAM_on_cpu.py`/`SLAM_on_gpu.py` の `Rotate_mat`/`P_roll_Rotate`/`P_pitch_Rotate`/`P_yaw_Rotate` | デフォルト(正しい参照式)モードでは **全て PROVED**。`--source` モードで検証すると、**`P_roll_Rotate[1][2]` のみ反例 (COUNTEREXAMPLE)** となり、既知バグを再現する。 |
| 3 | `eskf_attitude_jacobian.py` | `ESKF.py` の `H_dtheta = skew(u_cam)` と `H_dp` | **全て PROVED**。右摂動 `q_new=q⊗δq` に対し `(I-[δ]×)u = u + skew(u)δ` が恒等的に成立し、`H_dtheta` の符号が正しいことを証明。`H_dp` の核となる `-(I-uuᵀ)/d` の性質 ((I-uuᵀ)u=0、対称性、冪等性) も PROVED。 |
| 4 | `pixel_mapping.py` | `SLAM_on_cpu.py` の `_trans_func_vectorized` と `ExKalmanFilter.py` の `calc_obs` (ArUco) | **全て PROVED**。各ブランチで phi∈[-π,π], theta∈[-π/2,π/2]、x=1.5H でのブランチ一致 (theta は完全一致、phi は mod 2π で一致)、2つのマスクが [0,2H] を過不足なく分割すること、ArUco マッピングとの一致 (theta は完全一致、phi は mod 2π) を証明。 |
| 5 | `slam_index_bookkeeping.py` | `SLAM_on_cpu.py` の `search_correspond_LM_ID` のランドマーク削除・インデックス再割当 | **全て PROVED** (nLM≤8 の有界モデルで全数検証)。再割当後のインデックスが `minid - #{削除された minid 未満のインデックス}` に一致すること、`[0, nLM_new)` の範囲内に収まること、状態ベクトルのスライス `[6+3*id, 6+3*id+3)` が新しい状態長 `6+3*nLM_new` の範囲内に収まることを証明。 |

## 見つかったバグ

> 下記のバグはコミット eb49375 で修正済み。`--source` モードは修正前のソースの転記を検証し、バグを再現する。

- **`SLAM_on_cpu.py:20`** (`P_roll_Rotate` の `[1][2]` 成分) と
  **`SLAM_on_gpu.py:22`** (同じく `[1][2]` 成分、cupy 版):
  ```
  誤: -Sy*Sp*Cr-Cy*Cr
  正: -Sy*Sp*Sr-Cy*Cr
  ```
  正しい回転行列 `Rotate_mat` の roll に関する偏微分では `Sr` であるべき箇所が
  `Cr` になっている(タイプミス)。GPU 版にも同一のバグがそのままコピーされている。
  `euler_jacobian.py --source` で反例として再現される。

  他の成分・他の関数 (`P_pitch_Rotate`, `P_yaw_Rotate`, `dR_dphi/dtheta/dpsi`,
  四元数版の `PRPww/PRPwx/PRPwy/PRPwz`) にはバグは見つからなかった。

## 意外だった点

- 四元数のヤコビアン (`PRPww` 等) と `EKF_Euler.py` の偏微分はすべて数式的に
  正しく、バグは1件も見つからなかった。バグは `SLAM_on_cpu.py`/`SLAM_on_gpu.py`
  の `P_roll_Rotate` の1箇所のみで、しかも CPU 版・GPU 版に全く同じ誤りが
  コピーされていた(片方から他方へコードをコピーした際に一緒に持ち込まれたと
  推測される)。
- ピクセル→球面角のマッピング (`_trans_func_vectorized`) は、2つのブランチの
  `theta` が実は分岐に関係なく同一の式になっている(コード上、`y_values` の
  列0 (`phi`) だけがブランチで異なり、列1 (`theta`) は共通の質量的意味を持つ
  行に対して同じ式が適用されるため)。ArUco 側の `calc_obs` も `theta` は
  常に完全一致するが、`phi` はブランチ2側で `2π` のオフセットがあるため
  「mod 2π」でしか一致しない。これは実装として一貫しており、バグではない。
