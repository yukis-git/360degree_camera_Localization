# SlamIndex.lean — EKF-SLAM 状態ベクトルのインデックス検証

`SLAM_on_cpu.py` の `EKF_SLAM` クラスが状態ベクトル `believe` / 共分散 `cov` を
どのようにインデックスしているかを、Lean 4 (core のみ、Mathlib 不使用) で
形式的に検証したものです。数値計算そのもの（回転行列やヤコビアン）ではなく、
「配列の何番目が何を指しているか」というインデックス計算の正しさだけを対象にしています。

## 動かし方

```bash
lean verification/lean/SlamIndex.lean
```

exit code 0、`sorry` なしで全定理が証明されています（ワーニング1件のみ、未使用変数の注記で無害）。

## 証明した定理と、Python/Rust コードへの意味

### 1. レイアウト（`get_LM_Pos_from_state` / `pose_size` / `lm_size`）

状態ベクトルは `[ pose(6) | landmark_0(3) | landmark_1(3) | ... ]` という並びで、
ランドマーク `id` は半開区間 `[6+3*id, 6+3*id+3)` を占める
(`SLAM_on_cpu.py:215-216`, `:233`, `:241`, `:373` の `pose_size + lm_size*id` 計算に対応)。

- `lmSlice_subset_state` : `id < n` なら、そのランドマーク区間は `n` 個のランドマークを
  持つ状態ベクトル全体 `[0, 6+3n)` に完全に収まる。→ 添字が状態ベクトルの範囲外に
  出ないことの保証。
- `lmSlice_disjoint` : `id1 ≠ id2` の区間は重ならない。→ 異なるランドマークが
  同じメモリ領域を指してしまう（Rust に移植した際の aliasing バグにもつながる）
  ことがないことの保証。
- `pose_disjoint_lm` : pose 区間 `[0,6)` はどの `id` のランドマーク区間とも重ならない。
- `new_landmark_slice_is_tail`（ボーナス）: `n` 個から `n+1` 個に増えたとき、新しい
  ランドマーク (`id=n`) の区間はちょうど末尾に追加された `[6+3n, 6+3n+3)` に一致する。
  → `np.concatenate((self.believe, self.calc_LM_Pos(z, R_mat)))`
  (`SLAM_on_cpu.py:354-362`) が正しい位置に新規ランドマークを追加していることの保証。

### 2. 削除後の ID 付け替え（`search_correspond_LM_ID` の重複削除 & `lm_delete`）

`search_correspond_LM_ID` (`SLAM_on_cpu.py:248-319`) はマハラノビス距離が近い
「重複候補」ランドマークを一括削除してから、生き残った候補
`minid_candidate` の新しい ID を

```python
kept_lm_ids = np.where(~delete_candidates_mask)[0]
new_minid_candidate = np.where(kept_lm_ids == minid_candidate)[0][0]
```

(`:314-317`) で求めています。これを `mask : Nat → Bool`（`true`=削除）でモデル化し、

- `cnt m k` := `[0, k)` の中で削除されなかった（kept）個数
- `keptAux m n` := `[0, n)` の中で kept な添字を昇順に並べたリスト

と定義したうえで、次を証明:

- `mask_remap`（`List Bool` 版のラッパー。本質は `getElem?_keptAux_cnt` /
  `cnt_lt_length_keptAux`）: `mask[minid] = false`（削除されない）かつ
  `minid < n` なら、
  - `remap := cnt m minid` は `kept` の有効な添字（`remap < kept.length`）
  - `kept[remap]? = some minid`

  すなわち `cnt m minid` は Python の `np.where(kept_lm_ids == minid)[0][0]` が
  返す値と一致し、しかも **必ず見つかる**（out-of-bounds にならない）ことを保証します。
  Python 側は `if minid_candidate in kept_lm_ids:` で存在確認をした *あと* に
  `np.where(...)[0][0]` を呼んでいるので実行時には安全ですが、この定理はその
  「存在すれば必ず唯一の位置が見つかる」という論理を独立に裏付けます。

- `new_state_len` : `n` 個のランドマークのうち `mask` に従って削除した後の新しい
  状態ベクトル長は `6 + 3 * (kept 数)`。→ `self.believe = np.delete(...)` /
  `self.nLM -= len(indices_to_delete_lm)` (`:308-313`) の後の `nLM` と実際の配列長が
  一致し続けることの保証。単一ランドマーク削除の `lm_delete` (`:324-334`) は
  `mask` がちょうど1個だけ `true` の特殊ケースとして同じ枠組みに含まれます。

### 3. 画素列の分岐（`Observation._trans_func_vectorized`）

`_trans_func_vectorized` (`SLAM_on_cpu.py:103-120`) は正距円筒（equirectangular）画像の
横方向インデックス `x ∈ [0, 2H]`（`H = picv`、`picv` は動画フレーム高さ）を

```python
cond1: 0 <= x < p_val + picv        # = 1.5*H
cond2: p_val + picv <= x <= 2*picv  # 1.5*H <= x <= 2*H
```

の2分岐で処理しています。`1.5*H` は整数にならないことがあるので、Lean 側では
両辺を2倍して `Nat` のまま扱っています:

- `Branch1 H x := 2*x < 3*H`
- `Branch2 H x := 3*H ≤ 2*x ∧ x ≤ 2*H`
- `branch_partition` : `x ≤ 2*H` を満たすすべての `x` は Branch1 と Branch2 の
  **ちょうど一方**に入る（両方には入らない、どちらにも入らないことはない）。

  → `if cond1: ... elif cond2: ...` という Python の分岐が `x ∈ [0, 2H]` を
  重複も漏れもなくカバーしていることの保証。境界 `x` がちょうど `1.5*H`
  （`H` が偶数のとき整数）のときも cond1 側 (`<`) に落ちる、という Python の
  厳密な境界規則も `Branch1`/`Branch2` の定義（`<` と `≤` の非対称性）に
  そのまま反映されています。

## 気づいた点（Python コードについて）

- `_trans_func_vectorized` の分岐条件自体は本証明の意味で「漏れなく・重複なく」
  `[0, 2H]` をカバーしており、バグは見つかりませんでした
  (`SLAM_on_cpu.py:114`, `:117`)。
- `search_correspond_LM_ID` の ID 付け替え (`:314-318`) は、`minid_candidate` が
  削除対象でなければ必ず一意な新 ID が見つかるという意味で健全です。ただし
  `minid_candidate in kept_lm_ids` (`:315`, NumPy 配列に対する `in`) は要素数分の
  線形探索になっており、`Rust` 実装では削除マスクからの O(1) 変換
  （本証明の `cnt`/`remap` そのもの）に置き換えるのが自然です。
- レイアウト系の定理は、Rust 側でランドマーク `id -> 状態ベクトル添字` の変換を
  実装する際に、境界（`id < n` の外れ値や、隣接ランドマークとの重なり）に関する
  同じ主張をそのまま unit test / 型不変条件として移植できる形にしてあります。
