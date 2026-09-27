import cv2
import matplotlib.pyplot as plt
import matplotlib
try: matplotlib.use('TkAgg')
except ImportError: pass  # tkinter が無い環境 (テスト/CI) では既定バックエンドを使う
import matplotlib.animation as anm
import numpy as np
from scipy.stats import chi2
import time as Time

# 修正箇所
def Rotate_mat(Sr, Cr, Sp, Cp, Sy, Cy):
    return np.array([[Cy*Cp, Cy*Sp*Sr-Sy*Cr, Cy*Sp*Cr+Sy*Sr],
                     [Sy*Cp, Sy*Sp*Sr+Cy*Cr, Sy*Sp*Cr-Cy*Sr],
                     [-Sp, Cp*Sr, Cp*Cr]])

# 修正箇所
def P_roll_Rotate(Sr, Cr, Sp, Cp, Sy, Cy):
    return np.array([[0, Cy*Sp*Cr+Sy*Sr, -Cy*Sp*Sr+Sy*Cr],
                     [0, Sy*Sp*Cr-Cy*Sr, -Sy*Sp*Sr-Cy*Cr],  # 修正: d/dr(Sy*Sp*Cr-Cy*Sr) = -Sy*Sp*Sr-Cy*Cr
                     [0, Cp*Cr, -Cp*Sr]])

# 修正箇所
def P_pitch_Rotate(Sr, Cr, Sp, Cp, Sy, Cy):
    return np.array([[-Cy*Sp, Cy*Cp*Sr, Cy*Cp*Cr],
                     [-Sy*Sp, Sy*Cp*Sr, Sy*Cp*Cr],
                     [-Cp, -Sp*Sr, -Sp*Cr]])

# 修正箇所
def P_yaw_Rotate(Sr, Cr, Sp, Cp, Sy, Cy):
    return np.array([[-Sy*Cp, -Sy*Sp*Sr-Cy*Cr, -Sy*Sp*Cr+Cy*Sr],
                     [Cy*Cp, Cy*Sp*Sr-Sy*Cr, Cy*Sp*Cr+Sy*Sr],
                     [0, 0, 0]])

class World:
    def __init__(self, vision_pass, threshold, save=False, max_features=100):
        self.cap = cv2.VideoCapture(vision_pass)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.time_interval = 1/fps if fps > 0 else 1/30  # FPS が取得できない場合のゼロ除算対策
        self.totalframecount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.picv = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # FAST (記述子なし) -> ORB: 記述子で対応付けでき、1フレームの特徴点数も max_features で上限を設ける
        self.detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=threshold)
        self.scale = 0.2
        self.estimator = EKF_SLAM(picv=self.picv)
        self.observation = Observation()
        self.save = save

    def draw(self):
        fig = plt.figure(figsize=(6, 6))

        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)
        ax.set_zlabel("Z [m]", fontsize=10)

        self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(ax, ),
                                    frames=None, interval=int(self.time_interval*1000), repeat=False, save_count=self.totalframecount)

        if self.save:
            self.ani.save('test.mp4', writer="ffmpeg")
            plt.close(fig)
        else: plt.show()

    def one_step(self, i, ax):

        ret, frame = self.cap.read()

        if ret:

            time = self.time_interval*i

            current_frame_coords, descriptors = self.observation.preprocessing(frame, self.picv, self.scale, self.detector)
            observation = self.observation.data(self.picv, current_frame_coords)
            self.estimator.motion_update(self.time_interval)
            start = Time.time()
            self.estimator.observation_update(observation, descriptors)
            end = Time.time()
            self.estimator.draw(ax, time)
            print(end - start)


class Observation:
    def __init__(self, debug=False):
        self.lastdata = []
        self.debug = debug

    @staticmethod # ここを修正
    def preprocessing(frame, picv, scale, detector):
        '''特徴点の画素座標 (N,2) と ORB 記述子 (N,32) uint8 を返す'''
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray_frame, dtype=np.uint8)
        mask[int(picv*5/12):int(picv*7/12), 0:int(2*picv)] = 255
        kp, descriptors = detector.detectAndCompute(gray_frame, mask)
        if descriptors is None: kp, descriptors = (), np.zeros((0, 32), dtype=np.uint8)
        current_frame_coords = np.array([point.pt for point in kp]).reshape(-1, 2)
        frame_with_kp = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
        # text = f"FAST Corners: {len(kp)}"
        # cv2.putText(frame_with_kp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        width, height = int(frame_with_kp.shape[1] * scale), int(frame_with_kp.shape[0] * scale)
        resized_frame = cv2.resize(frame_with_kp, (width, height), interpolation=cv2.INTER_LINEAR)
        # if self.monitor and self.save is False: cv2.imshow('FAST Corner Detection in Video', resized_frame)
        cv2.imshow('ORB Features in Video', resized_frame)
        return current_frame_coords, descriptors

    def _trans_func_vectorized(self, corners, picv):
            if not isinstance(corners, np.ndarray): corners = np.asarray(corners, dtype=float)
            if corners.size == 0: return np.array([]).reshape(0, 2)

            p_val, q_val = picv * 0.5, picv * 0.5
            pi_div_picv = np.pi / picv
            base_term = -pi_div_picv * corners

            y_values = np.zeros_like(corners)
            corner_x = corners[:, 0]

            cond1_mask = (0 <= corner_x) & (corner_x < (p_val+picv))
            y_values[cond1_mask] = base_term[cond1_mask] + pi_div_picv * np.array([p_val, q_val])

            cond2_mask = ((p_val+picv) <= corner_x) & (corner_x <= 2*picv)
            y_values[cond2_mask] = base_term[cond2_mask] + pi_div_picv * np.array([p_val + picv, q_val]) + np.array([np.pi, 0])

            return y_values

    def get_observed_fully_vectorized(self, current_frame_coords):
        if current_frame_coords.size == 0:
            observed = np.array([]).reshape(0, 3)
            self.lastdata = observed
            if self.debug: print(f"sensor value : {observed}")
            return observed

    def data(self, picv, current_frame_coords):
        y_values = self._trans_func_vectorized(corners=current_frame_coords, picv=picv)
        x_components = np.cos(y_values[:, 1]) * np.cos(y_values[:, 0])
        y_components = np.cos(y_values[:, 1]) * np.sin(y_values[:, 0])
        z_components = np.sin(y_values[:, 1])
        observed = np.stack([x_components, y_components, z_components], axis=-1)
        self.lastdata = observed
        return observed

    @classmethod
    def observation_function_vectorized(cls, cam_pose, obj_pos_array):
        cam_pos_xyz = cam_pose[:3]

        diff_array = obj_pos_array - cam_pos_xyz
        norms = np.linalg.norm(diff_array, axis=1)

        roll, pitch, yaw = cam_pose[3:6]
        R_mat = Rotate_mat(np.sin(roll), np.cos(roll),
                           np.sin(pitch), np.cos(pitch),
                           np.sin(yaw), np.cos(yaw))

        d_inv_array_reshaped = np.where(norms > 1e-9, 1.0 / norms, 0.0)[:, np.newaxis]

        rotated_diff = (R_mat.T @ diff_array.T).T 

        return d_inv_array_reshaped * rotated_diff


class EKF_SLAM:
    def __init__(self, picv, debug=False, hamming_th=64, prune_age=30, prune_min_hits=2):
        '''
        hamming_th     : ORB 記述子 (256bit) の一致とみなすハミング距離の上限
        prune_age      : 生成からこのフレーム数経っても
        prune_min_hits : この回数未満しか再観測されないランドマークは削除する
        '''
        self.direction_dev = 10*np.pi/picv
        self.believe = np.asarray(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.cov = np.diag([1, 1, 1, np.pi/6, np.pi/6, np.pi/6])
        self.poses = [self.believe]
        self.pose_size = len(self.believe)
        self.lm_size = 3
        self.nLM = 0
        self.md_chi = chi2.ppf(0.95, self.pose_size)
        # 対応付けゲート: 正しい対応の 99% を受理する (旧 chi2.ppf(0.01, 3)~0.11 は正しい対応もほぼ棄却していた)
        self.M_DIST_TH = chi2.ppf(0.99, 3)
        self.Duplication = chi2.ppf(0.001, 3)
        self.color = "blue"
        self.debug = debug
        # ランドマークごとの付随情報 (状態ベクトルの並びと常に同期させる)
        self.hamming_th, self.prune_age, self.prune_min_hits = hamming_th, prune_age, prune_min_hits
        self.lm_desc = np.zeros((0, 32), dtype=np.uint8)  # ORB 記述子
        self.lm_hits = np.zeros(0, dtype=int)             # 既存ランドマークとして対応付いた回数
        self.lm_born = np.zeros(0, dtype=int)             # 生成フレーム
        self.frame = 0

    def motion_update(self, time_interval):
        # ランドマークは静止しているのでプロセスノイズは姿勢ブロックにだけ加える
        # (旧実装はランドマークにも毎秒分散 1 を加えており、地図が時間とともにぼやけていた。
        #  使われないランドマークの削除は observation_update の再観測回数による間引きで行う)
        motion_noise_diag = np.array([1, 1, 1, np.pi/6, np.pi/6, np.pi/6])
        ps = self.pose_size
        self.cov[:ps, :ps] += time_interval * np.diag(motion_noise_diag)

    def _matH_vectorized(self, cam_pos, all_lm_positions, R_mat_T, Proll_mat_T, Ppitch_mat_T, Pyaw_mat_T):
        nLM_batch = all_lm_positions.shape[0]
        
        if nLM_batch == 0: return np.zeros((0, self.lm_size, self.pose_size + self.nLM * self.lm_size)) # 全状態次元に合わせる
            
        diff_all = all_lm_positions - cam_pos
        norms_all = np.linalg.norm(diff_all, axis=1)
        
        d_inv_all = np.where(norms_all > 1e-9, 1.0 / norms_all, 0.0)

        d_inv_all_reshaped = d_inv_all[:, np.newaxis]

        d3_diff_all = (d_inv_all**3)[:, np.newaxis] * diff_all
        d_diff_all = d_inv_all_reshaped * diff_all
        
        Prx_all = (R_mat_T @ (diff_all[:, 0][:, np.newaxis] * d3_diff_all - np.concatenate([d_inv_all_reshaped, np.zeros((nLM_batch, 2))], axis=1)).T).T
        Pry_all = (R_mat_T @ (diff_all[:, 1][:, np.newaxis] * d3_diff_all - np.concatenate([np.zeros((nLM_batch, 1)), d_inv_all_reshaped, np.zeros((nLM_batch, 1))], axis=1)).T).T
        Prz_all = (R_mat_T @ (diff_all[:, 2][:, np.newaxis] * d3_diff_all - np.concatenate([np.zeros((nLM_batch, 2)), d_inv_all_reshaped], axis=1)).T).T

        pose_jacobian_all = np.stack([Prx_all, Pry_all, Prz_all,
                                      (Proll_mat_T @ d_diff_all.T).T,
                                      (Ppitch_mat_T @ d_diff_all.T).T,
                                      (Pyaw_mat_T @ d_diff_all.T).T], axis=2)

        lm_jacobian_core_all = -np.stack([Prx_all, Pry_all, Prz_all], axis=-1)

        return pose_jacobian_all, lm_jacobian_core_all

    def matQ(self):
        # 単位方向ベクトルの各成分の誤差分散 ~ (角度誤差)^2。
        # 旧式 diag[cos^2, sin(2d)/2, sin(d)] は x 成分の分散が ~1 となり x 成分の観測がほぼ無視されていた
        return np.eye(3) * self.direction_dev**2


    def get_LM_Pos_from_state(self, id):
        return self.believe[self.pose_size + self.lm_size*id: self.pose_size + self.lm_size*(id+1)]

    def calc_innovation(self, lm, z, lmid, R_mat, Proll_mat, Ppitch_mat, Pyaw_mat):
        pose_jacobian_core_batch, lm_jacobian_core_batch = self._matH_vectorized(
            self.believe[:3], lm[np.newaxis, :], R_mat.T, Proll_mat.T, Ppitch_mat.T, Pyaw_mat.T)
        
        # 形状を (3, 6) と (3, 3) に squeeze
        pose_jacobian_core = pose_jacobian_core_batch.squeeze(axis=0)
        lm_jacobian_core = lm_jacobian_core_batch.squeeze(axis=0)

        # H 行列 (3, 全状態次元) を構築
        H_for_single_lm = np.zeros((self.lm_size, self.pose_size+self.nLM*self.lm_size), dtype=float)

        H_for_single_lm[:, :self.pose_size] = pose_jacobian_core # 姿勢部分

        # ランドマーク部分 (lmid の位置に lm_jacobian_core を配置)
        if self.nLM > 0: # ランドマークが存在する場合のみ
            lm_start_idx = self.pose_size + lmid * self.lm_size
            lm_end_idx = lm_start_idx + self.lm_size
            H_for_single_lm[:, lm_start_idx:lm_end_idx] = lm_jacobian_core
        
        zp = Observation.observation_function_vectorized(self.believe[:6], lm[np.newaxis, :]).squeeze()
        y = z - zp
        
        # H の非ゼロ列 (姿勢 + 対象ランドマーク) だけで S を計算
        lm_start = self.pose_size + lmid * self.lm_size
        cols = np.r_[0:self.pose_size, lm_start:lm_start + self.lm_size]
        Hs = H_for_single_lm[:, cols]
        S = self.matQ() + Hs @ self.cov[np.ix_(cols, cols)] @ Hs.T
        
        return y, S, H_for_single_lm

    def search_correspond_LM_ID(self, z, desc=None):
        '''
        観測 z に対応するランドマーク番号を返す (新規なら self.nLM)。
        desc (ORB 記述子) が与えられた場合は、記述子が一致するランドマークだけを候補にする。
        '''
        if self.nLM == 0: return self.nLM

        roll, pitch, yaw = self.believe[3:6]
        Sr, Cr = np.sin(roll), np.cos(roll)
        Sp, Cp = np.sin(pitch), np.cos(pitch)
        Sy, Cy = np.sin(yaw), np.cos(yaw)
        R_mat = Rotate_mat(Sr, Cr, Sp, Cp, Sy, Cy)
        Proll_mat = P_roll_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)
        Ppitch_mat = P_pitch_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)
        Pyaw_mat = P_yaw_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)

        all_lm_positions = self.believe[self.pose_size:].reshape(self.nLM, self.lm_size)

        pose_jacobian_core_all, lm_jacobian_core_all = self._matH_vectorized(
            self.believe[:3], all_lm_positions, R_mat.T, Proll_mat.T, Ppitch_mat.T, Pyaw_mat.T)

        zps = Observation.observation_function_vectorized(self.believe[:6], all_lm_positions)
        ys = z - zps

        # H_i は姿勢列(6)と自身のランドマーク列(3)以外ゼロなので、密な H_all (nLM x 3 x N) を作らず
        # 共分散のブロックだけで S_i = Q + H_i P H_i^T を計算する (O(nLM^3) -> O(nLM))
        Hp, Hl = pose_jacobian_core_all, lm_jacobian_core_all
        ps = self.pose_size
        P_pp = self.cov[:ps, :ps]
        P_pl = self.cov[:ps, ps:].reshape(ps, self.nLM, self.lm_size).transpose(1, 0, 2)  # (nLM, 6, 3)
        lm_idx = np.arange(self.nLM)
        P_ll = self.cov[ps:, ps:].reshape(self.nLM, self.lm_size, self.nLM, self.lm_size)[lm_idx, :, lm_idx, :]  # (nLM, 3, 3)
        HpT, HlT = np.transpose(Hp, (0, 2, 1)), np.transpose(Hl, (0, 2, 1))
        cross = Hp @ P_pl @ HlT
        Q_mat = self.matQ()
        S_all = (Q_mat[np.newaxis, :, :] + Hp @ P_pp @ HpT + cross + np.transpose(cross, (0, 2, 1))
                 + Hl @ P_ll @ HlT)

        ys_reshaped = ys[:, :, np.newaxis]
        solve_results_all = np.linalg.solve(S_all, ys_reshaped)

        mahalanobis_distances = np.sum(ys_reshaped.transpose(0, 2, 1) @ solve_results_all, axis=(1, 2))

        # 記述子が一致しないランドマークは対応候補から外す (幾何だけの最近傍より誤対応が大幅に減る)
        match_distances = mahalanobis_distances
        if desc is not None:
            hamming = np.unpackbits(np.bitwise_xor(self.lm_desc, desc[np.newaxis, :]), axis=1).sum(axis=1)
            match_distances = np.where(hamming <= self.hamming_th, mahalanobis_distances, np.inf)

        # 最小のマハラノビス距離を持つランドマークを見つける
        minid_candidate = np.argmin(match_distances)
        minmd_actual = match_distances[minid_candidate]

        delete_candidates_mask = (mahalanobis_distances < self.Duplication)
        
        # minid_candidate は削除しないのでマスクから除外
        delete_candidates_mask[minid_candidate] = False 
        
        # 実際に削除するランドマークの元のインデックス (0からnLM-1)
        indices_to_delete_lm = np.where(delete_candidates_mask)[0]

        if len(indices_to_delete_lm) > 0:
            if self.debug:
                print(f"Removing {len(indices_to_delete_lm)} landmarks (IDs: {indices_to_delete_lm}) due to small Mahalanobis distance but not being the closest.")

            kept_lm_ids = self._delete_landmarks(delete_candidates_mask)
            if minid_candidate in kept_lm_ids:
                new_minid_candidate = np.where(kept_lm_ids == minid_candidate)[0][0]
                minid_candidate = new_minid_candidate
            else: return self.nLM 
        return minid_candidate if minmd_actual < self.M_DIST_TH else self.nLM

    def calc_LM_Pos(self, z, R_mat):
        return self.believe[:3] + 5*np.dot(R_mat, z)
    
    def _delete_landmarks(self, delete_mask):
        '''
        delete_mask (長さ nLM, True=削除) のランドマークを状態・共分散・付随情報からまとめて削除し、
        残ったランドマークの元の番号 (昇順) を返す。削除処理はすべてここを通して同期を保つ。
        '''
        delete_mask = np.asarray(delete_mask, dtype=bool)
        lm_ids = np.where(delete_mask)[0]
        if len(lm_ids) > 0:
            global_indices = (self.pose_size + lm_ids[:, np.newaxis] * self.lm_size + np.arange(self.lm_size)).ravel()
            self.believe = np.delete(self.believe, global_indices)
            self.cov = np.delete(np.delete(self.cov, global_indices, axis=0), global_indices, axis=1)
            self.lm_desc = self.lm_desc[~delete_mask]
            self.lm_hits = self.lm_hits[~delete_mask]
            self.lm_born = self.lm_born[~delete_mask]
            self.nLM -= len(lm_ids)
        return np.where(~delete_mask)[0]

    def lm_delete(self, id):
        mask = np.zeros(self.nLM, dtype=bool)
        mask[id] = True
        self._delete_landmarks(mask)


    def observation_update(self, observation, descriptors=None):
        '''
        observation : (N,3) 観測方向の単位ベクトル
        descriptors : (N,32) uint8 の ORB 記述子。None なら幾何 (マハラノビス距離) のみで対応付ける
        '''
        initP = np.eye(self.lm_size)
        self.frame += 1
        if descriptors is not None and len(descriptors) != len(observation): descriptors = None

        for k, z_input in enumerate(observation):
            z = np.asarray(z_input)
            desc = None if descriptors is None else descriptors[k]

            minid = self.search_correspond_LM_ID(z, desc)
            matched_existing = minid < self.nLM

            roll, pitch, yaw = self.believe[3:6]
            Sr, Cr = np.sin(roll), np.cos(roll)
            Sp, Cp = np.sin(pitch), np.cos(pitch)
            Sy, Cy = np.sin(yaw), np.cos(yaw)
            R_mat = Rotate_mat(Sr, Cr, Sp, Cp, Sy, Cy)
            Proll_mat = P_roll_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)
            Ppitch_mat = P_pitch_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)
            Pyaw_mat = P_yaw_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)

            if minid==self.nLM:
                self.nLM += 1
                new_believe = np.concatenate((self.believe, self.calc_LM_Pos(z, R_mat)))

                believe_len_cp = len(self.believe)
                new_cov = np.vstack((np.hstack((self.cov, np.zeros((believe_len_cp, self.lm_size), dtype=np.float32))), 
                                     np.hstack((np.zeros((self.lm_size, believe_len_cp), dtype=np.float32), initP))))
                
                self.believe, self.cov = new_believe, new_cov
                self.lm_desc = np.vstack((self.lm_desc, np.zeros((1, 32), dtype=np.uint8) if desc is None else desc[np.newaxis, :]))
                self.lm_hits = np.append(self.lm_hits, 0)
                self.lm_born = np.append(self.lm_born, self.frame)

            elif matched_existing:
                self.lm_hits[minid] += 1

            # 既存のランドマークの更新処理
            lm = self.get_LM_Pos_from_state(minid)
            # ランドマークがカメラに近すぎる場合の削除処理
            if np.linalg.norm(lm-self.believe[:3]) < 0.5: self.lm_delete(minid)
            else:
                y, S, H = self.calc_innovation(lm, z, minid, R_mat, Proll_mat, Ppitch_mat, Pyaw_mat)

                # H は姿勢列とランドマーク minid の列のみ非ゼロ: 疎な列だけで cov @ H.T を計算 (O(N^2) -> O(N))
                lm_start = self.pose_size + minid * self.lm_size
                cols = np.r_[0:self.pose_size, lm_start:lm_start + self.lm_size]
                PHt = self.cov[:, cols] @ H[:, cols].T
                K = PHt @ np.linalg.inv(S)
                new_believe = self.believe + K @ y

                d_believe = new_believe - self.believe
                dm = (d_believe[:self.pose_size].T @
                      np.linalg.solve(self.cov[:self.pose_size, :self.pose_size], d_believe[:self.pose_size]))

                if dm.item() < self.md_chi:
                    # (I - K H) P = P - K (P H^T)^T  (cov は対称)。丸め誤差で非対称化しないよう対称化する
                    new_cov = self.cov - K @ PHt.T
                    new_cov = (new_cov + new_cov.T) / 2
                    self.believe, self.cov = new_believe, new_cov

        self.poses.append(self.believe[:6])

        if self.nLM > 0:
            # 不確実性が大きすぎるランドマーク、および生成後 prune_age フレーム経っても
            # prune_min_hits 回未満しか再観測されないランドマーク (一過性の特徴点) を削除する
            all_lm_variances = np.diag(self.cov)[self.pose_size:].reshape(self.nLM, self.lm_size)
            too_uncertain = np.max(all_lm_variances, axis=1) >= 10
            rarely_seen = ((self.frame - self.lm_born) >= self.prune_age) & (self.lm_hits < self.prune_min_hits)
            self._delete_landmarks(too_uncertain | rarely_seen)

        if self.debug: print(f"camera pose : {self.believe[:6]}")

    def sigma_ellipse(self, ax, n=1):
        P_pp = (self.cov[:3, :3] + self.cov[:3, :3].T) / 2
        eig_vals, eig_vec = np.linalg.eigh(P_pp)  # 対称行列なので eigh (eig は複素数を返し得る)

        radii = n * np.sqrt(np.clip(eig_vals, 0, None))
        u_grid, v_grid = np.meshgrid(np.linspace(0, 2*np.pi, 40), np.linspace(0, np.pi, 40))

        x_sphere = radii[0]*np.cos(u_grid)*np.sin(v_grid)
        y_sphere = radii[1]*np.sin(u_grid)*np.sin(v_grid)
        z_sphere = radii[2]*np.cos(v_grid)

        points_sphere = np.stack([x_sphere, y_sphere, z_sphere], axis=-1)

        # x = V diag(r) s の行ベクトル表現は s_r @ V.T (V を掛けると回転が逆になる)
        transformed_points = (points_sphere @ eig_vec.T + self.believe[:3])

        return ax.plot_surface(transformed_points[..., 0],
                               transformed_points[..., 1],
                               transformed_points[..., 2],
                               rstride=4, cstride=4, color=self.color, alpha=0.3)

    def draw(self, ax, time):
        ax.cla()

        if self.believe is not None:
            r = 1.0
            believe_np = self.believe

            roll, pitch, yaw = believe_np[3:6]
            Sr, Cr = np.sin(roll), np.cos(roll)
            Sp, Cp = np.sin(pitch), np.cos(pitch)
            Sy, Cy = np.sin(yaw), np.cos(yaw)
            rotate_mat_np = Rotate_mat(Sr, Cr, Sp, Cp, Sy, Cy)

            ax.quiver(*believe_np[:3], *r*rotate_mat_np[:, 0], color=self.color)
            ax.quiver(*believe_np[:3], *r*rotate_mat_np[:, 1], color="red")
            ax.quiver(*believe_np[:3], *r*rotate_mat_np[:, 2], color="green")

            all_lm_np = self.believe[self.pose_size: self.pose_size+self.lm_size*self.nLM].reshape(self.nLM, self.lm_size)
            ax.scatter(all_lm_np[:, 0], all_lm_np[:, 1], all_lm_np[:, 2], s=0.7, color="black")

            poses_np_list = [p for p in self.poses]
            poses = np.array(poses_np_list)
            ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], linewidth=0.5, color=self.color)

        self.sigma_ellipse(ax)

        time_str = "t = %.2f[s]" % (time)
        ax.text(believe_np[:3][0]+1, believe_np[:3][1]+1, believe_np[:3][2]+0.1, time_str, fontsize=10) #believe_np[:3]が1D配列なのでインデックスアクセス修正

        xmin, xmax = believe_np[0]-5, believe_np[0]+5
        ymin, ymax = believe_np[1]-5, believe_np[1]+5
        zmin, zmax = believe_np[2]-2.5, believe_np[2]+2.5
        ax.set_xlim3d(xmin, xmax)
        ax.set_ylim3d(ymin, ymax)
        ax.set_zlim3d(zmin, zmax)
        ax.set_box_aspect((xmax-xmin,ymax-ymin,zmax-zmin))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EKF-SLAM (360度カメラ) 実行スクリプト")
    parser.add_argument('--video', default=r"C:\Users\sakata\Documents\VID_20250826_171914_00_012.mp4",
                        help="入力動画のパス")
    parser.add_argument('--threshold', type=float, default=100,
                        help="新規ランドマーク登録のしきい値")
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True,
                        help="アニメーションを保存する/しない")
    args = parser.parse_args()

    world = World(args.video, threshold=args.threshold, save=args.save)
    world.draw()


if __name__ == '__main__':
    main()