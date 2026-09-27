# import os
# os.environ['PYTHON_GIL'] = '0'

import cv2
import matplotlib.pyplot as plt
import matplotlib
try: matplotlib.use('TkAgg')
except ImportError: pass  # tkinter が無い環境 (テスト/CI) では既定バックエンドを使う
import matplotlib.animation as anm
import numpy as np
import cupy as cp
from scipy.stats import chi2
import time as Time

def Rotate_mat(Sr, Cr, Sp, Cp, Sy, Cy):
    return cp.vstack((cp.array([Cy*Cp, Cy*Sp*Sr-Sy*Cr, Cy*Sp*Cr+Sy*Sr], dtype=cp.float32),
                      cp.array([Sy*Cp, Sy*Sp*Sr+Cy*Cr, Sy*Sp*Cr-Cy*Sr], dtype=cp.float32),
                      cp.array([-Sp, Cp*Sr, Cp*Cr], dtype=cp.float32)))

def P_roll_Rotate(Sr, Cr, Sp, Cp, Sy, Cy):
    return cp.vstack((cp.array([cp.array(0), Cy*Sp*Cr+Sy*Sr, -Cy*Sp*Sr+Sy*Cr], dtype=cp.float32),
                      cp.array([cp.array(0), Sy*Sp*Cr-Cy*Sr, -Sy*Sp*Sr-Cy*Cr], dtype=cp.float32),  # 修正: -Sy*Sp*Cr -> -Sy*Sp*Sr
                      cp.array([cp.array(0), Cp*Cr, -Cp*Sr], dtype=cp.float32)))

def P_pitch_Rotate(Sr, Cr, Sp, Cp, Sy, Cy):
    return cp.vstack((cp.array([-Cy*Sp, Cy*Cp*Sr, Cy*Cp*Cr], dtype=cp.float32),
                      cp.array([-Sy*Sp, Sy*Cp*Sr, Sy*Cp*Cr], dtype=cp.float32),
                      cp.array([-Cp, -Sp*Sr, -Sp*Cr], dtype=cp.float32)))

def P_yaw_Rotate(Sr, Cr, Sp, Cp, Sy, Cy):
    return cp.vstack((cp.array([-Sy*Cp, -Sy*Sp*Sr-Cy*Cr, -Sy*Sp*Cr+Cy*Sr], dtype=cp.float32),
                      cp.array([Cy*Cp, Cy*Sp*Sr-Sy*Cr, Cy*Sp*Cr+Sy*Sr], dtype=cp.float32),
                      cp.array([cp.array(0), cp.array(0), cp.array(0)], dtype=cp.float32)))

class World:
    def __init__(self, vision_pass, threshold, save=False):
        self.cap = cv2.VideoCapture(vision_pass)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.time_interval = 1/fps if fps > 0 else 1/30  # FPS が取得できない場合のゼロ除算対策
        self.totalframecount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.picv = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)
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
            # while elems: elems.pop().remove()

            time = self.time_interval*i

            current_frame_coords = self.observation.preprocessing(frame, self.picv, self.scale, self.fast)
            observation = self.observation.data(self.picv, current_frame_coords)
            self.estimator.motion_update(self.time_interval)
            start = Time.time()
            self.estimator.observation_update(observation)
            end = Time.time()
            self.estimator.draw(ax, time)
            print(end - start)


class Observation:
    def __init__(self, debug=False):
        self.lastdata = []
        self.debug = debug

    @staticmethod # ここを修正
    def preprocessing(frame, picv, scale, fast):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray_frame, dtype=np.uint8)
        mask[int(picv*5/12):int(picv*7/12), 0:int(2*picv)] = 255
        kp = fast.detect(gray_frame, mask)
        current_frame_coords = cp.asarray([point.pt for point in kp])
        frame_with_kp = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
        # text = f"FAST Corners: {len(kp)}"
        # cv2.putText(frame_with_kp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        width, height = int(frame_with_kp.shape[1] * scale), int(frame_with_kp.shape[0] * scale)
        resized_frame = cv2.resize(frame_with_kp, (width, height), interpolation=cv2.INTER_LINEAR)
        # if self.monitor and self.save is False: cv2.imshow('FAST Corner Detection in Video', resized_frame)
        cv2.imshow('FAST Corner Detection in Video', resized_frame)
        return current_frame_coords

    def _trans_func_vectorized(self, corners, picv):
            if isinstance(corners, np.ndarray): corners = cp.asarray(corners)
            if corners.size == 0: return cp.array([]).reshape(0, 2)

            p_val, q_val = picv * 0.5, picv * 0.5
            pi_div_picv = cp.pi / picv
            base_term = -pi_div_picv * corners

            y_values = cp.zeros_like(corners)
            corner_x = corners[:, 0]

            cond1_mask = (0 <= corner_x) & (corner_x < (p_val+picv))
            y_values[cond1_mask] = base_term[cond1_mask] + pi_div_picv * cp.array([p_val, q_val])

            cond2_mask = ((p_val+picv) <= corner_x) & (corner_x <= 2*picv)
            y_values[cond2_mask] = base_term[cond2_mask] + pi_div_picv * cp.array([p_val + picv, q_val]) + cp.array([cp.pi, 0])

            return y_values

    def get_observed_fully_vectorized(self, current_frame_coords):
        if current_frame_coords.size == 0:
            observed = cp.array([]).reshape(0, 3)
            self.lastdata = observed
            if self.debug: print(f"sensor value : {observed}")
            return observed

    def data(self, picv, current_frame_coords):
        y_values = self._trans_func_vectorized(corners=current_frame_coords, picv=picv)
        x_components = cp.cos(y_values[:, 1]) * cp.cos(y_values[:, 0])
        y_components = cp.cos(y_values[:, 1]) * cp.sin(y_values[:, 0])
        z_components = cp.sin(y_values[:, 1])
        observed = cp.stack([x_components, y_components, z_components], axis=-1)
        self.lastdata = observed
        return observed

    @classmethod
    def observation_function_vectorized(cls, cam_pose, obj_pos_array):
        cam_pos_xyz = cam_pose[:3]

        diff_array = obj_pos_array - cam_pos_xyz
        norms = cp.linalg.norm(diff_array, axis=1)

        roll, pitch, yaw = cam_pose[3:6]
        R_mat = Rotate_mat(cp.sin(roll), cp.cos(roll),
                           cp.sin(pitch), cp.cos(pitch),
                           cp.sin(yaw), cp.cos(yaw))

        d_inv_array_reshaped = cp.where(norms > 1e-9, 1.0 / norms, 0.0)[:, cp.newaxis]
        rotated_diff = (R_mat.T @ diff_array.T).T

        return d_inv_array_reshaped * rotated_diff


class EKF_SLAM:
    def __init__(self, picv, debug=False):
        self.direction_dev = 10*cp.pi/picv
        self.believe = cp.asarray(cp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.cov = cp.diag([1, 1, 1, cp.pi/6, cp.pi/6, cp.pi/6])
        self.poses = [self.believe]
        self.pose_size = len(self.believe)
        self.lm_size = 3
        self.nLM = 0
        self.md_chi = chi2.ppf(0.95, self.pose_size)
        self.M_DIST_TH = chi2.ppf(0.99, 3)  # 正しい対応の 99% を受理 (旧 0.01 分位点はほぼ全て棄却)
        self.color = "blue"
        self.debug = debug

    def motion_update(self, time_interval):
        # ランドマークは静止しているのでプロセスノイズは姿勢ブロックにだけ加える
        motion_noise_diag = cp.array([1, 1, 1, cp.pi/6, cp.pi/6, cp.pi/6])
        new_cov_noise = cp.zeros_like(self.cov)
        new_cov_noise[:self.pose_size, :self.pose_size] = cp.diag(motion_noise_diag)

        self.cov = self.cov + time_interval * new_cov_noise

    def _matH_vectorized(self, cam_pos, all_lm_positions, R_mat_T, Proll_mat_T, Ppitch_mat_T, Pyaw_mat_T):
        nLM_batch = all_lm_positions.shape[0]

        if nLM_batch == 0: return cp.zeros((0, self.lm_size, self.pose_size + self.nLM * self.lm_size))

        diff_all = all_lm_positions - cam_pos
        norms_all = cp.linalg.norm(diff_all, axis=1)

        d_inv_all = cp.where(norms_all > 1e-9, 1.0 / norms_all, 0.0)
        d3_inv_all = d_inv_all**3

        d_inv_all_reshaped = d_inv_all[:, cp.newaxis]
        d3_inv_all_reshaped = d3_inv_all[:, cp.newaxis]

        d3_diff_all = d3_inv_all_reshaped * diff_all
        d_diff_all = d_inv_all_reshaped * diff_all

        # cp.arrayで明示的にdtype=cp.float32を指定
        term_Prx_combined = diff_all[:, 0][:, cp.newaxis] * d3_diff_all - cp.concatenate([d_inv_all_reshaped, cp.zeros((nLM_batch, 2), dtype=cp.float32)], axis=1)
        term_Pry_combined = diff_all[:, 1][:, cp.newaxis] * d3_diff_all - cp.concatenate([cp.zeros((nLM_batch, 1), dtype=cp.float32), d_inv_all_reshaped, cp.zeros((nLM_batch, 1), dtype=cp.float32)], axis=1)
        term_Prz_combined = diff_all[:, 2][:, cp.newaxis] * d3_diff_all - cp.concatenate([cp.zeros((nLM_batch, 2), dtype=cp.float32), d_inv_all_reshaped], axis=1)

        Prx_all = (R_mat_T @ term_Prx_combined.T).T
        Pry_all = (R_mat_T @ term_Pry_combined.T).T
        Prz_all = (R_mat_T @ term_Prz_combined.T).T

        pose_jacobian_components = [
            Prx_all, Pry_all, Prz_all,
            (Proll_mat_T @ d_diff_all.T).T,
            (Ppitch_mat_T @ d_diff_all.T).T,
            (Pyaw_mat_T @ d_diff_all.T).T
        ]
        pose_jacobian_all = cp.stack(pose_jacobian_components, axis=2)

        lm_jacobian_core_all = -cp.stack([Prx_all, Pry_all, Prz_all], axis=-1)

        return pose_jacobian_all, lm_jacobian_core_all

    def matQ(self):
        # 単位方向ベクトル各成分の誤差分散 ~ (角度誤差)^2 (旧式は x 成分の分散が ~1 で観測が効いていなかった)
        return cp.eye(3) * self.direction_dev**2

    def get_LM_Pos_from_state(self, id):
        return self.believe[self.pose_size + self.lm_size*id: self.pose_size + self.lm_size*(id+1)]

    def calc_innovation(self, lm, z, lmid, R_mat, Proll_mat, Ppitch_mat, Pyaw_mat):
        lm_cp = cp.asarray(lm)
        z_cp = cp.asarray(z)

        pose_jacobian_core, lm_jacobian_core = self._matH_vectorized(
            self.believe[:3], lm_cp[cp.newaxis, :], R_mat.T, Proll_mat.T, Ppitch_mat.T, Pyaw_mat.T)

        pose_jacobian_core = pose_jacobian_core.squeeze(axis=0)
        lm_jacobian_core = lm_jacobian_core.squeeze(axis=0)

        H_for_single_lm = cp.zeros((self.lm_size, self.pose_size+self.nLM*self.lm_size))

        H_for_single_lm[:, :self.pose_size] = pose_jacobian_core

        if self.nLM > 0:
            lm_start_idx = self.pose_size + lmid * self.lm_size
            lm_end_idx = lm_start_idx + self.lm_size
            H_for_single_lm[:, lm_start_idx:lm_end_idx] = lm_jacobian_core

        zp = Observation.observation_function_vectorized(self.believe[:6], lm_cp[cp.newaxis, :]).squeeze()
        y = (z_cp - zp)

        S = self.matQ() + H_for_single_lm @ self.cov @ H_for_single_lm.T

        return y, S, H_for_single_lm

    def search_correspond_LM_ID(self, z):
        if self.nLM == 0: return self.nLM

        z_cp = cp.asarray(z)

        roll, pitch, yaw = self.believe[3:6]
        Sr, Cr = cp.sin(roll), cp.cos(roll)
        Sp, Cp = cp.sin(pitch), cp.cos(pitch)
        Sy, Cy = cp.sin(yaw), cp.cos(yaw)
        R_mat = Rotate_mat(Sr, Cr, Sp, Cp, Sy, Cy)
        Proll_mat = P_roll_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)
        Ppitch_mat = P_pitch_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)
        Pyaw_mat = P_yaw_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)

        all_lm_positions = self.believe[self.pose_size:].reshape(self.nLM, self.lm_size)

        pose_jacobian_core_all, lm_jacobian_core_all = self._matH_vectorized(
            self.believe[:3], all_lm_positions, R_mat.T, Proll_mat.T, Ppitch_mat.T, Pyaw_mat.T)

        total_state_size = self.pose_size + self.nLM * self.lm_size
        H_all = cp.zeros((self.nLM, self.lm_size, total_state_size))

        H_all[:, :, :self.pose_size] = pose_jacobian_core_all

        for i in range(self.nLM):
            lm_start_idx = self.pose_size + i * self.lm_size
            lm_end_idx = lm_start_idx + self.lm_size
            H_all[i, :, lm_start_idx:lm_end_idx] = lm_jacobian_core_all[i, :, :]

        zps = Observation.observation_function_vectorized(self.believe[:6], all_lm_positions)
        ys = z_cp - zps

        Q_mat = self.matQ()
        S_all = Q_mat[cp.newaxis, :, :] + H_all @ self.cov @ cp.transpose(H_all, (0, 2, 1))

        ys_reshaped = ys[:, :, cp.newaxis]
        solve_results_all = cp.linalg.solve(S_all, ys_reshaped)

        mahalanobis_distances = cp.sum(ys_reshaped.transpose(0, 2, 1) @ solve_results_all, axis=(1, 2))

        minid_candidate = cp.argmin(mahalanobis_distances)
        minmd_actual = mahalanobis_distances[minid_candidate].item()

        return minid_candidate.item() if minmd_actual < self.M_DIST_TH else self.nLM

    def calc_LM_Pos(self, z, R_mat):
        return self.believe[:3] + 5*cp.dot(R_mat, z)

    def observation_update(self, observation):
        initP = cp.eye(self.lm_size)

        for z_input in observation:
            z = cp.asarray(z_input)

            minid = self.search_correspond_LM_ID(z)

            roll, pitch, yaw = self.believe[3:6]
            Sr, Cr = cp.sin(roll), cp.cos(roll)
            Sp, Cp = cp.sin(pitch), cp.cos(pitch)
            Sy, Cy = cp.sin(yaw), cp.cos(yaw)
            R_mat = Rotate_mat(Sr, Cr, Sp, Cp, Sy, Cy)
            Proll_mat = P_roll_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)
            Ppitch_mat = P_pitch_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)
            Pyaw_mat = P_yaw_Rotate(Sr, Cr, Sp, Cp, Sy, Cy)

            if minid==self.nLM:
                self.nLM += 1
                new_believe = cp.concatenate((self.believe, self.calc_LM_Pos(z, R_mat)))

                believe_len_cp = len(self.believe)
                new_cov = cp.vstack((cp.hstack((self.cov, cp.zeros((believe_len_cp, self.lm_size), dtype=cp.float32))), 
                                     cp.hstack((cp.zeros((self.lm_size, believe_len_cp), dtype=cp.float32), initP))))
                
                self.believe, self.cov = new_believe, new_cov

            lm = self.get_LM_Pos_from_state(minid)

            y, S, H = self.calc_innovation(lm, z, minid, R_mat, Proll_mat, Ppitch_mat, Pyaw_mat)

            K = self.cov @ H.T @ cp.linalg.inv(S)
            new_believe = self.believe + K @ y

            d_believe = new_believe - self.believe
            dm = (d_believe[:self.pose_size].T @
                  cp.linalg.solve(self.cov[:self.pose_size, :self.pose_size], d_believe[:self.pose_size]))

            if dm.item() < self.md_chi:
                new_cov = (cp.eye(len(self.believe)) - K @ H) @ self.cov
                self.believe, self.cov = new_believe, new_cov

        self.poses.append(self.believe[:6])
        if self.debug: print(f"camera pose : {self.believe[:6]}")

    def sigma_ellipse(self, ax, n=1):
        eig_vals, eig_vec = cp.linalg.eigh(self.cov[:3, :3])

        radii = n * cp.sqrt(cp.clip(eig_vals, 0, None))
        u_grid, v_grid = cp.meshgrid(cp.linspace(0, 2*cp.pi, 40), cp.linspace(0, cp.pi, 40))

        x_sphere = radii[0]*cp.cos(u_grid)*cp.sin(v_grid)
        y_sphere = radii[1]*cp.sin(u_grid)*cp.sin(v_grid)
        z_sphere = radii[2]*cp.cos(v_grid)

        points_sphere = cp.stack([x_sphere, y_sphere, z_sphere], axis=-1)

        transformed_points = (points_sphere @ eig_vec.T + self.believe[:3]).get()

        return ax.plot_surface(transformed_points[..., 0],
                               transformed_points[..., 1],
                               transformed_points[..., 2],
                               rstride=4, cstride=4, color=self.color, alpha=0.3)

    def draw(self, ax, time):
        ax.cla() # 以前の描画をクリア

        if self.believe is not None:
            r = 1.0
            believe_np = self.believe.get()

            roll, pitch, yaw = believe_np[3:6]
            Sr, Cr = cp.sin(roll), cp.cos(roll)
            Sp, Cp = cp.sin(pitch), cp.cos(pitch)
            Sy, Cy = cp.sin(yaw), cp.cos(yaw)
            rotate_mat_np = Rotate_mat(Sr, Cr, Sp, Cp, Sy, Cy).get()

            ax.quiver(*believe_np[:3], *r*rotate_mat_np[:, 0], color=self.color)
            ax.quiver(*believe_np[:3], *r*rotate_mat_np[:, 1], color="red")
            ax.quiver(*believe_np[:3], *r*rotate_mat_np[:, 2], color="green")

            all_lm_np = self.believe[self.pose_size: self.pose_size+self.lm_size*self.nLM].get().reshape(self.nLM, self.lm_size)
            ax.scatter(all_lm_np[:, 0], all_lm_np[:, 1], all_lm_np[:, 2], s=0.7, color="black")

            poses_np_list = [p.get() for p in self.poses]
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
    parser.add_argument('--video', default=r"C:\Users\sakata\Documents\VID_20250909_110359_00_023.mp4",
                        help="入力動画のパス")
    parser.add_argument('--threshold', type=int, default=150,
                        help="FAST のコーナー検出しきい値")
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=False,
                        help="アニメーションを保存する/しない")
    args = parser.parse_args()

    world = World(args.video, threshold=args.threshold, save=args.save)
    world.draw()


if __name__ == '__main__':
    main()