import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib
try: matplotlib.use('TkAgg')
except ImportError: pass  # tkinter が無い環境 (テスト/CI) では既定バックエンドを使う
import matplotlib.animation as anm
import numpy as np

# --- 1. オイラー角関連のヘルパー関数 ---

# ZYXオイラー角から回転行列 R を計算する関数 (Roll: phi, Pitch: theta, Yaw: psi)
# R = Rz(psi) Ry(theta) Rx(phi)
def euler_to_R(phi, theta, psi):
    c1, s1 = np.cos(psi), np.sin(psi)   # Yaw (psi, Z-axis)
    c2, s2 = np.cos(theta), np.sin(theta) # Pitch (theta, Y-axis)
    c3, s3 = np.cos(phi), np.sin(phi)     # Roll (phi, X-axis)

    R = np.array([
        [c1*c2, c1*s2*s3 - s1*c3, c1*s2*c3 + s1*s3],
        [s1*c2, s1*s2*s3 + c1*c3, s1*s2*c3 - c1*s3],
        [-s2, c2*s3, c2*c3]
    ])
    return R

# 回転行列RのRoll (phi) に関する偏微分 dR/d(phi)
def dR_dphi(phi, theta, psi):
    c1, s1 = np.cos(psi), np.sin(psi)
    c2, s2 = np.cos(theta), np.sin(theta)
    c3, s3 = np.cos(phi), np.sin(phi)

    return np.array([
        [0, c1*s2*c3 + s1*s3, -c1*s2*s3 + s1*c3],
        [0, s1*s2*c3 - c1*s3, -s1*s2*s3 - c1*c3],
        [0, c2*c3, -c2*s3]
    ])

# 回転行列RのPitch (theta) に関する偏微分 dR/d(theta)
def dR_dtheta(phi, theta, psi):
    c1, s1 = np.cos(psi), np.sin(psi)
    c2, s2 = np.cos(theta), np.sin(theta)
    c3, s3 = np.cos(phi), np.sin(phi)

    return np.array([
        [-c1*s2, c1*c2*s3, c1*c2*c3],
        [-s1*s2, s1*c2*s3, s1*c2*c3],
        [-c2, -s2*s3, -s2*c3]
    ])

# 回転行列RのYaw (psi) に関する偏微分 dR/d(psi)
def dR_dpsi(phi, theta, psi):
    c1, s1 = np.cos(psi), np.sin(psi)
    c2, s2 = np.cos(theta), np.sin(theta)
    c3, s3 = np.cos(phi), np.sin(phi)

    return np.array([
        [-s1*c2, -s1*s2*s3 - c1*c3, -s1*s2*c3 + c1*s3],
        [c1*c2, c1*s2*s3 - s1*c3, c1*s2*c3 + s1*s3],
        [0, 0, 0]
    ])

# Quaternion_Normalizationは不要になるため削除
# --- End Helper Functions ---


class Map:
    def __init__(self, map_pass):
        self.map = np.array([])
        self.landmark_scatter_plot = None
        self.landmark_colors = []
        self.map_pass = map_pass

    def load_map(self):
        self.map = np.loadtxt(self.map_pass, delimiter=',')
        self.map[:, 1] = -self.map[:, 1] 
    
    def settings(self, ax=None):
        self.load_map()
        
        self.landmark_colors = ["#272727ff"] * self.map.shape[0]
        
        self.landmark_scatter_plot = ax.scatter(self.map[:, 0], self.map[:, 1], self.map[:, 2], 
                                                s=100, marker="*", label="landmarks", c=self.landmark_colors)
        
        for i in range(self.map.shape[0]): ax.text(self.map[i, 0], self.map[i, 1], self.map[i, 2], f"id{i}", fontsize=10, color='black')

    def update_landmark_colors(self, current_observed_ids):
        if self.map.size == 0: return
        
        self.landmark_colors = ["#272727ff"] * self.map.shape[0]
        
        for observed_id in current_observed_ids:
            if 0 <= observed_id < self.map.shape[0]: self.landmark_colors[observed_id] = "#f3ba1c"
            
        if self.landmark_scatter_plot is not None:
            self.landmark_scatter_plot.set_facecolors(self.landmark_colors)
            self.landmark_scatter_plot.set_edgecolors(self.landmark_colors)


class Obsevation_AR:
    # (Observation_AR クラスは変更なし)
    def __init__(self):
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        # OpenCV>=4.7 では aruco.detectMarkers が廃止されたため ArucoDetector を使う
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.W = 1280

    def data(self, frame):
        resized_frame = self.resize_frame(frame)
        obs_data, resized_frame = self.detection(resized_frame)
        return obs_data, resized_frame

    
    def detection(self, imput_frame):
        masked_frame = self.apply_mask(imput_frame)
        corners, ids, _ = self.detector.detectMarkers(masked_frame)
        obs_data = self.calc_obs(corners, ids)
        detect_frame = aruco.drawDetectedMarkers(imput_frame, corners, ids)
        return obs_data, detect_frame

    def apply_mask(self, imput_frame):
        h, w, _ = imput_frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        mask_roi_x = 0
        mask_roi_y = h // 4
        mask_roi_width = w
        mask_roi_height = h // 2

        mask[mask_roi_y : mask_roi_y + mask_roi_height, 
             mask_roi_x : mask_roi_x + mask_roi_width] = 255
        
        masked_frame = cv2.bitwise_and(imput_frame, imput_frame, mask=mask)
        return masked_frame
    
    def calc_obs(self, corners, ids):
        if ids is not None and len(ids) > 0:
            ids = np.array(ids).flatten()
            means = np.mean(np.array(corners)[:, 0], axis=1) 
            
            y = -2*np.pi/self.W * means + np.array([np.pi*0.5, np.pi*0.5]) 
            y_phi_all, y_theta_all = y[:, 0], y[:, 1]
            
            return np.stack([np.cos(y_theta_all) * np.cos(y_phi_all), 
                             np.cos(y_theta_all) * np.sin(y_phi_all), 
                             np.sin(y_theta_all),
                             ids], axis=1)
        else: return np.array([])
        
    def resize_frame(self, input_frame):
        width, height = int(self.W), int(self.W*0.5)
        return cv2.resize(input_frame, (width, height), interpolation = cv2.INTER_AREA)

    def obs_update(self, frame):
        obs_data, detect_frame = self.data(frame)
        cv2.imshow('AR detect', detect_frame)
        return obs_data


class ExKalmanFilter:
    def __init__(self, map):
        '''
        self.pose: 状態ベクトル Tnom[phi, theta, psi, p_x, p_y, p_z, v_x, v_y, v_z] (9 dimensions)
        '''
        self.map = map
        self.pose = np.zeros(9) # 9次元に初期化 [phi, theta, psi, p_x, p_y, p_z, v_x, v_y, v_z]
        
        # P, Q, Rも9x9に調整
        self.P = np.eye(9) * 0.1 
        self.Q = np.eye(9) * 1e-2 
        self.R = np.eye(3) * 1e-3

    def EKF_update(self, time_interval, observations):
        self.prediction(time_interval)
        if observations.size > 0:
            for observation in observations: self.correction(observation)

    def prediction(self, time_interval):
        '''
        オイラー角は慣性系での角速度をゼロと仮定するため、F行列を適用するだけで良い（正規化不要）
        '''
        F = self.matF(time_interval=time_interval)
        self.pose = F @ self.pose
        
        # オイラー角を [-pi, pi] の範囲にラップする (オプションだが安定性のために推奨)
        self.pose[0:3] = np.arctan2(np.sin(self.pose[0:3]), np.cos(self.pose[0:3]))
        
        self.P = F @ self.P @ F.T + self.Q

    def matF(self, time_interval):
        # 状態ベクトル: [phi, theta, psi, p_x, p_y, p_z, v_x, v_y, v_z] (9 dim)
        
        I3 = np.eye(3)
        O33 = np.zeros((3, 3))

        # 姿勢 (phi, theta, psi) は変化しないと仮定 (I3)
        # 位置 p = p + dt * v
        # 速度 v = v
        
        F = np.block([[I3, O33, O33],
                      [O33, I3, time_interval*I3],
                      [O33, O33, I3]])
        return F

    def correction(self, observation):
        id = int(observation[3]) 
        landmark_data = self.map.map
        
        if landmark_data.size == 0 or id >= landmark_data.shape[0] or id < 0: 
            return

        lm = landmark_data[id] 

        # 観測関数のヤコビアン H を計算
        H = self.matH(lm)

        # 観測残差 (innovation) を計算: y - h(x_hat)
        innovation = observation[:3] - self.obs_func(lm)

        # 観測更新ステップ (3x3行列)
        S = H @ self.P @ H.T + self.R 
        
        try: S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError: return

        K = self.P @ H.T @ S_inv # Kalman gain (9x3行列)

        self.pose = self.pose + K @ innovation
        self.P = (np.eye(9) - K @ H) @ self.P

        # 修正後のオイラー角を [-pi, pi] にラップ
        self.pose[0:3] = np.arctan2(np.sin(self.pose[0:3]), np.cos(self.pose[0:3]))


    def matH(self, lm):
        eta = self.pose[0:3] # Euler angles (phi, theta, psi)
        p = self.pose[3:6]  # カメラの位置 (p_x, p_y, p_z)
        
        phi, theta, psi = eta[0], eta[1], eta[2]
        
        diff = lm - p 
        norm_diff = np.linalg.norm(diff)
        
        if norm_diff == 0: return np.zeros((3, 9))

        d_inv = 1 / norm_diff
        
        R = euler_to_R(phi, theta, psi)
        Rt = R.T
        u = diff / norm_diff # normalized vector in inertial frame
        
        # 1. 姿勢 (eta) に対するヤコビアン (3x3)
        # dh/d(eta_i) = (d(R^T)/d(eta_i)) * u
        H_phi = (dR_dphi(phi, theta, psi).T @ u).reshape(3, 1)
        H_theta = (dR_dtheta(phi, theta, psi).T @ u).reshape(3, 1)
        H_psi = (dR_dpsi(phi, theta, psi).T @ u).reshape(3, 1)
        
        H_eta = np.hstack([H_phi, H_theta, H_psi])

        # 2. 位置 (p) に対するヤコビアン (3x3)
        # H_p = - R^T * (I - u u^T) / ||d||
        H_p = - Rt @ (np.eye(3) - np.outer(u, u)) / norm_diff

        # 3. 速度 (v) に対するヤコビアン (3x3)
        O_v = np.zeros((3, 3)) 

        H = np.hstack([H_eta, H_p, O_v]) # 3x9 matrix
        
        return H

    def obs_func(self, lm):
        eta = self.pose[0:3]
        p = self.pose[3:6]
        
        R = euler_to_R(*eta)
        Rt = R.T
        
        diff = lm - p
        norm_diff = np.linalg.norm(diff)
        
        return np.zeros(3) if norm_diff == 0 else 1/norm_diff * Rt @ diff


class Camera():
    def __init__(self, map):
        self.poses = np.array([])
        self.quivers = None
        self.sigma_surface = None
        self.path_line = None
        self.estimator = ExKalmanFilter(map=map)
        self.observation = Obsevation_AR()

    def settings(self, ax=None):
        self.quivers_settings(ax)
        self.path_line_settings(ax)
        self.sigma_surface_settings(ax)
    
    def update(self, ax=None):
        self.quivers_update()
        self.path_line_update()
        self.sigma_surface_update(ax)

    def quivers_settings(self, ax):
        self.quivers = [ax.quiver(0,0,0,0,0,0, color="blue", arrow_length_ratio=0.3),
                        ax.quiver(0,0,0,0,0,0, color="red", arrow_length_ratio=0.3),
                        ax.quiver(0,0,0,0,0,0, color="green", arrow_length_ratio=0.3)]
    
    def path_line_settings(self, ax):
        self.path_line, = ax.plot([], [], [], linewidth=0.5, color="blue")

    def sigma_surface_settings(self, ax):
        self.sigma_surface = ax.plot_surface(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), 
                                             rstride=4, cstride=4, color="blue", alpha=0.3)
        self.sigma_surface.set_visible(False)
    
    def quivers_update(self):
        # 姿勢の取得インデックスを [0:3] に変更
        R, t = euler_to_R(*self.estimator.pose[0:3]), self.estimator.pose[3:6]
        if self.quivers:
            for i, q in enumerate(self.quivers):
                start_point = t
                end_point = t + R[:, i] * 0.5 
                q.set_segments([[start_point, end_point]])

    def path_line_update(self):
        # 位置の取得インデックスを [3:6] に変更
        current_pos = self.estimator.pose[3:6].reshape(1, -1)
        self.poses = current_pos if self.poses.size == 0 else np.vstack([self.poses, current_pos])
        
        if self.path_line: self.path_line.set_data_3d(self.poses[:, 0], self.poses[:, 1], self.poses[:, 2])

    def sigma_surface_update(self, ax=None, n=1):
        # 状態ベクトルが9次元であることを確認
        if self.estimator.P.shape[0] < 6:
            if self.sigma_surface is not None: self.sigma_surface.set_visible(False)
            return

        # P_pp (位置に関する共分散行列) のインデックスを [3:6, 3:6] に変更
        P_pp = self.estimator.P[3:6, 3:6] 
        
        try:
            P_pp = (P_pp + P_pp.T) / 2
            eig_vals, eig_vec = np.linalg.eig(P_pp)
        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix for position not positive definite. Skipping sigma surface.")
            if self.sigma_surface is not None: self.sigma_surface.set_visible(False)
            return

        eig_vals[eig_vals < 1e-9] = 1e-9 
        radii = n * np.sqrt(eig_vals) 

        u_grid, v_grid = np.meshgrid(np.linspace(0, 2*np.pi, 20), np.linspace(0, np.pi, 20))

        x_sphere = radii[0]*np.cos(u_grid)*np.sin(v_grid)
        y_sphere = radii[1]*np.sin(u_grid)*np.sin(v_grid)
        z_sphere = radii[2]*np.cos(v_grid)

        points_sphere = np.stack([x_sphere, y_sphere, z_sphere], axis=-1)

        # 現在の位置のインデックスを [3:6] に変更
        transformed_points = (points_sphere @ eig_vec.T) + self.estimator.pose[3:6]
        
        if self.sigma_surface is not None:
            self.sigma_surface.remove() 

        self.sigma_surface = ax.plot_surface(transformed_points[..., 0],
                                             transformed_points[..., 1],
                                             transformed_points[..., 2],
                                             rstride=4, cstride=4, color="blue", alpha=0.3)
        self.sigma_surface.set_visible(True)


class World():
    def __init__(self, vision_pass, map_pass):

        self.cap = cv2.VideoCapture(vision_pass)
        self.totalframecount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.time_interval = 1/fps if fps > 0 else 0.03 
        self.time_text_obj = None 
        self.fig = None
        self.ax = None
        self.map = Map(map_pass=map_pass)
        
        self.camera = Camera(map=self.map)
        
    def settings(self):
        self.world_settings()
        self.time_settings(self.ax)
        self.map.settings(ax=self.ax)
        self.camera.settings(self.ax) 

    def update(self, ax, frame, time):
        self.time_update(time=time)
        
        obs_data = self.camera.observation.obs_update(frame) 
        self.camera.estimator.EKF_update(time_interval=self.time_interval, observations=obs_data) 
        
        current_observed_ids = []
        if obs_data.size > 0: current_observed_ids = [int(obs[3]) for obs in obs_data]
        self.map.update_landmark_colors(current_observed_ids) 

        self.camera.update(ax=ax)
    
    def world_settings(self):
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("X [m]", fontsize=10)
        self.ax.set_ylabel("Y [m]", fontsize=10)
        self.ax.set_zlabel("Z [m]", fontsize=10)
        self.ax.set_xlim3d(-5, 5)
        self.ax.set_ylim3d(-5, 5)
        self.ax.set_zlim3d(-2.5, 2.5)
        self.ax.set_box_aspect((10,10,5))
    
    def time_settings(self, ax=None):
        self.time_text_obj = ax.text(0, 0, 0, "", fontsize=10, color='black')
    
    def time_update(self, time):
        time_str = "t = %.2f[s]" % (time)
        # 姿勢が9次元であることを確認
        if self.camera.estimator.pose is not None and self.camera.estimator.pose.size == 9:
            current_pos = self.camera.estimator.pose[3:6]
        else:
            current_pos = np.array([0,0,0])
            
        if self.time_text_obj:
            self.time_text_obj.set_position((current_pos[0]+0.5, current_pos[1]+0.5, current_pos[2]+0.5))
            self.time_text_obj.set_text(time_str)


class Processing():
    def __init__(self, vision_pass, map_pass):
        self.world = World(vision_pass=vision_pass, map_pass=map_pass)
        self.map_pass = map_pass
        self.save = False

    def play(self):
        self.world.settings()

        ani = anm.FuncAnimation(self.world.fig, self.one_step, fargs=(self.world.ax, ),
                                frames=self.world.totalframecount,
                                interval=int(self.world.time_interval*1000),
                                repeat=False, save_count=self.world.totalframecount)
        if self.save:
            ani.save(r"G:\test.mp4", writer="ffmpeg")
            plt.close(self.world.fig)
        else: plt.show()
    
    def one_step(self, i, ax):
        ret, frame = self.world.cap.read()

        if ret:
            time = self.world.time_interval * i
            self.world.update(ax=ax, frame=frame, time=time)
        else:
            print(f"End of video or failed to read frame at frame {i}.")
            self.world.cap.release()
            return


if __name__ == '__main__':

    # 実行環境に合わせてパスを適宜変更してください
    vision_pass = r"C:\Users\sakata\Documents\SLAM\map_base\VID_20250826_171914_00_012.mp4"
    map_pass = r"C:\Users\sakata\Documents\SLAM\map_base\map_20250826.csv"
    Processing(vision_pass=vision_pass, map_pass=map_pass).play()