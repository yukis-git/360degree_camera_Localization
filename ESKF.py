import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib
try: matplotlib.use('TkAgg')
except ImportError: pass  # tkinter が無い環境 (テスト/CI) では既定バックエンドを使う
import matplotlib.animation as anm
import numpy as np
import time as Time


def matR(ww, wx, wy, wz):
    return np.array([[1.0 - 2*(wy**2 + wz**2), 2*(wx*wy - ww*wz), 2*(ww*wy + wx*wz)],
                     [2*(ww*wz + wx*wy), 1.0 - 2*(wx**2 + wz**2), 2*(wy*wz - ww*wx)],
                     [2*(wx*wz - ww*wy), 2*(ww*wx + wy*wz), 1.0 - 2*(wx**2 + wy**2)]])

def Quaternion_Normalization(q):
    quat_norm = np.linalg.norm(q)
    return q / quat_norm if quat_norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quat_from_angle_axis(q):
    angle = np.linalg.norm(q)
    if angle < 1e-12: return np.array([1.0, 0.0, 0.0, 0.0])
    axis = q / angle  # 旧実装は q /= angle で呼び出し元の配列 (delta_x のビュー) を書き換えていた
    half_angle = 0.5 * angle
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


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


class ESKalmanFilter:
    def __init__(self, map_instance, q_theta=0.1, q_pos=1e-4, q_vel=0.1):
        '''
        q_theta : 姿勢のランダムウォーク強度 [rad^2/s]  (sqrt(0.1)~0.32 rad/sqrt(s))
        q_pos   : 位置の追加ランダムウォーク強度 [m^2/s]
        q_vel   : 速度のランダムウォーク強度 [(m/s)^2/s]
        '''
        self.map_instance = map_instance 

        self.pose = np.zeros(10)
        self.pose[0] = 1.0       

        self.P = np.eye(9) * 0.1 
        # 連続時間のプロセスノイズ密度。予測で dt を掛けるのでフレームレートに依存しない
        # (旧: 毎フレーム 1e-6 固定。姿勢はほぼ固定扱いとなり、カメラの回転に追従できなかった)
        self.Q_c = np.diag([q_theta]*3 + [q_pos]*3 + [q_vel]*3)
        self.R = np.eye(3) * 1e-3

        self._I3 = np.eye(3)
        self._O33 = np.zeros((3, 3))
        self._I9 = np.eye(9) 

    def EKF_update(self, time_interval, observations):
        self.prediction(time_interval)
        if observations.size > 0:
            for observation in observations: self.correction(observation)

    def prediction(self, time_interval):
        q_nom = self.pose[0:4]
        p_nom = self.pose[4:7]
        v_nom = self.pose[7:10]

        p_nom_pred = p_nom + v_nom * time_interval
        v_nom_pred = v_nom

        self.pose[0:4] = Quaternion_Normalization(q_nom) 
        self.pose[4:7] = p_nom_pred
        self.pose[7:10] = v_nom_pred

        F_matrix = self.matF_eskf(time_interval=time_interval)
        self.P = F_matrix @ self.P @ F_matrix.T + self.Q_c * time_interval

    def matF_eskf(self, time_interval):
        F = np.zeros((9, 9))
        F[0:3, 0:3] = self._I3
        F[3:6, 3:6] = self._I3
        F[3:6, 6:9] = time_interval * self._I3
        F[6:9, 6:9] = self._I3
        return F

    def correction(self, observation):
        id = int(observation[3])
        
        landmark_data = self.map_instance.map 
        
        if landmark_data.size == 0 or id >= landmark_data.shape[0] or id < 0: 
            return

        lm = landmark_data[id]

        predicted_observation = self.obs_func(lm)
        innovation = observation[:3] - predicted_observation

        H_matrix = self.matH_eskf(lm)

        S = H_matrix @ self.P @ H_matrix.T + self.R
        try: S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix in ESKF correction, skipping update.")
            return

        K = self.P @ H_matrix.T @ S_inv

        delta_x = K @ innovation

        q_nom = self.pose[0:4]
        p_nom = self.pose[4:7]
        v_nom = self.pose[7:10]

        dtheta = delta_x[0:3]
        dp     = delta_x[3:6]
        dv     = delta_x[6:9]

        p_nom_new = p_nom + dp
        v_nom_new = v_nom + dv

        dq_quat = quat_from_angle_axis(dtheta)
        q_nom_new = quat_multiply(q_nom, dq_quat)

        self.pose[0:4] = Quaternion_Normalization(q_nom_new)
        self.pose[4:7] = p_nom_new
        self.pose[7:10] = v_nom_new

        # Joseph 形式: 丸め誤差があっても P の対称性・半正定値性が保たれる
        I_KH = self._I9 - K @ H_matrix
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # ESKF のリセット: 誤差状態を名目状態へ注入した後、姿勢誤差の共分散を新しい名目姿勢の接空間へ写す
        G = self._I9.copy()
        G[0:3, 0:3] = self._I3 - skew(0.5 * dtheta)
        self.P = G @ self.P @ G.T

    def matH_eskf(self, lm):
        q_nom = self.pose[0:4]
        p_nom = self.pose[4:7]

        diff_inertial = lm - p_nom
        norm_diff_inertial = np.linalg.norm(diff_inertial)

        if norm_diff_inertial == 0: return np.zeros((3, 9))

        R_ic_T = matR(*q_nom).T

        u_cam_predicted = (1 / norm_diff_inertial) * R_ic_T @ diff_inertial

        H_dtheta = skew(u_cam_predicted)

        u_inertial = diff_inertial / norm_diff_inertial
        H_dp = -R_ic_T @ (self._I3 - np.outer(u_inertial, u_inertial)) / norm_diff_inertial

        H_dv = self._O33

        H = np.zeros((3, 9))
        H[0:3, 0:3] = H_dtheta
        H[0:3, 3:6] = H_dp
        H[0:3, 6:9] = H_dv
        return H

    def obs_func(self, lm):
        q_nom = self.pose[0:4]
        p_nom = self.pose[4:7]

        diff = lm - p_nom
        norm_diff = np.linalg.norm(diff)

        return np.zeros(3) if norm_diff == 0 else matR(*q_nom).T @ (diff / norm_diff)
    

class Camera:
    def __init__(self, map_instance):
        self.observation = Obsevation_AR()
        self.estimator = ESKalmanFilter(map_instance=map_instance)
        
        self.poses = np.array([]) 
        self.quivers = None
        self.sigma_surface = None
        self.path_line = None

    def settings(self, ax=None):
        self.quivers_settings(ax)
        self.path_line_settings(ax)
        self.sigma_surface_settings(ax)

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
        R, t = matR(*self.estimator.pose[:4]), self.estimator.pose[4:7]
        if self.quivers:
            for i, q in enumerate(self.quivers):
                start_point = t
                end_point = t + R[:, i] * 0.5
                q.set_segments([[start_point, end_point]])

    def path_line_update(self):
        current_pos = self.estimator.pose[4:7].reshape(1, -1)
        self.poses = current_pos if self.poses.size == 0 else np.vstack([self.poses, current_pos])
        
        if self.path_line: self.path_line.set_data_3d(self.poses[:, 0], self.poses[:, 1], self.poses[:, 2])

    def sigma_surface_update(self, ax=None, n=1):
        
        # Pのサイズが9x9未満の場合はエラー防止
        if self.estimator.P.shape[0] < 6:
            if self.sigma_surface is not None: self.sigma_surface.set_visible(False)
            return

        # --- 修正点: ESKFのP行列から位置誤差の共分散を取得 (インデックス 3:6) ---
        P_pp = self.estimator.P[3:6, 3:6] 

        try:
            P_pp = (P_pp + P_pp.T) / 2
            eig_vals, eig_vec = np.linalg.eigh(P_pp)  # 対称行列なので eigh (eig は複素数を返し得る)
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

        transformed_points = (points_sphere @ eig_vec.T) + self.estimator.pose[4:7]
        
        if self.sigma_surface is not None: self.sigma_surface.remove()

        self.sigma_surface = ax.plot_surface(transformed_points[..., 0],
                                             transformed_points[..., 1],
                                             transformed_points[..., 2],
                                             rstride=4, cstride=4, color="blue", alpha=0.3)
        self.sigma_surface.set_visible(True)


class World():
    def __init__(self, vision_pass, map_pass):
        self.map = Map(map_pass=map_pass)
        self.camera = Camera(map_instance=self.map)
        
        self.cap = cv2.VideoCapture(vision_pass)
        self.totalframecount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.time_interval = 1/fps if fps > 0 else 0.03
        self.time_text_obj = None
        self.fig = None
        self.ax = None
        
    def settings(self):
        self.world_settings()
        self.time_settings(self.ax)
        self.map.settings(ax=self.ax)
        self.camera.settings(self.ax) 

    def update(self, ax, frame, time):
        # 観測と推定
        obs_data = self.camera.observation.obs_update(frame)
        self.camera.estimator.EKF_update(time_interval=self.time_interval, observations=obs_data)
        
        # 描画の更新
        self.time_update(time=time)
        
        current_observed_ids = []
        if obs_data.size > 0: current_observed_ids = [int(obs[3]) for obs in obs_data]
        self.map.update_landmark_colors(current_observed_ids)
        
        self.camera.quivers_update()
        self.camera.path_line_update()
        self.camera.sigma_surface_update(ax)
    
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
        if self.time_text_obj:
            current_pos = self.camera.estimator.pose[4:7] if self.camera.estimator.pose.size == 10 else np.array([0,0,0])
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
            s = Time.perf_counter()
            self.world.update(ax=ax, frame=frame, time=time)
            e = Time.perf_counter()
            print(round(e-s, 4))
        else:
            print(f"End of video or failed to read frame at frame {i}.")
            self.world.cap.release()
            return


if __name__ == '__main__':

    vision_pass = r"C:\Users\sakata\Documents\SLAM\map_base\VID_20250826_171914_00_012.mp4"
    map_pass = r"C:\Users\sakata\Documents\SLAM\map_base\map_20250826.csv"
    Processing(vision_pass=vision_pass, map_pass=map_pass).play()