import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib
try: matplotlib.use('TkAgg')
except ImportError: pass  # tkinter が無い環境 (テスト/CI) では既定バックエンドを使う
import matplotlib.animation as anm
import numpy as np


def matR(ww, wx, wy, wz):
    return np.array([[1.0 - 2*(wy**2 + wz**2), 2*(wx*wy - ww*wz), 2*(ww*wy + wx*wz)],
                     [2*(ww*wz + wx*wy), 1.0 - 2*(wx**2 + wz**2), 2*(wy*wz - ww*wx)],
                     [2*(wx*wz - ww*wy), 2*(ww*wx + wy*wz), 1.0 - 2*(wx**2 + wy**2)]])

def PRPww(ww, wx, wy, wz):
    return 2*np.array([[0.0, -wz, wy],
                       [wz, 0.0, -wx],
                       [-wy, wx, 0.0]])

def PRPwx(ww, wx, wy, wz):
    return 2*np.array([[0.0, wy, wz],
                       [wy, -2*wx, -ww],
                       [wz, ww, -2*wx]])

def PRPwy(ww, wx, wy, wz):
    return 2*np.array([[-2*wy, wx, ww],
                       [wx, 0.0, wz],
                       [-ww, wz, -2*wy]])

def PRPwz(ww, wx, wy, wz):
    return 2*np.array([[-2*wz, -ww, wx],
                       [ww, -2*wz, wy],
                       [wx, wy, 0.0]])

def Quaternion_Normalization(q):
    quat_norm = np.linalg.norm(q)
    return q / quat_norm if quat_norm != 0 else np.array([1.0, 0.0, 0.0, 0.0])


class Map:
    def __init__(self, map_pass):
        self.map = np.array([])
        self.landmark_scatter_plot = None # To store the PathCollection object from matplotlib
        self.landmark_colors = []         # List to store colors for each landmark, updated dynamically
        self.map_pass = map_pass

    def load_map(self):
        self.map = np.loadtxt(self.map_pass, delimiter=',')
        self.map[:, 1] = -self.map[:, 1] # Adjust y-coordinate if necessary for display consistency
    
    def settings(self, ax=None):
        self.load_map()
        
        # Initialize all landmark colors to orange ('#ffa500')
        self.landmark_colors = ["#272727ff"] * self.map.shape[0]
        
        # Plot landmarks with initial colors
        self.landmark_scatter_plot = ax.scatter(self.map[:, 0], self.map[:, 1], self.map[:, 2], 
                                                s=100, marker="*", label="landmarks", c=self.landmark_colors)
        
        # Add text labels for landmarks
        for i in range(self.map.shape[0]): ax.text(self.map[i, 0], self.map[i, 1], self.map[i, 2], f"id{i}", fontsize=10, color='black')

    def update_landmark_colors(self, current_observed_ids):
        """
        Updates the color of landmarks: green for currently observed, orange for not observed.
        """
        if self.map.size == 0: return
        
        # Reset all colors to default orange
        self.landmark_colors = ["#272727ff"] * self.map.shape[0]
        
        # Set colors for currently observed landmarks to green
        for observed_id in current_observed_ids:
            if 0 <= observed_id < self.map.shape[0]: self.landmark_colors[observed_id] = "#f3ba1c"
            
        # Update the scatter plot's colors
        if self.landmark_scatter_plot is not None:
            self.landmark_scatter_plot.set_facecolors(self.landmark_colors)
            self.landmark_scatter_plot.set_edgecolors(self.landmark_colors) # Update edge color too for '*' marker


class Obsevation_AR:
    def __init__(self):
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        # OpenCV>=4.7 では aruco.detectMarkers が廃止されたため ArucoDetector を使う
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.W = 1280

    def data(self, frame):
        '''
        観測値とリサイズされたフレームを返す
        '''
        resized_frame = self.resize_frame(frame)
        obs_data, resized_frame = self.detection(resized_frame)
        return obs_data, resized_frame

    
    def detection(self, imput_frame):
        '''
        マーカーを検出し、観測値と検出したフレームを返す
        '''
        masked_frame = self.apply_mask(imput_frame)
        corners, ids, _ = self.detector.detectMarkers(masked_frame)
        obs_data = self.calc_obs(corners, ids)
        detect_frame = aruco.drawDetectedMarkers(imput_frame, corners, ids)
        return obs_data, detect_frame

    def apply_mask(self, imput_frame):
        """
        フレームにマスクを適用し、指定された領域のみを残す。
        """
        h, w, _ = imput_frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # --- マスク領域の定義 ---
        mask_roi_x = 0
        mask_roi_y = h // 4
        mask_roi_width = w
        mask_roi_height = h // 2

        # マスク領域を白くする
        mask[mask_roi_y : mask_roi_y + mask_roi_height, 
             mask_roi_x : mask_roi_x + mask_roi_width] = 255
        
        # マスクを適用
        masked_frame = cv2.bitwise_and(imput_frame, imput_frame, mask=mask)
        return masked_frame
    
    def calc_obs(self, corners, ids):
        '''
        マーカーの四つ角の座標から観測値ベクトルを計算する
        '''
        if ids is not None and len(ids) > 0:
            ids = np.array(ids).flatten()
            means = np.mean(np.array(corners)[:, 0], axis=1) # Get the center of each marker
            
            # Map pixel coordinates to spherical angles (simplified model)
            # This mapping might need refinement based on actual camera intrinsics
            y = -2*np.pi/self.W * means + np.array([np.pi*0.5, np.pi*0.5]) 
            y_phi_all, y_theta_all = y[:, 0], y[:, 1]
            
            # Convert spherical coordinates to 3D unit vector
            return np.stack([np.cos(y_theta_all) * np.cos(y_phi_all), 
                             np.cos(y_theta_all) * np.sin(y_phi_all), 
                             np.sin(y_theta_all),
                             ids], axis=1)
        else: return np.array([])
        
    def resize_frame(self, input_frame):
        '''
        観測後に扱いやすいサイズにフレームをリサイズする
        '''
        width, height = int(self.W), int(self.W*0.5)
        return cv2.resize(input_frame, (width, height), interpolation = cv2.INTER_AREA)

    def obs_update(self, frame):
        obs_data, detect_frame = self.data(frame)
        cv2.imshow('AR detect', detect_frame)
        return obs_data


class ExKalmanFilter:
    def __init__(self, map):
        '''
        self.pose        : 状態ベクトルTnom[q_w, q_x, q_y, q_z, p_x, p_y, p_z, v_x, v_y, v_z]
        '''
        self.map = map # Mapオブジェクト全体が格納される
        self.pose = np.zeros(10) # Initialize with zeros
        self.pose[0] = 1.0       # Initial quaternion ww component (identity rotation)
        self.P = np.eye(10) * 0.1 # 初期共分散行列の調整
        self.Q = np.eye(10) * 1e-2 # プロセスノイズ共分散行列
        self.R = np.eye(3) * 1e-3 # 観測ノイズ共分散行列

    def EKF_update(self, time_interval, observations):
        self.prediction(time_interval)
        if observations.size > 0:
            for observation in observations: self.correction(observation)

    def prediction(self, time_interval):
        '''
        x_hat_t|t-1  = f(x_hat_t-1|t-1, 0, u_t, 0)
        P_t|t-1      = F_t * P_t-1|t-1 * F_t.T + Q_t
        '''
        F = self.matF(time_interval=time_interval)
        self.pose = F @ self.pose
        
        # 予測されたクォータニオンを正規化
        self.pose[:4] = Quaternion_Normalization(self.pose[:4])

        self.P = F @ self.P @ F.T + self.Q

    def matF(self, time_interval):
        # 状態ベクトル: [q_w, q_x, q_y, q_z, p_x, p_y, p_z, v_x, v_y, v_z]
        # Assuming no angular velocity and linear acceleration for simplicity
        I4 = np.eye(4)
        I3 = np.eye(3)
        O43 = np.zeros((4, 3))
        O34 = np.zeros((3, 4))
        O33 = np.zeros((3, 3))

        F = np.block([[I4, O43, O43],
                      [O34, I3, time_interval*I3],
                      [O34, O33, I3]])
        return F

    def correction(self, observation):
        '''
        K_t = P_t|t-1 * H_t.T * (H_t * P_t|t-1 * H_t.T + R_t)^-1
        x_hat_t|t  <- x_hat_t|t-1 + K_t * (z_t - h(x_hat_t|t-1))
        P_t|t       = (I - K_t * H_T) * P_t|t-1
        '''
        id = int(observation[3]) # idは整数
        
        # --- 修正点: Mapオブジェクトの内部のNumPy配列にアクセスし、形状でチェックする ---
        landmark_data = self.map.map
        
        # データが存在しない、またはIDが範囲外の場合はスキップ
        if landmark_data.size == 0 or id >= landmark_data.shape[0] or id < 0: 
            return

        lm = landmark_data[id] # マップからランドマーク情報を取得
        # ------------------------------------------------------------------------

        # 観測関数のヤコビアン H を計算
        H = self.matH(lm)

        # 観測残差 (innovation) を計算: y - h(x_hat)
        innovation = observation[:3] - self.obs_func(lm)

        # 観測更新ステップ
        S = H @ self.P @ H.T + self.R # Innovation covariance
        
        # Sの逆行列が存在するか確認 (特異行列の場合の対処)
        try: S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError: return

        K = self.P @ H.T @ S_inv # Kalman gain

        self.pose = self.pose + K @ innovation
        # Joseph 形式: 丸め誤差があっても P の対称性・半正定値性が保たれる
        I_KH = np.eye(10) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # クォータニオンを正規化
        self.pose[:4] = Quaternion_Normalization(self.pose[:4])


    def matH(self, lm):
        p = self.pose[4:7]  # カメラの位置
        q = self.pose[:4]   # カメラの姿勢クォータニオン
        
        diff = lm - p # ランドマークからカメラ位置へのベクトル (lm_inertial - p_inertial)
        norm_diff = np.linalg.norm(diff)
        
        if norm_diff == 0: return np.zeros((3, 10))

        d_inv = 1 / norm_diff
        Rt = matR(*q).T # カメラ姿勢の回転行列の転置 (R_ci in formulas)

        dh_dq_w = d_inv * (PRPww(*q).T @ diff)
        dh_dq_x = d_inv * (PRPwx(*q).T @ diff)
        dh_dq_y = d_inv * (PRPwy(*q).T @ diff)
        dh_dq_z = d_inv * (PRPwz(*q).T @ diff)
        
        H_q = np.hstack([dh_dq_w[:, None], dh_dq_x[:, None], dh_dq_y[:, None], dh_dq_z[:, None]]) # 3x4 matrix

        # Jacobian of the bearing vector w.r.t. position p
        uv_inertial = diff / norm_diff # unit vector from p to lm in inertial frame
        H_p = Rt @ (np.outer(uv_inertial, uv_inertial) - np.eye(3)) / norm_diff

        # 速度に対する偏微分は0
        O_v = np.zeros((3, 3)) # 3x3 zero matrix for velocity part

        H = np.hstack([H_q, H_p, O_v]) # 3x10 matrix
        
        return H

    def obs_func(self, lm):
        
        diff = lm - self.pose[4:7]
        norm_diff = np.linalg.norm(diff)
        
        return np.zeros(3) if norm_diff == 0 else 1/norm_diff * matR(*self.pose[:4]).T @ diff


class Camera():
    def __init__(self, map):
        self.poses = np.array([]) # 経路を保存するための配列
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
        # 初期のpath_lineオブジェクトを作成
        self.path_line, = ax.plot([], [], [], linewidth=0.5, color="blue")

    def sigma_surface_settings(self, ax):
        # Initial dummy plot_surface, will be updated in sigma_surface_update
        self.sigma_surface = ax.plot_surface(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), 
                                             rstride=4, cstride=4, color="blue", alpha=0.3)
        self.sigma_surface.set_visible(False) # 最初は非表示
    
    def quivers_update(self):
        R, t = matR(*self.estimator.pose[:4]), self.estimator.pose[4:7]
        if self.quivers:
            for i, q in enumerate(self.quivers):
                start_point = t
                end_point = t + R[:, i] * 0.5 # 矢印の長さを調整
                q.set_segments([[start_point, end_point]])

    def path_line_update(self):
        current_pos = self.estimator.pose[4:7].reshape(1, -1)
        self.poses = current_pos if self.poses.size == 0 else np.vstack([self.poses, current_pos])
        
        if self.path_line: self.path_line.set_data_3d(self.poses[:, 0], self.poses[:, 1], self.poses[:, 2])

    def sigma_surface_update(self, ax=None, n=1):
        if self.estimator.P.shape[0] < 7:
            if self.sigma_surface is not None: self.sigma_surface.set_visible(False)
            return

        P_pp = self.estimator.P[4:7, 4:7] # 位置に関する共分散行列P_pp
        
        try:
            P_pp = (P_pp + P_pp.T) / 2 # Ensure symmetry
            eig_vals, eig_vec = np.linalg.eigh(P_pp)  # 対称行列なので eigh (eig は複素数を返し得る)
        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix for position not positive definite. Skipping sigma surface.")
            if self.sigma_surface is not None: self.sigma_surface.set_visible(False)
            return

        eig_vals[eig_vals < 1e-9] = 1e-9 # Prevent sqrt of negative and near zero
        radii = n * np.sqrt(eig_vals) # エリプソイドの半径 (e.g., 1-sigma boundary)

        u_grid, v_grid = np.meshgrid(np.linspace(0, 2*np.pi, 20), np.linspace(0, np.pi, 20))

        # 単位球面を生成
        x_sphere = radii[0]*np.cos(u_grid)*np.sin(v_grid)
        y_sphere = radii[1]*np.sin(u_grid)*np.sin(v_grid)
        z_sphere = radii[2]*np.cos(v_grid)

        # 球面上の点を結合
        points_sphere = np.stack([x_sphere, y_sphere, z_sphere], axis=-1)

        # 固有ベクトルで回転させ、現在の位置を中心に移動
        transformed_points = (points_sphere @ eig_vec.T) + self.estimator.pose[4:7]
        
        if self.sigma_surface is not None:
            self.sigma_surface.remove() # Remove old surface

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
        self.time_interval = 1/fps if fps > 0 else 0.03 # FPSが0の場合の対策 (30fps)
        self.time_text_obj = None # time_settingsで初期化されるようにする
        self.fig = None
        self.ax = None
        self.map = Map(map_pass=map_pass)
        
        # --- 修正点: Mapインスタンス全体を渡す ---
        self.camera = Camera(map=self.map)
        
    def settings(self):
        self.world_settings()
        self.time_settings(self.ax)
        self.map.settings(ax=self.ax)
        self.camera.settings(self.ax) 

    def update(self, ax, frame, time):
        self.time_update(time=time)
        
        obs_data = self.camera.observation.obs_update(frame) # Get observations
        self.camera.estimator.EKF_update(time_interval=self.time_interval, observations=obs_data) # Update EKF state
        # --- Updated: Update observed landmark colors based on current frame ---
        current_observed_ids = []
        if obs_data.size > 0: current_observed_ids = [int(obs[3]) for obs in obs_data]
        self.map.update_landmark_colors(current_observed_ids) # Call Map's method to update the plot color
        # --------------------------------------------------------------------

        # Now update camera's plotting elements
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
        if self.time_text_obj:
            # Display time near the camera's current position
            current_pos = self.camera.estimator.pose[4:7] if self.camera.estimator.pose is not None and self.camera.estimator.pose.size == 10 else np.array([0,0,0])
            self.time_text_obj.set_position((current_pos[0]+0.5, current_pos[1]+0.5, current_pos[2]+0.5))
            self.time_text_obj.set_text(time_str)


class Processing():
    def __init__(self, vision_pass, map_pass):
        self.world = World(vision_pass=vision_pass, map_pass=map_pass)
        self.map_pass = map_pass # Store map_pass
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
        ret, frame = self.world.cap.read() # self.cap を使う

        if ret:
            time = self.world.time_interval * i
            self.world.update(ax=ax, frame=frame, time=time)
        else:
            print(f"End of video or failed to read frame at frame {i}.")
            self.world.cap.release()
            # plt.close(self.fig) 
            return


if __name__ == '__main__':

    vision_pass = r"C:\Users\sakata\Documents\SLAM\map_base\VID_20250826_171914_00_012.mp4"
    map_pass = r"C:\Users\sakata\Documents\SLAM\map_base\map_20250826.csv"
    Processing(vision_pass=vision_pass, map_pass=map_pass).play()