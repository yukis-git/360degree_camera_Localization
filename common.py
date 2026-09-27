import cv2
import cv2.aruco as aruco
import numpy as np


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

