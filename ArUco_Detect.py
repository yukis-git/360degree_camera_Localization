import cv2
import cv2.aruco as aruco
import numpy as np # numpyをインポート

targetVideo = r"C:\Users\sakata\Documents\Python\estimation\1_ArUco_20250522.mp4"

cap = cv2.VideoCapture(targetVideo)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

while cap.isOpened():

    ret, frame = cap.read()

    # フレームが正しく読み込めたか確認
    if not ret:
        print("終了") if frame is None else print("フレームの読み込みに失敗しました 。")
        break

    # フレームをリサイズ
    scale_percent = 25  # %
    width, height = int(frame.shape[1] * scale_percent / 100), int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) # リサイズ後のフレームを別名で保持

    # ArUcoマーカーを検出
    corners, ids, rejectedImgPoints = aruco.detectMarkers(resized_frame, aruco_dict, parameters=parameters)

    # 検出されたマーカーを描画
    output_frame = aruco.drawDetectedMarkers(resized_frame.copy(), corners, ids) # resized_frameをコピーして描画

    # マーカーの中心座標とIDをprintする処理
    # if ids is not None: # マーカーが1つ以上検出された場合
    observed = []
    for i in range(len(ids)):
        # i番目のマーカーのコーナーを取得 (corners[i]は (1, 4, 2) の形状)
        marker_corners = corners[i][0] # (4, 2) の配列になる (角の座標)
        
        # 中心座標から変位角と天頂角を計算
        ratio =np.pi/2944*100/scale_percent
        center_x, center_y = -np.mean(marker_corners[:, 0])*ratio, np.mean(marker_corners[:, 1])*ratio
        z = np.array([center_x, center_y]).T.tolist()
        
        # IDを取得 (ids[i]は (1,) の形状なので ids[i][0] で値を取得)
        id = int(ids[i][0])
        observed.append((z, id))
    print(observed)

    # 結果のフレームを表示
    cv2.imshow('frame', output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# 全てのリソースを解放
cap.release()
cv2.destroyAllWindows()