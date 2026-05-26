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

    # 検出されたマーカーの描画とID/座標の出力
    # output_frame を resized_frame のコピーで初期化し、このフレームに描画していく
    output_frame = resized_frame.copy() 

    # マーカーの中心座標とIDをprintする処理
    observed = [] # 各フレームでリセット
    if ids is not None: # マーカーが1つ以上検出された場合
        for i in range(len(ids)):
            # i番目のマーカーのコーナーを取得 (corners[i]は (1, 4, 2) の形状)
            marker_corners = corners[i][0] # (4, 2) の配列になる (角の座標)
            
            # IDを取得 (ids[i]は (1,) の形状なので ids[i][0] で値を取得)
            marker_id = int(ids[i][0])

            # --- カスタムBBOX描画 ---
            # 角の座標を整数に変換し、cv2.polylinesで描画できる形式に整形
            points = marker_corners.astype(np.int32).reshape((-1, 1, 2))
            # BBOXの色を赤 (BGR: 0,0,255)、線の太さを2に設定
            cv2.polylines(output_frame, [points], True, (255, 255, 204), 1) # Trueで閉じたポリゴン

            # --- IDテキストのカスタム描画 ---
            # 表示用のマーカー中心座標を計算 (整数値)
            center_x_display = int(np.mean(marker_corners[:, 0]))
            center_y_display = int(np.mean(marker_corners[:, 1]))

            # 表示するテキストのフォーマットを設定
            text = f"ID: {marker_id}" # 例: "ID: 123"

            # テキストのフォント、スケール、色、太さを設定
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6 # フォントサイズ
            font_thickness = 1 # フォントの太さ
            text_color = (255, 255, 0) # BGR: (青, 緑, 赤) 例: シアン

            # テキストの描画位置を調整
            # テキストのサイズを取得して、中央揃えや適切な位置に配置するため
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = center_x_display - text_size[0] // 2 # マーカーの中心にテキストを横方向でセンタリング
            text_y = int(marker_corners[:, 1].min()) - 10 # マーカーの最上部から10ピクセル上に表示

            # テキストが画像の上端からはみ出す場合の調整
            if text_y < text_size[1] + 5: # text_size[1]はテキストの高さ。+5は余白。
                text_y = int(marker_corners[:, 1].max()) + text_size[1] + 10 # マーカーの最下部から下に表示

            cv2.putText(output_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            # --- 元の座標出力処理 (変更なし) ---
            ratio =np.pi/2944*100/scale_percent
            center_x_output, center_y_output = -np.mean(marker_corners[:, 0])*ratio, np.mean(marker_corners[:, 1])*ratio
            z = np.array([center_x_output, center_y_output]).T.tolist()
            
            observed.append((z, marker_id))
        print(observed)

    # 結果のフレームを表示
    cv2.imshow('frame', output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# 全てのリソースを解放
cap.release()
cv2.destroyAllWindows()