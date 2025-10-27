#png->jpg
#jpg->png

import cv2
import os

# 入力と出力ファイル名
input_path = "19.jpg"
output_path = "output.png"
# input_path = "input.png"
# output_path = "output.jpg"


# 画像を読み込み
img = cv2.imread(input_path)

# PNGとして保存（拡張子で自動判定）
cv2.imwrite(output_path, img)

print(f"✅ {os.path.basename(input_path)} → {os.path.basename(output_path)} に変換しました。")
