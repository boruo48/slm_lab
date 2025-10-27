# show_on_slm_opencv.py
import cv2, numpy as np

# SLM解像度（例：FHD）
H, W = 1080, 1920

# 例：位相 0..2π に対応する 0..255 の放射状グレースケール（Rに載せる）
yy, xx = np.ogrid[:H, :W]
r = np.sqrt((xx - W/2)**2 + (yy - H/2)**2)
img = np.uint8(np.clip((r / r.max()) * 255.0, 0, 255))

# BGR作成（R=img, G=B=0）
bgr = np.dstack([np.zeros_like(img), np.zeros_like(img), img])

cv2.namedWindow("SLM", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("SLM", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("SLM", bgr)

# ここでウィンドウを SLM 画面へドラッグして全画面化（WMにより自動でSLM側に出す設定も可）
cv2.waitKey(0)
