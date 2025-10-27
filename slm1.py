# 設定->システム->ディスプレイでslmの方のモニターを右側にしておく

import cv2
import numpy as np

# --- 表示する画像（ホログラム例） ---
"""
振幅画像
"""
# img = cv2.imread("18_gray.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("19.jpg", cv2.IMREAD_GRAYSCALE)


"""
位相画像
"""
# img = cv2.imread("hologram.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("hologram2.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("bun045.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("./phasepng/15/phase.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("./isoumidori-/18/phase_100.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("./resultmi+/19/phase_75.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("./resultmi+/19/phase_50.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("./resultmi+/15/phase_100.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("./resultmi+/18/phase_100.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("./resultmi+/17/phase_100.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("./result_midori++/19/d_150mm/phase_19_d150mm.png", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("C:/Users/fshib/Desktop/images/result_midori++/16/d_150mm/phase_16_d150mm.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("画像が読み込めません。ファイルパスを確認してください。")
    exit()

if img.shape != (1080, 1920):
    img = cv2.resize(img, (1920,1080), interpolation=cv2.INTER_NEAREST)
    print("oh")


# --- ウィンドウ作成 ---
cv2.namedWindow("SLM", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("SLM", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# --- もしディスプレイが2枚あるなら SLM 側に移動 ---
# SLMの位置が右側（メインディスプレイ1920x1080）にあると仮定
cv2.moveWindow("SLM", 1920, 0)
print("ok")

# --- 表示 ---
cv2.imshow("SLM", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
