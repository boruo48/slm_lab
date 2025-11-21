# 振幅slm用

import numpy as np
from PIL import Image

# ---- SLM の解像度に合わせて変更 ----
W, H = 1920, 1080
# ------------------------------------

# 黒画像 (0)
black = np.zeros((H, W), dtype=np.uint8)
Image.fromarray(black).save("slm_black.png")

# 白画像 (255)
white = np.ones((H, W), dtype=np.uint8) * 255
Image.fromarray(white).save("slm_white.png")

print("保存完了: slm_black.png / slm_white.png")


# # 配列を作成（H×W）
# img = np.zeros((H, W), dtype=np.uint8)

# # 左半分 → 黒(0)
# img[:, :W//2] = 0

# # 右半分 → 白(255)
# img[:, W//2:] = 255

# Image.fromarray(img).save("slm_half_bw.png")

# print("保存完了: slm_half_bw.png")


# 配列を作成（H×W）
img2 = np.zeros((H, W), dtype=np.uint8)

# 左半分 → 黒(0)
img2[:, :W//2] = 0.05 * 255

# 右半分 → 白(255)
img2[:, W//2:] = 0.95 * 255

Image.fromarray(img2).save("slm_half_bw_offset.png")

print("保存完了: slm_half_bw_offset.png")