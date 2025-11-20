import sys
import time

# import HEDS
# print("HEDS Loaded:", HEDS.__file__)
import numpy as np

# sys.path.append(r"C:/Program Files/HOLOEYE Photonics/SLM Display SDK (Python) v4.1.0/api/python")

# ① HOLOEYE SDK の python API へのパスを通す
sys.path.append(r"C:/Program Files/HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0/api/python")

# ② HEDS ではなく hedslib を import
import hedslib
from hedslib.heds_types import *

print("hedslib loaded from:", hedslib.__file__)


# --- SDK の初期化 ---
err = HEDS.SDK.Init(4, 1)                     # v4.1
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# --- SLM を開く ---
slm = HEDS.SLM.Init()
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# --- プレビューを開く（高速モード） ---
HEDS.SDK.libapi.heds_slmpreview_open(slm.window().id())

# --- テストパターン（1-1 縦バイナリ格子）を表示 ---
HEDS.SDK.libapi.heds_slm_show_grating_binary_vertical(slm.slm_id(), 1, 1, 0, 255)
time.sleep(2.0)

# --- 任意の NumPy 画像を表示する例 ---
img = (np.random.rand(slm.height(), slm.width()) * 255).astype(np.uint8)
handle = HEDS.SLM.ImageHandle()
handle.loadFromArrayU8(img)
slm.showDataHandle(handle)
time.sleep(2.0)

# --- 終了処理 ---
slm.window().close()
