# 振幅画像amp.pmg(0~255)を表示

import cv2
import numpy as np

import HEDS
from hedslib.heds_types import *

# ★ここを自分の振幅SLMの解像度に合わせて直す（LETO-3/PLUTO-2.1 なら 1920x1080）
# SLM_W = 1920
# SLM_H = 1080

def init_slm():
    # SDK 初期化
    err = HEDS.SDK.Init(4, 1)
    if err != HEDSERR_NoError:
        raise RuntimeError(f"SDK Init failed: {HEDS.SDK.ErrorString(err)}")

    # SLM 初期化（必要なら "name:LETO" や "index:0" に変更）
    slm = HEDS.SLM.Init()
    # slm = HEDS.SLM.Init("name:PLUTO", True, 0.0)
    if slm.errorCode() != HEDSERR_NoError:
        raise RuntimeError(f"SLM Init failed: {HEDS.SDK.ErrorString(slm.errorCode())}")

    print("SLM initialized.")

    # --- SLM解像度取得（公式サンプル準拠） ---
    W = int(slm.width_px())
    H = int(slm.height_px())

    # print("Resolution:", w, "x", h)
    print("Resolution:", W, "x", H)
    return slm

def show_amp_png_on_slm(png_path: str, slm, width, height):
    """振幅PNGを振幅SLMに表示"""
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("振幅PNGが読み込めません")

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    # SDK に numpy を渡す
    err, dataHandle = slm.loadImageData(img)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    err = dataHandle.show()
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    print("振幅パターンを表示しました。")

def main():
    slm = init_slm()
    SLM_W = int(slm.width_px())
    SLM_H = int(slm.height_px())
    show_amp_png_on_slm("amp.png", slm, SLM_W, SLM_H)

    input("Enter を押すと SLM を閉じます...")
    err = slm.window().close()
    if err != HEDSERR_NoError:
        print("SLM window close error:", HEDS.SDK.ErrorString(err))
    HEDS.SDK.WaitAllClosed()
    print("SLMウィンドウ終了")

if __name__ == "__main__":
    main()