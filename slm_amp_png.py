# 振幅画像amp.pmg(0~255)を表示

import cv2
import numpy as np

import HEDS
from hedslib.heds_types import *


def init_slm():
    # SDK 初期化
    err = HEDS.SDK.Init(4, 1)
    if err != HEDSERR_NoError:
        raise RuntimeError(f"SDK Init failed: {HEDS.SDK.ErrorString(err)}")

    # SLM 初期化（必要なら "name:LETO" や "index:0" に変更）
    slm = HEDS.SLM.Init()
    if slm.errorCode() != HEDSERR_NoError:
        raise RuntimeError(f"SLM Init failed: {HEDS.SDK.ErrorString(slm.errorCode())}")

    print("SLM initialized.")
    print("Resolution:", slm.width(), "x", slm.height())
    return slm


def show_amp_png_on_slm(png_path: str, slm):
    # 8bitグレースケールで読み込み
    amp_u8 = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if amp_u8 is None:
        raise ValueError(f"Failed to load PNG: {png_path}")

    print("Loaded PNG:", amp_u8.shape, amp_u8.dtype)

    slm_w = slm.width()
    slm_h = slm.height()

    # SLM解像度に合わせてリサイズ
    amp_u8_resized = cv2.resize(
        amp_u8,
        (slm_w, slm_h),
        interpolation=cv2.INTER_NEAREST
    )

    # 必要に応じて向き補正
    # amp_u8_resized = np.flipud(amp_u8_resized)
    # amp_u8_resized = np.fliplr(amp_u8_resized)

    HEDS.SLM.displayImage(slm, amp_u8_resized)
    # HEDS.SLM.displayDataArray(slm, amp_u8_resized)

    print("Amplitude PNG displayed on SLM.")

def main():
    slm = init_slm()
    show_amp_png_on_slm("amp.png", slm)

    input("Enter を押すと SLM を閉じます...")
    slm.window().close()
