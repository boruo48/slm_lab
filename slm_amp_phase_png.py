# 2台の SLM
# 振幅PNG & 位相PNG をそれぞれの SLM に送る

import HEDS
from hedslib.heds_types import *

import cv2
import numpy as np

def init_two_slms():
    # --- SDK 初期化（1回だけ） ---
    err = HEDS.SDK.Init(4, 1)
    if err != HEDSERR_NoError:
        raise RuntimeError(f"SDK Init failed: {HEDS.SDK.ErrorString(err)}")

    # --- 振幅SLM（例：index:0） ---
    # もしデバイス名がわかっているなら "name:LETO" とか "name:Tensor" などに変えてOK
    slm_amp = HEDS.SLM.Init("index:0")
    if slm_amp.errorCode() != HEDSERR_NoError:
        raise RuntimeError(f"Amp SLM Init failed: {HEDS.SDK.ErrorString(slm_amp.errorCode())}")

    print("Amp SLM initialized.")
    print("  Resolution:", slm_amp.width(), "x", slm_amp.height())

    # --- 位相SLM（例：index:1） ---
    slm_phase = HEDS.SLM.Init("index:1")
    if slm_phase.errorCode() != HEDSERR_NoError:
        raise RuntimeError(f"Phase SLM Init failed: {HEDS.SDK.ErrorString(slm_phase.errorCode())}")

    print("Phase SLM initialized.")
    print("  Resolution:", slm_phase.width(), "x", slm_phase.height())

    return slm_amp, slm_phase



def load_and_resize_u8(path, slm):
    """0〜255 PNG を読み込んで、その SLM 解像度にリサイズして返す"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    h, w = slm.height(), slm.width()
    img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    # 向き補正が必要ならここで flip
    # img_resized = np.flipud(img_resized)  # 上下反転
    # img_resized = np.fliplr(img_resized)  # 左右反転

    return img_resized


def show_amp_phase_png(amp_png_path, phase_png_path):
    slm_amp, slm_phase = init_two_slms()

    # --- 振幅画像を読み込み＆振幅SLMサイズに合わせる ---
    amp_u8 = load_and_resize_u8(amp_png_path, slm_amp)

    # --- 位相画像を読み込み＆位相SLMサイズに合わせる ---
    phase_u8 = load_and_resize_u8(phase_png_path, slm_phase)

    # --- 各SLMに表示 ---
    # 関数名は、あなたの SDK example に合わせて変更してね：
    HEDS.SLM.displayImage(slm_amp, amp_u8)      # 振幅SLM
    HEDS.SLM.displayImage(slm_phase, phase_u8)  # 位相SLM
    # もし example が displayDataArray を使っているなら：
    # HEDS.SLM.displayDataArray(slm_amp, amp_u8)
    # HEDS.SLM.displayDataArray(slm_phase, phase_u8)

    print("Amp & Phase patterns displayed on their SLMs.")

    input("Enter を押すと両方のSLMを閉じます...")

    # --- SLM ウィンドウを閉じる ---
    err = slm_amp.window().close()
    if err != HEDSERR_NoError:
        print("Amp SLM window close error:", HEDS.SDK.ErrorString(err))

    err = slm_phase.window().close()
    if err != HEDSERR_NoError:
        print("Phase SLM window close error:", HEDS.SDK.ErrorString(err))


if __name__ == "__main__":
    show_amp_phase_png("amp.png", "phase.png")
