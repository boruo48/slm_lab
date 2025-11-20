# 位相画像phase.png(0~255)を表示

import HEDS
from hedslib.heds_types import *
import cv2
import numpy as np


def init_slm():
    # SDK 初期化
    err = HEDS.SDK.Init(4, 1)
    if err != HEDSERR_NoError:
        raise RuntimeError(f"SDK Init failed: {HEDS.SDK.ErrorString(err)}")

    # SLM 初期化（必要なら "name:LETO" や "index:0" に変更可）
    slm = HEDS.SLM.Init()
    if slm.errorCode() != HEDSERR_NoError:
        raise RuntimeError(f"SLM Init failed: {HEDS.SDK.ErrorString(slm.errorCode())}")

    print("SLM initialized.")
    print("Resolution:", slm.width(), "x", slm.height())
    return slm


def show_png_phase_on_slm(png_path: str, slm):
    # --- ① PNG を 8bit グレースケールで読み込み ---
    phase_u8 = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if phase_u8 is None:
        raise ValueError(f"Failed to load PNG: {png_path}")

    print("Loaded PNG:", phase_u8.shape, phase_u8.dtype)  # 例: (H, W) uint8

    # --- ② SLM の解像度に合わせる ---
    slm_w = slm.width()
    slm_h = slm.height()

    # 単純リサイズ（まずはこれでOK。後でパディング方式に変えてもいい）
    phase_u8_resized = cv2.resize(
        phase_u8,
        (slm_w, slm_h),  # (width, height)
        interpolation=cv2.INTER_NEAREST
    )

    # --- ③ 必要なら向き補正（上下/左右反転など） ---
    # 実機の像を見ながら、必要に応じてコメントアウトを切り替える：
    # phase_u8_resized = np.flipud(phase_u8_resized)   # 上下反転
    # phase_u8_resized = np.fliplr(phase_u8_resized)   # 左右反転

    # --- ④ SLM に表示 ---
    # 関数名は SDK の example に合わせてね：
    HEDS.SLM.displayImage(slm, phase_u8_resized)
    # もし example が displayDataArray を使っているなら：
    # HEDS.SLM.displayDataArray(slm, phase_u8_resized)

    print("PNG phase displayed on SLM.")


def main():
    slm = init_slm()
    # 同じフォルダに置いた phase.png を表示する例
    show_png_phase_on_slm("phase.png", slm)

    input("Enter を押すと SLM を閉じます...")
    err = slm.window().close()
    if err != HEDSERR_NoError:
        print("SLM window close error:", HEDS.SDK.ErrorString(err))


if __name__ == "__main__":
    main()
