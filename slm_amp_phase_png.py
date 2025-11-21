# 2台の SLM
# 振幅PNG & 位相PNG をそれぞれの SLM に送る

# -*- coding: utf-8 -*-

import HEDS
from hedslib.heds_types import *
import cv2
import numpy as np


# --- ここに自分の SLM の解像度を設定（例：1920x1080） ---
# AMP_W, AMP_H = 1920, 1080
# PHASE_W, PHASE_H = 1920, 1080


def init_slm(index: int):
    """SLM を index 指定で初期化（index:0 = 振幅SLM、index:1 = 位相SLM）"""
    err = HEDS.SDK.Init(4, 1)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    # preselector に "index:x" を入れて複数SLMを指定
    slm = HEDS.SLM.Init(f"index:{index}", True, 0.0)
    assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

    print(f"SLM {index} initialized.")
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


def show_phase_png_on_slm(png_path: str, slm, width, height):
    """位相PNGを位相SLMに表示"""
    # SDK には「位相画像を直接ファイルから読む」関数がある
    # loadPhaseDataFromFile() を使うとミスが少ない
    err, dataHandle = slm.loadPhaseDataFromFile(png_path)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    err = dataHandle.show()
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    print("位相パターンを表示しました。")


def main():
    # --- 2 台の SLM を初期化 ---
    slm_amp = init_slm(index=0)    # 振幅SLM
    slm_phase = init_slm(index=1)  # 位相SLM

    SLM_AMP_W = int(slm_amp.width_px())
    SLM_AMP_H = int(slm_amp.height_px())
    SLM_PHS_W = int(slm_amp.width_px())
    SLM_PHS_H = int(slm_amp.height_px())

    # --- それぞれのSLMに画像を表示 ---
    show_amp_png_on_slm("amp.png", slm_amp, SLM_AMP_W, SLM_AMP_H)
    show_phase_png_on_slm("phase.png", slm_phase, SLM_PHS_W, SLM_PHS_H)

    print("両SLMへの表示が完了しました。")
    input("Enter を押すと SLM を閉じます...")
    err = slm_amp.window().close()
    if err != HEDSERR_NoError:
        print("SLM window close error:", HEDS.SDK.ErrorString(err))
    err = slm_phase.window().close()
    if err != HEDSERR_NoError:
        print("SLM window close error:", HEDS.SDK.ErrorString(err))
    # print("SLMウィンドウを閉じるまで待ちます…")

    HEDS.SDK.WaitAllClosed()
    print("SLMウィンドウ終了")


if __name__ == "__main__":
    main()
