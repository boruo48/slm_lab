# 位相画像phase.png(0~255)を表示

import HEDS
from hedslib.heds_types import *
import cv2
import numpy as np

# ★ここを自分の振幅SLMの解像度に合わせて直す（LETO-3/PLUTO-2.1 なら 1920x1080）
# SLM_W = 1920
# SLM_H = 1080

def init_slm():
    # SDK 初期化
    err = HEDS.SDK.Init(4, 1)
    if err != HEDSERR_NoError:
        raise RuntimeError(f"SDK Init failed: {HEDS.SDK.ErrorString(err)}")

    # SLM 初期化（必要なら "name:LETO" や "index:0" に変更可）
    slm = HEDS.SLM.Init()
    # slm = HEDS.SLM.Init("name:LETO", True, 0.0)
    if slm.errorCode() != HEDSERR_NoError:
        raise RuntimeError(f"SLM Init failed: {HEDS.SDK.ErrorString(slm.errorCode())}")

    print("SLM initialized.")
    SLM_W = int(slm.width_px())
    SLM_H = int(slm.height_px())
    print("Resolution:", SLM_W, "x", SLM_H)
    return slm


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
    slm = init_slm()
    SLM_W = int(slm.width_px())
    SLM_H = int(slm.height_px())
    # 同じフォルダに置いた phase.png を表示する例
    show_phase_png_on_slm("phase.png", slm, SLM_W, SLM_H)

    input("Enter を押すと SLM を閉じます...")
    err = slm.window().close()
    if err != HEDSERR_NoError:
        print("SLM window close error:", HEDS.SDK.ErrorString(err))
    HEDS.SDK.WaitAllClosed()
    print("SLMウィンドウ終了")


if __name__ == "__main__":
    main()
