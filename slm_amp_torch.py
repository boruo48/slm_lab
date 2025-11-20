# 振幅パイトーチファイルamp.thをslmに表示

import HEDS
from hedslib.heds_types import *

import numpy as np
import torch
import torch.nn.functional as F

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



def amp_array_to_slm_u8(
    amp, slm_h: int, slm_w: int
) -> np.ndarray:
    """
    振幅画像 (0〜1) を SLM に投げられる 0〜255 uint8 に変換する。
    amp: numpy.ndarray or torch.Tensor
         想定レンジ: 0.0〜1.0（超えていても clamp する）
         想定shape: (H,W), (1,H,W), (1,1,H,W)
    """
    # --- torch.Tensor に統一 ---
    if isinstance(amp, np.ndarray):
        t = torch.from_numpy(amp)
    elif isinstance(amp, torch.Tensor):
        t = amp
    else:
        raise TypeError("amp must be numpy.ndarray or torch.Tensor")

    t = t.float()

    # shape を (1,1,H,W) にそろえる
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)       # (1,1,H,W)
    elif t.dim() == 3:
        t = t.unsqueeze(0)                    # (1,C,H,W)
    elif t.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported amp shape: {t.shape}")

    t = t[0:1, 0:1, :, :]                     # (1,1,H,W)

    # 0〜1 に clamp
    t = torch.clamp(t, 0.0, 1.0)

    # SLM 解像度にあわせてリサイズ
    t = F.interpolate(t, size=(slm_h, slm_w), mode="nearest")

    # 0〜1 → 0〜255 に
    t = (t * 255.0).round()
    t = torch.clamp(t, 0.0, 255.0)

    amp_u8 = t.to(torch.uint8).squeeze().cpu().numpy()  # (slm_h, slm_w)
    return amp_u8

def show_amp_array_on_slm(amp, slm):
    """
    amp: 0〜1 の振幅画像 (numpy or torch)
    """
    slm_w = slm.width()
    slm_h = slm.height()

    amp_u8 = amp_array_to_slm_u8(amp, slm_h=slm_h, slm_w=slm_w)

    # 必要なら左右/上下反転（光学系に合わせて調整）
    # amp_u8 = np.flipud(amp_u8)
    # amp_u8 = np.fliplr(amp_u8)

    # SDK の example に合わせて displayImage or displayDataArray を使う
    HEDS.SLM.displayImage(slm, amp_u8)
    # HEDS.SLM.displayDataArray(slm, amp_u8) のパターンならこちら

    print("Amplitude pattern displayed on SLM.")

def main():
    slm = init_slm()

    amp_tensor = torch.load("amp.pt")

    # 例：ランダム振幅 (0〜1)
    H, W = 512, 512
    amp_tensor = torch.rand(1, 1, H, W)  # (1,1,H,W), 0〜1

    show_amp_array_on_slm(amp_tensor, slm)

    input("Enter を押すと SLM を閉じます...")
    slm.window().close()


if __name__ == "__main__":
    main()
