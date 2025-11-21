# 振幅パイトーチファイルamp.ptをslmに表示

# -*- coding: utf-8 -*-

import math
import torch
import torch.nn.functional as F
import numpy as np

import HEDS
from hedslib.heds_types import *


# ---- SLM解像度（あなたの振幅SLMに合わせて変更）----
AMP_W, AMP_H = 1920, 1080


def to_amp_u8(amp_tensor: torch.Tensor, W: int, H: int) -> np.ndarray:
    """
    amp_tensor: 0〜1 振幅（想定）
      shape: (H,W) / (1,H,W) / (1,1,H,W)
    -> SLM用 uint8画像 (H,W), 0..255
    """
    t = amp_tensor.detach().to("cpu").float()

    # shape を (1,1,H,W) に揃える
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(0)
    elif t.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported amp_tensor shape: {t.shape}")

    t = t[0:1, 0:1, :, :]  # (1,1,H,W)

    # 0..1 にクリップ(amp.ptが0~1のとき)
    t = torch.clamp(t, 0.0, 1.0)

    # SLM解像度にリサイズ
    t = F.interpolate(t, size=(H, W), mode="nearest")

    # 0..1 -> 0..255
    t = (t * 255.0).round()
    t = torch.clamp(t, 0.0, 255.0)

    return t.to(torch.uint8).squeeze().numpy()  # (H,W), uint8


def main():
    # ---- SDK init ----
    err = HEDS.SDK.Init(4, 1)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    # ---- 振幅SLM init（必要なら index や name 指定）----
    slm_amp = HEDS.SLM.Init("index:0", True, 0.0)
    # slm_amp = HEDS.SLM.Init("name:PLUTO", True, 0.0)
    assert slm_amp.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm_amp.errorCode())

    # ---- amp.pt 読み込み ----
    amp_tensor = torch.load("amp.pt", map_location="cpu")

    # dict保存だった場合の保険
    if isinstance(amp_tensor, dict):
        if "amp" in amp_tensor:
            amp_tensor = amp_tensor["amp"]
        else:
            # 最初に見つかったテンソルを使う
            amp_tensor = next(v for v in amp_tensor.values() if isinstance(v, torch.Tensor))

    print("amp_tensor:", amp_tensor.shape, amp_tensor.min().item(), amp_tensor.max().item())

    # ---- uint8振幅画像へ変換 ----
    amp_u8 = to_amp_u8(amp_tensor, AMP_W, AMP_H)

    # ---- numpy配列を SDKに渡して表示 ----
    err, dataHandle = slm_amp.loadImageData(amp_u8)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    err = dataHandle.show()
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    print("amp.pt を振幅SLMに表示しました。")
    input("Enter を押すと SLM を閉じます...")
    err = slm_amp.window().close()
    if err != HEDSERR_NoError:
        print("SLM window close error:", HEDS.SDK.ErrorString(err))
    HEDS.SDK.WaitAllClosed()
    print("SLMウィンドウ終了")


if __name__ == "__main__":
    main()
