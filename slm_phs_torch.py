# 位相パイトーチファイルphase.ptをslmに表示

import math
import torch
import torch.nn.functional as F
import numpy as np

import HEDS
from hedslib.heds_types import *


def phase_tensor_to_u8(phase_tensor: torch.Tensor, W: int, H: int,
                       phase_range: str = "minus_pi_pi") -> np.ndarray:
    """
    phase_tensor: 位相テンソル [rad] or 正規化位相
      shape: (H,W) / (1,H,W) / (1,1,H,W)
    phase_range:
      "minus_pi_pi" : [-π, π] 想定  -> [0,2π) にwrapして 0..255
      "zero_two_pi" : [0,2π] 想定   -> 0..255
      "zero_one"    : [0,1] 想定    -> 0..255
    """
    t = phase_tensor.detach().to("cpu").float()

    # shape -> (1,1,H,W)
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(0)
    elif t.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported phase_tensor shape: {t.shape}")

    t = t[0:1, 0:1, :, :]  # (1,1,h,w)

    # --- 位相レンジ処理 ---
    if phase_range in ["minus_pi_pi", "zero_two_pi"]:
        two_pi = 2.0 * math.pi
        if phase_range == "minus_pi_pi":
            t = torch.remainder(t + two_pi, two_pi)  # [-π,π] -> [0,2π)
        else:
            t = torch.remainder(t, two_pi)           # [0,2π] -> [0,2π)

        t = t / two_pi                              # [0,2π) -> [0,1]

    elif phase_range == "zero_one":
        t = torch.clamp(t, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown phase_range: {phase_range}")

    # --- SLM解像度にリサイズ ---
    t = F.interpolate(t, size=(H, W), mode="nearest")

    # [0,1] -> [0,255]
    t = torch.clamp(t, 0.0, 1.0)
    t = (t * 255.0).round()
    t = torch.clamp(t, 0.0, 255.0)

    return t.to(torch.uint8).squeeze().numpy()  # (H,W) uint8


def main():
    # --- SDK init ---
    err = HEDS.SDK.Init(4, 1)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    # --- 位相SLM init（index:1 を位相SLMにしてるならこれ） ---
    slm_phase = HEDS.SLM.Init("index:1", True, 0.0)
    assert slm_phase.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm_phase.errorCode())

    # --- SLM解像度取得（公式サンプル準拠） ---
    W = int(slm_phase.width_px())
    H = int(slm_phase.height_px())

    print("Phase SLM resolution:", W, "x", H)

    # --- phase.pt 読み込み ---
    phase_obj = torch.load("phase.pt", map_location="cpu")

    # dict保存の保険
    if isinstance(phase_obj, dict):
        if "phase" in phase_obj:
            phase_tensor = phase_obj["phase"]
        elif "phi" in phase_obj:
            phase_tensor = phase_obj["phi"]
        else:
            phase_tensor = next(v for v in phase_obj.values() if isinstance(v, torch.Tensor))
    else:
        phase_tensor = phase_obj

    # ここでptファイルの範囲を確認
    print("phase_tensor:", phase_tensor.shape,
          phase_tensor.min().item(), phase_tensor.max().item())

    # --- uint8 位相画像へ変換 ---
    phase_u8 = phase_tensor_to_u8(
        phase_tensor, W, H,
        phase_range="minus_pi_pi"  # 出力レンジに合わせて変更
    )

    # ==========================================================
    # ルートA: loadPhaseData(numpy) が存在するなら高速に表示
    # ==========================================================
    if hasattr(slm_phase, "loadPhaseData"):
        err, handle = slm_phase.loadPhaseData(phase_u8)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        err = handle.show()
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        print("phase.pt を loadPhaseData ルートで表示しました。")

    # ==========================================================
    # ルートB: 無ければ確実な SLMDataField → showPhaseData
    # ==========================================================
    # else:
    #     phaseData = HEDS.SLMDataField(W, H, HEDSDTFMT_INT_U8, HEDSSHF_PresentFitScreen)

    #     # 全ピクセル書き込み（確実だがやや遅い）
    #     for y in range(H):
    #         for x in range(W):
    #             err = phaseData.setPixel(x, y, int(phase_u8[y, x]))
    #             assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    #     err = slm_phase.showPhaseData(phaseData)
    #     assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    #     print("phase.pt を SLMDataField ルートで表示しました。")

    input("Enter を押すと SLM を閉じます...")
    err = slm_phase.window().close()
    if err != HEDSERR_NoError:
        print("SLM window close error:", HEDS.SDK.ErrorString(err))

    # --- ウィンドウを閉じるまで待機 ---
    
    HEDS.SDK.WaitAllClosed()
    print("SLMウィンドウ終了")


if __name__ == "__main__":
    main()
