# amp.ptとphase.ptを同時に映す

import math
import torch
import torch.nn.functional as F
import numpy as np

import HEDS
from hedslib.heds_types import *


# ====== SLM に合わせて解像度を手動設定 ======
# AMP_W, AMP_H = 1920, 1080       # 振幅 SLM（index 0 など）
# PHASE_W, PHASE_H = 1920, 1080   # 位相 SLM（index 1 など）


# ====== 振幅テンソル → uint8 ======
def amp_tensor_to_u8(t, W, H):
    t = t.detach().to("cpu").float()

    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(0)
    elif t.dim() != 4:
        raise ValueError("amp tensor shape error")

    t = t[0, 0]  # (H,W)
    t = torch.clamp(t, 0, 1)

    t = torch.nn.functional.interpolate(
        t.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="nearest"
    ).squeeze()

    t = (t * 255).round().clamp(0, 255)
    return t.to(torch.uint8).numpy()


# ====== 位相テンソル → uint8 ======
def phase_tensor_to_u8(t, W, H, mode="minus_pi_pi"):
    t = t.detach().to("cpu").float()

    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(0)
    elif t.dim() != 4:
        raise ValueError("phase tensor shape error")

    t = t[0, 0]  # (H,W)

    # ----- 位相レンジ処理 -----
    two_pi = 2 * math.pi
    if mode == "minus_pi_pi":
        t = torch.remainder(t + two_pi, two_pi)
    elif mode == "zero_two_pi":
        t = torch.remainder(t, two_pi)
    elif mode == "zero_one":
        t = torch.clamp(t, 0, 1)
    else:
        raise ValueError("unknown phase range")

    if mode in ["minus_pi_pi", "zero_two_pi"]:
        t = t / two_pi  # [0, 2pi] -> [0, 1]

    # ----- resize -----
    t = torch.nn.functional.interpolate(
        t.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="nearest"
    ).squeeze()

    # ----- 0..1 → 0..255 -----
    t = (t * 255).round().clamp(0, 255)
    return t.to(torch.uint8).numpy()


def main():

    # ====== SDK 初期化 ======
    err = HEDS.SDK.Init(4, 1)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    # ====== SLM 初期化（2台）=====
    slm_amp   = HEDS.SLM.Init("index:0", True, 0.0)
    slm_phase = HEDS.SLM.Init("index:1", True, 0.0)

    assert slm_amp.errorCode()   == HEDSERR_NoError
    assert slm_phase.errorCode() == HEDSERR_NoError

    SLM_AMP_W = int(slm_amp.width_px())
    SLM_AMP_H = int(slm_amp.height_px())
    SLM_PHS_W = int(slm_amp.width_px())
    SLM_PHS_H = int(slm_amp.height_px())

    # ====== amp.pt 読み込み ======
    amp_obj = torch.load("amp.pt", map_location="cpu")
    if isinstance(amp_obj, dict):
        amp_tensor = next(v for v in amp_obj.values() if isinstance(v, torch.Tensor))
    else:
        amp_tensor = amp_obj

    print("[amp.pt]", amp_tensor.shape, amp_tensor.min().item(), amp_tensor.max().item())

    # ====== phase.pt 読み込み ======
    phase_obj = torch.load("phase.pt", map_location="cpu")
    if isinstance(phase_obj, dict):
        phase_tensor = next(v for v in phase_obj.values() if isinstance(v, torch.Tensor))
    else:
        phase_tensor = phase_obj

    print("[phase.pt]", phase_tensor.shape, phase_tensor.min().item(), phase_tensor.max().item())

    # ====== uint8化 ======
    amp_u8   = amp_tensor_to_u8(amp_tensor, SLM_AMP_W, SLM_AMP_H)
    phase_u8 = phase_tensor_to_u8(phase_tensor, SLM_PHS_W, SLM_PHS_H, mode="minus_pi_pi")

    # ====== SLM に転送（video memoryへ） ======
    err, h_amp   = slm_amp.loadImageData(amp_u8)
    assert err == HEDSERR_NoError

    err, h_phase = slm_phase.loadPhaseData(phase_u8)
    assert err == HEDSERR_NoError

    # ====== 2台同時表示（公式 recommended）=====
    err = HEDS.SDK.ShowDataHandles([h_amp, h_phase])
    assert err == HEDSERR_NoError

    print("2台のSLMへ amp.pt + phase.pt を同時表示しました！")
    input("Enter を押すと SLM を閉じます...")
    err = slm_amp.window().close()
    if err != HEDSERR_NoError:
        print("SLM window close error:", HEDS.SDK.ErrorString(err))
    err = slm_phase.window().close()
    if err != HEDSERR_NoError:
        print("SLM window close error:", HEDS.SDK.ErrorString(err))

    HEDS.SDK.WaitAllClosed()
    print("SLMウィンドウ終了")


if __name__ == "__main__":
    main()
