# amp.phとphase.phを同時に映す
import math
import torch
import torch.nn.functional as F
import numpy as np

import HEDS
from hedslib.heds_types import *


# =============== 共通ヘルパ：テンソル形状を (1,1,H,W) にする ===============
def _to_1x1hw(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().to("cpu").float()
    if t.dim() == 2:            # (H, W)
        t = t.unsqueeze(0).unsqueeze(0)      # (1,1,H,W)
    elif t.dim() == 3:          # (C,H,W) or (1,H,W)
        t = t.unsqueeze(0)                  # (1,C,H,W)
    elif t.dim() == 4:          # (N,C,H,W)
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {t.shape}")
    t = t[0:1, 0:1, :, :]       # (1,1,H,W) を1枚抽出
    return t


# =============== 振幅テンソル(0〜1) → SLM用 uint8 (0〜255) ===============
def amp_tensor_to_slm_u8(amp_tensor: torch.Tensor, slm_h: int, slm_w: int) -> np.ndarray:
    """
    amp_tensor: 0〜1 の振幅 (torch.Tensor), shape は 2D〜4D どれでもOK
    """
    t = _to_1x1hw(amp_tensor)          # (1,1,H,W)
    t = torch.clamp(t, 0.0, 1.0)       # 0〜1 に制限

    # SLM 解像度にリサイズ
    t = F.interpolate(t, size=(slm_h, slm_w), mode="nearest")

    # 0〜1 -> 0〜255
    t = (t * 255.0).round()
    t = torch.clamp(t, 0.0, 255.0)

    amp_u8 = t.to(torch.uint8).squeeze().numpy()   # (slm_h, slm_w)
    return amp_u8


# =============== 位相テンソル(rad) → SLM用 uint8 (0〜255) ===============
def phase_tensor_to_slm_u8(
    phase_tensor: torch.Tensor,
    slm_h: int,
    slm_w: int,
    phase_range: str = "minus_pi_pi",
) -> np.ndarray:
    """
    phase_tensor: 位相 [rad] (torch.Tensor)
    phase_range:
        "minus_pi_pi" : [-π, π] を想定
        "zero_two_pi" : [0, 2π] を想定
        "zero_one"    : [0,1] 正規化済み位相
    """
    t = _to_1x1hw(phase_tensor)    # (1,1,H,W)

    if phase_range in ["minus_pi_pi", "zero_two_pi"]:
        two_pi = 2.0 * math.pi
        if phase_range == "minus_pi_pi":
            # [-π, π] -> [0, 2π)
            t = torch.remainder(t + two_pi, two_pi)
        else:  # "zero_two_pi"
            t = torch.remainder(t, two_pi)
        # [0, 2π) -> [0,1]
        t = t / two_pi
    elif phase_range == "zero_one":
        t = torch.clamp(t, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown phase_range: {phase_range}")

    # SLM 解像度にリサイズ
    t = F.interpolate(t, size=(slm_h, slm_w), mode="nearest")

    # 0〜1 -> 0〜255
    t = torch.clamp(t, 0.0, 1.0)
    t = (t * 255.0).round()
    t = torch.clamp(t, 0.0, 255.0)

    phase_u8 = t.to(torch.uint8).squeeze().numpy()   # (slm_h, slm_w)
    return phase_u8


# =============== 2枚のSLM初期化 ===============
def init_two_slms():
    # SDK 初期化
    err = HEDS.SDK.Init(4, 1)
    if err != HEDSERR_NoError:
        raise RuntimeError(f"SDK Init failed: {HEDS.SDK.ErrorString(err)}")

    # --- 振幅SLM: index:0 と仮定（必要なら name:◯◯ に変更） ---
    slm_amp = HEDS.SLM.Init("index:0")
    if slm_amp.errorCode() != HEDSERR_NoError:
        raise RuntimeError(f"Amp SLM Init failed: {HEDS.SDK.ErrorString(slm_amp.errorCode())}")

    print("[Amp SLM]")
    print("  Resolution:", slm_amp.width(), "x", slm_amp.height())

    # --- 位相SLM: index:1 と仮定 ---
    slm_phase = HEDS.SLM.Init("index:1")
    if slm_phase.errorCode() != HEDSERR_NoError:
        raise RuntimeError(f"Phase SLM Init failed: {HEDS.SDK.ErrorString(slm_phase.errorCode())}")

    print("[Phase SLM]")
    print("  Resolution:", slm_phase.width(), "x", slm_phase.height())

    return slm_amp, slm_phase


# =============== .pt を読みつつ dict/テンソルどちらでも対応 ===============
def load_tensor_from_pt(path: str, key_candidates=("amp", "phase", "tensor")) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for k in key_candidates:
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]
        # 見つからなければ最初のテンソルっぽいものを返す
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v
    raise ValueError(f"Could not find tensor in {path}, got type {type(obj)}")


# =============== メイン： amp.pt / phase.pt → 2枚のSLMへ表示 ===============
def main():
    # --- 2台のSLM初期化 ---
    slm_amp, slm_phase = init_two_slms()

    # --- amp.pt を読み込み（0〜1 振幅想定） ---
    amp_tensor = load_tensor_from_pt("amp.pt", key_candidates=("amp", "amplitude"))
    print("Loaded amp.pt:", amp_tensor.shape, amp_tensor.min().item(), amp_tensor.max().item())

    # --- phase.pt を読み込み（[-π, π] 位相想定） ---
    phase_tensor = load_tensor_from_pt("phase.pt", key_candidates=("phase", "phi"))
    print("Loaded phase.pt:", phase_tensor.shape, phase_tensor.min().item(), phase_tensor.max().item())

    # --- 各 SLM の解像度取得 ---
    amp_h, amp_w = slm_amp.height(), slm_amp.width()
    ph_h, ph_w = slm_phase.height(), slm_phase.width()

    # --- テンソル → uint8 画像に変換 ---
    amp_u8 = amp_tensor_to_slm_u8(amp_tensor, amp_h, amp_w)

    # phase_range は出力レンジに合わせて：
    #   ・[-π, π] → "minus_pi_pi"
    #   ・[0, 2π] → "zero_two_pi"
    #   ・[0, 1]  → "zero_one"
    phase_u8 = phase_tensor_to_slm_u8(
        phase_tensor,
        ph_h,
        ph_w,
        phase_range="minus_pi_pi",  # 必要ならここを変更
    )

    # --- 必要なら向きを合わせる（光学系を見ながら調整） ---
    # import numpy as _np
    # amp_u8   = _np.flipud(amp_u8)
    # amp_u8   = _np.fliplr(amp_u8)
    # phase_u8 = _np.flipud(phase_u8)
    # phase_u8 = _np.fliplr(phase_u8)

    # --- SLM に送信（SDK example の関数名に合わせて） ---
    HEDS.SLM.displayImage(slm_amp, amp_u8)      # 振幅SLM
    HEDS.SLM.displayImage(slm_phase, phase_u8)  # 位相SLM
    # もし displayDataArray の方なら：
    # HEDS.SLM.displayDataArray(slm_amp, amp_u8)
    # HEDS.SLM.displayDataArray(slm_phase, phase_u8)

    print("Amplitude & Phase patterns sent to their SLMs.")

    input("Enter を押すと両方のSLMウィンドウを閉じます...")

    err = slm_amp.window().close()
    if err != HEDSERR_NoError:
        print("Amp SLM window close error:", HEDS.SDK.ErrorString(err))

    err = slm_phase.window().close()
    if err != HEDSERR_NoError:
        print("Phase SLM window close error:", HEDS.SDK.ErrorString(err))


if __name__ == "__main__":
    main()
