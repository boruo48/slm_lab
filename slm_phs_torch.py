# 位相パイトーチファイルphase.thをslmに表示

import math
import torch
import torch.nn.functional as F
import numpy as np

import HEDS
from hedslib.heds_types import *

def phase_tensor_to_slm_u8(
    phase_tensor: torch.Tensor,
    slm_h: int,
    slm_w: int,
    phase_range: str = "minus_pi_pi",
) -> np.ndarray:
    """
    PyTorch の位相 tensor を HOLOEYE SLM に渡せる uint8 画像に変換する。

    Args:
        phase_tensor: 位相 [rad] の tensor
            想定 shape:
                (H, W)
                (1, H, W)
                (1, 1, H, W)
        slm_h, slm_w: SLM の縦横ピクセル数
        phase_range:
            "minus_pi_pi"  : [-π, π] の位相を想定
            "zero_two_pi"  : [0, 2π] を想定
            "zero_one"     : [0,1] をそのまま灰度とみなす（例：既に正規化済）

    Returns:
        np.ndarray shape=(slm_h, slm_w), dtype=uint8, 値域 0..255
    """

    # ---- 1) CPU float にして余計な次元を整理 ----
    t = phase_tensor.detach().to("cpu").float()

    if t.dim() == 2:
        # (H, W) -> (1,1,H,W)
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        # (C,H,W) or (1,H,W) とみなして (1,1,H,W) に
        t = t.unsqueeze(0)  # -> (1,C,H,W)
    elif t.dim() == 4:
        # (N,C,H,W) 想定。とりあえず [0,0] だけ使う
        pass
    else:
        raise ValueError(f"Unsupported phase_tensor dim: {t.shape}")

    # 1枚目・1チャネル目だけ使う
    t = t[0:1, 0:1, :, :]  # shape: (1,1,H,W)

    # ---- 2) 位相レンジを [0, 2π) or [0,1] にする ----
    if phase_range in ["minus_pi_pi", "zero_two_pi"]:
        two_pi = 2.0 * math.pi

        if phase_range == "minus_pi_pi":
            # [-π, π] → [0, 2π) にラップ
            t = torch.remainder(t + two_pi, two_pi)
        else:  # "zero_two_pi"
            t = torch.remainder(t, two_pi)

        # [0,2π) → [0,1]
        t = t / two_pi

    elif phase_range == "zero_one":
        # すでに 0..1 の正規化位相（もしくはグレースケール）とみなす
        t = torch.clamp(t, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown phase_range: {phase_range}")

    # ---- 3) SLM 解像度にリサイズ ----
    t = F.interpolate(t, size=(slm_h, slm_w), mode="nearest")  # (1,1,H,W')

    # ---- 4) [0,1] → [0,255] uint8 へ ----
    t = torch.clamp(t, 0.0, 1.0)
    t = (t * 255.0).round()
    t = torch.clamp(t, 0.0, 255.0)

    phase_u8 = t.to(torch.uint8).squeeze().numpy()  # shape=(slm_h, slm_w), np.uint8

    return phase_u8


def init_slm():
    """HEDS SDK と SLM を初期化して slm オブジェクトを返す"""
    err = HEDS.SDK.Init(4, 1)
    if err != HEDSERR_NoError:
        raise RuntimeError(f"SDK Init failed: {HEDS.SDK.ErrorString(err)}")

    # 必要なら "name:LETO" や "index:0" など指定してもOK
    slm = HEDS.SLM.Init()
    if slm.errorCode() != HEDSERR_NoError:
        raise RuntimeError(f"SLM Init failed: {HEDS.SDK.ErrorString(slm.errorCode())}")

    print("SLM initialized.")
    print("Resolution:", slm.width(), "x", slm.height())
    return slm


def show_phase_tensor_on_slm(
    phase_tensor: torch.Tensor,
    slm,
    phase_range: str = "minus_pi_pi",
):
    """PyTorch の位相 tensor を SLM に表示するラッパー"""

    slm_w = slm.width()
    slm_h = slm.height()
    print("slm_w, slm_h",slm_w, slm_h)

    phase_u8 = phase_tensor_to_slm_u8(
        phase_tensor,
        slm_h=slm_h,
        slm_w=slm_w,
        phase_range=phase_range,
    )

    # ★ここが SDK の example によって微妙に違う部分★
    # あなたの環境の「データ配列表示 example」と同じ関数名に合わせてね。
    # 例1: displayImage
    HEDS.SLM.displayImage(slm, phase_u8)

    # 例2: displayDataArray (もしこっちだったら↑をコメントアウトしてこっちを使う)
    # HEDS.SLM.displayDataArray(slm, phase_u8)

    print("Phase pattern sent to SLM.")


# --- モデル読込例（ここはあなたの環境に合わせて） ---
# model = YourModel(...)
# model.load_state_dict(torch.load("weights.pth", map_location="cpu"))
# model.eval()

# テスト入力（例）
# input_img = ...  # shape: (1, 1, H, W) など

def main():
    slm = init_slm()

    with torch.no_grad():
        # phase_pred: 例として [-π, π] の位相を出すモデルを仮定
        # phase_pred shape: (1, 1, H, W)
        # phase_pred = model(input_img)
        phase_tensor = torch.load("phase.pt")

        # ここではダミーで適当なサイン波位相を作る例：
        H, W = 512, 512
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij",
        )
        phase_tensor = torch.atan2(yy, xx)  # [-π, π] の位相
        phase_tensor = phase_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # SLM に表示
        show_phase_tensor_on_slm(
            phase_tensor,
            slm,
            phase_range="minus_pi_pi",  # モデルの出力レンジに合わせる
        )

    input("Enter を押すと SLM を閉じます...")

    err = slm.window().close()
    if err != HEDSERR_NoError:
        print("SLM window close error:", HEDS.SDK.ErrorString(err))


if __name__ == "__main__":
    main()
