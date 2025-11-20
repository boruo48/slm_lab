#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLMに“OpenCVだけ”で画像をピクセル等倍表示する最小スクリプト。

例:
  python show_on_slm_cv.py --img phase.png --monitor-index 1 --fullscreen --center
  python show_on_slm_cv.py --img phase_u16.tiff --monitor-index 2 --screen-w 1920 --screen-h 1080

特徴:
- OpenCV(cv2)のみ使用（GUIは imshow / namedWindow / moveWindow）
- 8/16bitグレースケール/カラー対応（自動で8bitグレーに変換）
- マルチモニタ: Windows は ctypes で解像度と原点座標を自動取得
  - 他OSは --screen-w/--screen-h で指定（座標は横並び想定の原点計算）
- フルスクリーン時も“ピクセル等倍”を維持するため、画面解像度に一致
  しない場合は黒キャンバスに貼り付け（拡大縮小しない）
- キー操作: Esc=終了 / F=フルスクリーン切替 / Space=黒トグル / +/-=ゲイン
            G=ガンマ2.2トグル / R=再読込

注意:
- OSの表示スケールは 100% 推奨（Windowsの拡大縮小が効くと1:1になりません）。
- SLM固有のCLUT/ガンマ補正は別途ドライバ/SDKにて調整してください。
"""

from __future__ import annotations
import argparse
import sys
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional

# ------------------ 画像IO ------------------
def imread_to_u8_gray(path: str | Path) -> np.ndarray:
    """ファイルを読み、8bitグレースケール(np.uint8, HxW)に変換。Unicodeパス対応。"""
    p = str(path)
    data = np.fromfile(p, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(p)
    if img.ndim == 3:  # BGR or BGRA -> Gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        # 16bit -> 8bit 等価縮約（65535/255=257）
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

# ------------------ モニタ情報 ------------------

def list_monitors_windows() -> Tuple[list[Tuple[int,int,int,int]], int]:
    """Windows: 各モニタの矩形(x0,y0,x1,y1)リストとプライマリのindexを返す。"""
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return [], 0

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()

    MONITORENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.POINTER(wintypes.RECT), ctypes.c_double)
    monitors = []

    def _callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
        r = lprcMonitor.contents
        monitors.append((r.left, r.top, r.right, r.bottom))
        return 1

    user32.EnumDisplayMonitors(0, 0, MONITORENUMPROC(_callback), 0)

    # プライマリの矩形（原点を含む）を探す
    primary_idx = 0
    for i, (x0,y0,x1,y1) in enumerate(monitors):
        if x0 <= 0 <= x1 and y0 <= 0 <= y1:
            primary_idx = i
            break
    return monitors, primary_idx

# ------------------ 表示ユーティリティ ------------------

def make_canvas_and_paste(img: np.ndarray, screen_w: int, screen_h: int, center: bool) -> np.ndarray:
    """imgを拡大縮小せず、screenサイズの黒キャンバスに貼り付ける"""
    H, W = img.shape
    canvas = np.zeros((screen_h, screen_w), np.uint8)
    if center:
        y0 = max(0, (screen_h - H)//2)
        x0 = max(0, (screen_w - W)//2)
    else:
        y0, x0 = 0, 0
    y1 = min(screen_h, y0 + H)
    x1 = min(screen_w, x0 + W)
    canvas[y0:y1, x0:x1] = img[0:(y1-y0), 0:(x1-x0)]
    return canvas

# ------------------ メイン ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', required=True, help='表示する画像パス (8/16bit, RGB可)')
    ap.add_argument('--monitor-index', type=int, default=0, help='表示先モニタの番号(0始まり)。Windowsは自動列挙。')
    ap.add_argument('--fullscreen', action='store_true', help='フルスクリーンで表示')
    ap.add_argument('--center', action='store_true', help='キャンバス中央に貼り付け（等倍）')
    ap.add_argument('--screen-w', type=int, default=None, help='表示先モニタの幅（非Windows用）')
    ap.add_argument('--screen-h', type=int, default=None, help='表示先モニタの高さ（非Windows用）')
    ap.add_argument('--gain', type=float, default=1.0, help='初期表示ゲイン')
    ap.add_argument('--gamma22', action='store_true', help='ガンマ2.2補正')
    args = ap.parse_args()

    img0 = imread_to_u8_gray(args.img)

    # --- モニタ情報 ---
    monitors, primary_idx = list_monitors_windows()
    if monitors:
        idx = max(0, min(args.monitor-index if hasattr(args, 'monitor-index') else args.monitor_index, len(monitors)-1))
        x0, y0, x1, y1 = monitors[idx]
        screen_w = x1 - x0
        screen_h = y1 - y0
    else:
        # 非Windows: 画面サイズは指定必須
        if args.screen_w is None or args.screen_h is None:
            print('[WARN] Windows以外では --screen-w と --screen-h を指定してください。例: --screen-w 1920 --screen-h 1080')
            screen_w = img0.shape[1]
            screen_h = img0.shape[0]
            x0, y0 = 0, 0
        else:
            screen_w, screen_h = int(args.screen_w), int(args.screen_h)
            # 横並び想定の座標
            x0 = screen_w * args.monitor_index
            y0 = 0

    # --- ウィンドウ作成 ---
    win_name = 'SLM'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win_name, x0, y0)
    cv2.resizeWindow(win_name, screen_w, screen_h)

    is_full = bool(args.fullscreen)

    def set_fullscreen(flag: bool):
        # フルスクリーン切り替え
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if flag else cv2.WINDOW_NORMAL)

    set_fullscreen(is_full)

    gain = float(args.gain)
    gamma22 = bool(args.gamma22)
    show_black = False

    def make_frame() -> np.ndarray:
        img = img0.astype(np.float32)
        if gamma22:
            img = (np.power(img/255.0, 1/2.2) * 255.0)
        if gain != 1.0:
            img = np.clip(img * gain, 0, 255)
        img = img.astype(np.uint8)
        if is_full:
            # フルスクリーン時はキャンバスに貼る（拡大縮小しない）
            frame = make_canvas_and_paste(img, screen_w, screen_h, center=args.center)
        else:
            frame = img  # 通常ウィンドウでは原寸表示（必要なら手動で拡大）
        if show_black:
            if is_full:
                return np.zeros((screen_h, screen_w), np.uint8)
            else:
                return np.zeros_like(frame)
        return frame

    frame = make_frame()
    while True:
        cv2.imshow(win_name, frame)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # Esc
            break
        elif key in (ord('f'), ord('F')):
            is_full = not is_full
            set_fullscreen(is_full)
            frame = make_frame()
        elif key == ord(' '):  # Space 黒トグル
            show_black = not show_black
            frame = make_frame()
        elif key in (ord('+'), ord('=')):
            gain = min(gain * 1.05, 10.0)
            frame = make_frame()
        elif key == ord('-'):
            gain = max(gain / 1.05, 0.01)
            frame = make_frame()
        elif key in (ord('g'), ord('G')):
            gamma22 = not gamma22
            frame = make_frame()
        elif key in (ord('r'), ord('R')):
            try:
                img0_new = imread_to_u8_gray(args.img)
                img0[:] = img0_new  # 同サイズ想定
                frame = make_frame()
            except Exception as e:
                print('[ERR] reload failed:', e)
        # 矢印キーで微調整（非必須）
        elif key == 81:  # ←
            cv2.moveWindow(win_name, x0-100, y0)
            x0 -= 100
        elif key == 83:  # →
            cv2.moveWindow(win_name, x0+100, y0)
            x0 += 100
        elif key == 82:  # ↑
            cv2.moveWindow(win_name, x0, y0-100)
            y0 -= 100
        elif key == 84:  # ↓
            cv2.moveWindow(win_name, x0, y0+100)
            y0 += 100

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
