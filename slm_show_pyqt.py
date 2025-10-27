#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import cv2

from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtCore import QEvent
from PyQt6.QtCore import QSize
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtCore import QUrl
from PyQt6.QtCore import QRect
from PyQt6.QtCore import QPoint
from PyQt6.QtCore import QLibraryInfo
from PyQt6.QtCore import QLocale
from PyQt6.QtCore import QTranslator
from PyQt6.QtCore import QLibraryInfo
from PyQt6.QtCore import Qt as QtCoreQt

# --------- ユーティリティ ---------
def pick_slm_screen(app, prefer_name="HDMI-0"):
    """
    PyQtのスクリーン一覧から、名前が prefer_name のスクリーンを優先して返す。
    見つからなければ、x座標が最大=一番右のスクリーンを返す。
    """
    screens = app.screens()
    # まず名前で探す（X11なら "HDMI-0" 等の名前が得られることが多い）
    for s in screens:
        try:
            if s.name() == prefer_name:
                return s
            # 一応 "HDMI" を含む名前も候補に
            if ("HDMI" in s.name()) and (prefer_name.startswith("HDMI")):
                return s
        except Exception:
            pass
    # 見つからなければ一番右側（x最大）を選ぶ
    return max(screens, key=lambda ss: ss.geometry().x())

def to_qimage_from_r8(img_u8):
    """
    1ch (H,W, uint8) を R チャネルに入れ、G/B=0 の RGB888 QImage へ。
    QImage は元配列に参照を持つため、返却後も numpy 配列を保持しておくこと。
    """
    if img_u8.ndim != 2 or img_u8.dtype != np.uint8:
        raise ValueError("img_u8 must be 2D uint8")
    H, W = img_u8.shape
    # BGR順で並べ替え（B=0, G=0, R=img）
    rgb = np.dstack([np.zeros_like(img_u8), np.zeros_like(img_u8), img_u8])
    # QImage に共有メモリとして渡す（ライフタイムに注意）
    qimg = QImage(rgb.data, W, H, 3*W, QImage.Format.Format_RGB888)
    # 念のためコピーして独立させる（numpy参照切り離し）
    return qimg.copy()

def make_test_pattern(h, w, kind="angular"):
    """
    簡易テストパターン生成 (uint8, 0..255)；Rチャネルに載せる。
    kind:
      - "radial": 中心から外側へ 0..255
      - "angular": 角度で 0..255
      - "checker": チェッカー
    """
    yy, xx = np.ogrid[:h, :w]
    if kind == "radial":
        r = np.sqrt((xx - w/2.0)**2 + (yy - h/2.0)**2)
        out = np.uint8(np.clip((r / r.max()) * 255.0, 0, 255))
    elif kind == "angular":
        ang = (np.arctan2(yy - h/2.0, xx - w/2.0) + np.pi) / (2*np.pi)  # 0..1
        out = np.uint8(np.clip(ang * 255.0, 0, 255))
    else:  # checker
        tile = 32
        out = (((yy // tile) + (xx // tile)) % 2) * 255
        out = out.astype(np.uint8)
    return out

# --------- メインウィンドウっぽい最低限のラベル ---------
class SLMWindow(QLabel):
    def __init__(self, target_screen, img_u8, keep_rgb=False, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setStyleSheet("background: black;")

        # 画像作成（Rに載せる）
        self._img_u8 = img_u8  # ライフタイム保持
        self._qimg = to_qimage_from_r8(self._img_u8)
        self.setPixmap(QPixmap.fromImage(self._qimg))

        # ターゲットスクリーンへ移動
        geo = target_screen.geometry()
        self.move(geo.x(), geo.y())
        self.resize(geo.width(), geo.height())
        self.showFullScreen()

    def keyPressEvent(self, e):
        # Esc/Q で終了、S でスクリーンショット保存、スペースでパターン切替
        if e.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            QCoreApplication.quit()
        elif e.key() == Qt.Key.Key_S:
            cv2.imwrite("slm_capture_r.png", self._img_u8)
            print("Saved: slm_capture_r.png (Rチャネル画像そのまま)")
        elif e.key() == Qt.Key.Key_Space:
            # 別パターンに切替（angular <-> radial）
            h, w = self._img_u8.shape
            new_kind = "radial" if np.mean(self._img_u8) < 127 else "angular"
            self._img_u8 = make_test_pattern(h, w, new_kind)
            self._qimg = to_qimage_from_r8(self._img_u8)
            self.setPixmap(QPixmap.fromImage(self._qimg))

def main():
    parser = argparse.ArgumentParser(description="Send R-channel 8-bit image to SLM (HDMI-0) in fullscreen.")
    parser.add_argument("--target", default="HDMI-0", help="preferred screen name (default: HDMI-0)")
    parser.add_argument("--image", default="", help="grayscale image path (optional). If omitted, a test pattern is used.")
    parser.add_argument("--pattern", default="angular", choices=["angular", "radial", "checker"], help="test pattern type")
    parser.add_argument("--fit", action="store_true", help="resize input image to target screen size (INTER_NEAREST)")
    args = parser.parse_args()

    # DPIスケーリング無効化（値が誤って変換されるのを防ぐ）
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_DisableHighDpiScaling, True)
    # 新:
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass  # Qtの古いバージョンでは無視
    
    app = QApplication(sys.argv)
    slm_screen = pick_slm_screen(app, args.target)

    geo = slm_screen.geometry()
    H, W = geo.height(), geo.width()

    if args.image:
        src = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if src is None:
            print(f"Failed to read image: {args.image}")
            sys.exit(1)
        if args.fit or (src.shape[0] != H or src.shape[1] != W):
            # 画素値を保つため INTER_NEAREST が無難
            src = cv2.resize(src, (W, H), interpolation=cv2.INTER_NEAREST)
        img_u8 = src.astype(np.uint8)
    else:
        img_u8 = make_test_pattern(H, W, args.pattern)

    win = SLMWindow(slm_screen, img_u8)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
