#!/usr/bin/env python3
import sys, argparse, numpy as np

# --- PyQt を最初に import（重要） ---
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

def pick_slm_screen(app, prefer_name="HDMI-0"):
    screens = app.screens()
    for s in screens:
        try:
            if s.name() == prefer_name:
                return s
            if ("HDMI" in s.name()) and prefer_name.startswith("HDMI"):
                return s
        except Exception:
            pass
    return max(screens, key=lambda ss: ss.geometry().x())

def to_qimage_from_r8(img_u8):
    H, W = img_u8.shape
    rgb = np.dstack([np.zeros_like(img_u8), np.zeros_like(img_u8), img_u8])
    qimg = QImage(rgb.data, W, H, 3*W, QImage.Format.Format_RGB888)
    return qimg.copy()

def make_test_pattern(h, w, kind="angular"):
    yy, xx = np.ogrid[:h, :w]
    if kind == "radial":
        r = np.sqrt((xx - w/2.0)**2 + (yy - h/2.0)**2)
        out = np.uint8(np.clip((r / r.max()) * 255.0, 0, 255))
    elif kind == "angular":
        ang = (np.arctan2(yy - h/2.0, xx - w/2.0) + np.pi) / (2*np.pi)
        out = np.uint8(np.clip(ang * 255.0, 0, 255))
    else:
        tile = 32
        out = ((((yy // tile) + (xx // tile)) % 2) * 255).astype(np.uint8)
    return out

class SLMWindow(QLabel):
    def __init__(self, target_screen, img_u8, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setStyleSheet("background: black;")

        self._img_u8 = img_u8
        self._qimg = to_qimage_from_r8(self._img_u8)
        self.setPixmap(QPixmap.fromImage(self._qimg))

        geo = target_screen.geometry()
        self.move(geo.x(), geo.y())
        self.resize(geo.width(), geo.height())
        self.showFullScreen()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            QApplication.quit()
        elif e.key() == Qt.Key.Key_S:
            # 保存時だけ遅延 import（OpenCV依存を最小化）
            try:
                import cv2
                cv2.imwrite("slm_capture_r.png", self._img_u8)
                print("Saved: slm_capture_r.png")
            except Exception as ex:
                print("Save failed:", ex)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="HDMI-0")
    parser.add_argument("--image", default="")
    parser.add_argument("--pattern", default="angular", choices=["angular","radial","checker"])
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args()

    # Qt6のDPI設定（旧AA_DisableHighDpiScalingは削除済）
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    app = QApplication(sys.argv)
    slm_screen = pick_slm_screen(app, args.target)
    geo = slm_screen.geometry()
    H, W = geo.height(), geo.width()

    if args.image:
        # OpenCVを使う場合はここで遅延 import
        import cv2
        src = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if src is None:
            print(f"Failed to read image: {args.image}")
            sys.exit(1)
        if args.fit or (src.shape[0] != H or src.shape[1] != W):
            src = cv2.resize(src, (W, H), interpolation=cv2.INTER_NEAREST)
        img_u8 = src.astype(np.uint8)
    else:
        img_u8 = make_test_pattern(H, W, args.pattern)

    win = SLMWindow(slm_screen, img_u8)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
