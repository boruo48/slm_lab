# show_on_slm_pyqt.py
import sys, numpy as np
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

def to_qimage_r(img_u8):
    H, W = img_u8.shape
    # R=img, G=B=0 の 24bit RGB
    rgb = np.dstack([np.zeros_like(img_u8), np.zeros_like(img_u8), img_u8]).astype(np.uint8)
    return QImage(rgb.data, W, H, 3*W, QImage.Format.Format_RGB888)

app = QApplication(sys.argv)

# ここで SLM 側スクリーンを選択（例: screens()[1] が SLM）
screens = app.screens()
slm_screen = screens[min(1, len(screens)-1)]  # 適宜変更（モニタ順序に依存）

# テストパターン生成
H, W = 1080, 1920
yy, xx = np.ogrid[:H, :W]
phase_u8 = np.uint8((np.mod(np.arctan2(yy-H/2, xx-W/2) + np.pi, 2*np.pi) / (2*np.pi)) * 255)

label = QLabel()
label.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
label.setWindowState(Qt.WindowState.WindowFullScreen)

img = to_qimage_r(phase_u8)
label.setPixmap(QPixmap.fromImage(img))

# SLMスクリーンに移動して全画面
geo = slm_screen.geometry()
label.move(geo.x(), geo.y())
label.resize(geo.width(), geo.height())
label.showFullScreen()

sys.exit(app.exec())
