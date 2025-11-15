import HEDS
from hedslib.heds_types import *


# --- SDK を初期化 ---
err = HEDS.SDK.Init(4, 1)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# --- SLM を初期化 ---
slm = HEDS.SLM.Init()
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# --- テストパターンを表示（白黒の縦グレーティング） ---
HEDS.SLM.showGratingBinaryVertical(slm, 20, 20, 0, 255)

input("Press Enter to close SLM window...")

# --- SLM ウィンドウを閉じる ---
err = slm.window().close()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
