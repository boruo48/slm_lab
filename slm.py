from holoeye import slmdisplaysdk

slm = slmdisplaysdk.SLMInstance()

# 初期化
result = slm.open()
print("Open result:", result)

# 画像を表示
import numpy as np
phase = np.linspace(0, 2*np.pi, slm.width*slm.height).reshape(slm.height, slm.width)
slm.showPhasevalues(phase)

input("Press Enter to close...")
slm.close()
