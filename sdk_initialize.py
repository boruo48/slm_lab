import sys
import os

# ① HOLOEYE SDK の python API へのパスを通す
sys.path.append(r"C:/Program Files/HOLOEYE Photonics/SLM Display SDK (Python) v4.1.0/api/python")

# ② HEDS ではなく hedslib を import
import hedslib

print("hedslib loaded from:", hedslib.__file__)

# ③ DLL のパス（※ここは自分の環境に合わせて修正）
# SDK フォルダの中の bin や lib の下にある DLL を探す：
# 例）slmdisplaysdk.dll, heds_*.dll など
dll_path = r"C:/Program Files/HOLOEYE Photonics/SLM Display SDK (Python) v4.1.0/bin/slmdisplaysdk.dll"

# ④ ライブラリインスタンスを作成
sdk = hedslib.holoeye_slmdisplaysdk_library(dll_path)

# ⑤ C の DLL 関数を叩く（.lib 経由）
ret = sdk.lib.heds_init()
print("heds_init returned:", ret)

# ここから先は heds_～ 関数を sdk.lib.heds_xxx(...) で呼んでいく
# sdk.lib.heds_get_version(...)
# sdk.lib.heds_slm_show_datahandle(...)
# など
