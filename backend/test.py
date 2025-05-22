import os
import time


script_dir = os.path.dirname(os.path.abspath(__file__))
# Create a temporary file to save the uploaded audio in the tmp directory relative to the script
tmp_dir = os.path.join(script_dir, "tmp")
os.makedirs(tmp_dir, exist_ok=True)
tmp_filename = f"input_{int(time.time())}.wav"
tmp_input_audio_path = os.path.join(tmp_dir, tmp_filename)

try:
    with open(tmp_input_audio_path, "w") as f:
        f.write("")
except PermissionError:
    print("❌ 没有权限写入该路径！")
