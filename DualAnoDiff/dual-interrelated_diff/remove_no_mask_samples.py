import glob
import os
from PIL import Image
import numpy as np
import shutil

source_dir = "generate_data-Edge"
save_dir = "generate_data"

files = glob.glob(os.path.join(source_dir,"*","*","fg","*.png"))
files.sort()

for file in files:
    mask = Image.open(file).convert("L")
    h, w = mask.size
    mask_np = np.array(mask)
    mask_np = mask_np > 0
    if np.sum(mask_np) < h * w * 0.0001:
        continue
    else:
        mask_file = file
        save_path = mask_file.replace(source_dir, save_dir)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shutil.copy(mask_file, save_path)
        img_file = file.replace("fg", "image")
        save_path = img_file.replace(source_dir, save_dir)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shutil.copy(img_file, save_path)
        ori_file = file.replace("fg", "origin")
        save_path = ori_file.replace(source_dir, save_dir)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shutil.copy(ori_file, save_path)



