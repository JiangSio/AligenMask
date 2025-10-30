import numpy as np
import torch
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from torchvision import transforms
from PIL import Image
from scipy.linalg import sqrtm
import glob
import os
import random
import shutil
from cleanfid import fid
import pandas as pd

def calculate_fid(real_images, generate_images):
    tmp_real_dir = "tmp_fid_real"
    tmp_gen_dir = "tmp_fid_gen"
    if os.path.exists(tmp_real_dir):
        os.system("rm -rf " + tmp_real_dir)
    if os.path.exists(tmp_gen_dir):
        os.system("rm -rf " + tmp_gen_dir)
    os.mkdir(tmp_real_dir)
    os.mkdir(tmp_gen_dir)
    for i,real_image in enumerate(real_images):
        img = Image.open(real_image).convert("RGB").resize((256,256))
        # mask = Image.open(real_image.replace("test","ground_truth")).convert("L").resize((256,256))
        # black_bg = Image.new("RGB", img.size, (0, 0, 0))
        # img = Image.composite(img, black_bg, mask)
        img.convert("RGB").save(os.path.join(tmp_real_dir, f"{i}.png"))
    for i,generate_image in enumerate(generate_images):
        img = Image.open(generate_image).convert("RGB").resize((256,256))
        # mask = Image.open(generate_image.replace("image","mask")).convert("L").resize((256,256))
        # black_bg = Image.new("RGB", img.size, (0, 0, 0))
        # img = Image.composite(img, black_bg, mask)
        img.convert("RGB").save(os.path.join(tmp_gen_dir, f"{i}.png"))
    
    fid_score = fid.compute_fid(tmp_real_dir, tmp_gen_dir)
    print(f"FID Score: {fid_score:.2f}")
    return fid_score



# 示例用法
if __name__ == "__main__":

    os.makedirs("results",exist_ok=True)
    save_path = "results/fid.csv"
    
    data_root = {
        "anomalydiffusion": ["/data1/gpt/jtj/AligenMask/anomalydiffusion/generated_dataset","/data1/gpt/jtj/AligenMask/anomalydiffusion/generated_matched_dataset"],
        "anogen": ["/data1/gpt/jtj/AligenMask/anogen/DIFFUSION/output_unmatched2","/data1/gpt/jtj/AligenMask/anogen/DIFFUSION/output_matched"],
        "FAST": ["/data1/gpt/jtj/AligenMask/fast-foreground-aware-anomaly-synthesis/samples","/data1/gpt/jtj/AligenMask/fast-foreground-aware-anomaly-synthesis/samples-matching"]
    }

    colums = ["class"]
    for x in data_root.keys():
        colums.extend([x, x+"+aligen"])
    results = pd.DataFrame(columns=colums)

    origin_path = "/data1/gpt/jtj/asynthesis_data"
    clsses = os.listdir(origin_path)
    clsses = [clss for clss in clsses if os.path.isdir(os.path.join(origin_path, clss))]
    clsses.sort()

    for i,clss_name in enumerate(clsses):
        results.loc[i] = [clss_name] + [0] * (len(colums)-1)

    results.to_csv(save_path, index=False, encoding="utf-8")
    
    for clss_name in clsses:

        for method, paths in data_root.items():
            
            generate_path = paths[0]
            generate_matching_path = paths[1]
        
        
            list1 = []
            anomalies = os.listdir(os.path.join(origin_path, clss_name, "ground_truth"))
            for anomaly in anomalies:
                list1.extend(glob.glob(os.path.join(origin_path, clss_name, "test",anomaly,"*.png"))) # 真实图片
            
            list2 = glob.glob(os.path.join(generate_path, clss_name,"*","image", "*"))  # 生成图片
            # import pdb;pdb.set_trace()
            
            length = min(len(list1), len(list2))
            list1 = random.sample(list1, length)
            list2 = random.sample(list2, length)

            res = calculate_fid(list1, list2)
            results.loc[results["class"] == clss_name, method] = round(res,4)

            list2 = glob.glob(os.path.join(generate_matching_path, clss_name,"*","image", "*"))
            length = min(len(list1), len(list2))
            list1 = random.sample(list1, length)
            list2 = random.sample(list2, length)

            res = calculate_fid(list1, list2)
            results.loc[results["class"] == clss_name, method+"+aligen"] = round(res,4)

            results.to_csv(save_path, index=False, encoding="utf-8")

    

    
    
    
    