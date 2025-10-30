import argparse
import os
import sys
import torch

from PIL import Image, ImageFile
from tqdm import tqdm
from torchvision.transforms import transforms
import lpips
import pandas as pd
import numpy as np
import glob
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

def compute_clpips(real_images,generate_images,resolution = 256,):
    """
    This function computes the IC-LPIPS score for given generated images.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    with torch.no_grad():
        # import pdb;pdb.set_trace()
        loss_fn_alex = lpips.LPIPS(net='vgg', verbose = False).to(device) # best forward scores

        data_list = []
        for real_img in real_images:
            img = Image.open(real_img).convert("RGB").resize((256,256))
            img = transform(img).unsqueeze(0)
            data_list.append(img)

        # calculate the features of all the real images, which is used to calculate the genrated images.
        
        data_list = torch.cat(data_list, dim = 0)
        data_list = data_list.to(device)
        cluster = [[] for _ in range(data_list.shape[0])]

        # Calculate the LPIPS between 1000 anomaly images and all real images one by one,
        # and form the cluster using the realimages as the center.
        for file_path in generate_images:
            img = Image.open(file_path).convert("RGB").resize((256,256))
            mask = Image.open(file_path.replace("image", "mask")).convert("L").resize((256,256))
            mask_np = np.array(mask)
            mask_np = mask_np > 0
            if mask_np.sum() >= 256*256*0.025:
                continue
            img = transform(img).unsqueeze(0)
            score_list = loss_fn_alex(img.repeat(data_list.shape[0], 1, 1, 1).to(device), data_list)

            closest_index = score_list.argmin().item()
            if len(cluster[closest_index]) < 140:
                cluster[closest_index].append(img)
           
        print("done!") 
        cluster_lpips = []
        i= 0

        # Calculate the LPIPS between all the images in the same cluster,
        # and get the IC-LPIPS
        iterator = tqdm(cluster, desc = 'Computing clustered LPIPS')
        for c in iterator:
            print("Cluster {} contains {} images".format(i, len(c)))
            if len(c) <= 1:
                cluster_lpips.append(0.0)
                i+=1
                continue
            c_lpips = 0.0
            img = torch.cat(c, dim = 0).to(device)
            ref_img = img.clone()
            for _ in range(img.shape[0] - 1):
                img = torch.cat([img[1:], img[0:1]], dim = 0)
                c_lpips += loss_fn_alex(img, ref_img).sum().item()
            cluster_lpips.append(c_lpips / (img.shape[0] * (img.shape[0] - 1)))
            i+=1

    print(cluster_lpips)
    clpips = sum(cluster_lpips) / len(cluster_lpips)
    rz_sum = 0.0
    n = 0
    for score in cluster_lpips:
        if score != 0.0:
            rz_sum += score
            n += 1
    clpips_rz = rz_sum / n
    return clpips, clpips_rz

def compute_clpips_from_list(real_images, generate_images, resolution = 256):
    """
    This function computes the IC-LPIPS score for given generated images.
    """
    
    clpips, clpips_rz = compute_clpips(real_images, generate_images)
    return clpips_rz

if __name__ == "__main__":
    os.makedirs("results",exist_ok=True)
    save_path = "results/ic-lpips.csv"
    
    data_root = {
        "anomalydiffusion": ["/data1/gpt/jtj/AligenMask/anomalydiffusion/generated_dataset","/data1/gpt/jtj/AligenMask/anomalydiffusion/generated_matched_dataset_canny"],
        "anogen": ["/data1/gpt/jtj/AligenMask/anogen/DIFFUSION/output_unmatched2","/data1/gpt/jtj/AligenMask/anogen/DIFFUSION/output_matched_canny"],
        "FAST": ["/data1/gpt/jtj/AligenMask/fast-foreground-aware-anomaly-synthesis/samples-strict","/data1/gpt/jtj/AligenMask/fast-foreground-aware-anomaly-synthesis/samples-matching0"]
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
                real_images = glob.glob(os.path.join(origin_path, clss_name, "test",anomaly,"*.png"))
                real_images = real_images[:len(real_images)//3]
                list1.extend(real_images) # 真实图片
                # print(len(list1))
            
            list2 = glob.glob(os.path.join(generate_path, clss_name,"*","image", "*"))  # 生成图片
            list2 = random.sample(list2, min(1000,len(list2)))
            # import pdb;pdb.set_trace()
            print(f"start compute ic-lpips of {generate_path}")
            res = compute_clpips_from_list(list1, list2)
            results.loc[results["class"] == clss_name, method] = round(res,4)

            list2 = glob.glob(os.path.join(generate_matching_path, clss_name,"*","image", "*"))  # 生成图片
            list2 = random.sample(list2, min(1000,len(list2)))

            print(f"start compute ic-lpips of {generate_matching_path}")
            res = compute_clpips_from_list(list1, list2)
            results.loc[results["class"] == clss_name, method+"+aligen"] = round(res,4)

            results.to_csv(save_path, index=False, encoding="utf-8")
    





    