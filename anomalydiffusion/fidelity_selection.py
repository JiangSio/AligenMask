import argparse
import os
import lpips
from PIL import Image
import torch
import numpy as np
import shutil

def cal_lpips(dir0,dir1):

    result={}
    ## Initializing the model
    lpips_model = lpips.LPIPS(net='alex', version='0.1')
    
    # the total list of images
    files = os.listdir(dir0)
    for file in files:
    
        image1 = Image.open(os.path.join(dir0,file))
        image2 = Image.open(os.path.join(dir1,file))
        mask = Image.open(os.path.join(dir0,file).replace("image","mask"))
        
        mask = mask.convert("L")
        # 将图像转换为PyTorch的Tensor格式
        image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image2_tensor = torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # 使用LPIPS模型计算距离
        distance = lpips_model(image1_tensor, image2_tensor)
        mask = np.array(mask)
        
        h,w = mask.shape
        
        mask_weight = mask.sum() / 255 / h / w + 0.2
        

        current_lpips_distance = distance.item() / mask_weight
        #print('%s: %.3f'%(file, current_lpips_distance))
        result[file] = current_lpips_distance
    
        
    
    sorted_items_desc = sorted(result.items(), key=lambda item: item[1], reverse=True)
    return sorted_items_desc

data_path = "/data4/jiangtianjia/Project/anomalydiffusion/generated_matched_random_dataset"
save_path = "/data4/jiangtianjia/Project/anomalydiffusion/generated_matched_choosedrandom_dataset"

for clss in os.listdir(data_path):
    if clss not in ["screw","capsule"]:
        continue
    clss_path = os.path.join(data_path,clss)
    for anomaly in os.listdir(clss_path):
        anomaly_path = os.path.join(clss_path,anomaly)
        image_path = os.path.join(anomaly_path,"image")
        ori_path = os.path.join(anomaly_path,"ori")
        result = cal_lpips(image_path,ori_path)
        # import pdb;pdb.set_trace()
        for image_sample in result[:500]:
            image = os.path.join(image_path,image_sample[0])
            mask = image.replace("image","mask")
            image_mask = image.replace("image","image-mask")
            ori = image.replace("image","ori")
            recon = image.replace("image","recon")
            os.makedirs(os.path.join(save_path,clss,anomaly,"image"),exist_ok=True)
            os.makedirs(os.path.join(save_path,clss,anomaly,"mask"),exist_ok=True)
            os.makedirs(os.path.join(save_path,clss,anomaly,"image-mask"),exist_ok=True)
            os.makedirs(os.path.join(save_path,clss,anomaly,"ori"),exist_ok=True)
            os.makedirs(os.path.join(save_path,clss,anomaly,"recon"),exist_ok=True)

            identifier = len(os.listdir(os.path.join(save_path,clss,anomaly,"image")))
            shutil.copy(image,image.replace(data_path,save_path).replace(image_sample[0],f"{identifier}.jpg"))
            shutil.copy(mask,mask.replace(data_path,save_path).replace(image_sample[0],f"{identifier}.jpg"))
            shutil.copy(ori,ori.replace(data_path,save_path).replace(image_sample[0],f"{identifier}.jpg"))
            shutil.copy(image_mask,image_mask.replace(data_path,save_path).replace(image_sample[0],f"{identifier}.jpg"))
            shutil.copy(recon,recon.replace(data_path,save_path).replace(image_sample[0],f"{identifier}.jpg"))

            identifier = len(os.listdir(os.path.join(save_path,clss,anomaly,"image")))
            shutil.copy(image,image.replace(data_path,save_path).replace(image_sample[0],f"{identifier}.jpg"))
            shutil.copy(mask,mask.replace(data_path,save_path).replace(image_sample[0],f"{identifier}.jpg"))
            shutil.copy(ori,ori.replace(data_path,save_path).replace(image_sample[0],f"{identifier}.jpg"))
            shutil.copy(image_mask,image_mask.replace(data_path,save_path).replace(image_sample[0],f"{identifier}.jpg"))
            shutil.copy(recon,recon.replace(data_path,save_path).replace(image_sample[0],f"{identifier}.jpg"))
        
# cal_lpips("/data4/jiangtianjia/Project/anomalydiffusion/generated_matched_dataset/bottle/broken_large/image","/data4/jiangtianjia/Project/anomalydiffusion/generated_matched_dataset/bottle/broken_large/ori")