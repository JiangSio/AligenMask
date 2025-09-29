import random

import torchmetrics

from share import *

import pytorch_lightning as pl
import torch
import os
import argparse
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from mvtecad_dataloader import mvtecGenerateFromMaskGenerator
from sgn.model import create_model, load_state_dict
from utils.eval_helper import dump, log_metrics, merge_together, performances
from torch.nn import functional as F
import logging
import timm
import glob
from scipy.ndimage import gaussian_filter
import cv2
from utils.util import cal_anomaly_map, log_local, create_logger, setup_seed
from visa_dataloader import VisaDataset

parser = argparse.ArgumentParser(description="DiAD")
parser.add_argument("--resume_path", default='val_ckpt/epoch=224-step=16874.ckpt')


args = parser.parse_args()

# Configs
resume_path = args.resume_path

batch_size = 1
learning_rate = 1e-5
only_mid_control = True
output_dir = "output1000_random"

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('models/diad.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.only_mid_control = only_mid_control

# Misc
dataset = mvtecGenerateFromMaskGenerator("../DualAnoDiff/dual-interrelated_diff/generate_data")

dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=False)

def gen_local(images, obj,an,name):
    pixel_mean = [0.485, 0.456, 0.406]
    pixel_std = [0.229, 0.224, 0.225]
    pixel_mean = torch.tensor(pixel_mean).cuda().unsqueeze(1).unsqueeze(1)  # 3 x 1 x 1
    pixel_std = torch.tensor(pixel_std).cuda().unsqueeze(1).unsqueeze(1)
    root = os.path.join(output_dir)
    for k in images:
        if k == "mask":
            mask = (images[k].squeeze() * 255).to('cpu').numpy()
            filename = "{}-{}.jpg".format(name, k)
            path = os.path.join(root, obj,an,filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            cv2.imwrite(path, mask)
            continue
        image = images[k].squeeze() * pixel_std + pixel_mean
        image = image * 255
        image = image.permute(1, 2, 0).to('cpu').numpy()
        filename = "{}-{}.jpg".format(name, k)
        path = os.path.join(root, obj,an,filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)

model.eval()
os.makedirs(output_dir, exist_ok=True)
with torch.no_grad():
    for i,input in enumerate(dataloader):
        input_img = input['jpg']
        input_mask = input["mask"]
        model = model.cuda()
        output= model.log_images_test(input)
        images = output
        images["mask"] = input_mask
        obj,an = input["type"][0].split("+")
        gen_local(images,obj,an,i)