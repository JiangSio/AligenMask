from PIL import Image
from torchvision import transforms
from torchvision import utils
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion
import numpy as np
import torch
import os
import cv2
import sys

def identify_mask(img_path):
    img = Image.open(img_path)
    trans = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    img = trans(img)
    # import pdb;pdb.set_trace()
    m0 = img[0] > 0.5
    m1 = img[1] <-0.5
    m2 = img[2] <-0.5
    mask = m0 & m1 & m2
    mask = mask.float()

    mask_np = mask.numpy()

    # # 标记连通区域
    # labeled_arr, num_features = label(mask_np, connectivity=2, return_num=True)

    # # 获取每个连通区域的属性
    # regions = regionprops(labeled_arr)

    # # 初始化最大连通区域的大小和标签
    # max_area = 0
    # max_label = 0

    # # 遍历所有连通区域，找到最大的一个
    # for region in regions:
    #     if region.area > max_area:
    #         max_area = region.area
    #         max_label = region.label

    # # 创建一个与原始数组形状相同的数组，用于存储最大连通分量
    # max_connected_component = np.zeros_like(mask_np)

    # # 将最大连通区域填充为1
    # max_connected_component[labeled_arr == max_label] = 1

    # # 将 NumPy 数组转换回 PyTorch 张量
    # mask = torch.from_numpy(max_connected_component)

    utils.save_image(mask, img_path.replace("image","fg"))

data_root = "generate_data"
args = sys.argv
mvtec_name = args[1]
mvtec_aomaly_name = args[2]
clss_path = os.path.join(data_root,mvtec_name)

anomaly_path = os.path.join(clss_path,mvtec_aomaly_name,"image")
files = os.listdir(anomaly_path)
for file in files:
    file_path = os.path.join(anomaly_path,file)
    identify_mask(file_path)