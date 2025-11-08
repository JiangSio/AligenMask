import argparse
import os
import time
from torch.utils.data import DataLoader

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from .pidinet import PiDiNet

def load_checkpoint(path):
    model_filename = path
        
    loadinfo = "=> loading checkpoint from '{}'".format(model_filename)
    print(loadinfo)

    state = None
    if os.path.exists(model_filename):
        state = torch.load(model_filename, map_location='cpu')
        loadinfo2 = "=> loaded checkpoint '{}' successfully".format(model_filename)
    else:
        loadinfo2 = "no checkpoint loaded"
    print(loadinfo2)

    return state

def convert_pdc(op, weight):
    if op == 'cv':
        return weight
    elif op == 'cd':
        shape = weight.shape
        weight_c = weight.sum(dim=[2, 3])
        weight = weight.view(shape[0], shape[1], -1)
        weight[:, :, 4] = weight[:, :, 4] - weight_c
        weight = weight.view(shape)
        return weight
    elif op == 'ad':
        shape = weight.shape
        weight = weight.view(shape[0], shape[1], -1)
        weight_conv = (weight - weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
        return weight_conv
    elif op == 'rd':
        shape = weight.shape
        buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device)
        weight = weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weight[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weight[:, :, 1:]
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return buffer
    raise ValueError("wrong op {}".format(str(op)))

def convert_pidinet(state_dict, pdcs):
    new_dict = {}
    for pname, p in state_dict.items():
        if 'init_block.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[0], p)
        elif 'block1_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[1], p)
        elif 'block1_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[2], p)
        elif 'block1_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[3], p)
        elif 'block2_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[4], p)
        elif 'block2_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[5], p)
        elif 'block2_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[6], p)
        elif 'block2_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[7], p)
        elif 'block3_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[8], p)
        elif 'block3_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[9], p)
        elif 'block3_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[10], p)
        elif 'block3_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[11], p)
        elif 'block4_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[12], p)
        elif 'block4_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[13], p)
        elif 'block4_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[14], p)
        elif 'block4_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[15], p)
        else:
            new_dict[pname] = p

    return new_dict


config = {'carv4': 
    {
        'layer0':  'cd',
        'layer1':  'ad',
        'layer2':  'rd',
        'layer3':  'cv',
        'layer4':  'cd',
        'layer5':  'ad',
        'layer6':  'rd',
        'layer7':  'cv',
        'layer8':  'cd',
        'layer9':  'ad',
        'layer10': 'rd',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'ad',
        'layer14': 'rd',
        'layer15': 'cv',
    }   
}
use_cuda = torch.cuda.is_available()
pdcs = []
for i in range(16):
    layer_name = 'layer%d' % i
    op = config['carv4'][layer_name]
    pdcs.append(op)
dil = 24

model = PiDiNet(60, pdcs, dil=dil, sa=True, convert=True)
checkpoint = load_checkpoint("pil2canny/table7_pidinet.pth")
if checkpoint is not None:
    state_dict = convert_pidinet(checkpoint['state_dict'], pdcs)
    new_state_dict = {}
    for key, value in state_dict.items():
        # 移除开头的 "module."
        if key.startswith('module.'):
            new_key = key[7:]  # 移除前7个字符（"module."）
        else:
            new_key = key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
else:
    raise ValueError('no checkpoint loaded')
if use_cuda:
    model = model.cuda()


transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

def pil_to_edge(img):
    img = img.convert("RGB")
    img = transform(img)
    with torch.no_grad():
        img = img.cuda() if use_cuda else img
        img = img.unsqueeze(0)
        results = model(img)
        result = torch.squeeze(results[-1]).cpu().numpy()
        result = Image.fromarray((result * 255).astype(np.uint8))
    return result

def pil_to_canny(pil_image, low_threshold=100, high_threshold=125):
    # 将 PIL 图像转换为 NumPy 数组（OpenCV 格式）
    numpy_image = np.array(pil_image)

    # 转换 RGB 为 BGR（OpenCV 默认格式）
    if numpy_image.ndim == 3:  # 彩色图像
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        # 转换为灰度图
        gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    else:  # 灰度图像
        gray_image = numpy_image

    # 应用 Canny 边缘检测
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)

    # 将边缘检测结果转换回 PIL 图像
    return Image.fromarray(edges)

if __name__ == '__main__':

    img = Image.open('../image/002.png')
    result = pil_to_edge(img)
    result.save('result.png')
