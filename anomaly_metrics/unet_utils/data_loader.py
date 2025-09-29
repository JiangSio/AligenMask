import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
from torchvision import transforms
import random
from PIL import Image
from glob import glob
import json
class MVTecDRAEMTestDataset_partial(Dataset):

    def __init__(self, root_dir,obj_name, resize_shape=None):
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.images=[]
        obj_path = os.path.join(self.root_dir,obj_name,"test")
        anomalies = os.listdir(obj_path)
        anomalies.remove("good")
        for anomaly in anomalies:
            anomaly_path = os.path.join(obj_path,anomaly)
            anomaly_files = os.listdir(anomaly_path)
            anomaly_files.sort()
            anomaly_files = anomaly_files[4:]
            self.images += [os.path.join(anomaly_path,x) for x in anomaly_files]
        good_path = os.path.join(self.root_dir,obj_name,"test","good")
        good_files = os.listdir(good_path)
        self.images += [os.path.join(good_path,x) for x in good_files]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            if not os.path.exists(os.path.join(mask_path, mask_file_name)):
                mask_file_name = file_name.split(".")[0] + ".png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample

class MVTec_Anomaly_Detection(Dataset):
    def __init__(self, args,sample_name,suffix):
        self.gooddata=[]
        self.anomalydata=[]
        self.good_path='%s/%s/train/good'%(args.mvtec_path,sample_name)
        self.good_files = [os.path.join(self.good_path,i) for i in os.listdir(self.good_path)]

        self.root_dir = '%s/%s'%(args.generated_data_path,sample_name)
        self.anomaly_names=os.listdir(self.root_dir)
        
        self.img_paths=glob(os.path.join(self.root_dir,f'*/image/*.{suffix}'))

        for impath in self.img_paths:
            anomaly = os.path.split(impath.replace(f'/image/{os.path.basename(impath)}',''))[1]
            self.anomalydata.append({'img_path':impath, 'mask_paths': impath.replace('image','mask'), 'anomaly_id':self.anomaly_names.index(anomaly)+1})
        for impath in self.good_files:
            self.gooddata.append({'img_path':impath, 'mask_paths': None, 'anomaly_id':-1})

        self.length=500
        random.shuffle(self.anomalydata)

        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256])
        ])
        print(f'Training {sample_name} with {len(self.anomaly_names)} anomaly types and {len(self.img_paths)} samples')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if random.random()>0.5:
            idx = idx % len(self.anomalydata)
            img_path=self.anomalydata[idx]['img_path']
            image = self.loader(Image.open(img_path).convert('RGB'))
            mask_path = self.anomalydata[idx]['mask_paths']
            anomaly_id= self.anomalydata[idx]['anomaly_id']
        else:
            idx = idx % len(self.gooddata)
            img_path=self.gooddata[idx]['img_path']
            image = self.loader(Image.open(img_path).convert('RGB'))
            mask_path = self.gooddata[idx]['mask_paths']
            anomaly_id= self.gooddata[idx]['anomaly_id']
        if mask_path is not None:
            mask = self.loader(Image.open(mask_path).convert('L'))
        else:
            mask=torch.zeros((1,image.size(-2),image.size(-1)))
        
        mask=(mask>0.5).float()
        if mask.sum()==0:
            has_anomaly = np.array([0], dtype=np.float32)
            anomaly_id=-1
        else:
            has_anomaly = np.array([1], dtype=np.float32)
        sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'anomay_id': anomaly_id}
        return sample

class MVTec_classification_train(Dataset):
    def __init__(self, args,sample_name):
        self.root_dir = '%s/%s'%(args.generated_data_path,sample_name)
        self.root_dir = '%s/%s'%(args.generated_data_path,sample_name)
        self.anomaly_names=os.listdir(self.root_dir)
        self.img_paths=[]
        self.labels=[]
        for idx,anomaly in enumerate(self.anomaly_names):
            self.img_paths+=glob(os.path.join(self.root_dir,f'{anomaly}/image/*.png'))
            self.labels+=[idx]*len(glob(os.path.join(self.root_dir,f'{anomaly}/image/*.png')))
        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256])
        ])
        self.length=len(self.img_paths)
        # import pdb;pdb.set_trace()
    def __len__(self):
        return self.length*5
    def class_num(self):
        return len(self.anomaly_names)
    def return_anomaly_names(self):
        return self.anomaly_names
    def __getitem__(self, idx):
        image=self.loader(Image.open(self.img_paths[idx%len(self.img_paths)]).convert('RGB'))
        label=self.labels[idx%len(self.img_paths)]
        return image,label

class MVTec_classification_test(Dataset):
    def __init__(self, args,sample_name,anomaly_names,json_file):
        root_dir = args.mvtec_path
        self.img_paths=[]
        self.labels=[]
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for c in data:
            adtype = c['atype']
            impath = c['impath']
            mask = c['mask']
            if adtype.split('+')[0]==sample_name:
                self.img_paths.append(os.path.join(root_dir,impath))
                self.labels+=[anomaly_names.index(adtype.split('+')[1])]
        self.anomaly_names=anomaly_names

        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256])
        ])
        self.length=len(self.img_paths)
        # import pdb;pdb.set_trace()
    def __len__(self):
        return self.length
    def class_num(self):
        return len(self.anomaly_names)
    def __getitem__(self, idx):
        image=self.loader(Image.open(self.img_paths[idx%len(self.img_paths)]).convert('RGB'))
        label=self.labels[idx%len(self.img_paths)]
        return image,label
