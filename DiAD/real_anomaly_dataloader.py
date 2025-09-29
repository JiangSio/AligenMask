import json
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import os
from glob import glob

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]
def data_transforms(size):
    datatrans =  transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.CenterCrop(size),
    #transforms.CenterCrop(args.input_size),
    transforms.Normalize(mean=mean_train,
                         std=std_train)])
    return datatrans
def gt_transforms(size):
    gttrans =  transforms.Compose([
    transforms.Resize((size, size)),
    transforms.CenterCrop(size),
    transforms.ToTensor()])
    return gttrans

class RealDataset(Dataset):
    def __init__(self,type, root):
        self.data = []
        self.label_to_idx = {'cable': '0', 'capsules': '1', 'macaroni2': '2', 
                             'metal_plate': '3', 'screw': '4', 'transistor': '5',
                             'tubes': '6'}
        
        if type == 'train':
            normal_images = glob(f"{root}/*/train/good/*")
            for image in normal_images:
                item = {}
                item['filename']=image
                item['label']=0
                item['label_name']='good'
                item['clsname']=image.split('/')[-4]
                self.data.append(item)
        else:
            for clss in self.label_to_idx:
                anomalies = os.listdir(f"{root}/{clss}/ground_truth/")
                for anomaly_name in anomalies:
                    anomaly_images = glob(f"{root}/{clss}/test/{anomaly_name}/*")
                    anomaly_masks = glob(f"{root}/{clss}/ground_truth/{anomaly_name}/*")
                    anomaly_images.sort()
                    anomaly_masks.sort()
                    for i in range(4):
                        item = {}
                        item['filename']=anomaly_images[i]
                        item['label']=1
                        item['label_name']='defective'
                        item['clsname']=clss
                        item['maskname']=anomaly_masks[i]
                        self.data.append(item)
        self.image_size = (256, 256)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['filename']
        target_filename = item['filename']
        label = item["label"]
        if item.get("maskname", None):
            mask = cv2.imread(item['maskname'], cv2.IMREAD_GRAYSCALE)
        else:
            if label == 0:  # good
                mask = np.zeros(self.image_size).astype(np.uint8)
            elif label == 1:  # defective
                mask = (np.ones(self.image_size)).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        prompt = ""
        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)
        source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        source = Image.fromarray(source, "RGB")
        target = Image.fromarray(target, "RGB")
        mask = Image.fromarray(mask, "L")
        # transform_fn = transforms.Resize(256, Image.BILINEAR)
        transform_fn = transforms.Resize(self.image_size)
        source = transform_fn(source)
        target = transform_fn(target)
        mask = transform_fn(mask)
        source = transforms.ToTensor()(source)
        target = transforms.ToTensor()(target)
        mask = transforms.ToTensor()(mask)
        normalize_fn = transforms.Normalize(mean=mean_train, std=std_train)
        source = normalize_fn(source)
        target = normalize_fn(target)
        clsname = item["clsname"]
        image_idx = self.label_to_idx[clsname]

        return dict(jpg=target, txt=prompt, hint=source, mask=mask, filename=source_filename, clsname=clsname, label=int(image_idx))

class mvtecGenerateDataset(Dataset):
    def __init__(self, jsonpath, mvtec_root,category,repeat):
        self.mvtec_root=mvtec_root
        self.image_size = (256, 256)
        
        with open(jsonpath, 'r',encoding='utf-8') as f:
            self.json = json.load(f) 
        
        self.categories = category
        self.repeat=repeat
        self.imgpaths=[]
        self.maskpaths=[]
        self.types=[]
        for c in self.json:
            adtype = c['atype']
            impath = os.path.join(mvtec_root,c['impath'])
            maskpath = os.path.join(mvtec_root,c['mask'])
            obj,an = adtype.split("+")
            if obj in category or category == "all":
                self.imgpaths.append(impath)
                self.maskpaths.append(maskpath)
                self.types.append(adtype)
        
        self.pretransform_img = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图像大小
            transforms.ToTensor(),           # 转换为 tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ])
        self.pretransform_mask = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图像大小
            transforms.ToTensor(),           # 转换为 tensor
        ])

    def __len__(self):
        return len(self.imgpaths)*self.repeat

    def __getitem__(self, idx):
        impath = self.imgpaths[idx%len(self.imgpaths)]
        maskpath = self.maskpaths[idx%len(self.imgpaths)]
        atype = self.types[idx%len(self.imgpaths)]
        img = Image.open(impath)
        img = img.convert("RGB")
        mask = Image.open(maskpath)
        mask = mask.convert("L")
        
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mask = np.array(mask)

        transformed_image, transformed_mask = random_transform(img, mask, atype.split("+")[0])

        transformed_image = Image.fromarray(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
        transformed_mask = Image.fromarray(transformed_mask)
        
        transformed_image = self.pretransform_img(transformed_image)
        transformed_mask = self.pretransform_mask(transformed_mask)
        hint = transformed_image
        # transformed_image.save(os.path.basename(impath))
        # transformed_mask.save(os.path.basename(maskpath))
        # hint.save(os.path.basename(impath)[:-4]+os.path.basename(maskpath)[:-4]+".jpg")
        return dict(jpg=transformed_image, txt="", hint=hint, mask=transformed_mask, filename=impath, type=atype)

class mvtecGenerateFromMaskGenerator(Dataset):
    def __init__(self, generate_root,repeat=1):
        self.generate_root=generate_root
        self.image_size = (256, 256)
        
        self.repeat=repeat
        self.imgpaths=[]
        self.maskpaths=[]
        self.imgpaths = glob.glob(f'{generate_root}/*/*/image/*.png')
        
        self.maskpaths = [x.replace("image","fg") for x in self.imgpaths]
        self.types = [x.split("/")[-4]+"+"+x.split("/")[-3] for x in self.imgpaths]
        
        self.pretransform_img = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图像大小
            transforms.ToTensor(),           # 转换为 tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ])
        self.pretransform_mask = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图像大小
            transforms.ToTensor(),           # 转换为 tensor
        ])

    def __len__(self):
        return len(self.imgpaths)*self.repeat

    def __getitem__(self, idx):
        impath = self.imgpaths[idx%len(self.imgpaths)]
        maskpath = self.maskpaths[idx%len(self.imgpaths)]
        atype = self.types[idx%len(self.imgpaths)]
        img = Image.open(impath)
        img = img.convert("RGB")
        mask = Image.open(maskpath)
        mask = mask.convert("L")
        
        transformed_image = self.pretransform_img(img)
        transformed_mask = self.pretransform_mask(mask)
        hint = transformed_image
        # transformed_image.save(os.path.basename(impath))
        # transformed_mask.save(os.path.basename(maskpath))
        # hint.save(os.path.basename(impath)[:-4]+os.path.basename(maskpath)[:-4]+".jpg")
        return dict(jpg=transformed_image, txt="", hint=hint, mask=transformed_mask, filename=impath, type=atype)
