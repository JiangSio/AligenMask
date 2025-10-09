import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,StableDiffusionPipeline_bg,DPMSolverMultistepScheduler
# import ipdb
import sys
import os
from PIL import Image
import torch

mvtec_path = '/data/gpt/real_aligen/AliGen-main/datasets/real_anomaly_set'


sys.path.append('.')
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", safety_checker=None
).to("cuda")
# #############
args = sys.argv
mvtec_name = args[1]
mvtec_aomaly_name = args[2]
# ##############
pipe.load_lora_weights('./generate_data/'+mvtec_name+'/'+mvtec_aomaly_name+'/checkpoint-2000')

target_path = './generate_data/'+mvtec_name+'/'+ mvtec_aomaly_name
if not os.path.exists(os.path.join(target_path,'image')):
    os.makedirs(os.path.join(target_path,'image'))
if not os.path.exists(os.path.join(target_path,'fg')):
    os.mkdir(os.path.join(target_path,'fg'))
cnt = len(os.listdir(os.path.join(target_path,'image')))

# for i in range(cnt,1000):
for i in range(cnt,500):
    
    outputs,origin_image = pipe(prompt_blend='a vfx with large red marker sks',num_inference_steps=100,guidance_scale=2.5,class_id = mvtec_name, data_dir = mvtec_path)
    outputs.images[0].save(os.path.join(target_path,'image',str(i)+".png"))
    
    os.makedirs(os.path.join(target_path,'origin'),exist_ok=True)
    origin_image = origin_image
    origin_image = origin_image.resize((512,512))
    origin_image.save(os.path.join(target_path,'origin',str(i)+".png"))

    def concat_images_horizontally(img1, img2):
        """
        水平拼接两张图片
        """
        # 获取两张图片的尺寸
        width1, height1 = img1.size
        width2, height2 = img2.size
        
        # 计算拼接后的新尺寸
        total_width = width1 + width2
        max_height = max(height1, height2)
        
        # 创建新图像
        new_img = Image.new('RGB', (total_width, max_height), color='white')
        
        # 粘贴图片
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (width1, 0))
        
        return new_img
    concat_image = concat_images_horizontally(outputs.images[0], origin_image)
    os.makedirs(os.path.join(target_path,'concat'),exist_ok=True)
    concat_image.save(os.path.join(target_path,'concat',str(i)+".png"))

    print(i)
 
