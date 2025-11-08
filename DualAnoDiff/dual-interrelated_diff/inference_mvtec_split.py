import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,StableDiffusionPipeline_bg,DPMSolverMultistepScheduler
# import ipdb
import sys
import os
from PIL import Image
import torch


sys.path.append('.')

# #############
args = sys.argv
mvtec_name = args[1]
mvtec_aomaly_name = args[2]
mvtec_path = args[3]
# ##############


target_path = './generate_data/'+mvtec_name+'/'+ mvtec_aomaly_name
if not os.path.exists(os.path.join(target_path,'image')):
    os.makedirs(os.path.join(target_path,'image'))
if not os.path.exists(os.path.join(target_path,'fg')):
    os.mkdir(os.path.join(target_path,'fg'))
cnt = len(os.listdir(os.path.join(target_path,'image')))
if cnt>=500:
    print(f"already generate 500 images {mvtec_name} {mvtec_aomaly_name}")
    exit()

os.makedirs(os.path.join(target_path,'origin'),exist_ok=True)
os.makedirs(os.path.join(target_path,'concat'),exist_ok=True)

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", safety_checker=None
).to("cuda")
pipe.load_lora_weights('./generate_data/'+mvtec_name+'/'+mvtec_aomaly_name+'/checkpoint-2000')

# for i in range(cnt,1000):
batch_size = 4
for _ in range(cnt,500):
    cnt = len(os.listdir(os.path.join(target_path,'image')))
    if cnt>=500:
        exit()
    
    text_prompt = ["a vfx with large sks and red sks and marker sks"]
    text_prompt = text_prompt * batch_size
    outputs,origin_images = pipe(prompt_blend=text_prompt,num_inference_steps=100,guidance_scale=8.5,class_id = mvtec_name, data_dir = mvtec_path)
    
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
    
    for b in range(batch_size):
        if b+cnt>=500:
            exit()
        outputs.images[b].save(os.path.join(target_path,'image',str(cnt+b)+".png"))
        origin_image = origin_images[b]
        origin_image = origin_image.resize((512,512))
        origin_image.save(os.path.join(target_path,'origin',str(cnt+b)+".png"))
        concat_image = concat_images_horizontally(outputs.images[b], origin_image)
        concat_image.save(os.path.join(target_path,'concat',str(cnt+b)+".png"))
        print(b+cnt)
    
    

    
    
 
