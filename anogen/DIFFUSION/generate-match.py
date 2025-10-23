import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange, repeat
from torchvision.utils import make_grid
import shutil
import random

import sys
sys.path.append('/apdcephfs/private_laurelgui/projects/textual_inversion')

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256,256))
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = (image/127.5 - 1.0).astype(np.float32)
    return image

def preprocess_mask(mask_path, h, w):
    mask = Image.open(mask_path).convert('1')
    mask_resize = mask.resize((w, h))
    return 1-np.array(mask_resize).astype(np.float32)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="*",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="/data/pretrained_models/ldm/text2img-large/model.ckpt", 
        help="Path to pretrained ldm text2img model")

    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")
    
    parser.add_argument(
        "--image_prompt",
        type=str,
        help="image to prompt with, must specify a mask",
        default=None
    )

    parser.add_argument(
        "--mask_prompt",
        type=str,
        help="mask to prompt with, must specify image prompt",
        default=None
    )

    opt = parser.parse_args()
    opt.ddim_eta=0.0
    opt.n_samples=1
    opt.n_iter=1 
    opt.scale=10.0 
    opt.ddim_steps=50 
    opt.ckpt_path="models/ldm/text2img-large/model.ckpt"
    opt.prompt="*"
    


    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval_with_tokens.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, opt.ckpt_path)  # TODO: check path
    

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    directory="../../DualAnoDiff/dual-interrelated_diff/generate_data"
    root_dir="/data1/gpt/jtj/asynthesis_data"
    
    # image_prompt = opt.image_prompt
    # mask_prompt = opt.mask_prompt
    
    
    objects=os.listdir(root_dir)
    for single_object in objects:
        
        anomalies=os.listdir(os.path.join(root_dir,single_object,"test"))
        anomalies.remove("good")
        for anomaly in anomalies:
            mask_prompt_list = glob.glob(os.path.join(directory,single_object,anomaly,'fg','*.png'),recursive=True)
            print(len(mask_prompt_list))
            image_save_dir=f"output_matched/{single_object}/{anomaly}/image"
            mask_save_dir=f"output_matched/{single_object}/{anomaly}/mask"
            os.makedirs(image_save_dir, exist_ok=True)
            os.makedirs(mask_save_dir, exist_ok=True)
            good_path=os.path.join(root_dir,single_object,"train","good")

            embedding_dir=[x for x in glob.glob(os.path.join("logs","*")) if x.endswith(f"{single_object}_{anomaly}")][0]
            opt.embedding_path = os.path.join(embedding_dir,"checkpoints","embeddings.pt")
            model.embedding_manager.load(opt.embedding_path)

            cur_num = len(os.listdir(image_save_dir))
            for identifier in range(cur_num,500):
                
                mask_prompt=random.choice(mask_prompt_list)
                # image_prompt=random.choice([os.path.join(good_path,x) for x in os.listdir(good_path)])
                image_prompt=mask_prompt.replace("fg", "origin")

                x0 = None
                mask = None
                print("Using image as x0: " + image_prompt)
                print("Using mask image: " + mask_prompt)
                image_prompt_input = preprocess_image(image_prompt)
                image_prompt_input = rearrange(image_prompt_input, 'h w c -> c h w')
                image_prompt_input = torch.from_numpy(image_prompt_input)
                image_prompt_input = image_prompt_input.to(memory_format=torch.contiguous_format).float()
                image_prompt_input = repeat(image_prompt_input, 'c h w -> b c h w', b=opt.n_samples).to(device)
                encoder_posterior = model.encode_first_stage(image_prompt_input)
                x0 = model.get_first_stage_encoding(encoder_posterior).detach()
                h = opt.H//opt.f
                w = opt.W//opt.f
                mask_prompt_input = preprocess_mask(mask_prompt, h, w)
                mask = torch.tensor(mask_prompt_input)
                mask = repeat(mask, 'h w -> b h w', b=opt.n_samples).to(device)
                mask = mask[:, None, ...]

                with torch.no_grad():
                    with model.ema_scope():
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(opt.n_samples * [""])
                        for n in trange(opt.n_iter, desc="Sampling"):
                            c = model.get_learned_conditioning(opt.n_samples * [opt.prompt])
                            shape = [4, opt.H//opt.f, opt.W//opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x0=x0,
                                                            mask=mask)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(image_save_dir, f"{identifier}.png"))
                                mask = Image.open(mask_prompt).convert("L").resize((opt.W,opt.H))
                                mask.save(os.path.join(mask_save_dir,f"{identifier}.png"))
                                identifier += 1