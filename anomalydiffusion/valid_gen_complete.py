import os
from glob import glob

with open('name-anomaly.txt', 'r') as f:
    names = f.read().splitlines()
# for name in names:
#     clss,anomaly = name.split('+')
#     root = "generated_mask"
#     save_path = os.path.join(root, clss, anomaly)
#     length = len(os.listdir(save_path))
#     print(f"{root}    {clss}+{anomaly}: {length}")
#     if length != 501:
#         print("ERROR")

# for name in names:
#     clss,anomaly = name.split('+')
#     save_file = f"logs/mask-checkpoints/{clss}-{anomaly}/checkpoints/embeddings.pt"
#     if not os.path.exists(save_file):
#         print(f"ERROR: {save_file} does not exist")
#     save_file = f"logs/mask-checkpoints/{clss}-{anomaly}/checkpoints/embeddings_gs-30000.pt"
#     if not os.path.exists(save_file):
#         print(f"ERROR: {save_file} does not exist")

# for name in names:
#     clss,anomaly = name.split('+')
#     root = "/data1/gpt/jtj/supp_exp/AligenMask/anogen/DIFFUSION/logs"

#     save_file = glob(os.path.join(root, f"*v2_{clss}_{anomaly}", "checkpoints", "embeddings.pt"))
#     if len(save_file) == 0:
#         print(f"ERROR: {clss}+{anomaly} embeddings.pt does not exist")
#     save_file = glob(os.path.join(root, f"*v2_{clss}_{anomaly}", "checkpoints", "embeddings_gs-6099.pt"))
#     if len(save_file) == 0:
#         print(f"ERROR: {clss}+{anomaly} embeddings_gs-6099.pt does not exist")

# for name in names:
#     clss,anomaly = name.split('+')
#     root = "/data1/gpt/jtj/supp_exp/AligenMask/fast-foreground-aware-anomaly-synthesis/logs"
#     save_file = f"{root}/{clss}_{anomaly}/checkpoints/embeddings_gs-9999.pt"
#     if not os.path.exists(save_file):
#         print(f"ERROR: {save_file} does not exist")

for name in names:
    clss,anomaly = name.split('+')
    root = "/data1/gpt/jtj/supp_exp/AligenMask/anogen/DIFFUSION/output_unmatched2"
    save_path = os.path.join(root, clss, anomaly,"image")
    save_mask_path = os.path.join(root, clss, anomaly,"mask")
    if len(os.listdir(save_path)) != 500 or len(os.listdir(save_mask_path)) != 500:
        print(f"ERROR: {save_path} or {save_mask_path} does not have 500 images")
    root = "/data1/gpt/jtj/supp_exp/AligenMask/anomalydiffusion/generated_dataset"
    save_path = os.path.join(root, clss, anomaly,"image")
    save_mask_path = os.path.join(root, clss, anomaly,"mask")
    if len(os.listdir(save_path)) != 500 or len(os.listdir(save_mask_path)) != 500:
        print(f"ERROR: {save_path} or {save_mask_path} does not have 500 images")
    root = "/data1/gpt/jtj/supp_exp/AligenMask/fast-foreground-aware-anomaly-synthesis/samples"
    save_path = os.path.join(root, clss, anomaly,"image")
    save_mask_path = os.path.join(root, clss, anomaly,"mask")
    if len(os.listdir(save_path)) != 500 or len(os.listdir(save_mask_path)) != 500:
        print(f"ERROR: {save_path} or {save_mask_path} does not have 500 images")


    
    

    
