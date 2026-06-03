import os

root_dir = "/data1/gpt/jtj/supp_exp/mvtec+visa"
save_file = "train.sh"
cuda_id = "5"

if os.path.exists(save_file):
    os.remove(save_file)

bash_script_template = '''
CUDA_VISIBLE_DEVICES={cuda_id} python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder-mvtec.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n mvtec_run \
    --gpus 0, \
    --init_word broken \
    --mvtec_path {root_dir}/{clss_name}/ground_truth/{anomaly}/ \
    --log_folder "{clss_name}_{anomaly}"
    
'''

clsses = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
clsses.sort()
for clss_name in clsses:
    clss_dir = os.path.join(root_dir, clss_name)
    test_path = os.path.join(clss_dir, "test")
    anomalies = [x for x in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, x))]
    anomalies.remove("good")
    anomalies.sort()
    for anomaly in anomalies:

        with open(save_file, "a") as f:
            path = os.path.join(test_path, anomaly)
            bash_text = bash_script_template.format(cuda_id=cuda_id, anomaly=anomaly, root_dir=root_dir, clss_name=clss_name)
            f.write(bash_text)
            f.write("\n")
        