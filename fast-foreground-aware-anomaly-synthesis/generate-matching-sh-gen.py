import os

root_dir = "/data1/gpt/jtj/asynthesis_data"
matching_img_mask_dir = "/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data"
save_file = "generate_matching.sh"
cuda_id = "4"

if os.path.exists(save_file):
    os.remove(save_file)

bash_script_template = '''
CUDA_VISIBLE_DEVICES={cuda_id} python generate_with_mask_mvtec.py \
--data_root='{matching_img_mask_dir}/{clss_name}/{anomaly}/origin/' \
--weight_idx 9999 \
--sample_name='samples-matching/{clss_name}/' \
--init_word broken \
--anomaly_name='{anomaly}' \
--pt_path='logs/{clss_name}_{anomaly}/checkpoints' \
--mask_path='{matching_img_mask_dir}/{clss_name}/{anomaly}/fg/' \
--matching
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
            bash_text = bash_script_template.format(cuda_id=cuda_id, anomaly=anomaly, matching_img_mask_dir=matching_img_mask_dir, clss_name=clss_name)
            f.write(bash_text)
            f.write("\n")
        