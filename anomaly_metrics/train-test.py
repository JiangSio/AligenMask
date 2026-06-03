import subprocess
import os

bash_segment_template='''
CUDA_VISIBLE_DEVICES={id} python train-localization.py --generated_data_path={generated_data_path}  --mvtec_path={mvtec_path} --name={name} --epochs=100 --clss_name={clss_name} --bs={bs} --lr={lr}
'''
mvtec_path = "/data1/gpt/jtj/supp_exp/mvtec+visa"
cuda_id = 5
generated_data_path = "/data1/gpt/jtj/supp_exp/AligenMask/fast-foreground-aware-anomaly-synthesis/samples"
name = "fast"
# ########

name_list = [x for x in os.listdir(generated_data_path) if os.path.isdir(os.path.join(generated_data_path, x))]
name_list.sort()

bash_file_path = "fast.sh"
if os.path.exists(bash_file_path):
    os.remove(bash_file_path)

for clss_name in name_list:
        
    with open(bash_file_path, 'a') as file:
        #测试
        
        bash_script = bash_segment_template.format(id=cuda_id, mvtec_path=mvtec_path, generated_data_path=generated_data_path, name=name, clss_name=clss_name, bs=16,lr=0.0001)
        file.write(bash_script)
        file.write('\n')
    

        
# subprocess.run(['chmod', '+x', bash_file_path])
# subprocess.run(["bash",bash_file_path])