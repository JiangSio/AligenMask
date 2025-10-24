import subprocess
import os

bash_segment_template='''
CUDA_VISIBLE_DEVICES={id} python train-localization.py --generated_data_path={generated_data_path}  --mvtec_path={mvtec_path} --suffix={suffix} --name={name} --epochs=100 --clss_name={clss_name} --bs={bs} --lr={lr}
'''
mvtec_path = "/data1/gpt/jtj/asynthesis_data"
cuda_id = 2
generated_data_path = "/data1/gpt/jtj/AligenMask/anogen/DIFFUSION/output_unmatched2"
name = "anogen2"
suffix = "png"
# ########

name_list = [
    "bottle", 
	"cable", 
	"capsule", 
	"capsules", 
	"cashew", 
	"chewinggum", 
	"fryum", 
	"hazelnut", 
	"macaroni", 
	"metal_nut", 
	"pcb1", 
	"pcb2", 
	"pcb3", 
	"pcb4", 
	"pill", 
	"pipe_fryum", 
	"screw", 
	"screw_single", 
	"toothbrush", 
	"tubes", 
    ]

bash_file_path = "run1.sh"
if os.path.exists(bash_file_path):
    os.remove(bash_file_path)

for clss_name in name_list:
        
    with open(bash_file_path, 'a') as file:
        #测试
        
        bash_script = bash_segment_template.format(id=cuda_id, mvtec_path=mvtec_path, generated_data_path=generated_data_path, suffix=suffix, name=name, clss_name=clss_name, bs=16,lr=0.0001)
        file.write(bash_script)
        file.write('\n')
    

        
# subprocess.run(['chmod', '+x', bash_file_path])
# subprocess.run(["bash",bash_file_path])