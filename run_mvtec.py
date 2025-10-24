import subprocess
import os

mvtec_path = '/data1/gpt/jtj/asynthesis_data'
dualanodiff_path = '/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff'
anomalydiffusion_path = '/data1/gpt/jtj/AligenMask/anomalydiffusion'
testmodel_path = "/data1/gpt/jtj/AligenMask/anomaly_metrics"
cuda_id = 7
# 定义要执行的Bash脚本模板
bash_script_template = '''

cd {dualanodiff_path}
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR={mvtec_path}

export NAME="{name}"
export ANOMALY="{anomaly}"
export OUTPUT_DIR="generate_data/$NAME/$ANOMALY"

CUDA_VISIBLE_DEVICES={id} accelerate launch \
    --main_process_port=30005 \
    train_dreambooth_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="a photo of hazelnut" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=2000 \
    --resume_from_checkpoint "latest" \
    --mvtec_name=$NAME \
    --mvtec_anamaly_name=$ANOMALY \
    --rank 32 \
    --seed 32 \
    --train_text_encoder \
    --attn_loss_weight 0.1
    
# sleep 2m
    
'''


bash_generate_data_template='''
cd {dualanodiff_path}
CUDA_VISIBLE_DEVICES={id} python inference_mvtec_split.py {name} {anomaly} {mvtec_path}
# sleep 2m
'''

bash_generate_mask_template='''
cd {dualanodiff_path}
CUDA_VISIBLE_DEVICES={id} python process.py {name} {anomaly}

'''

bash_anomaly_generate_template='''
cd {anomalydiffusion_path}
python run-mvtecgeneratematching1.py --gpu_id={id} --data_path={dualanodiff_path}/generate_data --clssname={name} --anomalyname={anomaly}

'''

bash_segment_template='''
cd {testmodel_path}
CUDA_VISIBLE_DEVICES={id} python train-localization.py --generated_data_path={anomalydiffusion_path}/generated_matched_dataset  --mvtec_path={mvtec_path} --suffix=jpg --name=Anomalydiffusion+aligen --epochs=100 --clss_name={name} --bs={bs} --lr={lr}
'''

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

bash_file_path = "train_shells/"+"run2.sh"
if os.path.exists(bash_file_path):
    os.remove(bash_file_path)
os.makedirs("train_shells/",exist_ok=True)

for name in name_list:
    
    anomalies=[]
    for anomaly in os.listdir(os.path.join(mvtec_path,name,'test')):
        if anomaly != 'good':
            anomalies.append(anomaly)
        
    
    with open(bash_file_path, 'a') as file:
        
        # 训练模型
        
        for anomaly in anomalies:
            bash_script = bash_script_template.format(name=name, anomaly=anomaly, id=cuda_id, dualanodiff_path=dualanodiff_path, mvtec_path=mvtec_path)
            file.write(bash_script)
            file.write('\n')
            
            # 生成数据：
            bash_script = bash_generate_data_template.format(name=name,id=cuda_id,anomaly=anomaly, dualanodiff_path=dualanodiff_path, mvtec_path=mvtec_path)
            file.write(bash_script)
            file.write('\n')

            # 生成mask
            bash_script = bash_generate_mask_template.format(name=name, anomaly=anomaly, id=cuda_id, dualanodiff_path=dualanodiff_path)
            file.write(bash_script)
            file.write('\n')

            #生成异常图像
            bash_script = bash_anomaly_generate_template.format(id=cuda_id, name=name, anomaly=anomaly, anomalydiffusion_path=anomalydiffusion_path, dualanodiff_path=dualanodiff_path)
            # file.write(bash_script)
            # file.write('\n')
        
        #测试
        
        bash_script = bash_segment_template.format(id=cuda_id, mvtec_path=mvtec_path, anomalydiffusion_path=anomalydiffusion_path, name=name, testmodel_path=testmodel_path,bs=16,lr=0.0001)
        # file.write(bash_script)
        # file.write('\n')
    

        
subprocess.run(['chmod', '+x', bash_file_path])
subprocess.run(["bash",bash_file_path])