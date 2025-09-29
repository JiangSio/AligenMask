

cd /data/gpt/real_aligen/AliGen-main/DualAnoDiff/dual-interrelated_diff
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="none"

export NAME="capsules"
export ANOMALY="defect"
export OUTPUT_DIR="generate_data/$NAME/$ANOMALY"

CUDA_VISIBLE_DEVICES=2 accelerate launch     --main_process_port=30005     train_dreambooth_lora.py     --pretrained_model_name_or_path=$MODEL_NAME     --instance_data_dir=$INSTANCE_DIR     --output_dir=$OUTPUT_DIR     --instance_prompt="a photo of hazelnut"     --resolution=512     --train_batch_size=1     --gradient_accumulation_steps=4     --learning_rate=5e-5     --lr_scheduler="constant"     --lr_warmup_steps=0     --max_train_steps=2000     --resume_from_checkpoint "latest"     --mvtec_name=$NAME     --mvtec_anamaly_name=$ANOMALY     --rank 32     --seed 32     --train_text_encoder     --attn_loss_weight 1
    
# sleep 2m
    

