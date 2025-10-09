
cd /data/gpt/real_aligen/AliGen-main/anomaly_metrics
CUDA_VISIBLE_DEVICES=3 python train-localization.py --generated_data_path=/data/gpt/real_aligen/AliGen-main/anomalydiffusion/generated_matched_dataset  --mvtec_path=/data/gpt/real_aligen/AliGen-main/datasets/real_anomaly_set --suffix=jpg --name=Anomalydiffusion+aligen --epochs=100 --clss_name=capsules

