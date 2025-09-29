path_to_mvtec_dataset="/home/guangpintao/jiangtianjia/datasets/jsons/mvtec/train_4_shot"
CUDA_VISIBLE_DEVICES=1 python main.py --spatial_encoder_embedding --data_enhance \
 --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml -t \
 --actual_resume models/ldm/text2img-large/model.ckpt -n test --gpus 0, \
 --init_word anomaly  --mvtec_path=$path_to_mvtec_dataset