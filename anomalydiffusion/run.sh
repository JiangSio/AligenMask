path_to_mvtec_dataset=/data4/jiangtianjia/datasets/jsons/mvtec/train_4_shot
python run-mvtec.py --data_path $path_to_mvtec_dataset --gpu_id 1


# conda deactivate
# conda activate ldm
# cd /data4/jiangtianjia/Project/anomalydiffusion
# path_to_mvtec_dataset=/data4/jiangtianjia/datasets/jsons/mvtec/train_4_shot
# python run-mvtec-transistor.py --data_path $path_to_mvtec_dataset --gpu_id 7

conda deactivate
conda activate ldm
cd /data4/jiangtianjia/Project/anomalydiffusion
path_to_mvtec_dataset=/data4/jiangtianjia/datasets/jsons/mvtec/train_4_shot
python run-mvtecgenerateunmatching2.py --data_path $path_to_mvtec_dataset --gpu_id 3