
path_to_the_generated_data="../anomalydiffusion/generated_matched_random_dataset"
path_to_mvtec="../datasets/real_anomaly_set"

#train and test the anomaly detection model
python train-localization.py --generated_data_path $path_to_the_generated_data  --mvtec_path=$path_to_mvtec --suffix jpg --name DiAD+Anomalydiffusion+random --epochs 100
# python train-classification.py --mvtec_path=$path_to_mvtec --generated_data_path=$path_to_the_generated_data --json $json