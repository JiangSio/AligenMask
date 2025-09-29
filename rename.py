import os

appendix = "_realdata"

dir_list=[
    "DualAnoDiff/dual-interrelated_diff/generate_data",
    "DiAD/output1000_random",
    "anomalydiffusion/generated_matched_random_dataset",
    "anomaly_metrics/logs",
]

for name in dir_list:
    origin_dir = name
    new_dir = name+appendix
    os.rename(origin_dir,new_dir)