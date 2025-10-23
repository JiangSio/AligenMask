
CUDA_VISIBLE_DEVICES=0 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/asynthesis_data/bottle/train/good/' --weight_idx 2999 --sample_name='samples/bottle/' --init_word broken --anomaly_name='broken_large' --pt_path='logs/bottle_broken_large/checkpoints' --mask_path='/data1/gpt/jtj/asynthesis_data/bottle/ground_truth/broken_large/'
    

