
CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_bottle_broken_large     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/bottle/test/broken_large     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_bottle_broken_small     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/bottle/test/broken_small     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_bottle_contamination     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/bottle/test/contamination     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_bent_wire     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/cable/test/bent_wire     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_cable_swap     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/cable/test/cable_swap     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_combined     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/cable/test/combined     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_cut_inner_insulation     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/cable/test/cut_inner_insulation     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_cut_outer_insulation     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/cable/test/cut_outer_insulation     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_missing_cable     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/cable/test/missing_cable     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_missing_wire     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/cable/test/missing_wire     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_poke_insulation     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/cable/test/poke_insulation     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_capsule_crack     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/capsule/test/crack     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_capsule_faulty_imprint     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/capsule/test/faulty_imprint     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_capsule_poke     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/capsule/test/poke     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_capsule_scratch     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/capsule/test/scratch     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_capsule_squeeze     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/capsule/test/squeeze     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_capsules_defect     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/capsules/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cashew_defect     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/cashew/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_chewinggum_defect     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/chewinggum/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_fryum_defect     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/fryum/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_hazelnut_crack     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/hazelnut/test/crack     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_hazelnut_cut     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/hazelnut/test/cut     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_hazelnut_hole     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/hazelnut/test/hole     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_hazelnut_print     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/hazelnut/test/print     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_macaroni_defect     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/macaroni/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_metal_nut_bent     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/metal_nut/test/bent     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_metal_nut_color     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/metal_nut/test/color     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_metal_nut_flip     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/metal_nut/test/flip     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_metal_nut_scratch     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/metal_nut/test/scratch     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pcb1_defect     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/pcb1/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pcb2_defect     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/pcb2/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pcb3_defect     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/pcb3/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pcb4_defect     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/pcb4/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_color     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/pill/test/color     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_combined     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/pill/test/combined     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_contamination     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/pill/test/contamination     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_crack     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/pill/test/crack     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_faulty_imprint     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/pill/test/faulty_imprint     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_pill_type     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/pill/test/pill_type     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_scratch     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/pill/test/scratch     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pipe_fryum_defect     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/pipe_fryum/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_manipulated_front     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/screw/test/manipulated_front     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_scratch_head     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/screw/test/scratch_head     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_scratch_neck     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/screw/test/scratch_neck     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_thread_side     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/screw/test/thread_side     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_thread_top     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/screw/test/thread_top     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_single_manipulated_front     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/screw_single/test/manipulated_front     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_single_scratch_head     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/screw_single/test/scratch_head     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_single_scratch_neck     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/screw_single/test/scratch_neck     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_single_thread_side     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/screw_single/test/thread_side     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_single_thread_top     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/screw_single/test/thread_top     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_toothbrush_defective     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/toothbrush/test/defective     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=6 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_tubes_anomalous     --gpus 0,     --data_root /data1/gpt/jtj/asynthesis_data/tubes/test/anomalous     --init_word "defect"
    

