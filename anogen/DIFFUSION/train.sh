
CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_bottle_broken_large     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/bottle/test/broken_large     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_bottle_broken_small     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/bottle/test/broken_small     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_bottle_contamination     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/bottle/test/contamination     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_bent_wire     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/cable/test/bent_wire     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_cable_swap     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/cable/test/cable_swap     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_combined     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/cable/test/combined     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_cut_inner_insulation     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/cable/test/cut_inner_insulation     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_cut_outer_insulation     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/cable/test/cut_outer_insulation     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_missing_cable     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/cable/test/missing_cable     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_missing_wire     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/cable/test/missing_wire     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cable_poke_insulation     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/cable/test/poke_insulation     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_candle_defect     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/candle/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_capsule_crack     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/capsule/test/crack     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_capsule_faulty_imprint     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/capsule/test/faulty_imprint     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_capsule_poke     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/capsule/test/poke     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_capsule_scratch     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/capsule/test/scratch     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_capsule_squeeze     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/capsule/test/squeeze     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_capsules_defect     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/capsules/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_carpet_color     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/carpet/test/color     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_carpet_cut     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/carpet/test/cut     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_carpet_hole     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/carpet/test/hole     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_carpet_metal_contamination     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/carpet/test/metal_contamination     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_carpet_thread     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/carpet/test/thread     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_cashew_defect     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/cashew/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_chewinggum_defect     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/chewinggum/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_fryum_defect     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/fryum/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_grid_bent     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/grid/test/bent     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_grid_broken     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/grid/test/broken     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_grid_glue     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/grid/test/glue     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_grid_metal_contamination     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/grid/test/metal_contamination     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_grid_thread     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/grid/test/thread     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_hazelnut_crack     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/hazelnut/test/crack     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_hazelnut_cut     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/hazelnut/test/cut     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_hazelnut_hole     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/hazelnut/test/hole     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_hazelnut_print     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/hazelnut/test/print     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_leather_color     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/leather/test/color     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_leather_cut     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/leather/test/cut     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_leather_fold     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/leather/test/fold     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_leather_glue     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/leather/test/glue     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_leather_poke     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/leather/test/poke     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_macaroni1_defect     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/macaroni1/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_macaroni2_defect     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/macaroni2/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_metal_nut_bent     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/metal_nut/test/bent     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_metal_nut_color     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/metal_nut/test/color     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_metal_nut_flip     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/metal_nut/test/flip     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_metal_nut_scratch     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/metal_nut/test/scratch     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pcb1_defect     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/pcb1/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pcb2_defect     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/pcb2/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pcb3_defect     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/pcb3/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pcb4_defect     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/pcb4/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_color     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/pill/test/color     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_combined     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/pill/test/combined     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_contamination     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/pill/test/contamination     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_crack     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/pill/test/crack     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_faulty_imprint     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/pill/test/faulty_imprint     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_pill_type     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/pill/test/pill_type     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pill_scratch     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/pill/test/scratch     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_pipe_fryum_defect     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/pipe_fryum/test/defect     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_manipulated_front     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/screw/test/manipulated_front     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_scratch_head     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/screw/test/scratch_head     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_scratch_neck     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/screw/test/scratch_neck     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_thread_side     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/screw/test/thread_side     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_screw_thread_top     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/screw/test/thread_top     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_tile_crack     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/tile/test/crack     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_tile_glue_strip     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/tile/test/glue_strip     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_tile_gray_stroke     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/tile/test/gray_stroke     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_tile_oil     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/tile/test/oil     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_tile_rough     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/tile/test/rough     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_toothbrush_defective     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/toothbrush/test/defective     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_transistor_bent_lead     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/transistor/test/bent_lead     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_transistor_cut_lead     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/transistor/test/cut_lead     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_transistor_damaged_case     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/transistor/test/damaged_case     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_transistor_misplaced     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/transistor/test/misplaced     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_wood_color     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/wood/test/color     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_wood_combined     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/wood/test/combined     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_wood_hole     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/wood/test/hole     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_wood_liquid     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/wood/test/liquid     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_wood_scratch     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/wood/test/scratch     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_zipper_broken_teeth     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/zipper/test/broken_teeth     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_zipper_combined     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/zipper/test/combined     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_zipper_fabric_border     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/zipper/test/fabric_border     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_zipper_fabric_interior     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/zipper/test/fabric_interior     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_zipper_rough     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/zipper/test/rough     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_zipper_split_teeth     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/zipper/test/split_teeth     --init_word "defect"
    


CUDA_VISIBLE_DEVICES=1 python main.py     --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml     -t --actual_resume models/ldm/text2img-large/model.ckpt     -n v2_zipper_squeezed_teeth     --gpus 0,     --data_root /data1/gpt/jtj/supp_exp/mvtec+visa/zipper/test/squeezed_teeth     --init_word "defect"
    

