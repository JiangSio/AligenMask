mvtec_root="/data1/gpt/jtj/asynthesis_data"
# 1. bottle
categories=("broken_large" "broken_small" "contamination")
for ((i=0;i<3;i++))
do
    name="v2_bottle_${categories[$i]}"
    path="${mvtec_root}/bottle/test/${categories[$i]}"
    CUDA_VISIBLE_DEVICES=6 python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 2. cable
categories=("bent_wire" "cable_swap" "combined" "cut_inner_insulation" "cut_outer_insulation" "missing_cable" "missing_wire" "poke_insulation")
for ((i=0;i<8;i++))
do
    name="v2_cable_${categories[$i]}"
    path="${mvtec_root}/cable/test/${categories[$i]}"
    CUDA_VISIBLE_DEVICES=6 python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 3. capsule
categories=("crack" "faulty_imprint" "poke" "scratch" "squeeze")
for ((i=0;i<5;i++))
do
    name="v2_capsule_${categories[$i]}"
    path="${mvtec_root}/capsule/test/${categories[$i]}"
    CUDA_VISIBLE_DEVICES=6 python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 4. hazelnut
categories=("crack" "cut" "hole" "print")
for ((i=0;i<4;i++)) 
do
    name="v2_hazelnut_${categories[$i]}"
    path="${mvtec_root}/hazelnut/test/${categories[$i]}"
    CUDA_VISIBLE_DEVICES=6 python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done


# 5. metal_nut
categories=("bent" "color" "flip" "scratch")
for ((i=0;i<4;i++)) 
do
    name="v2_metal_nut_${categories[$i]}"
    path="${mvtec_root}/metal_nut/test/${categories[$i]}"
    CUDA_VISIBLE_DEVICES=6 python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 6. pill
categories=("color" "combined" "contamination" "crack" "faulty_imprint" "pill_type" "scratch")
for ((i=0;i<7;i++)) 
do
    name="v2_pill_${categories[$i]}"
    path="${mvtec_root}/pill/test/${categories[$i]}"
    CUDA_VISIBLE_DEVICES=6 python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 7. screw
categories=("manipulated_front" "scratch_head" "scratch_neck" "thread_side" "thread_top")
for ((i=0;i<5;i++)) 
do
    name="v2_screw_${categories[$i]}"
    path="${mvtec_root}/screw/test/${categories[$i]}"
    CUDA_VISIBLE_DEVICES=6 python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 8.capsules
categories=("defect")
for ((i=0;i<1;i++)) 
do
    name="v2_capsules_${categories[$i]}"
    path="${mvtec_root}/capsule/test/${categories[$i]}"
    CUDA_VISIBLE_DEVICES=6 python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"

#9.macaroni2
categories=("defect")
for ((i=0;i<1;i++)) 
do
    name="v2_macaroni2_${categories[$i]}"
    path="${mvtec_root}/macaroni2/test/${categories[$i]}"
    CUDA_VISIBLE_DEVICES=6 python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"

#10.tubes
categories=("anomalous")
for ((i=0;i<1;i++))
do
    name="v2_tubes_${categories[$i]}"
    path="${mvtec_root}/tubes/test/${categories[$i]}"
    CUDA_VISIBLE_DEVICES=6 python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"