CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=bottle --anomaly_name=contamination

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=bottle --anomaly_name=contamination --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=bottle --anomaly_name=broken_large

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=bottle --anomaly_name=broken_large --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=bottle --anomaly_name=broken_small

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=bottle --anomaly_name=broken_small --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=cable --anomaly_name=cut_outer_insulation

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=cable --anomaly_name=cut_outer_insulation --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=cable --anomaly_name=cable_swap

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=cable --anomaly_name=cable_swap --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=cable --anomaly_name=missing_cable

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=cable --anomaly_name=missing_cable --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=cable --anomaly_name=combined

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=cable --anomaly_name=combined --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=cable --anomaly_name=cut_inner_insulation

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=cable --anomaly_name=cut_inner_insulation --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=cable --anomaly_name=poke_insulation

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=cable --anomaly_name=poke_insulation --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=cable --anomaly_name=bent_wire

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=cable --anomaly_name=bent_wire --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=cable --anomaly_name=missing_wire

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=cable --anomaly_name=missing_wire --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=capsule --anomaly_name=squeeze

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=capsule --anomaly_name=squeeze --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=capsule --anomaly_name=crack

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=capsule --anomaly_name=crack --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=capsule --anomaly_name=scratch

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=capsule --anomaly_name=scratch --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=capsule --anomaly_name=faulty_imprint

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=capsule --anomaly_name=faulty_imprint --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=capsule --anomaly_name=poke

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=capsule --anomaly_name=poke --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=capsules --anomaly_name=defect

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=capsules --anomaly_name=defect --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=cashew --anomaly_name=defect

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=cashew --anomaly_name=defect --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=chewinggum --anomaly_name=defect

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=chewinggum --anomaly_name=defect --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=fryum --anomaly_name=defect

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=fryum --anomaly_name=defect --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=hazelnut --anomaly_name=cut

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=hazelnut --anomaly_name=cut --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=hazelnut --anomaly_name=print

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=hazelnut --anomaly_name=print --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=hazelnut --anomaly_name=crack

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=hazelnut --anomaly_name=crack --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=hazelnut --anomaly_name=hole

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=hazelnut --anomaly_name=hole --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=macaroni --anomaly_name=defect

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=macaroni --anomaly_name=defect --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=metal_nut --anomaly_name=scratch

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=metal_nut --anomaly_name=scratch --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=metal_nut --anomaly_name=bent

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=metal_nut --anomaly_name=bent --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=metal_nut --anomaly_name=color

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=metal_nut --anomaly_name=color --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=metal_nut --anomaly_name=flip

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=metal_nut --anomaly_name=flip --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=pcb1 --anomaly_name=defect

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=pcb1 --anomaly_name=defect --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=pcb2 --anomaly_name=defect

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=pcb2 --anomaly_name=defect --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=pcb3 --anomaly_name=defect

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=pcb3 --anomaly_name=defect --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=pcb4 --anomaly_name=defect

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=pcb4 --anomaly_name=defect --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=pill --anomaly_name=contamination

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=pill --anomaly_name=contamination --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=pill --anomaly_name=pill_type

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=pill --anomaly_name=pill_type --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=pill --anomaly_name=combined

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=pill --anomaly_name=combined --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=pill --anomaly_name=crack

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=pill --anomaly_name=crack --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=pill --anomaly_name=scratch

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=pill --anomaly_name=scratch --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=pill --anomaly_name=faulty_imprint

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=pill --anomaly_name=faulty_imprint --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=pill --anomaly_name=color

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=pill --anomaly_name=color --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=pipe_fryum --anomaly_name=defect

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=pipe_fryum --anomaly_name=defect --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=screw --anomaly_name=scratch_head

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=screw --anomaly_name=scratch_head --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=screw --anomaly_name=thread_top

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=screw --anomaly_name=thread_top --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=screw --anomaly_name=manipulated_front

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=screw --anomaly_name=manipulated_front --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=screw --anomaly_name=thread_side

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=screw --anomaly_name=thread_side --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=screw --anomaly_name=scratch_neck

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=screw --anomaly_name=scratch_neck --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=screw_single --anomaly_name=scratch_head

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=screw_single --anomaly_name=scratch_head --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=screw_single --anomaly_name=thread_top

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=screw_single --anomaly_name=thread_top --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=screw_single --anomaly_name=manipulated_front

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=screw_single --anomaly_name=manipulated_front --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=screw_single --anomaly_name=thread_side

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=screw_single --anomaly_name=thread_side --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=screw_single --anomaly_name=scratch_neck

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=screw_single --anomaly_name=scratch_neck --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=toothbrush --anomaly_name=defective

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=toothbrush --anomaly_name=defective --data_root=/data1/gpt/jtj/asynthesis_data

CUDA_VISIBLE_DEVICES=3 python train_mask.py --mvtec_path=/data1/gpt/jtj/asynthesis_data --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=tubes --anomaly_name=anomalous

CUDA_VISIBLE_DEVICES=3 python generate_mask.py --data_root=/data1/gpt/jtj/asynthesis_data --sample_name=tubes --anomaly_name=anomalous --data_root=/data1/gpt/jtj/asynthesis_data

