
CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/bottle/broken_large/origin/' --weight_idx 9999 --sample_name='samples-matching/bottle/' --init_word broken --anomaly_name='broken_large' --pt_path='logs/bottle_broken_large/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/bottle/broken_large/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/bottle/broken_small/origin/' --weight_idx 9999 --sample_name='samples-matching/bottle/' --init_word broken --anomaly_name='broken_small' --pt_path='logs/bottle_broken_small/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/bottle/broken_small/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/bottle/contamination/origin/' --weight_idx 9999 --sample_name='samples-matching/bottle/' --init_word broken --anomaly_name='contamination' --pt_path='logs/bottle_contamination/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/bottle/contamination/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/bent_wire/origin/' --weight_idx 9999 --sample_name='samples-matching/cable/' --init_word broken --anomaly_name='bent_wire' --pt_path='logs/cable_bent_wire/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/bent_wire/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/cable_swap/origin/' --weight_idx 9999 --sample_name='samples-matching/cable/' --init_word broken --anomaly_name='cable_swap' --pt_path='logs/cable_cable_swap/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/cable_swap/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/combined/origin/' --weight_idx 9999 --sample_name='samples-matching/cable/' --init_word broken --anomaly_name='combined' --pt_path='logs/cable_combined/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/combined/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/cut_inner_insulation/origin/' --weight_idx 9999 --sample_name='samples-matching/cable/' --init_word broken --anomaly_name='cut_inner_insulation' --pt_path='logs/cable_cut_inner_insulation/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/cut_inner_insulation/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/cut_outer_insulation/origin/' --weight_idx 9999 --sample_name='samples-matching/cable/' --init_word broken --anomaly_name='cut_outer_insulation' --pt_path='logs/cable_cut_outer_insulation/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/cut_outer_insulation/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/missing_cable/origin/' --weight_idx 9999 --sample_name='samples-matching/cable/' --init_word broken --anomaly_name='missing_cable' --pt_path='logs/cable_missing_cable/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/missing_cable/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/missing_wire/origin/' --weight_idx 9999 --sample_name='samples-matching/cable/' --init_word broken --anomaly_name='missing_wire' --pt_path='logs/cable_missing_wire/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/missing_wire/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/poke_insulation/origin/' --weight_idx 9999 --sample_name='samples-matching/cable/' --init_word broken --anomaly_name='poke_insulation' --pt_path='logs/cable_poke_insulation/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cable/poke_insulation/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/capsule/crack/origin/' --weight_idx 9999 --sample_name='samples-matching/capsule/' --init_word broken --anomaly_name='crack' --pt_path='logs/capsule_crack/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/capsule/crack/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/capsule/faulty_imprint/origin/' --weight_idx 9999 --sample_name='samples-matching/capsule/' --init_word broken --anomaly_name='faulty_imprint' --pt_path='logs/capsule_faulty_imprint/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/capsule/faulty_imprint/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/capsule/poke/origin/' --weight_idx 9999 --sample_name='samples-matching/capsule/' --init_word broken --anomaly_name='poke' --pt_path='logs/capsule_poke/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/capsule/poke/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/capsule/scratch/origin/' --weight_idx 9999 --sample_name='samples-matching/capsule/' --init_word broken --anomaly_name='scratch' --pt_path='logs/capsule_scratch/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/capsule/scratch/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/capsule/squeeze/origin/' --weight_idx 9999 --sample_name='samples-matching/capsule/' --init_word broken --anomaly_name='squeeze' --pt_path='logs/capsule_squeeze/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/capsule/squeeze/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/capsules/defect/origin/' --weight_idx 9999 --sample_name='samples-matching/capsules/' --init_word broken --anomaly_name='defect' --pt_path='logs/capsules_defect/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/capsules/defect/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cashew/defect/origin/' --weight_idx 9999 --sample_name='samples-matching/cashew/' --init_word broken --anomaly_name='defect' --pt_path='logs/cashew_defect/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/cashew/defect/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/chewinggum/defect/origin/' --weight_idx 9999 --sample_name='samples-matching/chewinggum/' --init_word broken --anomaly_name='defect' --pt_path='logs/chewinggum_defect/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/chewinggum/defect/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/fryum/defect/origin/' --weight_idx 9999 --sample_name='samples-matching/fryum/' --init_word broken --anomaly_name='defect' --pt_path='logs/fryum_defect/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/fryum/defect/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/hazelnut/crack/origin/' --weight_idx 9999 --sample_name='samples-matching/hazelnut/' --init_word broken --anomaly_name='crack' --pt_path='logs/hazelnut_crack/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/hazelnut/crack/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/hazelnut/cut/origin/' --weight_idx 9999 --sample_name='samples-matching/hazelnut/' --init_word broken --anomaly_name='cut' --pt_path='logs/hazelnut_cut/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/hazelnut/cut/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/hazelnut/hole/origin/' --weight_idx 9999 --sample_name='samples-matching/hazelnut/' --init_word broken --anomaly_name='hole' --pt_path='logs/hazelnut_hole/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/hazelnut/hole/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/hazelnut/print/origin/' --weight_idx 9999 --sample_name='samples-matching/hazelnut/' --init_word broken --anomaly_name='print' --pt_path='logs/hazelnut_print/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/hazelnut/print/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/macaroni/defect/origin/' --weight_idx 9999 --sample_name='samples-matching/macaroni/' --init_word broken --anomaly_name='defect' --pt_path='logs/macaroni_defect/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/macaroni/defect/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/metal_nut/bent/origin/' --weight_idx 9999 --sample_name='samples-matching/metal_nut/' --init_word broken --anomaly_name='bent' --pt_path='logs/metal_nut_bent/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/metal_nut/bent/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/metal_nut/color/origin/' --weight_idx 9999 --sample_name='samples-matching/metal_nut/' --init_word broken --anomaly_name='color' --pt_path='logs/metal_nut_color/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/metal_nut/color/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/metal_nut/flip/origin/' --weight_idx 9999 --sample_name='samples-matching/metal_nut/' --init_word broken --anomaly_name='flip' --pt_path='logs/metal_nut_flip/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/metal_nut/flip/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/metal_nut/scratch/origin/' --weight_idx 9999 --sample_name='samples-matching/metal_nut/' --init_word broken --anomaly_name='scratch' --pt_path='logs/metal_nut_scratch/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/metal_nut/scratch/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pcb1/defect/origin/' --weight_idx 9999 --sample_name='samples-matching/pcb1/' --init_word broken --anomaly_name='defect' --pt_path='logs/pcb1_defect/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pcb1/defect/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pcb2/defect/origin/' --weight_idx 9999 --sample_name='samples-matching/pcb2/' --init_word broken --anomaly_name='defect' --pt_path='logs/pcb2_defect/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pcb2/defect/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pcb3/defect/origin/' --weight_idx 9999 --sample_name='samples-matching/pcb3/' --init_word broken --anomaly_name='defect' --pt_path='logs/pcb3_defect/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pcb3/defect/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pcb4/defect/origin/' --weight_idx 9999 --sample_name='samples-matching/pcb4/' --init_word broken --anomaly_name='defect' --pt_path='logs/pcb4_defect/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pcb4/defect/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/color/origin/' --weight_idx 9999 --sample_name='samples-matching/pill/' --init_word broken --anomaly_name='color' --pt_path='logs/pill_color/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/color/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/combined/origin/' --weight_idx 9999 --sample_name='samples-matching/pill/' --init_word broken --anomaly_name='combined' --pt_path='logs/pill_combined/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/combined/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/contamination/origin/' --weight_idx 9999 --sample_name='samples-matching/pill/' --init_word broken --anomaly_name='contamination' --pt_path='logs/pill_contamination/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/contamination/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/crack/origin/' --weight_idx 9999 --sample_name='samples-matching/pill/' --init_word broken --anomaly_name='crack' --pt_path='logs/pill_crack/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/crack/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/faulty_imprint/origin/' --weight_idx 9999 --sample_name='samples-matching/pill/' --init_word broken --anomaly_name='faulty_imprint' --pt_path='logs/pill_faulty_imprint/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/faulty_imprint/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/pill_type/origin/' --weight_idx 9999 --sample_name='samples-matching/pill/' --init_word broken --anomaly_name='pill_type' --pt_path='logs/pill_pill_type/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/pill_type/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/scratch/origin/' --weight_idx 9999 --sample_name='samples-matching/pill/' --init_word broken --anomaly_name='scratch' --pt_path='logs/pill_scratch/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pill/scratch/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pipe_fryum/defect/origin/' --weight_idx 9999 --sample_name='samples-matching/pipe_fryum/' --init_word broken --anomaly_name='defect' --pt_path='logs/pipe_fryum_defect/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/pipe_fryum/defect/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw/manipulated_front/origin/' --weight_idx 9999 --sample_name='samples-matching/screw/' --init_word broken --anomaly_name='manipulated_front' --pt_path='logs/screw_manipulated_front/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw/manipulated_front/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw/scratch_head/origin/' --weight_idx 9999 --sample_name='samples-matching/screw/' --init_word broken --anomaly_name='scratch_head' --pt_path='logs/screw_scratch_head/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw/scratch_head/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw/scratch_neck/origin/' --weight_idx 9999 --sample_name='samples-matching/screw/' --init_word broken --anomaly_name='scratch_neck' --pt_path='logs/screw_scratch_neck/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw/scratch_neck/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw/thread_side/origin/' --weight_idx 9999 --sample_name='samples-matching/screw/' --init_word broken --anomaly_name='thread_side' --pt_path='logs/screw_thread_side/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw/thread_side/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw/thread_top/origin/' --weight_idx 9999 --sample_name='samples-matching/screw/' --init_word broken --anomaly_name='thread_top' --pt_path='logs/screw_thread_top/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw/thread_top/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw_single/manipulated_front/origin/' --weight_idx 9999 --sample_name='samples-matching/screw_single/' --init_word broken --anomaly_name='manipulated_front' --pt_path='logs/screw_single_manipulated_front/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw_single/manipulated_front/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw_single/scratch_head/origin/' --weight_idx 9999 --sample_name='samples-matching/screw_single/' --init_word broken --anomaly_name='scratch_head' --pt_path='logs/screw_single_scratch_head/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw_single/scratch_head/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw_single/scratch_neck/origin/' --weight_idx 9999 --sample_name='samples-matching/screw_single/' --init_word broken --anomaly_name='scratch_neck' --pt_path='logs/screw_single_scratch_neck/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw_single/scratch_neck/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw_single/thread_side/origin/' --weight_idx 9999 --sample_name='samples-matching/screw_single/' --init_word broken --anomaly_name='thread_side' --pt_path='logs/screw_single_thread_side/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw_single/thread_side/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw_single/thread_top/origin/' --weight_idx 9999 --sample_name='samples-matching/screw_single/' --init_word broken --anomaly_name='thread_top' --pt_path='logs/screw_single_thread_top/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/screw_single/thread_top/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/toothbrush/defective/origin/' --weight_idx 9999 --sample_name='samples-matching/toothbrush/' --init_word broken --anomaly_name='defective' --pt_path='logs/toothbrush_defective/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/toothbrush/defective/fg/' --matching


CUDA_VISIBLE_DEVICES=5 python generate_with_mask_mvtec.py --data_root='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/tubes/anomalous/origin/' --weight_idx 9999 --sample_name='samples-matching/tubes/' --init_word broken --anomaly_name='anomalous' --pt_path='logs/tubes_anomalous/checkpoints' --mask_path='/data1/gpt/jtj/AligenMask/DualAnoDiff/dual-interrelated_diff/generate_data-60%/tubes/anomalous/fg/' --matching

