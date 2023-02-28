#!/bin/bash

#$ -l rt_G.large=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd

# rt_G.large=1
# rt_AF=1

source /etc/profile.d/modules.sh
module load singularitypro/3.9

# export OMP_NUM_THREADS=1

# vt 
# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/train_clevr_ablation_vf.py \
#         --learning_rate 2e-5 --save_name clevr_ablation_vt_2e5_dldhsa_wm_n1_0 \
#         --tgt clevr --dynamic_layers --dynamic_head --split_args \
#         --num_epochs 30 \
#         --seed 0

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/train_clevr_ablation_vf.py \
#         --learning_rate 2e-5 --save_name clevr_ablation_vt_2e5_dldhsa_wm_n1_0 \
#         --tgt clevr --dynamic_layers --dynamic_head --split_args \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/neurips22/clevr_ablation_vt_2e5_dldhsa_wm_n1_0/pytorch_model_29.bin \
#         --start_epoch 30 --num_epochs 40 \
#         --seed 0

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/train_clevr_ablation_vf.py \
#         --learning_rate 2e-5 --save_name clevr_ablation_vt_2e5_dldh_wm_n1_0 \
#         --num_epochs 30 \
#         --tgt clevr --dynamic_layers --dynamic_head \
#         --seed 0

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/train_clevr_ablation_vf.py \
#         --learning_rate 4e-5 --save_name clevr_ablation_vt_4e5_dldh_wm_f_0 \
#         --tgt clevr --dynamic_layers --dynamic_head \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/neurips22/clevr_ablation_vt_4e5_dldh_wm_f_0/pytorch_model_16.bin \
#         --start_epoch 17 --num_epochs 30
#         --seed 0

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/train_clevr_ablation_vf.py \
#         --learning_rate 2e-5 --save_name clevr_ablation_vt_2e5_dlsa_wm_n1_2 \
#         --num_epochs 30 \
#         --tgt clevr --dynamic_layers --split_args \
#         --seed 2

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/train_clevr_ablation_vf.py \
#         --learning_rate 2e-5 --save_name clevr_ablation_vt_2e5_dlsa_wm_n1_0 \
#         --tgt clevr --dynamic_layers --split_args \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/neurips22/clevr_ablation_vt_2e5_dlsa_wm_n1_0/pytorch_model_29.bin \
#         --start_epoch 30 --num_epochs 40 \
#         --seed 0

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/train_clevr_ablation_vf.py \
#         --learning_rate 2e-5 --save_name clevr_ablation_vt_2e5_dl_wm_n1_0 \
#         --num_epochs 30 \
#         --tgt clevr --dynamic_layers \
#         --seed 0

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/train_clevr_ablation_vf.py \
#         --learning_rate 4e-5 --save_name clevr_ablation_vt_4e5_dl_wm_f_0 \
#         --tgt clevr --dynamic_layers \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/neurips22/clevr_ablation_vt_4e5_dl_wm_f_0/pytorch_model_17.bin \
#         --start_epoch 18 --num_epochs 30 \
#         --seed 0

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/train_clevr_ablation_vf.py \
#         --learning_rate 2e-5 --save_name clevr_ablation_vt_2e5_dh_wm_n1_0 \
#         --num_epochs 30 \
#         --tgt clevr --dynamic_head \
#         --seed 0

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/train_clevr_ablation_vf.py \
#         --learning_rate 4e-5 --save_name clevr_ablation_vt_4e5_dh_wm_f_0 \
#         --tgt clevr --dynamic_head \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/neurips22/clevr_ablation_vt_4e5_dh_wm_f_0/pytorch_model_16.bin \
#         --start_epoch 17 --num_epochs 30 \
#         --seed 0

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/train_clevr_ablation_vf.py \
#         --learning_rate 2e-5 --save_name clevr_ablation_vt_2e5_wm_n1_0 \
#         --num_epochs 30 \
#         --tgt clevr \
#         --seed 0

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/train_clevr_ablation_vf.py \
#         --learning_rate 4e-5 --save_name clevr_ablation_vt_4e5_wm_f_0 \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/neurips22/clevr_ablation_vt_4e5_wm_f_0/pytorch_model_29.bin \
#         --tgt clevr \
#         --start_epoch 30 --num_epochs 40 \
#         --seed 0



# eval

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_clevr_base.py \
#         --tgt valA --from_pretrained /groups/gcb50257/results/transformer_clevr/ijcai22/clevr_cgt_std_4e5_12l_2/pytorch_model_19.bin

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_clevr_base_vf.py \
#         --tgt valA --vf region --from_pretrained /groups/gcb50257/results/transformer_clevr/ijcai22/clevr_cgt_std_ext_4e5_12l_2/pytorch_model_19.bin

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_clevr_base_vf.py \
#         --tgt valB --vf vt --dump \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/ijcai22/clevr_cgt_vt_2e5_12l_2/pytorch_model_19.bin

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_clevr_base_pg.py \
#         --tgt valA --from_pretrained /groups/gcb50257/results/transformer_clevr/ijcai22/clevr_cgt_pg_4e5_12l_2/pytorch_model_19.bin

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_clevr_base_pg_vf.py \
#         --tgt valA --vf region --from_pretrained /groups/gcb50257/results/transformer_clevr/ijcai22/clevr_cgt_pg_ext_4e5_12l_2/pytorch_model_19.bin

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_clevr_base_pg_vf.py \
#         --tgt valB --vf vt --dump \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/ijcai22/clevr_cgt_pg_vt_2e5_12l_0/pytorch_model_19.bin

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_closure_ablation_vf.py \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/neurips22/clevr_ablation_vt_2e5_dldhsa_wm_n1_0/pytorch_model_29.bin \
#         --dynamic_layers --dynamic_head --split_args

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_closure_base_pg_vf.py \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/ijcai22/clevr_pg_vt_2e5_12l_45t_0/pytorch_model_19.bin

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_closure_base_pg.py \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/ijcai22/clevr_pg_4e5_12l_0/pytorch_model_19.bin


# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_gqa_base.py \
#         --test ood --dump \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/ijcai22/gqa_std_1e4_12l_trainval_2/pytorch_model_19.bin

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_gqa_base.py \
#         --test ind --dump \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/ijcai22/gqa_std_1e4_12l_trainval_2/pytorch_model_19.bin


# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_gqa_base_pg.py \
#         --test ood \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/ijcai22/gqa_pg_1e4_12l_trainval_4/pytorch_model_19.bin

# singularity exec --nv -B /groups/gcb50257 ~/containers/modular-tf.sif \
#     python3 ~/dev/modular_transformer/eval_gqa_base_pg.py \
#         --test ood --dump \
#         --from_pretrained /groups/gcb50257/results/transformer_clevr/ijcai22/gqa_pg_1e4_12l_trainval_0/pytorch_model_19.bin

echo "finish"
