#!/usr/bin/env sh
# SAM-DD 6-fold cross-validation training script (Single GPU version)
# - Uses main_fold_rewrite.py (modified for single GPU training)
# - Writes per-fold checkpoints/logs under: ${dir}/fold_00, ${dir}/fold_01, ...

set -e

now=$(date +"%Y%m%d_%H%M%S")
dir="output_dir/dmd/16f_05131427"
#dir="output_dir/samdd/clip_vit_base_patch16_multimodal_adapter12x384_cv6"
mkdir -p "${dir}"

# GPU selection
export CUDA_VISIBLE_DEVICES="1"

# Optional: run a single fold by setting fold_id to 0..(N_FOLDS-1)
# fold_id=-1 means run all folds
fold_id=-1

# Single GPU training - no torchrun needed
python ../main_fold_floaps_metric.py \
    --model clip_vit_base_patch16_multimodal_adapter12x384 \
    --save_dir "${dir}" \
    --auto_remove \
    --dataset dmd \
    --fold "${fold_id}" \
    --num_frames 16 \
    --sampling_rate 0 \
    --num_spatial_views 1 \
    --num_temporal_views 1 \
    --resize_type random_resized_crop \
    --scale_range 0.08 1.0 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --batch_size 8 \
    --epochs 50 \
    --warmup_epochs 2 \
    --eval_freq 2 \
    --label_csv "../lists/dmd/dmd_labels.csv" \
    --mlm_label "../lists/dmd/dmd_mlm_labels.txt" \
    2>&1 | tee "${dir}/train.log"