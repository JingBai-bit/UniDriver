#!/usr/bin/env sh
# Cross-validation training script (single GPU)
# - Writes checkpoints, metrics, and console logs under logs/

set -e

now=$(date +"%Y%m%d_%H%M%S")
dir="logs/samdd/${now}"
mkdir -p "${dir}"

# GPU selection
export CUDA_VISIBLE_DEVICES="0"

# Optional: run a single fold by setting fold_id to 0..(N_FOLDS-1)
# fold_id=-1 means run all folds
fold_id=-1

# Single GPU training - no torchrun needed
python main.py \
    --model clip_vit_base_patch16_multimodal_adapter12x384 \
    --save_dir "${dir}" \
    --auto_remove \
    --dataset samdd \
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
    --label_csv "lists/samdd/samdd_labels.csv" \
    --mlm_label "lists/samdd/samdd_mlm_labels.txt" \
    2>&1 | tee "${dir}/console.log"
