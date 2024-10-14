#!/bin/bash

python3 src/main.py \
    --modality image \
    --validation_type input \
    --reference_type none \
    --evaluation_level instance \
    --target_data data/raw/ldm_dataset_new.jsonl \
    --batch_size 50 \
    --output_dir reports/image_metrics/ \
    --metric_group quality \
    --specific_metrics CLIPIQA Brightness Colorfulness Tenengrad EME
