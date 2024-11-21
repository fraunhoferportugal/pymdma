#!/bin/bash

python3 src/main.py \
    --modality text \
    --validation_domain input \
    --evaluation_level instance \
    --reference_type none \
    --target_data data/test/text/input_val/dataset \
    --batch_size 1\
    --output_dir reports/text_metrics/ \
    --metric_category privacy

# python3 src/main.py \
    #     --modality text \
    #     --validation_domain synth \
    #     --evaluation_level dataset \
    #     --reference_type none \
    #     --target_data data/test/text/input_val/dataset \
    #     --batch_size 1\
    #     --output_dir reports/text_metrics/ \
    #     --metric_category feature
