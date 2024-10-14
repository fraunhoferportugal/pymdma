#!/bin/bash

python3 src/main.py \
    --modality time_series \
    --validation_type synth \
    --evaluation_level dataset \
    --reference_type dataset \
    --target_data data/test/time_series/synthesis_val/dataset \
    --reference_data data/test/time_series/synthesis_val/reference \
    --batch_size 2\
    --output_dir reports/tabular_metrics/ \
    --metric_group feature

# python3 src/main.py \
    #     --modality time_series \
    #     --validation_type synth \
    #     --evaluation_level dataset \
    #     --reference_type dataset \
    #     --reference_data data/test/time_series/synthesis_val/reference \
    #     --target_data data/test/time_series/input_val/dataset \
    #     --batch_size 10 \
    #     --output_dir reports/time_series/ \
    #     --metric_group feature
