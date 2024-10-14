#!/bin/bash


pymdma --modality image \
    --validation_type synth \
    --reference_type dataset \
    --evaluation_level dataset \
    --reference_data data/test/image/synthesis_val/reference \
    --target_data data/test/image/synthesis_val/dataset \
    --batch_size 3\
    --metric_group feature \
    --output_dir reports/image_metrics/ \
    # --extractor_model_name inception_v3

# python3 src/main.py \
    #     --modality image \
    #     --validation_type input \
    #     --reference_type none \
    #     --evaluation_level instance \
    #     --target_data data/test/image/input_val/dataset \
    #     --reference_data data/test/image/input_val/reference \
    #     --batch_size 3\
    #     --output_dir reports/image_metrics/ \
    #     --metric_group quality \
    #     --annotation_file data/test/image/input_val/annotations/COCO_annotation_example_mask_exp.json
# --extractor_model_name inception
# --reference_data data/test/image/synthesis_val/reference \
