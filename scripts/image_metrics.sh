#!/bin/bash

pymdma \
        --modality image \
        --validation_type input \
        --reference_type none \
        --evaluation_level instance \
        --target_data data/test/image/input_val/dataset \
        --reference_data data/test/image/input_val/reference \
        --batch_size 3\
        --output_dir reports/image_metrics/ \
        --annotation_file data/test/image/input_val/annotations/COCO_annotation_example_mask_exp.json

pymdma \
        --modality image \
        --validation_type synth \
        --reference_type dataset \
        --evaluation_level dataset \
        --target_data data/test/image/synthesis_val/dataset \
        --reference_data data/test/image/synthesis_val/reference \
        --batch_size 3\
        --output_dir reports/image_metrics/ \
