#!/bin/bash

pymdma --modality tabular \
    --validation_domain input \
    --evaluation_level dataset \
    --reference_type none \
    --target_data data/test/tabular/input_val/dataset \
    --batch_size 1\
    --output_dir reports/tabular_metrics/ \

pymdma --modality tabular \
    --validation_domain synth \
    --evaluation_level dataset \
    --reference_type dataset \
    --reference_data data/test/tabular/synthesis_val/reference \
    --target_data data/test/tabular/synthesis_val/dataset \
    --batch_size 1\
    --output_dir reports/tabular_metrics/ \
