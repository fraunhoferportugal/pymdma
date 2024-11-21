#!/bin/bash

pymdma --modality time_series \
    --validation_domain synth \
    --evaluation_level dataset \
    --reference_type dataset \
    --target_data data/test/time_series/synthesis_val/dataset \
    --reference_data data/test/time_series/synthesis_val/reference \
    --batch_size 2\
    --output_dir reports/time_series/

# pymdma --modality time_series \
#     --validation_domain input \
#     --evaluation_level instance \
#     --reference_type none \
#     --target_data data/test/time_series/input_val/dataset \
#     --output_dir reports/time_series/
