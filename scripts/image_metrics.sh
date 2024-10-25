#!/bin/bash
#SBATCH --job-name=sd20_eval
#SBATCH --output=logs/sd20_eval.log
#SBATCH --partition=gpu32
#SBATCH --mem=32G
#SBATCH --time=8-24:00:00
#SBATCH --cpus-per-task=40
#SBATCH --gpus=1
#SBATCH --mail-user=ivo.facoco@fraunhofer.pt
#SBATCH --mail-type=ALL


pymdma --modality image \
    --validation_type synth \
    --reference_type dataset \
    --evaluation_level dataset \
    --reference_data "$HOME/ldm_dataset_test.jsonl" \
    --target_data "$HOME/cdiffusion_dretinopathy/reports/figures/sd20_da_concepts" \
    --batch_size 30 \
    --metric_group feature \
    --output_dir reports/sd20_da_concepts_vit32/ \
    --device cuda
    # --extractor_model_name inception_v3

echo Done.

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