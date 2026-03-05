#!/usr/bin/env bash
VIDEO_DIR="data/4ddress/00122_Inner/Take2"
MASK_PART="upper_body"
OUT_DIR="data/4ddress_processed/$(echo "${VIDEO_DIR}" | sed 's|data/4ddress/||' | tr '/' '_')_${MASK_PART}"

export CUDA_VISIBLE_DEVICES=4

# # video into image
# python inference/customize/video2image.py --input "${VIDEO_DIR}/video.mp4" --output "${OUT_DIR}/images" --part "${MASK_PART}"

# # get densepose
# bash inference/customize/detectron2/projects/DensePose/run.sh ${OUT_DIR}

# # image into video
# python inference/customize/image2video.py --input ${OUT_DIR}


# # garment caption from cloth.png
# python inference/customize/get_garment_caption.py --image "${OUT_DIR}/cloth.png" --output "${OUT_DIR}/cloth_caption.json"

# AniLines on cloth image (output to OUT_DIR/anilines to avoid overwriting cloth.png)
python inference/customize/AniLines/infer.py --dir_in "${OUT_DIR}/cloth.png" --dir_out "${OUT_DIR}/" --mode detail --binarize -1 --fp16 True --device cuda:0