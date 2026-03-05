#!/bin/bash

# 定义要处理的文件夹编号列表
FOLDER_IDS=(1)

# 模型配置与权重
# Path from repo root (when run via video_preprocess.sh) or use configs/ when run from DensePose dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/configs/densepose_rcnn_R_101_FPN_s1x.yaml"
MODEL_URL="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl"

# When called from video_preprocess.sh: first arg is OUT_DIR (e.g. data/4ddress_processed/00122_Inner_Take2_upper_body)
if [[ -n "${1:-}" ]]; then
  INPUT_DIR="$1/images"
  OUTPUT_DIR="$1/"
  echo "Processing: $INPUT_DIR -> $OUTPUT_DIR"
  python inference/customize/detectron2/projects/DensePose/apply_net.py show "$CONFIG" "$MODEL_URL" "$INPUT_DIR" "$OUTPUT_DIR" dp_segm -v
else
  # Legacy: use hardcoded base path and folder list
  BASE_PATH="datasets/person/customize/video"
  for ID in "${FOLDER_IDS[@]}"; do
    FOLDER_ID=$(printf "%05d" "$ID")
    INPUT_DIR="${BASE_PATH}/${FOLDER_ID}/images"
    OUTPUT_DIR="${BASE_PATH}/${FOLDER_ID}/"
    echo "Processing folder: $FOLDER_ID"
    python inference/customize/detectron2/projects/DensePose/apply_net.py show "$CONFIG" "$MODEL_URL" "$INPUT_DIR" "$OUTPUT_DIR" dp_segm -v
  done
fi
