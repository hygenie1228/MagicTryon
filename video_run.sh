#!/usr/bin/env bash
BASE_IN="data/4ddress_processed"
BASE_OUT="results"
export CUDA_VISIBLE_DEVICES=4

for dir in "${BASE_IN}"/*/; do
  name=$(basename "$dir")
  IN_DIR="${BASE_IN}/${name}"
  OUTPUT_DIR="${BASE_OUT}"
  OUTPUT_VIDEO="${OUTPUT_DIR}/sanity_check/${name}/${name}_cloth.mp4"
  if [ -f "${OUTPUT_VIDEO}" ]; then
    echo "Skipping ${name}: ${OUTPUT_VIDEO} already exists"
    continue
  fi
  echo "Running: ${IN_DIR} -> ${OUTPUT_DIR}"
  python inference/video_tryon/predict_video_tryon_customize.py --input_dir "${IN_DIR}" --output_dir "${OUTPUT_DIR}"
done