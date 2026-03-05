#!/usr/bin/env bash
BASE_IN="data/4ddress_processed"
BASE_OUT="results"
export CUDA_VISIBLE_DEVICES=4

dirs=("${BASE_IN}"/*/)
for ((i = 1; i < ${#dirs[@]}; i += 2)); do
  dir="${dirs[i]}"
  name=$(basename "$dir")
  IN_DIR="${BASE_IN}/${name}"
  OUTPUT_DIR="${BASE_OUT}/${name}"
  echo "Running: ${IN_DIR} -> ${OUTPUT_DIR}"
  python inference/video_tryon/predict_video_tryon_customize.py --input_dir "${IN_DIR}" --output_dir "${OUTPUT_DIR}"
done