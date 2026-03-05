#!/usr/bin/env bash
# Run video_preprocess pipeline for all data/4ddress/<subject>/Take* directories.
# Runs for both MASK_PART=upper_body and lower_body.
# Usage: ./video_preprocess_all.sh
#        SKIP_EXISTING=1 ./video_preprocess_all.sh   # skip if OUT_DIR already has cloth.png

set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

BASE="data/4ddress"
# for MASK_PART in upper_body lower_body; do
for MASK_PART in upper_body; do
  echo "========== MASK_PART=${MASK_PART} =========="
  for subject_dir in "${BASE}"/*/; do
  subject=$(basename "$subject_dir")
  for video_dir in "${subject_dir}"Take*/; do
    [ -d "$video_dir" ] || continue
    VIDEO_DIR="${video_dir%/}"
    if [ ! -f "${VIDEO_DIR}/video.mp4" ]; then
      echo "Skipping ${VIDEO_DIR}: no video.mp4"
      continue
    fi
    # Same OUT_DIR logic as video_preprocess.sh
    rel="${VIDEO_DIR#${BASE}/}"
    OUT_DIR="data/4ddress_processed/$(echo "$rel" | tr '/' '_')_${MASK_PART}"
    if [ -n "${SKIP_EXISTING:-}" ] && [ -f "${OUT_DIR}/cloth.png" ]; then
      echo "Skipping ${VIDEO_DIR}: ${OUT_DIR} already exists"
      continue
    fi
    if [ -f "${OUT_DIR}/cloth_anilines.png" ]; then
      echo "Skipping ${VIDEO_DIR}: ${OUT_DIR}/cloth_anilines.png already exists"
      continue
    fi
    echo "=== Processing: ${VIDEO_DIR} -> ${OUT_DIR} ==="

    # video into image
    python inference/customize/video2image.py --input "${VIDEO_DIR}/video.mp4" --output "${OUT_DIR}/images" --part "${MASK_PART}"

    # get agnostic
    python inference/customize/get_masked_person.py --image "${OUT_DIR}/images" --mask "${OUT_DIR}/masks" --output "${OUT_DIR}/agnostic"

    # get densepose
    bash inference/customize/detectron2/projects/DensePose/run.sh "${OUT_DIR}"

    # image into video
    python inference/customize/image2video.py --input "${OUT_DIR}"

    # garment caption from cloth.png
    python inference/customize/get_garment_caption.py --image "${OUT_DIR}/cloth.png" --output "${OUT_DIR}/cloth_caption.json"

    # AniLines on cloth image
    python inference/customize/AniLines/infer.py --dir_in "${OUT_DIR}/cloth.png" --dir_out "${OUT_DIR}/" --mode detail --binarize -1 --fp16 True --device cuda:0
  done
  done
done

echo "Done processing all 4ddress videos (upper_body and lower_body)."
