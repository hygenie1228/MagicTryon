VIDEO_DIR="data/4ddress/00122_Inner/Take2"
MASK_PART="lower_body"
OUT_DIR="data/4ddress_processed/$(echo "${VIDEO_DIR}" | sed 's|data/4ddress/||' | tr '/' '_')_${MASK_PART}"

export CUDA_VISIBLE_DEVICES=2

python inference/customize/video2image.py --input "${VIDEO_DIR}/video.mp4" --output "${OUT_DIR}/images" --part "${MASK_PART}"

# get masks
python inference/customize/gen_mask/app_mask.py --input ${OUT_DIR}/images --output ${OUT_DIR}/masks --part ${MASK_PART}
