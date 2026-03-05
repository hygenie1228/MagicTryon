import cv2
import os
import argparse
import numpy as np

def video_to_images(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    os.system(f"cp {video_path} {output_dir}/..")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f"{frame_idx:04d}.png")
        
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"视频处理完成，共保存了 {frame_idx} 帧图像到 {output_dir}")

# 示例调用
if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/home/namhj/X_nfs/dataset/aa/videos/B15_-__Walk_turn_around_stageii/B15_-__Walk_turn_around_stageii.mp4')
    parser.add_argument('--part', default="upper_body", choices=["upper_body", "lower_body"])
    parser.add_argument('--output', default='images')
    args = parser.parse_args()

    # if os.path.exists(args.output + "/../agnostics"):
    #     assert 0, "Output directory already exists"

    video_to_images(args.input, args.output)

    ## Masks
    masks = np.load(args.input[:-9] + "mask.npz")
    masks = masks['data']

    if args.part == "upper_body":
        if 'Outer' in args.input:
            part = 6
        else:
            part = 4
    elif args.part == "lower_body":
        part = 5

    masks = (masks[:,:,:,0] == part) * 255


    # os.makedirs(args.output + "/../mask", exist_ok=True)
    # for frame_idx, mask in enumerate(masks):
    #     cv2.imwrite(args.output + f"/../mask/{frame_idx:04d}.png", mask)


    ## Agnostic
    agnostics = np.load(args.input[:-9] + "agnostic.npz")
    agnostics = agnostics['data']
    agnostics = (agnostics[:,:,:,0] == 1) * 255

    os.makedirs(args.output + "/../masks", exist_ok=True)
    for frame_idx, mask in enumerate(agnostics):
        cv2.imwrite(args.output + f"/../masks/{frame_idx:04d}.png", mask)

    image = cv2.imread(args.output + "/0000.png")
    mask0 = masks[0]
    cloth_image = image * (mask0[:, :, None] > 128) + (255) * (mask0[:, :, None] <= 128)

    # Zoom to mask bounding box with 20% margin
    ys, xs = np.where(mask0 > 128)
    if len(ys) > 0 and len(xs) > 0:
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        w, h = x_max - x_min, y_max - y_min
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        x_min = max(0, x_min - margin_x)
        x_max = min(cloth_image.shape[1], x_max + margin_x)
        y_min = max(0, y_min - margin_y)
        y_max = min(cloth_image.shape[0], y_max + margin_y)
        cloth_image = cloth_image[y_min:y_max, x_min:x_max]

    # Resize to fit 1024x768 keeping aspect ratio, then place on white canvas
    target_w, target_h = 768, 1024
    crop_h, crop_w = cloth_image.shape[0], cloth_image.shape[1]
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    cloth_resized = cv2.resize(cloth_image.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = cloth_resized
    cloth_image = canvas

    cv2.imwrite(args.output + f"/../cloth.png", cloth_image)
    
    
    
    
    

    
