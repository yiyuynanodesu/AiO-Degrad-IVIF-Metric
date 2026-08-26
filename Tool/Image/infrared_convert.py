import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

infrared_dir = './RM3DV'

def apply_heatmap(filename, video_path, save_path, colormap=cv2.COLORMAP_JET, is_infrared=True):
    image_path = os.path.join(video_path, filename)
    save_path = os.path.join(save_path, filename)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {filename}")
    
    gray = img
    
    if gray.dtype != np.uint8:
        gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        gray_norm = gray
    
    heatmap_bgr = cv2.applyColorMap(gray_norm, colormap)
    cv2.imwrite(save_path,heatmap_bgr)

# ==================== 使用示例 ====================
if __name__ == "__main__":
    for degrad in os.listdir(infrared_dir):
        degrad_path = os.path.join(infrared_dir, degrad)
        ir_path = os.path.join(degrad_path, 'ir')
        save_path = os.path.join(degrad, 'new_ir')
        for video in os.listdir(ir_path):
            video_path = os.path.join(ir_path, video)
            save_video_path = os.path.join(save_path, video)
            os.makedirs(save_video_path, exist_ok=True)
            for file in os.listdir(video_path):
                apply_heatmap(file, video_path, save_video_path, colormap=cv2.COLORMAP_JET, is_infrared=True)