import cv2
import os
import numpy as np

def read_path(vis_path, save_path):
    for filename in os.listdir(vis_path):
        print(f'process {filename}')
        img = cv2.imread(vis_path +'/'+filename)
        image = np.power(img, 0.8)			# 对像素值指数变换
        cv2.imwrite(save_path + "/" + filename, image)
vis_path = './Visible_haze'
save_path = './Visible_haze_lowlight'
read_path(vis_path, save_path)
