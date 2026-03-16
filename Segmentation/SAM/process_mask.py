# coding:gbk

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from PIL import ImageColor
def process_masks(img_path, save_path):

    masks_file_list = os.listdir(img_path)

    for mask_file in masks_file_list:
        mask_file_path = os.path.join(img_path, mask_file)
        mask_list = os.listdir(mask_file_path)

        png_num = len(mask_list)-1
        img_list = [] 
        color_list = []
        
        img = np.array(Image.open(os.path.join(mask_file_path,mask_list[0])))  
        img_list.append(img)
        image_name = mask_file + ".png"
        save_path_ = os.path.join(save_path, mask_file)
        os.makedirs(save_path_, exist_ok=True)
        save_path_ = os.path.join(save_path_, image_name)
        new_img = np.zeros_like(img_list[0])
        Image.fromarray(new_img).save(save_path_)
        image_pil = Image.open(save_path_)
        image_pil = image_pil.convert("RGB")
        
        for root,dirs,files in os.walk(mask_file_path):
            for image in files:  
                print(image)
                if image.endswith(".png"): 
                    image_path =  os.path.join(mask_file_path, image)
                    img = Image.open(image_path)
                    img_rgb = img.convert("RGB")
                    for i in range(img.size[0]):
                        for j in range(img.size[1]):
                            if img_rgb.getpixel((i,j)) == (255,255,255):
                                image_pil.putpixel((i,j),(255,255,255)) 

        image_pil.convert("RGB")
        image_pil.save(save_path_)
        


img_path ="./mask/" 
save_path = './processed_mask/'

process_masks(img_path, save_path)