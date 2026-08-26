import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def process_masks(mask_dataset_path, save_path):
    masks_file_list = os.listdir(mask_dataset_path)

    for mask_file in tqdm(masks_file_list):
        mask_file_path = os.path.join(mask_dataset_path, mask_file)

        png_files = [f for f in os.listdir(mask_file_path) if f.endswith('.png')]
        if not png_files:
            continue

        first_img = Image.open(os.path.join(mask_file_path, png_files[0])).convert('RGB')
        h, w = first_img.size[1], first_img.size[0]
        accum_mask = np.zeros((h, w), dtype=bool)

        for png_file in png_files:
            img_path = os.path.join(mask_file_path, png_file)
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            white_mask = np.all(img_array == 255, axis=-1)
            accum_mask |= white_mask

        output_array = np.stack([accum_mask * 255] * 3, axis=-1).astype(np.uint8)
        output_img = Image.fromarray(output_array, mode='RGB')

        save_path_ = os.path.join(save_path, mask_file)
        output_img.save(save_path_)

def process_masks_color(mask_dataset_path, save_path):
    masks_file_list = os.listdir(mask_dataset_path)

    for mask_file in tqdm(masks_file_list):
        mask_file_path = os.path.join(mask_dataset_path, mask_file)

        png_files = [f for f in os.listdir(mask_file_path) if f.endswith('.png')]
        if not png_files:
            continue

        first_img = Image.open(os.path.join(mask_file_path, png_files[0])).convert('RGB')
        h, w = first_img.size[1], first_img.size[0]
        accum_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for png_file in png_files:
            img_path = os.path.join(mask_file_path, png_file)
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            non_white = np.any(img_array != 255, axis=-1)
            accum_mask[non_white] = img_array[non_white]

        output_img = Image.fromarray(accum_mask, mode='RGB')
        save_path_ = os.path.join(save_path, mask_file)
        output_img.save(save_path_)


if __name__ == "__main__":
    dataset = 'RM3DV'
    mask_path = "./mask/"
    mask_dataset_path = os.path.join(mask_path, dataset)
    save_dir = './processed_mask/'
    save_path = os.path.join(save_dir, dataset)
    os.makedirs(save_path, exist_ok=True)
    process_masks_color(mask_dataset_path, save_path)