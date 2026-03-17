import torchvision
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from tqdm import tqdm

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

#######################指定某个点（可选）##################################
def generate_image_with_point(image, save_path):
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.savefig('add_point.png')

#######################生成Mask（彩色）##################################
def generate_colorful_mask(image, save_path):
    masks = mask_generator.generate(image)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig('result.png')

#######################生成Mask（黑白）##################################
def generate_gray_mask(image, save_path):
    masks = mask_generator.generate(image)
    index = 0
    for mask in masks:
        m = mask['segmentation']
        m = m + 255
        m_save_path = os.path.join(save_path, f'{index}.png')
        plt.imsave(m_save_path,m,cmap='gray')
        index = index + 1

#######################生成Mask（仅保留分割的物体）##################################
def generate_only_object(image, save_path):
    save_path = ''
    masks = mask_generator.generate(image)
    index = 0
    for mask in masks:
        m = mask['segmentation']
        m = ~m
        m = m + 255
        m = np.repeat(m[:, :, np.newaxis], 3, axis=2)
        m = m.astype(np.uint8)
        res = cv2.bitwise_and(image, m)
        res[res == 0] = 255
        plt.imshow(res)
        plt.savefig(f'{index}.png')
        index = index + 1

#######################加载输入和SAM##################################

sam_checkpoint = "sam_vit_h_4b8939.pth"     #改为已下载的模型的存放路径
device = "cuda"     #默认是cuda，如果是用cpu的话就改为cpu
model_type = "default"      #default默认代表的是vit_h模型，可将其改为自己下载的模型名称（vit_h/vit_l/vit_b）
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

dataset = 'MSRS'
# use vi or ir (vi ==> background ir ==> frontground)
vi_path = '../Dataset/MSRS/test/vi'
ir_path = '../Dataset/MSRS/test/ir'
select_path = ir_path
save_path = './mask'
dataset_path = os.path.join(save_path, dataset)
os.makedirs(dataset_path, exist_ok=True)
for filename in tqdm(os.listdir(select_path)):
    file_path = os.path.join(select_path, filename)
    file_save_path = os.path.join(dataset_path, filename)
    os.makedirs(file_save_path, exist_ok=True)

    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    generate_gray_mask(image, file_save_path)
 





