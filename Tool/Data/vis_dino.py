from model.Dino_Clip import DINOiser
from hydra import compose, initialize
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_frequency_domain_tensor(image):
        if image is None:
            raise ValueError("input image should be None")
        if len(image.shape) != 4 or image.shape[1] != 1:
            raise ValueError(f"should be [b, 1, h, w], but get {image.shape}")
    
        f = torch.fft.fft2(image, dim=(-2, -1))
        fshift = torch.fft.fftshift(f, dim=(-2, -1))
        magnitude_spectrum = torch.abs(fshift)
    
        # Add a small epsilon to avoid log(0)
        magnitude_spectrum = torch.log(magnitude_spectrum + 1e-8)
        return magnitude_spectrum
    
def get_rgb_frequency_domain_tensor(image):
    if image is None:
        raise ValueError("input image should be None")
    
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    
    (b, g, r) = image.split()
    b_tensor, g_tensor, r_tensor = to_tensor(b).unsqueeze(0), to_tensor(g).unsqueeze(0), to_tensor(r).unsqueeze(0)

    mag_image_b = get_frequency_domain_tensor(b_tensor)
    mag_image_g = get_frequency_domain_tensor(g_tensor)
    mag_image_r = get_frequency_domain_tensor(r_tensor)
    
    mag_image = torch.concat([mag_image_b, mag_image_g, mag_image_r],dim=1)
    return mag_image.squeeze(0)

def show_channel_mean(feature_map):
    """
    显示所有通道的平均特征图
    """
    plt.figure(figsize=(12, 9))
    
    # 处理PyTorch张量
    if isinstance(feature_map, torch.Tensor):
        # 使用PyTorch的mean函数，dim=0表示对通道维度取平均
        mean_feat = feature_map.mean(dim=0)  # 对第一个维度（通道维度）取平均
        mean_feat = mean_feat.detach().cpu().numpy()
    else:
        # 如果是numpy数组，使用axis=0
        mean_feat = np.mean(feature_map, axis=0)  # 对所有通道取平均
    
    plt.imshow(mean_feat, cmap='viridis')
    plt.axis('off')
    plt.savefig('./dino_freq.png')


initialize(config_path="configs", version_base=None)            
cfg = compose(config_name="clip_dinoiser.yaml")

image_path = './00103D_extreme_Haze.png'
img = Image.open(image_path).convert('RGB')
freq_tensor = get_rgb_frequency_domain_tensor(img)
freq_tensor = freq_tensor.unsqueeze(0).cuda()
model_ex = DINOiser(cfg.model).to(device)
model_ex.load_dino()
model_ex.to(device)

feature = model_ex(freq_tensor)
feature = feature.squeeze()
show_channel_mean(feature)

