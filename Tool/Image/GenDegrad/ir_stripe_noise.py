import os
import cv2
import numpy as np
from tqdm import tqdm
def add_gaussian_noise(image, mean=0, std=14):
    """
    Add Gaussian noise to the image.

    Parameters:
        image: Input image.
        mean: Mean of the Gaussian distribution.
        std: Standard deviation of the Gaussian distribution.

    Returns:
        Noisy image.
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_poisson_noise(image):
    """
    Add Poisson noise to the image.

    Parameters:
        image: Input image.

    Returns:
        Noisy image.
    """
    #value = 80
    value = np.random.randint(70, 91)
    noisy_image = np.random.poisson(image / 255.0 * value) / value * 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.0000001, pepper_prob=0.0002):
    """
    Add salt and pepper noise to the image.

    Parameters:
        image: Input image.
        salt_prob: Probability of adding salt noise.
        pepper_prob: Probability of adding pepper noise.

    Returns:
        Noisy image.
    """
    noisy_image = np.copy(image)
    salt_mask = np.random.rand(*image.shape) < salt_prob
    pepper_mask = np.random.rand(*image.shape) < pepper_prob
    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0
    return noisy_image

def add_realistic_stripe_noise(image):
    """
    生成更贴近原图的条纹噪声
    核心调整：降低频率（条纹更稀疏）、动态幅度（模拟真实不均匀条纹）、添加轻微偏移
    """
    h, w = image.shape
    noisy_image = image.copy().astype(np.float32)
    
    # 1. 核心参数调整（适配原图条纹特征）
    # 频率：5-8 条/图像高度（原图条纹稀疏，频率远低于20）
    frequency = np.random.uniform(5, 8)
    # 幅度：8-15（原图条纹对比度适中，避免过亮/过暗）
    base_amplitude = np.random.uniform(8, 15)
    # 随机偏移：模拟真实条纹的轻微错位
    phase_shift = np.random.uniform(0, np.pi/4)

    # 2. 生成带动态幅度的水平条纹（更贴近真实噪声）
    y = np.linspace(0 + phase_shift, 2 * np.pi * frequency + phase_shift, h)
    # 动态幅度：让不同位置的条纹强度略有差异，避免机械感
    dynamic_amplitude = base_amplitude * (1 + 0.1 * np.random.rand(h, 1))
    stripe_pattern = dynamic_amplitude * np.sin(y)[:, np.newaxis]

    # 3. 叠加少量垂直条纹（模拟原图轻微的横竖混合条纹）
    vertical_freq = np.random.uniform(1, 3)
    vertical_amp = base_amplitude * 0.3  # 垂直条纹强度仅为水平的30%
    x = np.linspace(0, 2 * np.pi * vertical_freq, w)
    vertical_pattern = vertical_amp * np.sin(x)[np.newaxis, :]
    
    # 4. 合并条纹并叠加到图像
    total_pattern = stripe_pattern + vertical_pattern
    noisy_image += total_pattern
    
    # 5. 限制像素范围，模拟真实图像的亮度约束
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    # 6. 轻微模糊条纹边缘（避免条纹过于锐利，更贴近真实拍摄的条纹噪声）
    noisy_image = cv2.GaussianBlur(noisy_image, (3, 3), 0.5)
    
    return noisy_image

# Input and output directories
input_dir = ''
output_dir = ''

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all image files in the input directory
file_bar = tqdm(os.listdir(input_dir))
for filename in file_bar:
    if filename.endswith('.png') or filename.endswith('.jpg'):  # Adjust extensions as needed
        # Load image
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Add noise
        noisy_image = add_gaussian_noise(image)
        noisy_image = add_poisson_noise(noisy_image)
        noisy_image = add_salt_and_pepper_noise(noisy_image)
        noisy_image = add_realistic_stripe_noise(noisy_image)

        # Save noisy image
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, noisy_image)        
        file_bar.set_description("{} |".format(filename))
        # print(f"Noise added to {filename} and saved to {output_path}")