import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import matplotlib.pylab as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model.Adapter_one import Adapter as one


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(seed=721)

def get_label(file):
    if "Haze" in file and "Rain" in file:
        return 0
    if "Haze" in file and "Low" in file:
        return 1
    if "Rain" in file:
        return 2
    if "Haze" in file:
        return 3
    if "exposure" in file:
        return 4
    if "light" in file:
        return 5   

class CustomDataset(Dataset):
    def __init__(self, vis_path):
        super().__init__()
        self.vis_path = vis_path
        self.filename_path = os.listdir(vis_path)
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filename_path)

    def __getitem__(self, idx):
        filename = self.filename_path[idx]
        image_path = os.path.join(self.vis_path, filename)
        img = Image.open(image_path)
        img_tensor = self.toTensor(img)
        label = get_label(filename)
        return img_tensor, label

def plot_tsne3d(features, labels, class_text, save_path=None):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # 指定3维，并初始化
    tsne = TSNE(n_components=3, 
                init='pca', 
                random_state=42,
                perplexity=min(30, features.shape[0]-1),  # 自适应perplexity
                max_iter=1000,
                learning_rate='auto')
    
    try:
        tsne_features = tsne.fit_transform(features)  # 将特征使用t-SNE降维至3维
        print(f"t-SNE features shape: {tsne_features.shape}")
        print(f"t-SNE features range: [{tsne_features.min():.3f}, {tsne_features.max():.3f}]")
        
        # 检查t-SNE结果是否有效
        if np.any(np.isnan(tsne_features)):
            print("Error: t-SNE features contain NaN")
            return
        if np.all(tsne_features == 0):
            print("Error: All t-SNE features are zero")
            return
            
    except Exception as e:
        print(f"Error in t-SNE: {e}")
        return
    
    # 对数据进行归一化操作（添加小的epsilon避免除以0）
    x_min, x_max = np.min(tsne_features, axis=0), np.max(tsne_features, axis=0)
    if np.all((x_max - x_min) == 0):
        print("Warning: All features are identical, skipping normalization")
        embedded = tsne_features
    else:
        embedded = (tsne_features - x_min) / (x_max - x_min + 1e-8)
    
    hex = ["#c957db", "#dd5f57", "#b9db57", "#57db30", "#5784db", "#dc8a78"]  # 粉红，暗红，浅绿，绿，蓝
    
    # 创建显示的figure - 使用更现代的API
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置3D视图角度
    ax.view_init(elev=20, azim=45)
    
    # 为每个类别绘制散点图
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        if label < len(hex):  # 确保索引在范围内
            mask = labels == label
            if np.sum(mask) > 0:
                ax.scatter(embedded[mask, 0], 
                          embedded[mask, 1], 
                          embedded[mask, 2],
                          c=hex[label],  # 使用颜色列表
                          marker="o",    # 使用实心圆点
                          s=30,          # 增大点的大小
                          alpha=0.8,     # 设置透明度
                          label=f'Class {label}')
    
    # 设置坐标轴标签
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    ax.set_title(f't-SNE 3D Visualization')
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))
    
    # 设置图形布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig)  # 重要：关闭图形释放内存
    print(f"Saved plot to: {save_path}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_text = ['HazeRain', 'HazeLow', 'Rain', 'Haze', 'Exposure', 'LowLight']
    vis_path = '../dataset/LightDDL/train/Visible'
    save_path = 'tsne_visualization.png'

    
    dataset = CustomDataset(vis_path)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1)
    
    model = one()
    classification_model_path = 'pretrained_weights/one.pth'
    model.load_state_dict(torch.load(classification_model_path), strict=False)
    model.to(device)
    model.eval()
    
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            feature = model(data)
            features.append(feature.cpu().numpy())
            labels.append(target.cpu().numpy())
            
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"Test features shape: {features.shape}")
    print(f"Test labels shape: {labels.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    plot_tsne3d(features, labels, class_text, save_path=save_path)