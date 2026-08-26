import numpy as np 
import os
from PIL import Image
import random
import matplotlib.pyplot as plt

def _vis_feat(x, title, idx):
    save_path = os.path.join('vis_feat_before', f'{title}_{idx}.png')
    img_out = x.squeeze(0).permute(1, 2, 0).mean(dim=-1).detach().cpu().numpy()
    img_out = np.clip(img_out, 0, 1)

    fig = plt.figure(figsize=(3, 3))
    plt.title(title)
    plt.imshow(img_out, cmap='inferno', vmin=0.0, vmax=1.0)  # 固定颜色范围
    plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])  # 固定刻度
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # img_pil = Image.fromarray((img_out * 255).astype(np.uint8))
    # img_pil.save(save_path)