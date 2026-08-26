
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

model = 'ControlFusion'
image_dir = f'./{model}'

from_width = 550
end_width = 600

from_height = 50
end_height = 100

file_list = os.listdir(image_dir)
file_list.sort()

# 创建图形
plt.figure(figsize=(10, 6))

for file in file_list:
    level = file.split('_')[0]
    image_path = os.path.join(image_dir, file)
    heights = []
    brightness_row = []
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    H, W, _ = image.shape

    for i in range(from_height, end_height + 1):
        sum = 0
        cnt = 0
        for j in range(from_width, end_width + 1):
            pixel = image[i][j]
            mean = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
            sum = sum + mean
            cnt = cnt + 1
            if i == from_height or i == end_height:
                image[i][j] = [255, 0, 0]
            if j == from_width or j == end_width:
                image[i][j] = [255, 0, 0]
        heights.append(i)
        brightness_row.append(sum / cnt)
        
    plt.plot(heights, brightness_row, linewidth=2, label=f'Level {level}')
    save_image = Image.fromarray(image)
    save_image.save(f'./{model}_out/{file}')


# 设置标题和轴标签
plt.title('Average Brightness', fontsize=14)
plt.xlabel('Height', fontsize=12)
plt.ylabel('Brightness', fontsize=12)

# 添加网格和图例
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')

# 自动调整布局并显示
plt.tight_layout()
plt.savefig(f'./brightness_{model}.png') 