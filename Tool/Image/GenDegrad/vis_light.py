import numpy as np
from skimage import io, img_as_float, img_as_ubyte
import os

from Lime import LIME 

def add_illumination_degradation(image_path, gamma, output_path=None):
    I = img_as_float(io.imread(image_path))
    lime = LIME(gamma=1)
    lime.load(image_path)
    L = lime.illumMap()
    eps = 1e-6
    L_safe = np.maximum(L, eps)
    L_exp = L_safe[..., np.newaxis]
    D = I * (L_exp ** (gamma - 1))
    D = np.clip(D, 0, 1)
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_gamma_{gamma:.2f}{ext}"
    io.imsave(output_path, img_as_ubyte(D)) 
    return D

if __name__ == "__main__":
    img_path = "./030104_gt.png"
    gamma = 3.0
    add_illumination_degradation(img_path, gamma)