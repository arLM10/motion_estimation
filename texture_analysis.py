import cv2
import numpy as np

def compute_texture(block):
    gx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)

    texture = np.mean(np.abs(gx) + np.abs(gy))
    return texture
