import numpy as np

def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100
    MAX_I = 255.0
    return 10 * np.log10(MAX_I**2 / mse)
