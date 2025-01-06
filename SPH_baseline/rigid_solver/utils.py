import numpy as np
import math

class CubicKernel():
    def __init__(self):
        pass

    def weight(self, r_mod, h):
        q = r_mod / h
        w = 0.0
        sigma = 8.0 / (math.pi * h**3.0)

        if q <= 1.0:
            if q < 0.5:
                q2 = q * q
                w = sigma * (6 * q * q2 - 6 * q2 + 1)
            else:
                w = sigma * 2 * (1 - q)**3.0
        return w
    
    def gradient(self, r, h):
        r_mod = np.linalg.norm(r)
        q = r_mod / h
        sigma = 48.0 / (math.pi * h**3.0)
        grad = np.zeros(3)

        if r_mod > 1e-5 and q <= 1.0:
            if q < 0.5:
                grad = sigma * (3.0 * q - 2.0) * r / (h * h)
            else:
                grad = -sigma * (1 - q) * (1 - q) * r / (r_mod * h)
        return grad