import numpy as np
from numba import cuda, float32
import math

@cuda.jit(device=True)
def _weight(r_mod, h, out_weight):
    """计算核函数权重
    Args:
        r_mod: 距离
        h: 光滑长度
        out_weight: 输出权重数组
    """
    q = r_mod / h
    sigma = 8.0 / (math.pi * h**3)

    if q <= 1.0:
        if q < 0.5:
            q2 = q * q
            out_weight[0] = sigma * (6.0 * q * q2 - 6.0 * q2 + 1.0)
        else:
            out_weight[0] = sigma * 2.0 * (1.0 - q)**3
    else:
        out_weight[0] = 0.0

@cuda.jit(device=True)
def _gradient(r, h, out_grad):
    """计算核函数梯度
    Args:
        r: 位置向量
        h: 光滑长度
        out_grad: 输出梯度数组
    """
    r_mod = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    q = r_mod / h
    sigma = 48.0 / (math.pi * h**3)
    
    out_grad[0] = 0.0
    out_grad[1] = 0.0
    out_grad[2] = 0.0

    if r_mod > 1e-5 and q <= 1.0:
        if q < 0.5:
            factor = sigma * (3.0 * q - 2.0) / (h * h)
            for i in range(3):
                out_grad[i] = factor * r[i]
        else:
            factor = -sigma * (1.0 - q)**2 / (r_mod * h)
            for i in range(3):
                out_grad[i] = factor * r[i]

class CubicSpline:
    def __init__(self):
        self.weight = _weight
        self.gradient = _gradient