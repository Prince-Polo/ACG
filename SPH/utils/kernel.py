import taichi as ti
import math

@ti.data_oriented
class CubicKernel():
    def __init__(self):
        pass

    @ti.func
    def weight(self, r_mod, h):
        q = r_mod / h
        w = ti.cast(0.0, ti.f32)
        sigma = 8.0 / (math.pi * ti.pow(h, 3.0))

        if q <= 1.0:
            if q < 0.5:
                q2 = q * q
                w = sigma * (6 * q * q2 - 6 * q2 + 1)
            else:
                w = sigma * 2 * ti.pow(1 - q, 3.0)
        return w
    
    @ti.func
    def gradient(self, r, h):
        r_mod = r.norm()
        q = r_mod / h
        sigma = 48.0 / (math.pi * ti.pow(h, 3.0))
        grad = ti.Vector([0.0, 0.0, 0.0])

        if r_mod > 1e-5 and q <= 1.0:
            if q < 0.5:
                grad = sigma * (3.0 * q - 2.0) * r / (h * h)
            else:
                grad = -sigma * (1 - q) * (1 - q) * r / (r_mod * h)
        return grad

