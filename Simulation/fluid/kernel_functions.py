import taichi as ti
import numpy as np

@ti.data_oriented
class KernelFunctions:
    def __init__(self, h, dim):
        self.h = h
        self.dim = dim

    @ti.func
    def kernel_W(self, R_mod):
        # cubic kernel
        res = ti.cast(0.0, ti.f32)
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.dim == 1:
            k = 4 / 3
        elif self.dim == 2:
            k = 40 / 7 / np.pi
        elif self.dim == 3:
            k = 8 / np.pi
        k /= self.h ** self.dim
        q = R_mod / self.h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def kernel_gradient(self, R):
        # cubic kernel gradient
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.dim == 1:
            k = 4 / 3
        elif self.dim == 2:
            k = 40 / 7 / np.pi
        elif self.dim == 3:
            k = 8 / np.pi
        k = 6. * k / self.h ** self.dim
        R_mod = R.norm()
        q = R_mod / self.h
        res = ti.Vector([0.0 for _ in range(self.dim)])
        if R_mod > 1e-5 and q <= 1.0:
            grad_q = R / (R_mod * self.h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res