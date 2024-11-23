import taichi as ti
from .kernel_functions import KernelFunctions

@ti.data_oriented
class DensityUtils:
    def __init__(self, container):
        self.container = container
        self.kernel_functions = KernelFunctions(container.dh, container.dim)

    @ti.kernel
    def compute_density(self):
        """
        compute density for each particle from mass of neighbors in SPH standard way.
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_densities[p_i] = self.container.particle_rest_volumes[p_i] * self.kernel_functions.kernel_W(0.0)
                ret_i = 0.0
                self.container.for_all_neighbors(p_i, lambda p_i, p_j, ret=ret_i: self.compute_density_task(p_i, p_j, ret))
                self.container.particle_densities[p_i] += ret_i
                self.container.particle_densities[p_i] *= self.container.density_0

    @ti.func
    def compute_density_task(self, p_i, p_j, ret: ti.template()):
        # Fluid neighbors and rigid neighbors are treated the same
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        R_mod = R.norm()
        ret += self.container.particle_rest_volumes[p_j] * self.kernel_functions.kernel_W(R_mod)