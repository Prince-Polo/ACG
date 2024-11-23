import taichi as ti
from .kernel_functions import KernelFunctions

@ti.data_oriented
class RigidParticleUtils:
    def __init__(self, container, density_0, g_upper):
        self.container = container
        self.density_0 = density_0
        self.g_upper = g_upper
        self.kernel_functions = KernelFunctions(container.dh, container.dim)

    @ti.kernel
    def compute_rigid_particle_volume(self):
        # implemented as paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid:
                if self.container.particle_positions[p_i][1] <= self.g_upper:
                    ret = self.kernel_functions.kernel_W(0.0)
                    self.container.for_all_neighbors(p_i, lambda p_i, p_j, ret=ret: self.compute_rigid_particle_volumn_task(p_i, p_j, ret))
                    self.container.particle_rest_volumes[p_i] = 1.0 / ret 
                    self.container.particle_masses[p_i] = self.density_0 * self.container.particle_rest_volumes[p_i]

    @ti.func
    def compute_rigid_particle_volumn_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        if self.container.particle_object_ids[p_j] == self.container.particle_object_ids[p_i]:
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R_mod = R.norm()
            ret += self.kernel_functions.kernel_W(R_mod)