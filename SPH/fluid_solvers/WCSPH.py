import taichi as ti
from .base_solver import BaseSolver
from ..containers.base_container import BaseContainer
from ..utils.kernel import *

@ti.data_oriented
class WCSPHSolver(BaseSolver):
    def __init__(self, container:BaseContainer):
        super().__init__(container)

        # wcsph related parameters
        self.gamma = 7.0
        self.stiffness = 50000.0
        
    @ti.kernel
    def compute_pressure(self):
        # use equation of state to compute pressure
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                rho_i = self.container.particle_densities[p_i]
                rho_i = ti.max(rho_i, self.density_0)
                self.container.particle_densities[p_i] = rho_i
                self.container.particle_pressures[p_i] = self.stiffness * (ti.pow(rho_i / self.density_0, self.gamma) - 1.0)
                
                
    def _step(self):
        self.container.prepare_neighbor_search()
        self.compute_density()
        self.compute_non_pressure_acceleration()
        self.update_fluid_velocity()

        self.compute_pressure()
        self.compute_pressure_acceleration()
        self.update_fluid_velocity()
        self.update_fluid_position()

 
        self.rigid_solver.step()
        
        self.container.insert_object()
        self.rigid_solver.insert_rigid_object()
        self.renew_rigid_particle_state()
    
        self.boundary.enforce_domain_boundary(self.container.material_fluid)
