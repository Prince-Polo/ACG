import taichi as ti
from .base_solver import BaseSolver
from ..containers.base_container import BaseContainer
from ..utils.kernel import *

@ti.data_oriented
class WCSPHSolver(BaseSolver):
    def __init__(self, container:BaseContainer):
        super().__init__(container)
        
    @ti.kernel
    def compute_pressure(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                rho_i = self.container.particle_densities[p_i]
                if rho_i < self.density_0:
                    rho_i = self.density_0
                
                self.container.particle_densities[p_i] = rho_i
                rho_ratio = rho_i / self.density_0
                pressure_term = ti.pow(rho_ratio, self.container.gamma) - 1.0
                self.container.particle_pressures[p_i] = self.container.stiffness * pressure_term
    
    @ti.kernel
    def compute_pressure_acceleration(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                ret_i = ti.Vector([0.0, 0.0, 0.0])
                self.container.for_all_neighbors(i, self.compute_pressure_acceleration_task, ret_i)
                self.container.particle_accelerations[i] = ret_i

    @ti.func
    def compute_pressure_acceleration_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        den_i = self.container.particle_densities[p_i]
        nabla_ij = self.kernel.gradient(pos_i - pos_j, self.container.dh)

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            den_j = self.container.particle_densities[p_j]
            ret += self._compute_fluid_pressure_acc(p_i, p_j, den_i, den_j, nabla_ij)
        else:
            ret += self._compute_rigid_pressure_acc(p_i, p_j, den_i, nabla_ij)

    @ti.kernel
    def update_rigid_body_force(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                self.container.for_all_neighbors(i, self.update_rigid_body_task, None)

    @ti.func
    def update_rigid_body_task(self, p_i, p_j, ret: ti.template()):
        if (self.container.particle_materials[p_j] == self.container.material_rigid and 
            self.container.particle_is_dynamic[p_j]):
            self._apply_rigid_body_force(p_i, p_j)

    @ti.func
    def _compute_fluid_pressure_acc(self, p_i, p_j, den_i, den_j, nabla_ij):
        pressure_term = (self.container.particle_pressures[p_i] / (den_i * den_i) + 
                        self.container.particle_pressures[p_j] / (den_j * den_j))
        return -self.container.particle_masses[p_j] * pressure_term * nabla_ij

    @ti.func
    def _compute_rigid_pressure_acc(self, p_i, p_j, den_i, nabla_ij):
        return (-self.density_0 * self.container.particle_rest_volumes[p_j] * 
                self.container.particle_pressures[p_i] / (den_i * den_i) * nabla_ij)

    @ti.func
    def _apply_rigid_body_force(self, p_i, p_j):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        den_i = self.container.particle_densities[p_i]
        nabla_ij = self.kernel.gradient(pos_i - pos_j, self.container.dh)
        
        object_j = self.container.particle_object_ids[p_j]
        center_of_mass_j = self.container.rigid_body_com[object_j]
        
        force_j = (self.density_0 * self.container.particle_rest_volumes[p_j] * 
                  self.container.particle_pressures[p_i] / (den_i * den_i) * 
                  nabla_ij * (self.density_0 * self.container.particle_rest_volumes[p_i]))

        torque_j = ti.math.cross(pos_i - center_of_mass_j, force_j)
        self.container.rigid_body_forces[object_j] += force_j
        self.container.rigid_body_torques[object_j] += torque_j

    def _step(self):
        self.container.prepare_neighbor_search()
        self.compute_density()
        self.compute_non_pressure_acceleration()
        self.update_fluid_velocity()

        self.compute_pressure()
        self.compute_pressure_acceleration()
        self.update_rigid_body_force()
        self.update_fluid_velocity()
        self.update_fluid_position()

        self.rigid_solver.step()
        
        self.container.insert_object()
        self.rigid_solver.insert_rigid_object()
        self.renew_rigid_particle_state()
    
        self.boundary.enforce_domain_boundary(self.container.material_fluid)
