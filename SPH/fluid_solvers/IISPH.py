import taichi as ti
from .base_solver import BaseSolver
from ..containers.iisph_container import IISPHContainer
from ..utils.kernel import *

@ti.data_oriented
class IISPHSolver(BaseSolver):
    @ti.kernel
    def compute_sum_dij(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                ret = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(i, self.compute_sum_dij_task, ret)
                self.container.iisph_sum_dij[i] = ret
              
    @ti.func  
    def compute_sum_dij_task(self, i, j, ret: ti.template()):
        nabla_kernel = self.kernel.gradient(self.container.particle_positions[i] - self.container.particle_positions[j], self.container.dh)
        
        regular_volumn_j = self.container.particle_masses[j] / self.container.particle_densities[j]
        ret += regular_volumn_j * nabla_kernel / self.container.particle_densities[j]

    @ti.kernel
    def compute_aii(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                ret = 0.0
                self.container.for_all_neighbors(i, self.compute_aii_task, ret)
                self.container.iisph_a_ii[i] = -ret * self.dt[None] * self.dt[None]
                
    @ti.func
    def compute_aii_task(self, i, j, ret: ti.template()):
        nabla_kernel = self.kernel.gradient(self.container.particle_positions[i] - self.container.particle_positions[j], self.container.dh)
        
        regular_volumn_i = self.container.particle_masses[i] /self.container.particle_densities[i]
        d_ii = regular_volumn_i * nabla_kernel / self.container.particle_densities[i]
        kernel_ii = ti.math.dot(self.container.iisph_sum_dij[i] + d_ii, nabla_kernel)
        
        ret += self.container.particle_masses[j] * kernel_ii

    @ti.kernel
    def compute_source_term(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                ret = 0.0
                self.container.for_all_neighbors(i, self.compute_source_term_task, ret)
                density_error = self.container.particle_rest_densities[i] - self.container.particle_densities[i]
                self.container.iisph_source[i] = density_error - self.dt[None] * ret

    @ti.func
    def compute_source_term_task(self, i, j, ret: ti.template()):
        pos_i = self.container.particle_positions[i]
        pos_j = self.container.particle_positions[j]
        
        vel_i = self.container.particle_velocities[i]
        vel_j = self.container.particle_velocities[j]
        nabla_kernel = self.kernel.gradient(pos_i - pos_j, self.container.dh)
        
        ret += self.container.particle_masses[j] * ti.math.dot(vel_i - vel_j, nabla_kernel)

    @ti.kernel
    def init_pressure(self):
        for i in range(self.container.particle_num[None]):
            self.container.particle_pressures[i] = 0.0
                
    @ti.kernel
    def compute_pressure_acceleration(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                ret = ti.Vector([0.0, 0.0, 0.0])
                self.container.for_all_neighbors(i, self.compute_pressure_acceleration_task, ret)
                self.container.iisph_pressure_accelerations[i] = ret
                
    @ti.func
    def compute_pressure_acceleration_task(self, i, j, ret: ti.template()):
        pos_i = self.container.particle_positions[i]
        pos_j = self.container.particle_positions[j]
        nabla_kernel = self.kernel.gradient(pos_i - pos_j, self.container.dh)
        regular_iisph_pressure_i = self.container.particle_pressures[i] / self.container.particle_densities[i]
        regular_iisph_pressure_j = self.container.particle_pressures[j] / self.container.particle_densities[j]
       
        if self.container.particle_materials[j] == self.container.material_fluid:
            ret -= self.container.particle_masses[j] * (regular_iisph_pressure_i / self.container.particle_densities[i] + regular_iisph_pressure_j / self.container.particle_densities[j]) * nabla_kernel
            
        elif self.container.particle_materials[j] == self.container.material_rigid:
            ret -= self.container.particle_masses[j] * regular_iisph_pressure_i * nabla_kernel / self.container.particle_densities[j]
            
    @ti.kernel
    def compute_Laplacian(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                ret = 0.0
                self.container.for_all_neighbors(i, self.compute_Laplacian_task, ret)
                self.container.iisph_laplacian[i] = ret * self.dt[None] * self.dt[None]
                
    @ti.func
    def compute_Laplacian_task(self, i, j, ret: ti.template()):
        nabla_kernel = self.kernel.gradient(self.container.particle_positions[i] - self.container.particle_positions[j], self.container.dh)
        
        if self.container.particle_materials[j] == self.container.material_fluid:
            ret += self.container.particle_masses[j] * ti.math.dot(self.container.iisph_pressure_accelerations[i] - self.container.iisph_pressure_accelerations[j], nabla_kernel)
        
        elif self.container.particle_materials[j] == self.container.material_rigid:
            ret += self.container.particle_masses[j] * ti.math.dot(self.container.iisph_pressure_accelerations[i], nabla_kernel)
            
    @ti.kernel
    def update_pressure(self):
        error = 0.0
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                new_pressure = 0.0
                if self.container.iisph_a_ii[i] > 1e-8 or self.container.iisph_a_ii[i] < -1e-8:
                    extra_pressure = self.container.omega * (self.container.iisph_source[i] - self.container.iisph_laplacian[i]) / self.container.iisph_a_ii[i]
                    new_pressure = self.container.particle_pressures[i] + extra_pressure
                    if new_pressure < 0.0:
                        new_pressure = 0.0
                    
                self.container.particle_pressures[i] = new_pressure
                
                if new_pressure > 1e-8:
                    error += ti.abs((self.container.iisph_laplacian[i] - self.container.iisph_source[i]) / self.container.particle_rest_densities[i])
            
        self.container.density_error[None] = error / self.container.fluid_particle_num[None]
    
    def refine(self):
        max_iter = self.container.max_iterations
        eta = self.container.eta
        error = 0.0
        
        for iter_count in range(max_iter):
            self.compute_pressure_acceleration()
            self.compute_Laplacian()
            self.update_pressure()
            
            error = self.container.density_error[None]
            if error < eta:
                break
                
        print(f"IISPH - iteration: {iter_count + 1}, "
              f"Avg density err: {error * self.density_0:.4f}")
    
    @ti.kernel
    def update_rigid_body_force(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                ret = self.container.iisph_pressure_accelerations[i]
                self.container.for_all_neighbors(i, self.update_rigid_body_task, ret)
                self.container.particle_accelerations[i] = ret
              
    @ti.func  
    def update_rigid_body_task(self, i, j, ret: ti.template()):
        if self.container.rigid_body_is_dynamic[i]:
            pos_i = self.container.particle_positions[i]
            pos_j = self.container.particle_positions[j]
            nabla_kernel = self.kernel.gradient(pos_i - pos_j, self.container.dh)
            regular_iisph_pressure_i = self.container.particle_pressures[i] / self.container.particle_densities[i]
            object_j = self.container.particle_object_ids[j]
            center_of_mass_j = self.container.rigid_body_com[object_j]
                
            force_j = self.container.particle_masses[j] * regular_iisph_pressure_i * nabla_kernel * self.container.particle_masses[i] / self.container.particle_densities[i]
            torque_j = ti.math.cross(self.container.particle_positions[i] - center_of_mass_j, force_j)
                
            self.container.rigid_body_forces[object_j] += force_j
            self.container.rigid_body_torques[object_j] += torque_j
                
    def _step(self):
        self.compute_density()
        self.init_pressure()
        self.compute_non_pressure_acceleration()
        self.update_fluid_velocity()
        
        self.compute_sum_dij()
        self.compute_aii()
        self.compute_source_term()
        
        self.refine()
        self.update_rigid_body_force()
        self.update_fluid_velocity()  
        self.update_fluid_position()
        
        self.rigid_solver.step()

        self.container.insert_object()
        self.rigid_solver.insert_rigid_object()
        self.renew_rigid_particle_state()
        self.boundary.enforce_domain_boundary(self.container.material_fluid)
        self.container.prepare_neighbor_search()
