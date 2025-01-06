import numpy as np
from .base_solver import BaseSolverBaseline
from ..containers.dfsph_container import DFSPHContainerBaseline
from .utils import *

class DFSPHSolverBaseline(BaseSolverBaseline):
    def __init__(self, container:DFSPHContainerBaseline):
        super().__init__(container)
        self.m_max_iterations_v = 1000
        self.m_max_iterations = 1000
        self.m_eps = 1e-5
        self.max_error_V = 0.001
        self.max_error = 0.0001
    
    def compute_derivative_density(self):
        for i in range(self.container.particle_num):
            if self.container.particle_materials[i] == 1:
                derivative_density = 0.0
                num_neighbors = 0
                
                def task(p_j, ret):
                    nonlocal derivative_density, num_neighbors
                    pos_i = self.container.particle_positions[i]
                    pos_j = self.container.particle_positions[p_j]
                    vel_i = self.container.particle_velocities[i]
                    vel_j = self.container.particle_velocities[p_j]
                    
                    nabla_kernel = self.kernel.gradient(pos_i - pos_j, self.container.dh)
                    derivative_density += self.container.particle_masses[p_j] * np.dot(vel_i - vel_j, nabla_kernel)
                    num_neighbors += 1
                
                self.container.for_all_neighbors(i, task)
                
                derivative_density = max(derivative_density, 0.0)
                if num_neighbors < 20:
                    derivative_density = 0.0
                
                self.container.particle_dfsph_derivative_densities[i] = derivative_density
                
    def compute_factor_k(self):
        for i in range(self.container.particle_num):
            if self.container.particle_materials[i] == self.container.material_fluid:
                grad_sum = np.zeros(self.container.dim)
                grad_norm_sum = 0.0
                
                def task(p_j, ret):
                    nonlocal grad_sum, grad_norm_sum
                    if self.container.particle_materials[p_j] == self.container.material_fluid:
                        grad_p_j = self.container.particle_masses[p_j] * self.kernel.gradient(
                            self.container.particle_positions[i] - self.container.particle_positions[p_j], 
                            self.container.dh
                        )
                        grad_norm_sum += np.sum(grad_p_j * grad_p_j)
                        grad_sum += grad_p_j
                    elif self.container.particle_materials[p_j] == self.container.material_rigid:
                        grad_p_j = self.container.particle_masses[p_j] * self.kernel.gradient(
                            self.container.particle_positions[i] - self.container.particle_positions[p_j], 
                            self.container.dh
                        )
                        grad_sum += grad_p_j
                
                self.container.for_all_neighbors(i, task)
                
                grad_sum_norm = np.sum(grad_sum * grad_sum)
                sum_grad = grad_sum_norm + grad_norm_sum
                
                threshold = 1e-6 * self.container.particle_densities[i] * self.container.particle_densities[i]
                factor_k = 0.0
                if sum_grad > threshold:
                    factor_k = self.container.particle_densities[i] * self.container.particle_densities[i] / sum_grad
                
                self.container.particle_dfsph_factor_k[i] = factor_k
    
    def compute_pressure_for_DFS(self):
        for i in range(self.container.particle_num):
            if self.container.particle_materials[i] == self.container.material_fluid:
                self.container.particle_dfsph_pressure_v[i] = (
                    self.container.particle_dfsph_derivative_densities[i] * 
                    self.container.particle_dfsph_factor_k[i]
                )
    
    def compute_predict_density(self):
        for p_i in range(self.container.particle_num):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                derivative_density = 0.0
                
                def task(p_j, ret):
                    nonlocal derivative_density
                    pos_i = self.container.particle_positions[p_i]
                    pos_j = self.container.particle_positions[p_j]
                    vel_i = self.container.particle_velocities[p_i]
                    vel_j = self.container.particle_velocities[p_j]
                    
                    nabla_kernel = self.kernel.gradient(pos_i - pos_j, self.container.dh)
                    derivative_density += self.container.particle_masses[p_j] * np.dot(vel_i - vel_j, nabla_kernel)
                
                self.container.for_all_neighbors(p_i, task)
                
                predict_density = self.container.particle_densities[p_i] + self.dt * derivative_density
                self.container.particle_dfsph_predict_density[p_i] = max(predict_density, 1000.0)
                
    def compute_pressure_for_CDS(self):
        for p_i in range(self.container.particle_num):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_dfsph_pressure[p_i] = (
                    (self.container.particle_dfsph_predict_density[p_i] - self.density_0) * 
                    self.container.particle_dfsph_factor_k[p_i] / self.dt
                )

    def correct_divergence_error(self):
        num_iterations = 0
        average_density_derivative_error = 0.0
        
        while num_iterations < 1 or num_iterations < self.m_max_iterations:
            self.compute_derivative_density()
            self.compute_pressure_for_DFS()
            self.update_velocity_DFS()
            average_density_derivative_error = self.compute_density_derivative_error()

            eta = self.max_error_V * self.density_0 / self.dt
            num_iterations += 1

            if average_density_derivative_error <= eta:
                break
            
        print(f"DFSPH - iteration V: {num_iterations} Avg density err: {average_density_derivative_error}")
    
    def update_velocity_DFS(self):
        for i in range(self.container.particle_num):
            if self.container.particle_materials[i] == self.container.material_fluid:
                pressure_i = self.container.particle_dfsph_pressure_v[i]
                dv = np.zeros(3)
                
                def task(p_j, ret):
                    nonlocal dv
                    if self.container.particle_materials[p_j] == self.container.material_fluid:
                        pressure_j = self.container.particle_dfsph_pressure_v[p_j]
                        regular_pressure_i = pressure_i / self.container.particle_densities[i]
                        regular_pressure_j = pressure_j / self.container.particle_densities[p_j]
                        pressure_sum = pressure_i + pressure_j
                        
                        if abs(pressure_sum) > self.m_eps * self.container.particle_densities[i] * self.dt:
                            nabla_kernel = self.kernel.gradient(
                                self.container.particle_positions[i] - self.container.particle_positions[p_j], 
                                self.container.dh
                            )
                            dv += self.container.particle_masses[p_j] * (
                                regular_pressure_i / self.container.particle_densities[i] + 
                                regular_pressure_j / self.container.particle_densities[p_j]
                            ) * nabla_kernel
                    
                    elif self.container.particle_materials[p_j] == self.container.material_rigid:
                        pressure_j = pressure_i
                        regular_pressure_i = pressure_i / self.container.particle_densities[i]
                        regular_pressure_j = pressure_j / self.container.particle_densities[p_j]
                        pressure_sum = pressure_i + pressure_j
                        density_i = self.container.particle_densities[i]
                        
                        if abs(pressure_sum) > self.m_eps * self.container.particle_densities[i] * self.dt:
                            nabla_kernel = self.kernel.gradient(
                                self.container.particle_positions[i] - self.container.particle_positions[p_j], 
                                self.container.dh
                            )
                            dv += self.container.particle_masses[p_j] * (regular_pressure_i / density_i) * nabla_kernel
                            
                            if self.container.particle_is_dynamic[p_j]:
                                object_j = self.container.particle_object_ids[p_j]
                                center_of_mass_j = self.container.rigid_body_com[object_j]
                                force_j = (
                                    self.container.particle_masses[p_j] * 
                                    (regular_pressure_i / density_i) * 
                                    nabla_kernel * 
                                    self.container.particle_masses[i] / self.dt
                                )
                                torque_j = np.cross(
                                    self.container.particle_positions[p_j] - center_of_mass_j, 
                                    force_j
                                )
                                self.container.rigid_body_forces[object_j] += force_j
                                self.container.rigid_body_torques[object_j] += torque_j
                
                self.container.for_all_neighbors(i, task)
                self.container.particle_velocities[i] -= dv

    def compute_density_derivative_error(self) -> float:
        density_error = 0.0
        for idx_i in range(self.container.particle_num):
            if self.container.particle_materials[idx_i] == self.container.material_fluid:
                density_error += self.container.particle_dfsph_derivative_densities[idx_i]
        return density_error / self.container.particle_num

    def correct_density_error(self):
        num_itr = 0
        average_density_error = 0.0

        while num_itr < 1 or num_itr < self.m_max_iterations:
            self.compute_predict_density()
            self.compute_pressure_for_CDS()
            self.update_velocity_CDS()
            
            average_density_error = self.compute_density_error()

            eta = self.max_error
            num_itr += 1

            if average_density_error <= eta * self.density_0:
                break
            
        print(f"DFSPH - iterations: {num_itr} Avg density Err: {average_density_error :.4f}")
    
    def update_velocity_CDS(self):
        for i in range(self.container.particle_num):
            if self.container.particle_materials[i] == self.container.material_fluid:
                pressure_i = self.container.particle_dfsph_pressure[i]
                dv = np.zeros(3)
                
                def task(p_j, ret):
                    nonlocal dv
                    if self.container.particle_materials[p_j] == self.container.material_fluid:
                        pressure_j = self.container.particle_dfsph_pressure[p_j]
                        regular_pressure_i = pressure_i / self.container.particle_densities[i]
                        regular_pressure_j = pressure_j / self.container.particle_densities[p_j]
                        pressure_sum = pressure_i + pressure_j
                        
                        if abs(pressure_sum) > self.m_eps * self.container.particle_densities[i] * self.dt:
                            nabla_kernel = self.kernel.gradient(
                                self.container.particle_positions[i] - self.container.particle_positions[p_j], 
                                self.container.dh
                            )
                            dv += self.container.particle_masses[p_j] * (
                                regular_pressure_i / self.container.particle_densities[i] + 
                                regular_pressure_j / self.container.particle_densities[p_j]
                            ) * nabla_kernel
                    
                    elif self.container.particle_materials[p_j] == self.container.material_rigid:
                        pressure_j = pressure_i
                        regular_pressure_i = pressure_i / self.container.particle_densities[i]
                        regular_pressure_j = pressure_j / self.container.particle_densities[p_j]
                        pressure_sum = pressure_i + pressure_j
                        density_i = self.container.particle_densities[i]
                        
                        if abs(pressure_sum) > self.m_eps * self.container.particle_densities[i] * self.dt:
                            nabla_kernel = self.kernel.gradient(
                                self.container.particle_positions[i] - self.container.particle_positions[p_j], 
                                self.container.dh
                            )
                            dv += self.container.particle_masses[p_j] * (regular_pressure_i / density_i) * nabla_kernel
                            
                            if self.container.particle_is_dynamic[p_j]:
                                object_j = self.container.particle_object_ids[p_j]
                                center_of_mass_j = self.container.rigid_body_com[object_j]
                                force_j = (
                                    self.container.particle_masses[p_j] * 
                                    (regular_pressure_i / density_i) * 
                                    nabla_kernel * 
                                    self.container.particle_masses[i] / self.dt
                                )
                                torque_j = np.cross(
                                    self.container.particle_positions[p_j] - center_of_mass_j, 
                                    force_j
                                )
                                self.container.rigid_body_forces[object_j] += force_j
                                self.container.rigid_body_torques[object_j] += torque_j
                
                self.container.for_all_neighbors(i, task)
                self.container.particle_velocities[i] -= dv

    def compute_density_error(self) -> float:
        density_error = 0.0
        for i in range(self.container.particle_num):
            if self.container.particle_materials[i] == self.container.material_fluid:
                density_error += self.container.particle_dfsph_predict_density[i] - self.density_0
        return density_error / self.container.particle_num
    
    def _step(self):
        self.compute_non_pressure_acceleration()
        self.update_fluid_velocity()
        self.correct_density_error()

        self.update_fluid_position()

        self.rigid_solver.step()

        self.container.insert_object()
        self.rigid_solver.insert_rigid_object()
        self.renew_rigid_particle_state()
        
        self.boundary.enforce_domain_boundary(self.container.material_fluid)

        self.container.prepare_neighbor_search()
        self.compute_density()
        self.compute_factor_k()
        self.correct_divergence_error()

    def prepare(self):
        super().prepare()
        self.compute_density()
        
        self.compute_factor_k()