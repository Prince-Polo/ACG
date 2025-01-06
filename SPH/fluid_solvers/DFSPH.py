import taichi as ti
from .base_solver import BaseSolver
from ..containers.dfsph_container import DFSPHContainer
from ..utils.kernel import *

@ti.data_oriented
class DFSPHSolver(BaseSolver):
    @ti.kernel
    def compute_derivative_density(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == 1:
                ret = ti.Struct(derivative_density=0.0, num_neighbors=0)
                self.container.for_all_neighbors(i, self.compute_derivative_density_task, ret)

                derivative_density = 0.0
                if ret.num_neighbors > 20 and ret.derivative_density > 0.0:
                    derivative_density = ret.derivative_density
               
                self.container.particle_dfsph_derivative_densities[i] = derivative_density 
                
    @ti.func
    def compute_derivative_density_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        vel_i = self.container.particle_velocities[p_i]
        vel_j = self.container.particle_velocities[p_j]
        
        nabla_kernel = self.kernel.gradient(pos_i - pos_j, self.container.dh)
        ret.derivative_density += self.container.particle_masses[p_j] * ti.math.dot(vel_i - vel_j, nabla_kernel)
        
        ret.num_neighbors += 1
        
    @ti.kernel
    def compute_factor_k(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:

                ret = ti.Struct(grad_sum=ti.Vector([0.0 for _ in range(self.container.dim)]), grad_norm_sum=0.0)
                self.container.for_all_neighbors(i, self.compute_factor_k_task, ret)
                
                grad_sum = ti.Vector([0.0, 0.0, 0.0])
                
                grad_norm_sum = ret.grad_norm_sum
                for i in ti.static(range(self.container.dim)):
                    grad_sum[i] = ret.grad_sum[i] 
                grad_sum_norm = grad_sum.norm_sqr()
                
                sum_grad = grad_sum_norm + grad_norm_sum
                
                threshold = 1e-6 * self.container.particle_densities[i] * self.container.particle_densities[i]
                factor_k = 0.0
                if sum_grad > threshold:
                    factor_k = self.container.particle_densities[i] * self.container.particle_densities[i] / sum_grad
                
                self.container.particle_dfsph_factor_k[i]= factor_k                
                
    @ti.func
    def compute_factor_k_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            grad_p_j = self.container.particle_masses[p_j] * self.kernel.gradient(pos_i - pos_j, self.container.dh)
            ret.grad_norm_sum += grad_p_j.norm_sqr()
            ret.grad_sum += grad_p_j
        else:
            grad_p_j = self.container.particle_masses[p_j] * self.kernel.gradient(pos_i - pos_j, self.container.dh)
            ret.grad_sum += grad_p_j
    
    @ti.kernel
    def compute_pressure_for_DFS(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                self.container.particle_dfsph_pressure_v[i] = self.container.particle_dfsph_derivative_densities[i] * self.container.particle_dfsph_factor_k[i]
        
    @ti.kernel
    def compute_predict_density(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                derivative_density = 0.0
                self.container.for_all_neighbors(p_i, self.compute_predict_density_task, derivative_density)
                
                predict_density = self.container.particle_densities[p_i] + self.dt[None] * derivative_density
                if predict_density < self.density_0:
                    predict_density = self.density_0
                    
                self.container.particle_dfsph_predict_density[p_i] = predict_density
                
    @ti.func
    def compute_predict_density_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        vel_i = self.container.particle_velocities[p_i]
        vel_j = self.container.particle_velocities[p_j]
        
        nabla_kernel = self.kernel.gradient(pos_i - pos_j, self.container.dh)
        ret += self.container.particle_masses[p_j] * ti.math.dot(vel_i - vel_j, nabla_kernel)
                
    @ti.kernel
    def compute_pressure_for_CDS(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                error_density = self.container.particle_dfsph_predict_density[p_i] - self.density_0
                self.container.particle_dfsph_pressure[p_i] = error_density * self.container.particle_dfsph_factor_k[p_i] / self.dt[None]

    ############## Divergence-free Solver ################
    def correct_divergence_error(self):
        eta = self.container.max_error_V * self.density_0 / self.dt[None]
        max_iter = self.container.m_max_iterations
        
        for iter_count in range(max_iter):
            self.compute_derivative_density()
            self.compute_pressure_for_DFS()
            self.update_velocity_DFS()
            
            error = self.compute_density_derivative_error()
            if error <= eta and iter_count >= 0:
                break
            
        print(f"DFSPH - iteration DFS: {iter_count + 1}, DFS error: {error}")
            
    @ti.kernel
    def update_velocity_DFS(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                pressure_i = self.container.particle_dfsph_pressure_v[i]
                ret = ti.Struct(dv=ti.Vector([0.0, 0.0, 0.0]), pressure = pressure_i)
                self.container.for_all_neighbors(i, self.update_velocity_DFS_task, ret)
                self.container.particle_velocities[i] -= ret.dv 
    
    @ti.func  
    def update_velocity_DFS_task(self, p_i, p_j, ret: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            pressure_j = self.container.particle_dfsph_pressure_v[p_j]
            pressure_i = ret.pressure
            regular_pressure_i = pressure_i / self.container.particle_densities[p_i]
            regular_pressure_j = pressure_j / self.container.particle_densities[p_j]
            pressure_sum = pressure_i + pressure_j
            if ti.abs(pressure_sum) > self.container.m_eps * self.container.particle_densities[p_i] * self.dt[None]:
                nabla_kernel = self.kernel.gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j], self.container.dh)
                ret.dv += self.container.particle_masses[p_j] * (regular_pressure_i / self.container.particle_densities[p_i] + regular_pressure_j / self.container.particle_densities[p_j]) * nabla_kernel
        
        # This is 
        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            pressure_i = ret.pressure
            pressure_j = pressure_i
            regular_pressure_i = pressure_i / self.container.particle_densities[p_i]
            regular_pressure_j = pressure_j / self.container.particle_densities[p_j]
            pressure_sum = pressure_i + pressure_j
            density_i = self.container.particle_densities[p_i]
            if ti.abs(pressure_sum) > self.container.m_eps * self.container.particle_densities[p_i] * self.dt[None]:
                nabla_kernel = self.kernel.gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j], self.container.dh) 
                ret.dv += self.container.particle_masses[p_j] * (regular_pressure_i / density_i) * nabla_kernel
                
                if self.container.particle_is_dynamic[p_j]:
                    object_j = self.container.particle_object_ids[p_j]
                    center_of_mass_j = self.container.rigid_body_com[object_j]
                    force_j = self.container.particle_masses[p_j] * (regular_pressure_i / density_i) * nabla_kernel * self.container.particle_masses[p_i] / self.dt[None]
                    torque_j = ti.math.cross(self.container.particle_positions[p_j] - center_of_mass_j, force_j)
                    self.container.rigid_body_forces[object_j] += force_j
                    self.container.rigid_body_torques[object_j] += torque_j
                    
    @ti.kernel
    def compute_density_derivative_error(self) -> float:
        density_error = 0.0
        for idx_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[idx_i] == self.container.material_fluid:
                density_error += self.container.particle_dfsph_derivative_densities[idx_i]
        return density_error / self.container.particle_num[None]
    
    ################# Constant Density Solver #################
    def correct_density_error(self):
        max_iter = self.container.m_max_iterations
        eta = self.container.max_error * self.density_0
        error = 0.0
        
        for iter_count in range(max_iter):
            self.compute_predict_density()
            self.compute_pressure_for_CDS()
            self.update_velocity_CDS()
            
            error = self.compute_density_error()
            if error <= eta and iter_count >= 1:
                break
                
        print(f"DFSPH - CDS iterations: {iter_count + 1}, CDS Err: {error:.4f}")
    
    @ti.kernel
    def update_velocity_CDS(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                pressure_i = self.container.particle_dfsph_pressure[i]
                ret = ti.Struct(dv=ti.Vector([0.0, 0.0, 0.0]), pressure = pressure_i)
                self.container.for_all_neighbors(i, self.update_velocity_CDS_task, ret)
                self.container.particle_velocities[i] -= ret.dv
    
    @ti.func
    def update_velocity_CDS_task(self, p_i, p_j, ret: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            pressure_j = self.container.particle_dfsph_pressure[p_j]
            pressure_i = ret.pressure
            regular_pressure_i = pressure_i / self.container.particle_densities[p_i]
            regular_pressure_j = pressure_j / self.container.particle_densities[p_j]
            pressure_sum = pressure_i + pressure_j
            if ti.abs(pressure_sum) > self.container.m_eps * self.container.particle_densities[p_i] * self.dt[None]:
                nabla_kernel = self.kernel.gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j], self.container.dh)
                ret.dv += self.container.particle_masses[p_j] * (regular_pressure_i / self.container.particle_densities[p_i] + regular_pressure_j / self.container.particle_densities[p_j]) * nabla_kernel
        
        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            pressure_i = ret.pressure
            pressure_j = pressure_i
            regular_pressure_i = pressure_i / self.container.particle_densities[p_i]
            regular_pressure_j = pressure_j / self.container.particle_densities[p_j]
            pressure_sum = pressure_i + pressure_j
            density_i = self.container.particle_densities[p_i]
            
            if ti.abs(pressure_sum) > self.container.m_eps * self.container.particle_densities[p_i] * self.dt[None]:
                nabla_kernel = self.kernel.gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j], self.container.dh)
                ret.dv += self.container.particle_masses[p_j] * (regular_pressure_i / density_i) * nabla_kernel
                
                if self.container.particle_is_dynamic[p_j]:
                    object_j = self.container.particle_object_ids[p_j]
                    center_of_mass_j = self.container.rigid_body_com[object_j]
                    force_j = self.container.particle_masses[p_j] * (regular_pressure_i / density_i) * nabla_kernel * self.container.particle_masses[p_i] / self.dt[None]
                    torque_j = ti.math.cross(self.container.particle_positions[p_j] - center_of_mass_j, force_j)
                    self.container.rigid_body_forces[object_j] += force_j
                    self.container.rigid_body_torques[object_j] += torque_j

    @ti.kernel
    def compute_density_error(self) -> float:
        density_error = 0.0
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                density_error += self.container.particle_dfsph_predict_density[i] - self.density_0
        return density_error / self.container.particle_num[None]
    
    
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