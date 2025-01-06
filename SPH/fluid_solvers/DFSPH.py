import taichi as ti
from .base_solver import BaseSolver
from ..containers.dfsph_container import DFSPHContainer
from ..utils.kernel import *

@ti.data_oriented
class DFSPHSolver(BaseSolver):
    def __init__(self, container:DFSPHContainer):
        super().__init__(container)
        self.m_max_iterations_v = 1000
        self.m_max_iterations = 1000

        self.m_eps = 1e-5

        self.max_error_V = 0.001
        self.max_error = 0.0001
    
    @ti.kernel
    def compute_derivative_density(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == 1:
                ret = ti.Struct(derivative_density=0.0, num_neighbors=0)
                self.container.for_all_neighbors(i, self.compute_derivative_density_task, ret)
               
                derivative_density = ti.max(ret.derivative_density, 0.0)
                num_neighbors = ret.num_neighbors

                if num_neighbors < 20:
                    derivative_density = 0.0
               
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
                else:
                    factor_k = 0.0
                
                self.container.particle_dfsph_factor_k[i]= factor_k                
                
    @ti.func
    def compute_factor_k_task(self, p_i, p_j, ret: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            grad_p_j = self.container.particle_masses[p_j] * self.kernel.gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j], self.container.dh)
            ret.grad_norm_sum += grad_p_j.norm_sqr()
            ret.grad_sum += grad_p_j

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            grad_p_j = self.container.particle_masses[p_j] * self.kernel.gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j], self.container.dh)
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
                self.container.particle_dfsph_predict_density[p_i] = ti.max(predict_density, 1000.0)
                
    @ti.func
    def compute_predict_density_task(self, p_i, p_j, ret: ti.template()):
        # here we use partilce rest volume instead of mass
        # Fluid neighbor and rigid neighbor are treated the same
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
                self.container.particle_dfsph_pressure[p_i] = (self.container.particle_dfsph_predict_density[p_i] - self.density_0) * self.container.particle_dfsph_factor_k[p_i] / self.dt[None]
                

    ############## Divergence-free Solver ################
    def correct_divergence_error(self):
        num_iterations = 0
        average_density_derivative_error = 0.0
        
        while num_iterations < 1 or num_iterations < self.m_max_iterations:
            self.compute_derivative_density()
            self.compute_pressure_for_DFS()
            self.update_velocity_DFS()
            average_density_derivative_error = self.compute_density_derivative_error()

            eta = self.max_error_V * self.density_0 / self.dt[None]
            num_iterations += 1

            if average_density_derivative_error <= eta:
                break
            
        print(f"DFSPH - iteration V: {num_iterations} Avg density err: {average_density_derivative_error}")
            
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
            if ti.abs(pressure_sum) > self.m_eps * self.container.particle_densities[p_i] * self.dt[None]:
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
            if ti.abs(pressure_sum) > self.m_eps * self.container.particle_densities[p_i] * self.dt[None]:
                nabla_kernel = self.kernel.gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j], self.container.dh) 
                ret.dv += self.container.particle_masses[p_j] * (regular_pressure_i / density_i) * nabla_kernel
                
                if self.container.particle_is_dynamic[p_j]:
                    object_j = self.container.particle_object_ids[p_j]
                    center_of_mass_j = self.container.rigid_body_com[object_j]
                    force_j = (
                        self.container.particle_masses[p_j] * (regular_pressure_i / density_i) * nabla_kernel * self.container.particle_masses[p_i] / self.dt[None]
                    )
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
            if ti.abs(pressure_sum) > self.m_eps * self.container.particle_densities[p_i] * self.dt[None]:
                nabla_kernel = self.kernel.gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j], self.container.dh)
                ret.dv += self.container.particle_masses[p_j] * (regular_pressure_i / self.container.particle_densities[p_i] + regular_pressure_j / self.container.particle_densities[p_j]) * nabla_kernel
        
        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            pressure_i = ret.pressure
            pressure_j = pressure_i
            regular_pressure_i = pressure_i / self.container.particle_densities[p_i]
            regular_pressure_j = pressure_j / self.container.particle_densities[p_j]
            pressure_sum = pressure_i + pressure_j
            density_i = self.container.particle_densities[p_i]
            
            if ti.abs(pressure_sum) > self.m_eps * self.container.particle_densities[p_i] * self.dt[None]:
                nabla_kernel = self.kernel.gradient(self.container.particle_positions[p_i] - self.container.particle_positions[p_j], self.container.dh)
                ret.dv += self.container.particle_masses[p_j] * (regular_pressure_i / density_i) * nabla_kernel
                
                if self.container.particle_is_dynamic[p_j]:
                    object_j = self.container.particle_object_ids[p_j]
                    center_of_mass_j = self.container.rigid_body_com[object_j]
                    force_j = (
                        self.container.particle_masses[p_j] * (regular_pressure_i / density_i) * nabla_kernel * self.container.particle_masses[p_i] / self.dt[None]
                    )
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