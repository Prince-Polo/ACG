import taichi as ti
from .base_solver import BaseSolverBaseline
from ..containers.iisph_container import IISPHContainerBaseline
from .utils import *

@ti.data_oriented
class IISPHSolverBaseline(BaseSolverBaseline):
    def __init__(self, container: IISPHContainerBaseline):
        super().__init__(container)
        self.max_iterations = 2000
        self.eta = 0.001 # This criterion is given by our reference paper
        self.omega = 0.08

    @ti.kernel
    def compute_sum_dij(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                ret = ti.Vector([0.0, 0.0, 0.0])
                self.container.for_all_neighbors(i, self.compute_sum_dij_task, ret)
                self.container.iisph_sum_dij[i] = ret
              
    @ti.func  
    def compute_sum_dij_task(self, i, j, ret: ti.template()):
        nabla_kernel = self.kernel.gradient(self.container.particle_positions[i] - self.container.particle_positions[j], self.container.dh)
        
        ret += self.container.particle_masses[j] * nabla_kernel / self.container.particle_densities[j] / self.container.particle_densities[j]

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
        d_ii = self.container.particle_masses[i] * nabla_kernel / self.container.particle_densities[i] / self.container.particle_densities[i]
        
        ret += self.container.particle_masses[j] * ti.math.dot(self.container.iisph_sum_dij[i] + d_ii, nabla_kernel)

    @ti.kernel
    def compute_source_term(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                ret = 0.0
                self.container.for_all_neighbors(i, self.compute_source_term_task, ret)
                self.container.iisph_source[i] = self.container.particle_rest_densities[i] - self.container.particle_densities[i] - self.dt[None] * ret

    @ti.func
    def compute_source_term_task(self, i, j, ret: ti.template()):
        nabla_kernel = self.kernel.gradient(self.container.particle_positions[i] - self.container.particle_positions[j], self.container.dh)
        ret += self.container.particle_masses[j] * ti.math.dot(self.container.particle_velocities[i] - self.container.particle_velocities[j], nabla_kernel)

    @ti.kernel
    def init_pressure(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
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
        nabla_kernel = self.kernel.gradient(self.container.particle_positions[i] - self.container.particle_positions[j], self.container.dh)
        regular_iisph_pressure_i = self.container.particle_pressures[i] / self.container.particle_densities[i] / self.container.particle_densities[i]
        regular_iisph_pressure_j = self.container.particle_pressures[j] / self.container.particle_densities[j] / self.container.particle_densities[j]
       
        if self.container.particle_materials[j] == self.container.material_fluid:
            ret -= self.container.particle_masses[j] * (regular_iisph_pressure_i + regular_iisph_pressure_j) * nabla_kernel
            
        elif self.container.particle_materials[j] == self.container.material_rigid:
            ret -= self.container.particle_masses[j] * regular_iisph_pressure_i * nabla_kernel
            
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
                    new_pressure = self.container.particle_pressures[i] + (self.omega * (self.container.iisph_source[i] - self.container.iisph_laplacian[i]) / self.container.iisph_a_ii[i])
                    new_pressure = ti.max(0.0, new_pressure)
                    
                self.container.particle_pressures[i] = new_pressure
                
                if new_pressure > 1e-8:
                    error += ti.abs((self.container.iisph_laplacian[i] - self.container.iisph_source[i]) / self.container.particle_rest_densities[i])
            
        if self.container.fluid_particle_num[None] > 0:
            self.container.density_error[None] = error / self.container.fluid_particle_num[None]
        else:
            self.container.density_error[None] = 0.0
    
    def refine(self):
        num_iters = 0
        
        while num_iters < self.max_iterations:
            self.compute_pressure_acceleration()
            self.compute_Laplacian()
            self.update_pressure()
            
            num_iters += 1
            
            if self.container.density_error[None] < self.eta:
                break
        
        print(f"IISPH - iteration: {num_iters} Avg density err: {self.container.density_error[None] * self.density_0}")
    
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
            nabla_kernel = self.kernel.gradient(self.container.particle_positions[i] - self.container.particle_positions[j], self.container.dh)
            regular_iisph_pressure_i = self.container.particle_pressures[i] / self.container.particle_densities[i] / self.container.particle_densities[i]
            object_j = self.container.particle_object_ids[j]
            center_of_mass_j = self.container.rigid_body_com[object_j]
                
            force_j = self.container.particle_masses[j] * regular_iisph_pressure_i * nabla_kernel * self.container.particle_masses[i]
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
