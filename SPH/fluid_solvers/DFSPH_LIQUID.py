import taichi as ti
from .base_solver import BaseSolver
from ..containers.dfsph_container import DFSPHContainer
from ..utils.kernel import *

cnt = 0

@ti.data_oriented
class DFSPH_LSolver(BaseSolver):
    def __init__(self, container:DFSPHContainer):
        super().__init__(container)
        
        self.fluid_density_error = ti.field(float, shape=self.container.max_object_num)
        self.fluid_density_count = ti.field(int, shape=self.container.max_object_num)
        self.fluid_object_num = ti.field(int, shape=())
        self.fluid_object_list = ti.field(int, shape=self.container.max_object_num)  # 例如最多128个流体对象
    
    @ti.kernel
    def identify_fluid_objects(self):
        count = 0
        for obj_id in range(self.container.max_object_num):
            if self.container.object_materials[obj_id] == self.container.material_fluid:
                self.fluid_object_list[count] = obj_id
                count += 1
        self.fluid_object_num[None] = count
    
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
                if predict_density < self.container.particle_rest_densities[p_i]:
                    predict_density = self.container.particle_rest_densities[p_i]
                    
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
                error_density = self.container.particle_dfsph_predict_density[p_i] - self.container.particle_rest_densities[p_i]
                self.container.particle_dfsph_pressure[p_i] = error_density * self.container.particle_dfsph_factor_k[p_i] / self.dt[None]
                
    ############## Divergence-free Solver ################
    def correct_divergence_error(self):
        num_iterations = 0
        while num_iterations < 1 or num_iterations < self.container.m_max_iterations:
            self.compute_derivative_density()
            self.compute_pressure_for_DFS()
            self.update_velocity_DFS()
            all_errors = self.compute_density_derivative_error() 

            all_pass = True
            for i, e in enumerate(all_errors):
                oid = self.fluid_object_list[i]
                fluid_density = self.container.object_densities[oid]
                if e > (self.container.max_error_V * fluid_density / self.dt[None]):
                    all_pass = False
                    break

            num_iterations += 1
            if all_pass:
                break

        print(f"DFSPH - iteration V: {num_iterations} Fluid Errors: {all_errors}")
            
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
    def compute_density_derivative_error_kernel(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                oid = self.container.particle_object_ids[i]
                err = ti.abs(self.container.particle_dfsph_derivative_densities[i])
                ti.atomic_add(self.fluid_density_error[oid], err)
                ti.atomic_add(self.fluid_density_count[oid], 1)

    def compute_density_derivative_error(self):
        for i in range(self.container.max_object_num):
            self.fluid_density_error[i] = 0.0
            self.fluid_density_count[i] = 0
        self.compute_density_derivative_error_kernel()

        errors = []
        fluid_num = self.container.fluid_object_num[None]
        for idx in range(fluid_num):
            oid = self.fluid_object_list[idx]
            if self.fluid_density_count[oid] > 0:
                errors.append(self.fluid_density_error[oid] / self.fluid_density_count[oid])
            else:
                errors.append(0.0)
        return errors
    
    ################# Constant Density Solver #################
    def correct_density_error(self):
        num_itr = 0
        while num_itr < 1 or num_itr < self.container.m_max_iterations:
            self.compute_predict_density()
            self.compute_pressure_for_CDS()
            self.update_velocity_CDS()

            all_errors = self.compute_density_error()

            all_pass = True

            fluid_num = self.container.fluid_object_num[None]
            for idx, e in enumerate(all_errors):
                oid = self.fluid_object_list[idx]
                if e > self.container.max_error * self.container.object_densities[oid]:
                    all_pass = False
                    break

            num_itr += 1
            if all_pass:
                break
        print(f"DFSPH - iterations: {num_itr} Fluid Errors: {all_errors}")
    
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
                    force_j =  self.container.particle_masses[p_j] * (regular_pressure_i / density_i) * nabla_kernel * self.container.particle_masses[p_i] / self.dt[None]
                    torque_j = ti.math.cross(self.container.particle_positions[p_j] - center_of_mass_j, force_j)
                    self.container.rigid_body_forces[object_j] += force_j
                    self.container.rigid_body_torques[object_j] += torque_j

    @ti.kernel
    def compute_density_error_kernel(self):
        for i in range(self.container.particle_num[None]):
            if self.container.particle_materials[i] == self.container.material_fluid:
                oid = self.container.particle_object_ids[i]
                err = ti.abs(self.container.particle_dfsph_predict_density[i] - self.container.particle_rest_densities[i])
                ti.atomic_add(self.fluid_density_error[oid], err)
                ti.atomic_add(self.fluid_density_count[oid], 1)

    def compute_density_error(self):
        for i in range(self.container.max_object_num):
            self.fluid_density_error[i] = 0.0
            self.fluid_density_count[i] = 0

        self.compute_density_error_kernel()

        errors = []
        fluid_num = self.container.fluid_object_num[None]
        for idx in range(fluid_num):
            oid = self.fluid_object_list[idx]
            if self.fluid_density_count[oid] > 0:
                errors.append(self.fluid_density_error[oid] / self.fluid_density_count[oid])
            else:
                errors.append(0.0)
        return errors
    
    
    def _step(self):
        global cnt
        if cnt == 0:
            print("计算非压力加速度")
        self.compute_non_pressure_acceleration()
        if cnt == 0:
            print("更新速度")
        self.update_fluid_velocity()
        if cnt == 0:
            print("修正密度误差")
        self.correct_density_error()

        if cnt == 0:
            print("更新位置")
        self.update_fluid_position()

        if cnt == 0:
            print("刚体求解器")
        self.rigid_solver.step()

        self.container.insert_object()
        self.rigid_solver.insert_rigid_object()
        if cnt == 0:
            print("更新刚体位置")
        self.renew_rigid_particle_state()

        if cnt == 0:
            print("边界处理")
        self.boundary.enforce_domain_boundary(self.container.material_fluid)

        if cnt == 0:
            print("邻居搜索")
        self.container.prepare_neighbor_search()
        if cnt == 0:
            print("计算密度")
        self.compute_density()
        if cnt == 0:
            print("计算因子k")
        self.compute_factor_k()
        if cnt == 0:
            print("修正散度误差")
        self.correct_divergence_error()
        cnt += 1

    def prepare(self):
        super().prepare()
        self.compute_factor_k()
        self.compute_density()
        self.identify_fluid_objects()