import numpy as np
from numba import cuda, float32, int32
from .base_solver_numba import BaseSolver
from ..containers.dfsph_container_numba import DFSPHContainer
import math

@cuda.jit(device=True)
def kernel_gradient(r, h, out_grad):
    """核函数梯度"""
    r_norm = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    if r_norm > 0.0 and r_norm <= h:
        q = r_norm / h
        grad_q = -3.0 * (1.0 - q) * (1.0 - q) / (h * r_norm)
        for i in range(3):
            out_grad[i] = grad_q * r[i]
    else:
        for i in range(3):
            out_grad[i] = 0.0

class DFSPHSolver(BaseSolver):
    def __init__(self, container: DFSPHContainer):
        super().__init__(container)
        print("DFSPH Solver")
        self.m_eps = 1e-5
        self.max_error_V = 0.001
        self.max_error = 0.0001
        self.m_max_iterations_v = 1000
        self.m_max_iterations = 1000

    @staticmethod
    @cuda.jit
    def _compute_derivative_density_kernel(positions, masses, velocities, materials,
                                        dfsph_derivative_densities, material_fluid,
                                        particle_num, neighbors, neighbor_num, h):
        """计算密度导数的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
            
        if materials[idx] == material_fluid:
            derivative_density = 0.0
            num_neighbors = 0
            pos_i = positions[idx]
            vel_i = velocities[idx]
            
            # 遍历邻居粒子
            num_neighbors = neighbor_num[idx]
            for n_idx in range(num_neighbors):
                j = neighbors[idx, n_idx]
                
                if j == idx:
                    continue
                    
                r = cuda.local.array(3, dtype=float32)
                for d in range(3):
                    r[d] = pos_i[d] - positions[j][d]
                
                nabla_kernel = cuda.local.array(3, dtype=float32)
                kernel_gradient(r, h, nabla_kernel)
                vel_diff = cuda.local.array(3, dtype=float32)
                for d in range(3):
                    vel_diff[d] = vel_i[d] - velocities[j][d]
                
                dot_product = 0.0
                for d in range(3):
                    dot_product += vel_diff[d] * nabla_kernel[d]
                    
                derivative_density += masses[j] * dot_product
                num_neighbors += 1
            
            # 处理结果
            if num_neighbors < 20:
                derivative_density = 0.0
            else:
                derivative_density = max(derivative_density, 0.0)
                
            dfsph_derivative_densities[idx] = derivative_density

    def compute_derivative_density(self):
        """计算密度导数"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._compute_derivative_density_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_masses,
            self.container.particle_velocities,
            self.container.particle_materials,
            self.container.particle_dfsph_derivative_densities,
            self.container.material_fluid,
            self.container.particle_num,
            self.container.particle_neighbors,
            self.container.particle_neighbor_num,
            self.container.dh
        )

    @staticmethod
    @cuda.jit
    def _compute_factor_k_kernel(positions, masses, materials, densities, dfsph_factor_k,
                            material_fluid, particle_num, neighbors, neighbor_num, h):
        """计算factor k的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
            
        if materials[idx] == material_fluid:
            grad_sum = cuda.local.array(3, dtype=float32)
            for i in range(3):
                grad_sum[i] = 0.0
            grad_norm_sum = 0.0
            pos_i = positions[idx]
            
            # 遍历邻居粒子
            num_neighbors = neighbor_num[idx]
            for n_idx in range(num_neighbors):
                j = neighbors[idx, n_idx]
                
                if j == idx:
                    continue
                    
                if materials[j] == material_fluid:
                    # 流体粒子
                    r = cuda.local.array(3, dtype=float32)
                    for d in range(3):
                        r[d] = pos_i[d] - positions[j][d]
                    
                    grad_p_j = cuda.local.array(3, dtype=float32)
                    kernel_gradient(r, h, grad_p_j)
                    for d in range(3):
                        grad_p_j[d] *= masses[j]
                    
                    # 计算平方和
                    grad_norm = 0.0
                    for d in range(3):
                        grad_norm += grad_p_j[d] * grad_p_j[d]
                    grad_norm_sum += grad_norm
                    
                    # 累加梯度
                    for d in range(3):
                        grad_sum[d] += grad_p_j[d]
                        
            # 计算grad_sum的平方范数
            grad_sum_norm = 0.0
            for d in range(3):
                grad_sum_norm += grad_sum[d] * grad_sum[d]
                
            sum_grad = grad_sum_norm + grad_norm_sum
            
            # 计算factor k
            threshold = 1e-6 * densities[idx] * densities[idx]
            if sum_grad > threshold:
                dfsph_factor_k[idx] = densities[idx] * densities[idx] / sum_grad
            else:
                dfsph_factor_k[idx] = 0.0

    def compute_factor_k(self):
        """计算factor k"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._compute_factor_k_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_masses,
            self.container.particle_materials,
            self.container.particle_densities,
            self.container.particle_dfsph_factor_k,
            self.container.material_fluid,
            self.container.particle_num,
            self.container.particle_neighbors,
            self.container.particle_neighbor_num,
            self.container.dh
        )

    @staticmethod
    @cuda.jit
    def _compute_pressure_for_DFS_kernel(dfsph_derivative_densities, dfsph_factor_k, 
                                       dfsph_pressure_v, materials, material_fluid, particle_num):
        """计算DFS的压力的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
            
        if materials[idx] == material_fluid:
            dfsph_pressure_v[idx] = dfsph_derivative_densities[idx] * dfsph_factor_k[idx]

    def compute_pressure_for_DFS(self):
        """计算DFS的压力"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._compute_pressure_for_DFS_kernel[blocks, threads_per_block](
            self.container.particle_dfsph_derivative_densities,
            self.container.particle_dfsph_factor_k,
            self.container.particle_dfsph_pressure_v,
            self.container.particle_materials,
            self.container.material_fluid,
            self.container.particle_num
        )

    @staticmethod
    @cuda.jit
    def _compute_predict_density_kernel(positions, velocities, masses, materials, densities,
                                    dfsph_predict_density, material_fluid, particle_num,
                                    neighbors, neighbor_num, h, dt):
        """计算预测密度的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
            
        if materials[idx] == material_fluid:
            derivative_density = 0.0
            pos_i = positions[idx]
            vel_i = velocities[idx]
            
            # 遍历邻居粒子
            num_neighbors = neighbor_num[idx]
            for n_idx in range(num_neighbors):
                j = neighbors[idx, n_idx]
                
                if j == idx:
                    continue
                    
                r = cuda.local.array(3, dtype=float32)
                for d in range(3):
                    r[d] = pos_i[d] - positions[j][d]
                
                nabla_kernel = cuda.local.array(3, dtype=float32)
                kernel_gradient(r, h, nabla_kernel)
                vel_diff = cuda.local.array(3, dtype=float32)
                for d in range(3):
                    vel_diff[d] = vel_i[d] - velocities[j][d]
                
                dot_product = 0.0
                for d in range(3):
                    dot_product += vel_diff[d] * nabla_kernel[d]
                    
                derivative_density += masses[j] * dot_product
            
            predict_density = densities[idx] + dt * derivative_density
            dfsph_predict_density[idx] = max(predict_density, 1000.0)

    def compute_predict_density(self):
        """计算预测密度"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._compute_predict_density_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_velocities,
            self.container.particle_masses,
            self.container.particle_materials,
            self.container.particle_densities,
            self.container.particle_dfsph_predict_density,
            self.container.material_fluid,
            self.container.particle_num,
            self.container.particle_neighbors,
            self.container.particle_neighbor_num,
            self.container.dh,
            self.dt[0]
        )

    @staticmethod
    @cuda.jit
    def _compute_pressure_for_CDS_kernel(dfsph_predict_density, dfsph_factor_k, dfsph_pressure,
                                       materials, material_fluid, particle_num, density_0, dt):
        """计算CDS的压力的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
            
        if materials[idx] == material_fluid:
            dfsph_pressure[idx] = ((dfsph_predict_density[idx] - density_0[()]) * 
                                  dfsph_factor_k[idx] / dt)

    def compute_pressure_for_CDS(self):
        """计算CDS的压力"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._compute_pressure_for_CDS_kernel[blocks, threads_per_block](
            self.container.particle_dfsph_predict_density,
            self.container.particle_dfsph_factor_k,
            self.container.particle_dfsph_pressure,
            self.container.particle_materials,
            self.container.material_fluid,
            self.container.particle_num,
            self.density_0,
            self.dt[0]
        )

    @staticmethod
    @cuda.jit
    def _update_velocity_DFS_kernel(positions, velocities, masses, materials, densities,
                                dfsph_pressure_v, is_dynamic, object_ids, rigid_body_forces,
                                rigid_body_torques, rigid_body_com, material_fluid, 
                                material_rigid, particle_num, neighbors, neighbor_num,
                                h, m_eps, dt):
        """更新DFS速度的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
            
        if materials[idx] == material_fluid:
            pressure_i = dfsph_pressure_v[idx]
            dv = cuda.local.array(3, dtype=float32)
            for i in range(3):
                dv[i] = 0.0
            
            pos_i = positions[idx]
            
            # 遍历邻居粒子
            num_neighbors = neighbor_num[idx]
            for n_idx in range(num_neighbors):
                j = neighbors[idx, n_idx]
                
                if j == idx:
                    continue
                
                if materials[j] == material_fluid:
                    # 流体-流体相互作用
                    pressure_j = dfsph_pressure_v[j]
                    regular_pressure_i = pressure_i / densities[idx]
                    regular_pressure_j = pressure_j / densities[j]
                    pressure_sum = pressure_i + pressure_j
                    
                    if abs(pressure_sum) > m_eps * densities[idx] * dt:
                        r = cuda.local.array(3, dtype=float32)
                        for d in range(3):
                            r[d] = pos_i[d] - positions[j][d]
                        
                        nabla_kernel = cuda.local.array(3, dtype=float32)
                        kernel_gradient(r, h, nabla_kernel)
                        factor = masses[j] * (regular_pressure_i / densities[idx] + 
                                            regular_pressure_j / densities[j])
                        
                        for d in range(3):
                            dv[d] += factor * nabla_kernel[d]
                            
                elif materials[j] == material_rigid:
                    # 流体-刚体相互作用
                    pressure_j = pressure_i
                    regular_pressure_i = pressure_i / densities[idx]
                    regular_pressure_j = pressure_j / densities[j]
                    pressure_sum = pressure_i + pressure_j
                    density_i = densities[idx]
                    
                    if abs(pressure_sum) > m_eps * densities[idx] * dt:
                        r = cuda.local.array(3, dtype=float32)
                        for d in range(3):
                            r[d] = pos_i[d] - positions[j][d]
                        
                        nabla_kernel = cuda.local.array(3, dtype=float32)
                        kernel_gradient(r, h, nabla_kernel)
                        factor = masses[j] * regular_pressure_i / density_i
                        
                        for d in range(3):
                            dv[d] += factor * nabla_kernel[d]
                        
                        if is_dynamic[j]:
                            # 计算作用在刚体上的力和扭矩
                            object_j = object_ids[j]
                            force = cuda.local.array(3, dtype=float32)
                            for d in range(3):
                                force[d] = (factor * nabla_kernel[d] * 
                                        masses[idx] / dt)
                                cuda.atomic.add(rigid_body_forces[object_j], 
                                            d, force[d])
                            
                            # 计算扭矩
                            r_com = cuda.local.array(3, dtype=float32)
                            for d in range(3):
                                r_com[d] = positions[j][d] - rigid_body_com[object_j][d]
                            
                            torque = cuda.local.array(3, dtype=float32)
                            torque[0] = r_com[1]*force[2] - r_com[2]*force[1]
                            torque[1] = r_com[2]*force[0] - r_com[0]*force[2]
                            torque[2] = r_com[0]*force[1] - r_com[1]*force[0]
                            
                            for d in range(3):
                                cuda.atomic.add(rigid_body_torques[object_j], 
                                            d, torque[d])
        
            # 更新速度
            for d in range(3):
                velocities[idx][d] -= dv[d]

    def update_velocity_DFS(self):
        """更新DFS速度"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._update_velocity_DFS_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_velocities,
            self.container.particle_masses,
            self.container.particle_materials,
            self.container.particle_densities,
            self.container.particle_dfsph_pressure_v,
            self.container.particle_is_dynamic,
            self.container.particle_object_ids,
            self.container.rigid_body_forces,
            self.container.rigid_body_torques,
            self.container.rigid_body_com,
            self.container.material_fluid,
            self.container.material_rigid,
            self.container.particle_num,
            self.container.particle_neighbors,
            self.container.particle_neighbor_num,
            self.container.dh,
            self.m_eps,
            self.dt[0]
        )

    @staticmethod
    @cuda.jit
    def _compute_density_error_kernel(densities, materials, material_fluid, particle_num, density_0, error_sum):
        """计算密度误差的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
            
        if materials[idx] == material_fluid:
            error = abs(densities[idx] - density_0) / density_0 / 10.0
            cuda.atomic.add(error_sum, 0, error)

    def compute_density_error(self) -> float:
        """计算密度误差"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        # 使用CUDA reduction来计算总和
        error_sum = cuda.device_array(1, dtype=np.float32)
        error_sum[0] = 0.0
        
        self._compute_density_error_kernel[blocks, threads_per_block](
            self.container.particle_densities,
            self.container.particle_materials,
            self.container.material_fluid,
            self.container.particle_num,
            self.density_0[()],
            error_sum
        )
        
        return error_sum[0] / self.container.particle_num

    @staticmethod
    @cuda.jit
    def _compute_density_derivative_error_kernel(dfsph_derivative_densities, materials, material_fluid, particle_num, error_sum):
        """计算密度导数误差的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
            
        if materials[idx] == material_fluid:
            cuda.atomic.add(error_sum, 0, abs(dfsph_derivative_densities[idx]))

    def compute_density_derivative_error(self) -> float:
        """计算密度导数误差"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        # 使用CUDA reduction来计算总和
        error_sum = cuda.device_array(1, dtype=np.float32)
        error_sum[0] = 0.0
        
        self._compute_density_derivative_error_kernel[blocks, threads_per_block](
            self.container.particle_dfsph_derivative_densities,
            self.container.particle_materials,
            self.container.material_fluid,
            self.container.particle_num,
            error_sum
        )
        
        return error_sum[0] / self.container.particle_num

    @staticmethod
    @cuda.jit
    def _update_velocity_CDS_kernel(positions, velocities, masses, materials, densities,
                                dfsph_pressure, is_dynamic, object_ids, rigid_body_forces,
                                rigid_body_torques, rigid_body_com, material_fluid, 
                                material_rigid, particle_num, neighbors, neighbor_num,
                                h, m_eps, dt):
        """更新CDS速度的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
            
        if materials[idx] == material_fluid:
            pressure_i = dfsph_pressure[idx]
            dv = cuda.local.array(3, dtype=float32)
            for i in range(3):
                dv[i] = 0.0
            
            pos_i = positions[idx]
            
            # 遍历邻居粒子
            num_neighbors = neighbor_num[idx]
            for n_idx in range(num_neighbors):
                j = neighbors[idx, n_idx]
                
                if j == idx:
                    continue
                
                if materials[j] == material_fluid:
                    # 流体-流体相互作用
                    pressure_j = dfsph_pressure[j]
                    regular_pressure_i = pressure_i / densities[idx]
                    regular_pressure_j = pressure_j / densities[j]
                    pressure_sum = pressure_i + pressure_j
                    
                    if abs(pressure_sum) > m_eps * densities[idx] * dt:
                        r = cuda.local.array(3, dtype=float32)
                        for d in range(3):
                            r[d] = pos_i[d] - positions[j][d]
                        
                        nabla_kernel = cuda.local.array(3, dtype=float32)
                        kernel_gradient(r, h, nabla_kernel)
                        factor = masses[j] * (regular_pressure_i / densities[idx] + 
                                            regular_pressure_j / densities[j])
                        
                        for d in range(3):
                            dv[d] += factor * nabla_kernel[d]
                            
                elif materials[j] == material_rigid:
                    # 流体-刚体相互作用
                    pressure_j = pressure_i
                    regular_pressure_i = pressure_i / densities[idx]
                    regular_pressure_j = pressure_j / densities[j]
                    pressure_sum = pressure_i + pressure_j
                    density_i = densities[idx]
                    
                    if abs(pressure_sum) > m_eps * densities[idx] * dt:
                        r = cuda.local.array(3, dtype=float32)
                        for d in range(3):
                            r[d] = pos_i[d] - positions[j][d]
                        
                        nabla_kernel = cuda.local.array(3, dtype=float32)
                        kernel_gradient(r, h, nabla_kernel)
                        factor = masses[j] * regular_pressure_i / density_i
                        
                        for d in range(3):
                            dv[d] += factor * nabla_kernel[d]
                        
                        if is_dynamic[j]:
                            # 计算作用在刚体上的力和扭矩
                            object_j = object_ids[j]
                            force = cuda.local.array(3, dtype=float32)
                            for d in range(3):
                                force[d] = (factor * nabla_kernel[d] * 
                                        masses[idx] / dt)
                                cuda.atomic.add(rigid_body_forces[object_j], 
                                            d, force[d])
                            
                            # 计算扭矩
                            r_com = cuda.local.array(3, dtype=float32)
                            for d in range(3):
                                r_com[d] = positions[j][d] - rigid_body_com[object_j][d]
                            
                            torque = cuda.local.array(3, dtype=float32)
                            torque[0] = r_com[1]*force[2] - r_com[2]*force[1]
                            torque[1] = r_com[2]*force[0] - r_com[0]*force[2]
                            torque[2] = r_com[0]*force[1] - r_com[1]*force[0]
                            
                            for d in range(3):
                                cuda.atomic.add(rigid_body_torques[object_j], 
                                            d, torque[d])
        
            # 更新速度
            for d in range(3):
                velocities[idx][d] -= dv[d]

    def update_velocity_CDS(self):
        """更新CDS速度"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._update_velocity_CDS_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_velocities,
            self.container.particle_masses,
            self.container.particle_materials,
            self.container.particle_densities,
            self.container.particle_dfsph_pressure,
            self.container.particle_is_dynamic,
            self.container.particle_object_ids,
            self.container.rigid_body_forces,
            self.container.rigid_body_torques,
            self.container.rigid_body_com,
            self.container.material_fluid,
            self.container.material_rigid,
            self.container.particle_num,
            self.container.particle_neighbors,
            self.container.particle_neighbor_num,
            self.container.dh,
            self.m_eps,
            self.dt[0]
        )

    def _step(self):
        """执行一个时间步的模拟"""
        # 1. 计算非压力加速度（包括重力、粘度和表面张力）
        print("计算重力加速度")   
        self.compute_gravity_acceleration()
        print("计算粘度加速度")
        self.compute_viscosity_acceleration()
        print("计算表面张力加速度")
        self.compute_surface_tension_acceleration()
        
        # 2. 更新流体速度（半步）
        print("更新流体速度")
        self.update_fluid_velocity()
        
        # 3. 密度修正迭代
        print("修正密度误差")
        num_itr = 0
        while num_itr < 1 or num_itr < self.m_max_iterations:
            self.compute_predict_density()
            if num_itr == 0:
                print("计算CDS压力")
            self.compute_pressure_for_CDS()
            if num_itr == 0:
                print("更新CDS速度")
            self.update_velocity_CDS()
            average_density_error = self.compute_density_error()
            eta = self.max_error
            num_itr += 1

            if average_density_error <= eta * self.density_0[()]:
                break
        
        # 4. 更新流体位置
        print("更新流体位置")
        self.update_fluid_position()

        # 5. 刚体求解器步进
        print("刚体求解器步进")
        self.rigid_solver.step()
        
        # 6. 插入物体
        print("插入物体")
        self.container.insert_object()
        print("插入刚体")
        self.rigid_solver.insert_rigid_object()
        print("更新刚体粒子状态")
        self.renew_rigid_particle_state()
        
        # 7. 边界处理
        print("边界处理")
        self.boundary.enforce_domain_boundary(self.container.material_fluid)
        
        # 8. 准备邻居搜索
        print("准备邻居搜索")
        self.container.prepare_neighbor_search()
        
        # 9. 计算密度
        print("计算密度")
        self.compute_density()
        
        # 10. 计算因子k
        print("计算因子k")
        self.compute_factor_k()
        
        # 11. 无散度修正迭代
        print("修正散度误差")
        num_iterations = 0
        while num_iterations < 1 or num_iterations < self.m_max_iterations_v:
            if num_iterations == 0:
                print("计算密度导数")
            self.compute_derivative_density()
            if num_iterations == 0:
                print("计算DFS压力")
            self.compute_pressure_for_DFS()
            if num_iterations == 0:
                print("更新DFS速度")
            self.update_velocity_DFS()
            average_density_derivative_error = self.compute_density_derivative_error()

            eta = self.max_error_V * self.density_0[()] / self.dt[()]
            num_iterations += 1

            if average_density_derivative_error <= eta:
                break
        
        print("一轮模拟结束")