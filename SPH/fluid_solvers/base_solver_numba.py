import numpy as np
from numba import cuda, float32, int32
from ..kernels.cubic_spline_numba import CubicSpline, _weight, _gradient
from ..containers.base_container_numba import BaseContainer, pos_to_index, flatten_grid_index
from ..rigid_solver.rigid_solver_numba import RigidSolver
from .boundary_numba import Boundary
import math

@cuda.jit(device=True)
def kernel_weight(r, h, out_weight):
    """核函数权重"""
    q = r / h
    if q <= 1.0:
        out_weight[0] = (1.0 - q * q) * (1.0 - q * q) * (1.0 - q * q)
    else:
        out_weight[0] = 0.0

@cuda.jit(device=True)
def pos_to_index(pos, grid_size, grid_num, out_index):
    """将位置转换为网格索引"""
    for i in range(3):
        out_index[i] = int(pos[i] / grid_size[i])
        # 确保索引在有效范围内
        if out_index[i] < 0:
            out_index[i] = 0
        if out_index[i] >= grid_num[i]:
            out_index[i] = grid_num[i] - 1

@cuda.jit(device=True)
def get_flatten_grid_index(pos, grid_size, grid_num, temp_index):
    """获取位置对应的扁平化网格索引"""
    pos_to_index(pos, grid_size, grid_num, temp_index)
    ret = 0
    for i in range(3):
        ret_p = temp_index[i]
        for j in range(i+1, 3):
            ret_p *= grid_num[j]
        ret += ret_p
    return ret

class BaseSolver:
    def __init__(self, container: BaseContainer):
        self.container = container
        self.cfg = container.cfg
        self.total_time = 0.0

        # Gravity
        self.g = cuda.to_device(np.array(self.container.cfg.get_cfg("gravitation")))

        # density
        self.density = 1000.0
        self.density_0 = cuda.to_device(np.array(self.container.cfg.get_cfg("density0")))

        # surface tension
        if self.container.cfg.get_cfg("surface_tension"):
            self.surface_tension = cuda.to_device(np.array(self.container.cfg.get_cfg("surface_tension")))
        else:
            self.surface_tension = cuda.to_device(np.array([0.01], dtype=np.float32))

        # viscosity
        self.viscosity = cuda.to_device(np.array(self.container.cfg.get_cfg("viscosity")))
        if self.container.cfg.get_cfg("viscosity_b"):
            self.viscosity_b = cuda.to_device(np.array(self.container.cfg.get_cfg("viscosity_b")))
        else:
            self.viscosity_b = self.viscosity

        # time step
        self.dt = cuda.to_device(np.array([self.container.cfg.get_cfg("timeStepSize")], dtype=np.float32))

        # kernel
        self.kernel = CubicSpline()

        # boundary
        self.boundary = Boundary(self.container)

        # others
        if self.container.cfg.get_cfg("gravitationUpper"):
            self.g_upper = cuda.to_device(np.array(self.container.cfg.get_cfg("gravitationUpper")))
        else:
            self.g_upper = cuda.to_device(np.array([10000.0], dtype=np.float32))

        # rigid solver
        self.rigid_solver = RigidSolver(self.container, gravity=self.g, dt=self.dt[0])

        self.fluid_object_id = 0
        self.rigid_object_id = 1

    @staticmethod
    @cuda.jit
    def _compute_density_kernel(positions, masses, densities, materials, material_fluid,
                              particle_num, grid_ids, grid_num_particles, grid_size, 
                              grid_num, h):
        """计算密度的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
            
        if materials[idx] == material_fluid:
            density = 0.0
            pos_i = positions[idx]
            
            # 获取当前粒子的网格索引
            center_cell = cuda.local.array(3, dtype=int32)
            pos_to_index(pos_i, grid_size, grid_num, center_cell)
            
            # 遍历相邻网格
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        neighbor_cell = cuda.local.array(3, dtype=int32)
                        neighbor_cell[0] = center_cell[0] + i
                        neighbor_cell[1] = center_cell[1] + j
                        neighbor_cell[2] = center_cell[2] + k
                        
                        grid_idx = flatten_grid_index(neighbor_cell, grid_num)
                        start_idx = 0 if grid_idx == 0 else grid_num_particles[grid_idx - 1]
                        end_idx = grid_num_particles[grid_idx]
                        
                        # 遍历网格内的粒子
                        for j in range(start_idx, end_idx):
                            if j != idx:
                                r = cuda.local.array(3, dtype=float32)
                                r_norm_sq = 0.0
                                for d in range(3):
                                    r[d] = pos_i[d] - positions[j][d]
                                    r_norm_sq += r[d] * r[d]
                                r_norm = math.sqrt(r_norm_sq)
                                
                                if r_norm < h:
                                    weight = cuda.local.array(1, dtype=float32)
                                    kernel_weight(r_norm, h, weight)  # 直接使用文件中定义的kernel_weight
                                    density += masses[j] * weight[0]
            
            densities[idx] = density

    def compute_density(self):
        """计算密度"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._compute_density_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_masses,
            self.container.particle_densities,
            self.container.particle_materials,
            self.container.material_fluid,
            self.container.particle_num,
            self.container.grid_ids,
            self.container.grid_num_particles,
            self.container.grid_size,
            self.container.grid_num,
            self.container.dh
        )

    @staticmethod
    @cuda.jit
    def _compute_rigid_particle_volume_kernel(positions, object_ids, materials, rest_volumes, 
                                            masses, material_rigid, g_upper, particle_num,
                                            grid_ids, grid_num_particles, grid_size, grid_num,
                                            h, density_0):
        """计算刚体粒子体积的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        if materials[idx] == material_rigid and positions[idx][1] <= g_upper[0]:
            # 初始化体积
            weight = cuda.local.array(1, dtype=float32)
            _weight(0.0, h, weight)  # 自身贡献
            volume = weight[0]
            
            # 获取当前粒子的网格索引
            center_cell = cuda.local.array(3, dtype=int32)
            pos_to_index(positions[idx], grid_size, grid_num, center_cell)
            
            # 遍历相邻网格
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        neighbor_cell = cuda.local.array(3, dtype=int32)
                        neighbor_cell[0] = center_cell[0] + i
                        neighbor_cell[1] = center_cell[1] + j
                        neighbor_cell[2] = center_cell[2] + k
                        
                        # 确保在网格范围内
                        if (neighbor_cell[0] >= 0 and neighbor_cell[0] < grid_num[0] and
                            neighbor_cell[1] >= 0 and neighbor_cell[1] < grid_num[1] and
                            neighbor_cell[2] >= 0 and neighbor_cell[2] < grid_num[2]):
                            
                            # 计算网格ID
                            grid_idx = (neighbor_cell[0] + 
                                      neighbor_cell[1] * grid_num[0] + 
                                      neighbor_cell[2] * grid_num[0] * grid_num[1])
                            start_idx = 0 if grid_idx == 0 else grid_num_particles[grid_idx - 1]
                            end_idx = grid_num_particles[grid_idx]
                            
                            # 遍历网格内的粒子
                            for j in range(start_idx, end_idx):
                                if j != idx and object_ids[j] == object_ids[idx]:
                                    r = cuda.local.array(3, dtype=float32)
                                    r_norm_sq = 0.0
                                    for d in range(3):
                                        r[d] = positions[idx, d] - positions[j, d]
                                        r_norm_sq += r[d] * r[d]
                                    r_norm = math.sqrt(r_norm_sq)
                                    
                                    if r_norm < h:
                                        kernel_weight(r_norm, h, weight)
                                        volume += weight[0]
                
            # 计算体积和质量
            rest_volumes[idx] = 1.0 / volume
            masses[idx] = density_0[()] * rest_volumes[idx]

    def compute_rigid_particle_volume(self):
        """计算刚体粒子体积"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._compute_rigid_particle_volume_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_object_ids,
            self.container.particle_materials,
            self.container.particle_rest_volumes,
            self.container.particle_masses,
            self.container.material_rigid,
            self.g_upper,
            self.container.particle_num,
            self.container.grid_ids,
            self.container.grid_num_particles,
            self.container.grid_size,
            self.container.grid_num,
            self.container.dh,
            self.density_0
        )

    @cuda.jit
    def _compute_pressure_acceleration_kernel(positions, velocities, masses, densities, 
                                           pressures, materials, is_dynamic, object_ids,
                                           accelerations, rigid_body_forces, rigid_body_torques,
                                           rigid_body_com, material_fluid, material_rigid,
                                           particle_num, grid_ids, grid_num_particles, 
                                           grid_size, grid_num, kernel_gradient, h):
        """计算压力加速度的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        if not is_dynamic[idx] or materials[idx] != material_fluid:
            return
        
        # 初始化加速度
        acc = cuda.local.array(3, dtype=float32)
        for i in range(3):
            acc[i] = 0.0
        
        pos_i = positions[idx]
        center_cell = pos_to_index(pos_i, grid_size)
        
        # 遍历相邻网格
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    neighbor_cell = cuda.local.array(3, dtype=int32)
                    neighbor_cell[0] = center_cell[0] + i
                    neighbor_cell[1] = center_cell[1] + j
                    neighbor_cell[2] = center_cell[2] + k
                    
                    grid_idx = flatten_grid_index(neighbor_cell, grid_num)
                    start_idx = 0 if grid_idx == 0 else grid_num_particles[grid_idx - 1]
                    end_idx = grid_num_particles[grid_idx]
                    
                    # 遍历网格内的粒子
                    for j in range(start_idx, end_idx):
                        if j == idx:
                            continue
                        
                        # 计算
                        r = cuda.local.array(3, dtype=float32)
                        r_norm_sq = 0.0
                        for d in range(3):
                            r[d] = pos_i[d] - positions[j][d]
                            r_norm_sq += r[d] * r[d]
                        r_norm = math.sqrt(r_norm_sq)
                        
                        if r_norm < h:
                            # 计算核函数梯度
                            nabla_ij = kernel_gradient(r, h)
                            
                            if materials[j] == material_fluid:
                                # 流体-流体相互作用
                                factor = masses[j] * (pressures[idx]/(densities[idx]**2) + 
                                                    pressures[j]/(densities[j]**2))
                                for d in range(3):
                                    acc[d] -= factor * nabla_ij[d]
                                    
                            elif materials[j] == material_rigid:
                                # 流体-刚体相互作用
                                factor = masses[j] * pressures[idx]/(densities[idx]**2)
                                for d in range(3):
                                    acc[d] -= factor * nabla_ij[d]
                                
                                if is_dynamic[j]:
                                    # 计算作用在刚体上的力和扭矩
                                    object_j = object_ids[j]
                                    force = cuda.local.array(3, dtype=float32)
                                    for d in range(3):
                                        force[d] = factor * nabla_ij[d] * masses[idx]
                                        cuda.atomic.add(rigid_body_forces[object_j], d, force[d])
                                    
                                    # 计算扭矩
                                    r_com = cuda.local.array(3, dtype=float32)
                                    for d in range(3):
                                        r_com[d] = pos_i[d] - rigid_body_com[object_j][d]
                                    
                                    torque = cuda.local.array(3, dtype=float32)
                                    torque[0] = r_com[1]*force[2] - r_com[2]*force[1]
                                    torque[1] = r_com[2]*force[0] - r_com[0]*force[2]
                                    torque[2] = r_com[0]*force[1] - r_com[1]*force[0]
                                    
                                    for d in range(3):
                                        cuda.atomic.add(rigid_body_torques[object_j], d, torque[d])
        
        # 更新速度
        for i in range(3):
            accelerations[idx][i] = acc[i]

    def compute_pressure_acceleration(self):
        """计算压力加速度"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._compute_pressure_acceleration_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_velocities,
            self.container.particle_masses,
            self.container.particle_densities,
            self.container.particle_pressures,
            self.container.particle_materials,
            self.container.particle_is_dynamic,
            self.container.particle_object_ids,
            self.container.particle_accelerations,
            self.container.rigid_body_forces,
            self.container.rigid_body_torques,
            self.container.rigid_body_com,
            self.container.material_fluid,
            self.container.material_rigid,
            self.container.particle_num,
            self.container.grid_ids,
            self.container.grid_num_particles,
            self.container.grid_size,
            self.container.grid_num,
            self.kernel.gradient,
            self.container.dh
        )

    @staticmethod
    @cuda.jit
    def _init_acceleration_kernel(accelerations, particle_num):
        """初始化加速度的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        for i in range(3):
            accelerations[idx][i] = 0.0

    def init_acceleration(self):
        """初始化加速度"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._init_acceleration_kernel[blocks, threads_per_block](
            self.container.particle_accelerations,
            self.container.particle_num
        )

    @staticmethod
    @cuda.jit
    def _init_rigid_body_force_and_torque_kernel(forces, torques, rigid_body_num):
        """初始化刚体力和扭矩的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= rigid_body_num:
            return
        
        for i in range(3):
            forces[idx][i] = 0.0
            torques[idx][i] = 0.0

    def init_rigid_body_force_and_torque(self):
        """初始化刚体力和扭矩"""
        threads_per_block = 256
        blocks = (self.container.rigid_body_num + threads_per_block - 1) // threads_per_block
        
        self._init_rigid_body_force_and_torque_kernel[blocks, threads_per_block](
            self.container.rigid_body_forces,
            self.container.rigid_body_torques,
            self.container.rigid_body_num
        )

    @staticmethod
    @cuda.jit
    def _compute_gravity_acceleration_kernel(accelerations, materials, material_fluid, 
                                          particle_num, gravity):
        """计算重力加速度的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        if materials[idx] == material_fluid:
            for i in range(3):
                accelerations[idx][i] = gravity[i]

    def compute_gravity_acceleration(self):
        """计算重力加速度"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._compute_gravity_acceleration_kernel[blocks, threads_per_block](
            self.container.particle_accelerations,
            self.container.particle_materials,
            self.container.material_fluid,
            self.container.particle_num,
            self.g
        )

    @staticmethod
    @cuda.jit
    def _compute_surface_tension_acceleration_kernel(positions, masses, materials, is_dynamic,
                                                   accelerations, material_fluid, particle_num,
                                                   grid_ids, grid_num_particles, grid_size,
                                                   grid_num, h, surface_tension):
        """计算表面张力加速度的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        if materials[idx] == material_fluid:
            # 初始化加速度
            a_i = cuda.local.array(3, dtype=float32)
            for i in range(3):
                a_i[i] = 0.0
            
            pos_i = positions[idx]
            temp_index = cuda.local.array(3, dtype=int32)
            pos_to_index(pos_i, grid_size, grid_num, temp_index)  # 修正参数顺序
            
            # 遍历相邻网格
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        neighbor_cell = cuda.local.array(3, dtype=int32)
                        neighbor_cell[0] = temp_index[0] + i
                        neighbor_cell[1] = temp_index[1] + j
                        neighbor_cell[2] = temp_index[2] + k
                        
                        grid_idx = flatten_grid_index(neighbor_cell, grid_num)
                        start_idx = 0 if grid_idx == 0 else grid_num_particles[grid_idx - 1]
                        end_idx = grid_num_particles[grid_idx]
                        
                        # 遍历网格内的粒子
                        for j in range(start_idx, end_idx):
                            if j != idx and materials[j] == material_fluid:
                                r = cuda.local.array(3, dtype=float32)
                                r_norm_sq = 0.0
                                for d in range(3):
                                    r[d] = pos_i[d] - positions[j][d]
                                    r_norm_sq += r[d] * r[d]
                                r_norm = math.sqrt(r_norm_sq)
                                
                                if r_norm < h:
                                    # 计算质量比
                                    mass_ratio = masses[j] / masses[idx]
                                    # 计算权重
                                    weight = cuda.local.array(1, dtype=float32)
                                    kernel_weight(r_norm, h, weight)
                                    # 计算系数
                                    coef = surface_tension[0] * mass_ratio * weight[0]  # 先计算所有标量的乘积
                                    # 更新加速度
                                    for d in range(3):
                                        a_i[d] -= coef * r[d]  # 然后与向量分量相乘
            
            # 更新速度
            for i in range(3):
                cuda.atomic.add(accelerations[idx], i, a_i[i])

    def compute_surface_tension_acceleration(self):
        """计算表面张力加速度"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._compute_surface_tension_acceleration_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_masses,
            self.container.particle_materials,
            self.container.particle_is_dynamic,
            self.container.particle_accelerations,
            self.container.material_fluid,
            self.container.particle_num,
            self.container.grid_ids,
            self.container.grid_num_particles,
            self.container.grid_size,
            self.container.grid_num,
            self.container.dh,
            self.surface_tension
        )

    @staticmethod
    @cuda.jit
    def _compute_viscosity_acceleration_kernel(positions, velocities, masses, materials,
                                        is_dynamic, object_ids, accelerations,
                                        rigid_body_forces, rigid_body_torques,
                                        rigid_body_com, material_fluid, material_rigid,
                                        particle_num, grid_ids, grid_num_particles,
                                        grid_size, grid_num, h,
                                        viscosity, viscosity_b, density_0,
                                        rest_densities, rest_volumes):
        """计算粘度加速度的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        if materials[idx] == material_fluid:
            # 初始化加速度
            a_i = cuda.local.array(3, dtype=float32)
            for i in range(3):
                a_i[i] = 0.0
            
            pos_i = positions[idx]
            temp_index = cuda.local.array(3, dtype=int32)
            pos_to_index(pos_i, grid_size, grid_num, temp_index)  # 获取当前粒子的网格索引
            
            # 遍历相邻网格
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        neighbor_cell = cuda.local.array(3, dtype=int32)
                        neighbor_cell[0] = temp_index[0] + i
                        neighbor_cell[1] = temp_index[1] + j
                        neighbor_cell[2] = temp_index[2] + k
                        
                        # 确保在网格范围内
                        if (neighbor_cell[0] >= 0 and neighbor_cell[0] < grid_num[0] and
                            neighbor_cell[1] >= 0 and neighbor_cell[1] < grid_num[1] and
                            neighbor_cell[2] >= 0 and neighbor_cell[2] < grid_num[2]):
                            
                            grid_idx = flatten_grid_index(neighbor_cell, grid_num)
                            start_idx = 0 if grid_idx == 0 else grid_num_particles[grid_idx - 1]
                            end_idx = grid_num_particles[grid_idx]
                            
                            # 遍历网格内的粒子
                            for j in range(start_idx, end_idx):
                                if j == idx:
                                    continue
                                
                                # 计算距离向量
                                r = cuda.local.array(3, dtype=float32)
                                r_norm_sq = 0.0
                                for d in range(3):
                                    r[d] = pos_i[d] - positions[j][d]
                                    r_norm_sq += r[d] * r[d]
                                r_norm = math.sqrt(r_norm_sq)
                                
                                if r_norm < h:
                                    nabla_ij = cuda.local.array(3, dtype=float32)
                                    _gradient(r, h, nabla_ij)  # 直接使用导入的函数
                                    
                                    # 计算速度点积
                                    v_xy = 0.0
                                    for d in range(3):
                                        v_xy += (velocities[idx][d] - velocities[j][d]) * r[d]
                                    
                                    if materials[j] == material_fluid:
                                        # 流体-流体粘度
                                        factor = (2 * (3 + 2) * viscosity[()] * 
                                                (masses[idx] + masses[j]) / 2 /
                                                rest_densities[j] /
                                                (r_norm_sq + 0.01 * h * h) * v_xy)
                                        
                                        for d in range(3):
                                            a_i[d] += factor * nabla_ij[d]
                                            
                                    elif materials[j] == material_rigid:
                                        # 流体-刚体粘度
                                        acc = cuda.local.array(3, dtype=float32)
                                        factor = (2 * (3 + 2) * viscosity_b[()] * 
                                                (density_0[()] * rest_volumes[j]) /
                                                rest_densities[idx] /
                                                (r_norm_sq + 0.01 * h * h) * v_xy)
                                        
                                        for d in range(3):
                                            acc[d] = factor * nabla_ij[d]
                                            a_i[d] += acc[d]
                                        
                                        if is_dynamic[j]:
                                            # 计算作用在刚体上的力和扭矩
                                            object_j = object_ids[j]
                                            force = cuda.local.array(3, dtype=float32)
                                            for d in range(3):
                                                force[d] = -acc[d] * masses[idx] / density_0[()]
                                                cuda.atomic.add(rigid_body_forces[object_j], d, force[d])
                                            
                                            # 计算扭矩
                                            r_com = cuda.local.array(3, dtype=float32)
                                            for d in range(3):
                                                r_com[d] = positions[j][d] - rigid_body_com[object_j][d]
                                            
                                            torque = cuda.local.array(3, dtype=float32)
                                            torque[0] = r_com[1]*force[2] - r_com[2]*force[1]
                                            torque[1] = r_com[2]*force[0] - r_com[0]*force[2]
                                            torque[2] = r_com[0]*force[1] - r_com[1]*force[0]
                                            
                                            for d in range(3):
                                                cuda.atomic.add(rigid_body_torques[object_j], d, torque[d])
            
            # 更新加速度
            for i in range(3):
                accelerations[idx][i] += a_i[i] / rest_densities[idx]

    def compute_viscosity_acceleration(self):
        """计算粘度加速度"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._compute_viscosity_acceleration_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_velocities,
            self.container.particle_masses,
            self.container.particle_materials,
            self.container.particle_is_dynamic,
            self.container.particle_object_ids,
            self.container.particle_accelerations,
            self.container.rigid_body_forces,
            self.container.rigid_body_torques,
            self.container.rigid_body_com,
            self.container.material_fluid,
            self.container.material_rigid,
            self.container.particle_num,
            self.container.grid_ids,
            self.container.grid_num_particles,
            self.container.grid_size,
            self.container.grid_num,
            self.container.dh,
            self.viscosity,
            self.viscosity_b,
            self.density_0,
            self.container.particle_rest_densities,
            self.container.particle_rest_volumes
        )

    @staticmethod
    @cuda.jit
    def _update_fluid_velocity_kernel(velocities, accelerations, materials, material_fluid, 
                                particle_num, dt):
        """更新流体粒子速度的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        if materials[idx] == material_fluid:
            for i in range(3):
                velocities[idx][i] += dt * accelerations[idx][i]

    def update_fluid_velocity(self):
        """更新流体粒子速度"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._update_fluid_velocity_kernel[blocks, threads_per_block](
            self.container.particle_velocities,
            self.container.particle_accelerations,
            self.container.particle_materials,
            self.container.material_fluid,
            self.container.particle_num,
            self.dt[0]
        )

    @staticmethod
    @cuda.jit
    def _update_fluid_position_kernel(positions, velocities, materials, object_ids, 
                                object_materials, material_fluid, particle_num, 
                                g_upper, dt):
        """更新流体粒子位置的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        if materials[idx] == material_fluid:
            # 更新流体粒子位置
            for i in range(3):
                positions[idx][i] += dt * velocities[idx][i]
        elif positions[idx][1] > g_upper[0]:
            # 处理发射器部分
            obj_id = object_ids[idx]
            if object_materials[obj_id] == material_fluid:
                for i in range(3):
                    positions[idx][i] += dt * velocities[idx][i]
                if positions[idx][1] <= g_upper[0]:
                    materials[idx] = material_fluid

    def update_fluid_position(self):
        """更新流体粒子位置"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._update_fluid_position_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_velocities,
            self.container.particle_materials,
            self.container.particle_object_ids,
            self.container.object_materials,
            self.container.material_fluid,
            self.container.particle_num,
            self.g_upper,
            self.dt[0]
        )

    @staticmethod
    @cuda.jit
    def _prepare_emitter_kernel(materials, positions, material_fluid, material_rigid, 
                          particle_num, g_upper):
        """准备发射器的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        if materials[idx] == material_fluid:
            if positions[idx][1] > g_upper[0]:
                materials[idx] = material_rigid

    def prepare_emitter(self):
        """准备发射器"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._prepare_emitter_kernel[blocks, threads_per_block](
            self.container.particle_materials,
            self.container.particle_positions,
            self.container.material_fluid,
            self.container.material_rigid,
            self.container.particle_num,
            self.g_upper
        )

    @staticmethod
    @cuda.jit
    def _init_object_id_kernel(particle_object_ids, particle_materials, 
                          material_fluid, material_rigid,
                          fluid_object_id, rigid_object_id, particle_num):
        """初始化粒子的象ID"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        if particle_materials[idx] == material_fluid:
            particle_object_ids[idx] = fluid_object_id
        elif particle_materials[idx] == material_rigid:
            particle_object_ids[idx] = rigid_object_id

    def init_object_id(self):
        """初始化对象ID"""
        if self.container.particle_num == 0:
            return
        
        threads_per_block = min(256, self.container.particle_num)
        blocks = max(1, (self.container.particle_num + threads_per_block - 1) // threads_per_block)
        
        print(f"Debug - particle_num: {self.container.particle_num}, blocks: {blocks}, threads_per_block: {threads_per_block}")
        
        self._init_object_id_kernel[blocks, threads_per_block](
            self.container.particle_object_ids,
            self.container.particle_materials,
            self.container.material_fluid,
            self.container.material_rigid,
            self.fluid_object_id,
            self.rigid_object_id,
            self.container.particle_num
        )

    @staticmethod
    @cuda.jit
    def _renew_rigid_particle_state_kernel(positions, velocities, materials, is_dynamic,
                                         object_ids, rigid_body_is_dynamic, rigid_body_rotations,
                                         rigid_particle_original_positions, rigid_body_original_com,
                                         rigid_body_com, rigid_body_velocities, 
                                         rigid_body_angular_velocities, material_rigid,
                                         particle_num):
        """更新刚体粒子状态的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        if (materials[idx] == material_rigid and 
            is_dynamic[idx] and 
            rigid_body_is_dynamic[object_ids[idx]]):
            
            object_id = object_ids[idx]
            
            # 计算旋转后的位置
            p = cuda.local.array(3, dtype=float32)
            orig_p = cuda.local.array(3, dtype=float32)
            
            # 计算相对于原始重心的位置
            for i in range(3):
                orig_p[i] = (rigid_particle_original_positions[idx][i] - 
                           rigid_body_original_com[object_id][i])
            
            # 应用旋转
            for i in range(3):
                p[i] = 0.0
                for j in range(3):
                    p[i] += rigid_body_rotations[object_id][i][j] * orig_p[j]
            
            # 更新位置
            for i in range(3):
                positions[idx][i] = rigid_body_com[object_id][i] + p[i]
            
            # 计算速度
            cross_product = cuda.local.array(3, dtype=float32)
            cross_product[0] = (rigid_body_angular_velocities[object_id][1] * p[2] - 
                              rigid_body_angular_velocities[object_id][2] * p[1])
            cross_product[1] = (rigid_body_angular_velocities[object_id][2] * p[0] - 
                              rigid_body_angular_velocities[object_id][0] * p[2])
            cross_product[2] = (rigid_body_angular_velocities[object_id][0] * p[1] - 
                              rigid_body_angular_velocities[object_id][1] * p[0])
            
            # 更新速度
            for i in range(3):
                velocities[idx][i] = (rigid_body_velocities[object_id][i] + 
                                    cross_product[i])

    def renew_rigid_particle_state(self):
        """更新刚体粒子状态"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        self._renew_rigid_particle_state_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_velocities,
            self.container.particle_materials,
            self.container.particle_is_dynamic,
            self.container.particle_object_ids,
            self.container.rigid_body_is_dynamic,
            self.container.rigid_body_rotations,
            self.container.rigid_particle_original_positions,
            self.container.rigid_body_original_com,
            self.container.rigid_body_com,
            self.container.rigid_body_velocities,
            self.container.rigid_body_angular_velocities,
            self.container.material_rigid,
            self.container.particle_num
        )
        
        # 更新网格（如果需要）
        if self.cfg.get_cfg("exportObj"):
            for obj_i in range(self.container.object_num[None]):
                if (self.container.rigid_body_is_dynamic[obj_i] and 
                    self.container.object_materials[obj_i] == self.container.material_rigid):
                    # 更新网格顶点
                    rotated_vertices = (self.container.rigid_body_rotations[obj_i].to_numpy() @ 
                                     (self.container.object_collection[obj_i]["restPosition"] - 
                                      self.container.object_collection[obj_i]["restCenterOfMass"]).T).T
                    self.container.object_collection[obj_i]["mesh"].vertices = (
                        rotated_vertices + self.container.rigid_body_com[obj_i].to_numpy())

    def prepare(self):
        """准备模拟"""
        print("initializing object id")
        self.init_object_id()
        print("inserting object")
        self.container.insert_object()
        print("inserting emitter")
        self.prepare_emitter()
        print("inserting rigid object")
        self.rigid_solver.insert_rigid_object()
        print("renewing rigid particle state")
        self.renew_rigid_particle_state()
        print("preparing neighborhood search")
        self.container.prepare_neighbor_search()
        print("computing volume")
        self.compute_rigid_particle_volume()
        print("preparing finished")
        
        # 更新计数器
        fluid_num = self.container.fluid_object_num.copy_to_host()[0]
        rigid_num = self.container.rigid_object_num.copy_to_host()[0]
        total_num = fluid_num + rigid_num
        
        self.container.fluid_object_num = cuda.to_device(np.array([fluid_num], dtype=np.int32))
        self.container.rigid_object_num = cuda.to_device(np.array([rigid_num], dtype=np.int32))
        self.container.object_num = cuda.to_device(np.array([total_num], dtype=np.int32))

        print(f"Total object num: {self.container.object_num.copy_to_host()[0]}")

    def step(self):
        """执行一个时间步"""
        self._step()
        self.container.total_time += self.dt[0]
        self.rigid_solver.total_time += self.dt[0]
        self.compute_rigid_particle_volume()