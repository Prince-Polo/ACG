import numpy as np
from numba import cuda, float32
from ..containers.base_container_numba import BaseContainer
import math

@cuda.jit
def _enforce_domain_boundary_kernel(positions, velocities, materials, is_dynamic,
                                  domain_size, padding, particle_num, particle_type):
    """强制边界约束的CUDA kernel"""
    idx = cuda.grid(1)
    if idx >= particle_num:
        return
        
    if materials[idx] == particle_type and is_dynamic[idx]:
        pos = positions[idx]
        collision_normal = cuda.local.array(3, dtype=float32)
        collision_normal[0] = 0.0
        collision_normal[1] = 0.0
        collision_normal[2] = 0.0
        
        # X轴边界
        if pos[0] > domain_size[0] - padding:
            collision_normal[0] += 1.0
            positions[idx][0] = domain_size[0] - padding
        if pos[0] <= padding:
            collision_normal[0] += -1.0
            positions[idx][0] = padding
            
        # Y轴边界
        if pos[1] > domain_size[1] - padding:
            collision_normal[1] += 1.0
            positions[idx][1] = domain_size[1] - padding
        if pos[1] <= padding:
            collision_normal[1] += -1.0
            positions[idx][1] = padding
            
        # Z轴边界
        if pos[2] > domain_size[2] - padding:
            collision_normal[2] += 1.0
            positions[idx][2] = domain_size[2] - padding
        if pos[2] <= padding:
            collision_normal[2] += -1.0
            positions[idx][2] = padding
            
        # 计算碰撞法线长度
        collision_normal_length = math.sqrt(collision_normal[0]**2 + 
                                         collision_normal[1]**2 + 
                                         collision_normal[2]**2)
        
        # 如果发生碰撞，处理碰撞响应
        if collision_normal_length > 1e-6:
            c_f = 0.5  # 碰撞系数
            norm_factor = 1.0 / collision_normal_length
            normalized_normal = cuda.local.array(3, dtype=float32)
            for i in range(3):
                normalized_normal[i] = collision_normal[i] * norm_factor
                
            # 内联_simulate_collisions_kernel的逻辑
            dot_product = 0.0
            for i in range(3):
                dot_product += velocities[idx][i] * normalized_normal[i]
            
            for i in range(3):
                velocities[idx][i] -= (1.0 + c_f) * dot_product * normalized_normal[i]

class Boundary:
    def __init__(self, container: BaseContainer):
        """初始化边界处理器"""
        self.container = container
        
    def enforce_domain_boundary(self, particle_type):
        """强制域边界约束"""
        threads_per_block = 256
        blocks = (self.container.particle_num + threads_per_block - 1) // threads_per_block
        
        _enforce_domain_boundary_kernel[blocks, threads_per_block](
            self.container.particle_positions,
            self.container.particle_velocities,
            self.container.particle_materials,
            self.container.particle_is_dynamic,
            self.container.domain_size,
            self.container.padding,
            self.container.particle_num,
            particle_type
        ) 