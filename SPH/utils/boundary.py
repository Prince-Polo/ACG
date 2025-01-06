import taichi as ti
import numpy as np
import math
from ..containers import BaseContainer
from .kernel import *

@ti.data_oriented
class Boundary():
    def __init__(self, container: BaseContainer):
        self.container = container

    @ti.func
    def _check_boundary_dim(self, p_i: int, pos: ti.template(), 
                           dim: ti.template(), collision_normal: ti.template()):
        """检查单个维度的边界"""
        # 静态获取域边界和填充值
        domain_max = ti.static(self.container.domain_size[dim])
        padding = ti.static(self.container.padding)
        
        # 检查上边界
        if pos[dim] > domain_max - padding:
            collision_normal[dim] += 1.0
            self.container.particle_positions[p_i][dim] = domain_max - padding
            
        # 检查下边界
        if pos[dim] <= padding:
            collision_normal[dim] += -1.0
            self.container.particle_positions[p_i][dim] = padding

    @ti.kernel
    def enforce_domain_boundary(self, particle_type: int):
        """边界约束主函数"""
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == particle_type and \
               self.container.particle_is_dynamic[p_i]:
                
                # 局部变量存储
                pos = self.container.particle_positions[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                
                # 静态循环检查三个维度
                for dim in ti.static(range(3)):
                    self._check_boundary_dim(p_i, pos, dim, collision_normal)
                
                # 计算碰撞响应
                collision_length = collision_normal.norm()
                if collision_length > 1e-6:
                    c_f = 0.5
                    norm_vec = collision_normal / collision_length
                    vel = self.container.particle_velocities[p_i]
                    dot_prod = vel.dot(norm_vec)
                    self.container.particle_velocities[p_i] -= (1.0 + c_f) * dot_prod * norm_vec