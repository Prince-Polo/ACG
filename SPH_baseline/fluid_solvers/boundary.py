import numpy as np
from ..containers import BaseContainerBaseline

class BoundaryBaseline():
    def __init__(self, container: BaseContainerBaseline):
        self.container = container

    def _check_boundary_face(self, pos, p_i, collision_normal):
        """检查六个面的边界碰撞"""
        # 定义面检查参数: (轴向, 是否最大边界, 边界位置)
        faces = [
            (0, 1, self.container.domain_size[0] - self.container.padding),  # x+
            (0, 0, self.container.padding),                                  # x-
            (1, 1, self.container.domain_size[1] - self.container.padding),  # y+
            (1, 0, self.container.padding),                                  # y-
            (2, 1, self.container.domain_size[2] - self.container.padding),  # z+
            (2, 0, self.container.padding)                                   # z-
        ]
        
        for axis, is_max, bound in faces:
            if (is_max and pos[axis] > bound) or (not is_max and pos[axis] <= bound):
                collision_normal[axis] += 1.0 if is_max else -1.0
                self.container.particle_positions[p_i][axis] = bound

    def simulate_collisions(self, p_i, normal_vec):
        """处理碰撞响应"""
        c_f = 0.5  # 碰撞系数
        vel = self.container.particle_velocities[p_i]
        dot_product = np.dot(vel, normal_vec)
        self.container.particle_velocities[p_i] -= (1.0 + c_f) * dot_product * normal_vec

    def enforce_domain_boundary(self, particle_type):
        """边界约束主函数"""
        for p_i in range(self.container.particle_num):
            if (self.container.particle_materials[p_i] == particle_type and 
                self.container.particle_is_dynamic[p_i]):
                
                # 获取粒子位置和初始化碰撞法线
                pos = self.container.particle_positions[p_i]
                collision_normal = np.zeros(3)
                
                # 检查所有边界面
                self._check_boundary_face(pos, p_i, collision_normal)
                
                # 处理碰撞响应
                collision_length = np.linalg.norm(collision_normal)
                if collision_length > 1e-6:
                    self.simulate_collisions(p_i, collision_normal / collision_length)