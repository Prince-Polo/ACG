import numpy as np
from ..containers import BaseContainerBaseline
from .utils import *

class BoundaryBaseline():
    def __init__(self, container: BaseContainerBaseline):
        self.container = container

    def simulate_collisions(self, p_i, vec):
        # 碰撞因子，假设碰撞后损失(1-c_f)的速度
        c_f = 0.5

        self.container.particle_velocities[p_i] -= (1.0 + c_f) * np.dot(self.container.particle_velocities[p_i], vec) * vec

    def enforce_domain_boundary(self, particle_type):
        for p_i in range(self.container.particle_num):
            if self.container.particle_materials[p_i] == particle_type and self.container.particle_is_dynamic[p_i]:
                pos = self.container.particle_positions[p_i]
                collision_normal = np.zeros(3)
                
                if pos[0] > self.container.domain_size[0] - self.container.padding:
                    collision_normal[0] += 1.0
                    self.container.particle_positions[p_i][0] = self.container.domain_size[0] - self.container.padding
                if pos[0] <= self.container.padding:
                    collision_normal[0] += -1.0
                    self.container.particle_positions[p_i][0] = self.container.padding

                if pos[1] > self.container.domain_size[1] - self.container.padding:
                    collision_normal[1] += 1.0
                    self.container.particle_positions[p_i][1] = self.container.domain_size[1] - self.container.padding
                if pos[1] <= self.container.padding:
                    collision_normal[1] += -1.0
                    self.container.particle_positions[p_i][1] = self.container.padding

                if pos[2] > self.container.domain_size[2] - self.container.padding:
                    collision_normal[2] += 1.0
                    self.container.particle_positions[p_i][2] = self.container.domain_size[2] - self.container.padding
                if pos[2] <= self.container.padding:
                    collision_normal[2] += -1.0
                    self.container.particle_positions[p_i][2] = self.container.padding

                collision_normal_length = np.linalg.norm(collision_normal)
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(p_i, collision_normal / collision_normal_length)