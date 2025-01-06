import taichi as ti
import numpy as np
import math
from typing import Tuple
from ..containers import BaseContainer

@ti.data_oriented
class RigidSolver():
    def __init__(self, container: BaseContainer, gravity: Tuple[float, float, float] = (0, -9.8, 0), dt: float = 1e-3):
        # 基础属性初始化
        solver_attrs = {
            'container': container,
            'total_time': 0.0,
            'gravity': np.array(gravity),
            'dt': dt,
            'cfg': container.cfg,
            'present_rigid_object': [],
            'rigid_body_scales': {}
        }
        for name, value in solver_attrs.items():
            setattr(self, name, value)
            
        # 配置物体
        config_attrs = {
            'rigid_bodies': self.cfg.get_rigid_bodies(),
            'rigid_blocks': self.cfg.get_rigid_blocks(),
        }
        for name, value in config_attrs.items():
            setattr(self, name, value)
            
    def _compute_rotation_matrix(self, angle: float, direction: list) -> np.ndarray:
        """计算旋转矩阵"""
        rad = angle / 360 * (2 * math.pi)
        euler = np.array([d * rad for d in direction])
        c = np.cos(euler)
        s = np.sin(euler)
        
        return np.array([
            [c[1] * c[2], -c[1] * s[2], s[1]],
            [s[0] * s[1] * c[2] + c[0] * s[2], -s[0] * s[1] * s[2] + c[0] * c[2], -s[0] * c[1]],
            [-c[0] * s[1] * c[2] + s[0] * s[2], c[0] * s[1] * s[2] + s[0] * c[2], c[0] * c[1]]
        ])
        
    def create_boundary(self, thickness: float = 0.01):
        eps = self.container.diameter + self.container.boundary_thickness 
        self.start = np.array(self.container.domain_start) + eps
        self.end = np.array(self.container.domain_end) - eps

    def insert_rigid_object(self):
        for body in self.rigid_bodies:
            self.init_rigid_body(body)
        for block in self.rigid_blocks:
            self.init_rigid_block(block)

    def init_rigid_body(self, rigid_body: dict):
        """初始化刚体"""
        obj_id = rigid_body["objectId"]
        
        # 检查条件
        if (obj_id in self.present_rigid_object or 
            rigid_body["entryTime"] > self.total_time or 
            not rigid_body["isDynamic"]):
            return
            
        # 初始化属性
        body_attrs = {
            'scale': rigid_body["scale"],
            'velocity': rigid_body["velocity"],
            'translation': rigid_body["translation"],
            'rotation': self._compute_rotation_matrix(
                rigid_body["rotationAngle"],
                rigid_body["rotationAxis"]
            )
        }
        
        # 设置刚体属性
        self.rigid_body_scales[obj_id] = np.array(body_attrs['scale'], dtype=np.float32)
        rigid_attrs = {
            'rigid_body_velocities': body_attrs['velocity'],
            'rigid_body_angular_velocities': np.zeros(3),
            'rigid_body_original_com': np.zeros(3),
            'rigid_body_com': body_attrs['translation'],
            'rigid_body_rotations': body_attrs['rotation']
        }
        
        for name, value in rigid_attrs.items():
            setattr(self.container, name, obj_id, np.array(value, dtype=np.float32))
            
        self.present_rigid_object.append(obj_id)
        
    def init_rigid_block(self, rigid_block):
        pass

    def update_velocity(self, index):
        self.container.rigid_body_velocities[index] += self.gravity * self.dt + self.container.rigid_body_forces[index] / self.container.rigid_body_masses[index] * self.dt

    def update_angular_velocity(self, index):
        self.container.rigid_body_angular_velocities[index] += self.container.rigid_body_torques[index] / (0.4 * self.container.rigid_body_masses[index] * (self.rigid_body_scales[index][0]**2 + self.rigid_body_scales[index][1]**2+ self.rigid_body_scales[index][2]**2)) * self.dt

    def update_position(self, index):
        self.container.rigid_body_com[index] += self.container.rigid_body_velocities[index] * self.dt
    
        rotation_vector = self.container.rigid_body_angular_velocities[index] * self.dt

        theta = np.linalg.norm(rotation_vector)

        if theta > 0:
            omega = rotation_vector / theta

            omega_cross = np.array([
                [0.0, -omega[2], omega[1]],
                [omega[2], 0.0, -omega[0]],
                [-omega[1], omega[0], 0.0]
            ])
            rotation_matrix = np.eye(3) + np.sin(theta) * omega_cross + (1 - np.cos(theta)) * np.dot(omega_cross, omega_cross)

            self.container.rigid_body_rotations[index] = np.dot(rotation_matrix, self.container.rigid_body_rotations[index])
        

    def step(self):
        for index in range(self.container.object_num[None]):
            if self.container.rigid_body_is_dynamic[index] and self.container.object_materials[index] == self.container.material_rigid:
                self.update_velocity(index)
                self.update_angular_velocity(index)
                self.update_position(index)
                self.container.rigid_body_forces[index] = np.array([0.0, 0.0, 0.0])
                self.container.rigid_body_torques[index] = np.array([0.0, 0.0, 0.0])


    def get_rigid_body_states(self, index):
        # ! here we use the information of base frame. We assume the center of mass is exactly the base position.
        linear_velocity = self.container.rigid_body_velocities[index]
        angular_velocity = self.container.rigid_body_angular_velocities[index]
        position = self.container.rigid_body_com[index]
        rotation_matrix = self.container.rigid_body_rotations[index]
        
        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "position": position,
            "rotation_matrix": rotation_matrix
        }