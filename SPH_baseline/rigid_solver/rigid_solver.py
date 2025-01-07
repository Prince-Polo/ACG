import numpy as np
import math
from typing import Tuple
from ..containers import BaseContainerBaseline

class RigidSolverBaseline():
    def __init__(self, container: BaseContainerBaseline, gravity: Tuple[float, float, float] = (0, -9.8, 0), dt: float = 1e-3):
        # 使用字典初始化基础属性
        solver_attrs = {
            'container': container,
            'total_time': 0.0,
            'present_rigid_object': [],
            'rigid_body_scales': {},
            'gravity': np.array(gravity),
            'dt': dt,
            'cfg': container.cfg
        }
        
        for name, value in solver_attrs.items():
            setattr(self, name, value)
            
        config_bodies = {
            'rigid_bodies': self.cfg.get_rigid_bodies(),
            'rigid_blocks': self.cfg.get_rigid_blocks()
        }
        for name, value in config_bodies.items():
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

    def insert_rigid_object(self):
        for body in self.rigid_bodies:
            self.init_rigid_body(body)
        for block in self.rigid_blocks:
            self.init_rigid_block(block)

    def init_rigid_body(self, rigid_body: dict):
        """初始化刚体对象"""
        obj_id = rigid_body["objectId"]
        
        # 检查时间和存在性条件
        if obj_id in self.present_rigid_object or rigid_body["entryTime"] > self.total_time:
            return

        # 处理动态刚体
        if not rigid_body["isDynamic"]:
            return
            
        # 初始化刚体属性
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
        self.container.rigid_body_velocities[obj_id] = np.array(body_attrs['velocity'], dtype=np.float32)
        self.container.rigid_body_angular_velocities[obj_id] = np.zeros(3, dtype=np.float32)
        self.container.rigid_body_original_com[obj_id] = np.zeros(3, dtype=np.float32)
        self.container.rigid_body_com[obj_id] = np.array(body_attrs['translation'])
        self.container.rigid_body_rotations[obj_id] = body_attrs['rotation']
        
        self.present_rigid_object.append(obj_id)
        
    def init_rigid_block(self, rigid_block):
        pass

    def update_velocity(self, index):
        self.container.rigid_body_velocities[index] += (
            self.gravity * self.dt + 
            self.container.rigid_body_forces[index] / self.container.rigid_body_masses[index] * self.dt
        )

    def update_angular_velocity(self, index):
        self.container.rigid_body_angular_velocities[index] += (
            self.container.rigid_body_torques[index] / 
            (0.4 * self.container.rigid_body_masses[index] * 
             (self.rigid_body_scales[index][0]**2 + 
              self.rigid_body_scales[index][1]**2 + 
              self.rigid_body_scales[index][2]**2)) * self.dt
        )

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
            rotation_matrix = (np.eye(3) + 
                             np.sin(theta) * omega_cross + 
                             (1 - np.cos(theta)) * np.dot(omega_cross, omega_cross))

            self.container.rigid_body_rotations[index] = np.dot(rotation_matrix, 
                                                              self.container.rigid_body_rotations[index])

    def step(self):
        for index in range(self.container.object_num):
            if (self.container.rigid_body_is_dynamic[index] and 
                self.container.object_materials[index] == self.container.material_rigid):
                self.update_velocity(index)
                self.update_angular_velocity(index)
                self.update_position(index)
                self.container.rigid_body_forces[index] = np.array([0.0, 0.0, 0.0])
                self.container.rigid_body_torques[index] = np.array([0.0, 0.0, 0.0])

    def get_rigid_body_states(self, index):
        return {
            "linear_velocity": self.container.rigid_body_velocities[index],
            "angular_velocity": self.container.rigid_body_angular_velocities[index],
            "position": self.container.rigid_body_com[index],
            "rotation_matrix": self.container.rigid_body_rotations[index]
        }