import numpy as np
import math
from typing import Tuple

class RigidSolver:
    def __init__(self, container, gravity: Tuple[float, float, float] = (0, -9.8, 0), dt: float = 1e-3):
        """初始化刚体求解器"""
        # 使用字典初始化基础属性
        solver_attrs = {
            'container': container,
            'total_time': 0.0,
            'gravity': np.array(gravity),
            'dt': dt,
            'present_rigid_object': [],
            'rigid_body_scales': {},
            'cfg': container.cfg
        }
        for name, value in solver_attrs.items():
            setattr(self, name, value)
            
        # 加载配置
        config_attrs = {
            'rigid_bodies': self.cfg.get_rigid_bodies(),
            'rigid_blocks': self.cfg.get_rigid_blocks()
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

    def insert_rigid_object(self):
        """插入刚体对象"""
        for body in [*self.rigid_bodies, *self.rigid_blocks]:
            method = self.init_rigid_body if body in self.rigid_bodies else self.init_rigid_block
            method(body)

    def init_rigid_body(self, rigid_body: dict):
        """初始化刚体"""
        obj_id = rigid_body["objectId"]
        
        # 检查条件
        if (obj_id in self.present_rigid_object or 
            rigid_body["entryTime"] > self.total_time or 
            not rigid_body["isDynamic"]):
            return
            
        # 设置刚体属性
        rigid_attrs = {
            'scale': rigid_body["scale"],
            'velocity': rigid_body["velocity"],
            'translation': rigid_body["translation"],
            'rotation': self._compute_rotation_matrix(
                rigid_body["rotationAngle"],
                rigid_body["rotationAxis"]
            )
        }
        
        # 应用属性
        self.rigid_body_scales[obj_id] = np.array(rigid_attrs['scale'], dtype=np.float32)
        container_attrs = {
            'rigid_body_velocities': rigid_attrs['velocity'],
            'rigid_body_angular_velocities': np.zeros(3),
            'rigid_body_original_com': np.zeros(3),
            'rigid_body_com': rigid_attrs['translation'],
            'rigid_body_rotations': rigid_attrs['rotation']
        }
        
        for name, value in container_attrs.items():
            setattr(self.container, name, obj_id, np.array(value, dtype=np.float32))
            
        self.present_rigid_object.append(obj_id)

    def init_rigid_block(self, rigid_block):
        """初始化刚体块"""
        pass

    def update_velocity(self, index):
        """更新线速度"""
        self.container.rigid_body_velocities[index] += (
            self.gravity * self.dt + 
            self.container.rigid_body_forces[index] / self.container.rigid_body_masses[index] * self.dt
        )

    def update_angular_velocity(self, index):
        """更新角速度"""
        scale = self.rigid_body_scales[index]
        inertia = 0.4 * self.container.rigid_body_masses[index] * (scale[0]**2 + scale[1]**2 + scale[2]**2)
        self.container.rigid_body_angular_velocities[index] += (
            self.container.rigid_body_torques[index] / inertia * self.dt
        )

    def update_position(self, index):
        """更新位置和旋转"""
        # 更新位置
        self.container.rigid_body_com[index] += self.container.rigid_body_velocities[index] * self.dt
        
        # 更新旋转
        rotation_vector = self.container.rigid_body_angular_velocities[index] * self.dt
        theta = np.linalg.norm(rotation_vector)

        if theta > 0:
            omega = rotation_vector / theta
            omega_cross = np.array([
                [0.0, -omega[2], omega[1]],
                [omega[2], 0.0, -omega[0]],
                [-omega[1], omega[0], 0.0]
            ])
            rotation_matrix = (
                np.eye(3) + 
                np.sin(theta) * omega_cross + 
                (1 - np.cos(theta)) * np.dot(omega_cross, omega_cross)
            )
            self.container.rigid_body_rotations[index] = np.dot(
                rotation_matrix, 
                self.container.rigid_body_rotations[index]
            )

    def step(self):
        """执行一个时间步"""
        for index in range(self.container.object_num[0]):
            if (self.container.rigid_body_is_dynamic[index] and 
                self.container.object_materials[index] == self.container.material_rigid):
                self.update_velocity(index)
                self.update_angular_velocity(index)
                self.update_position(index)
                self.container.rigid_body_forces[index] = np.array([0.0, 0.0, 0.0])
                self.container.rigid_body_torques[index] = np.array([0.0, 0.0, 0.0]) 