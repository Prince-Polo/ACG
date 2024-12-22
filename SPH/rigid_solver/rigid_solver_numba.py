import numpy as np
import math
from typing import Tuple

class RigidSolver:
    def __init__(self, container, gravity: Tuple[float, float, float] = (0, -9.8, 0), dt: float = 1e-3):
        """初始化刚体求解器"""
        self.container = container
        self.total_time = 0.0
        self.present_rigid_object = []
        self.rigid_body_scales = {}
        self.gravity = np.array(gravity)

        self.cfg = container.cfg
        self.rigid_bodies = self.cfg.get_rigid_bodies()
        self.rigid_blocks = self.cfg.get_rigid_blocks()
        num_rigid_bodies = len(self.rigid_bodies) + len(self.rigid_blocks)
        self.dt = dt

    def insert_rigid_object(self):
        """插入刚体对象"""
        for rigid_body in self.rigid_bodies:
            self.init_rigid_body(rigid_body)

        for rigid_block in self.rigid_blocks:
            self.init_rigid_block(rigid_block)

    def init_rigid_body(self, rigid_body):
        """初始化刚体"""
        index = rigid_body["objectId"]

        # 处理入场时间
        if index in self.present_rigid_object:
            return
        if rigid_body["entryTime"] > self.total_time:
            return

        is_dynamic = rigid_body["isDynamic"]
        if not is_dynamic:
            return
        else:
            # 设置旋转
            angle = rigid_body["rotationAngle"] / 360 * (2 * math.pi)
            direction = rigid_body["rotationAxis"]
            euler = np.array([direction[0] * angle, direction[1] * angle, direction[2] * angle])
            cx, cy, cz = np.cos(euler)
            sx, sy, sz = np.sin(euler)

            # 初始化刚体属性
            self.rigid_body_scales[index] = np.array(rigid_body["scale"], dtype=np.float32)
            self.container.rigid_body_velocities[index] = np.array(rigid_body["velocity"], dtype=np.float32)
            self.container.rigid_body_angular_velocities[index] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.container.rigid_body_original_com[index] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.container.rigid_body_com[index] = np.array(rigid_body["translation"])
            self.container.rigid_body_rotations[index] = np.array([
                [cy * cz, -cy * sz, sy],
                [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
                [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy]
            ])

        self.present_rigid_object.append(index)

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