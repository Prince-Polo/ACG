import taichi as ti
import numpy as np
import os
import math
from ..containers import BaseContainer
from ..utils import create_urdf
from typing import List, Tuple, Dict, Union

class RigidSolver():
    def __init__(self, container: BaseContainer, gravity: Tuple[float, float, float] = (0, -9.8, 0), dt: float = 1e-3):
        self.container = container
        self.total_time = 0.0
        self.present_rigid_object = []
        assert container.dim == 3, "RigidSolver only supports 3D simulation currently"

        self.cfg = container.cfg
        self.rigid_bodies = self.cfg.get_rigid_bodies()
        self.rigid_blocks = self.cfg.get_rigid_blocks()
        num_rigid_bodies = len(self.rigid_bodies) + len(self.rigid_blocks)
        self.dt = dt
        self.gravity = np.array(gravity)

        # mapping between container index and rigid index
        self.container_idx_to_rigid_idx = {}
        self.rigid_idx_to_container_idx = {}

        if num_rigid_bodies != 0:
            self.create_boundary()
        else:
            print("No rigid body in the scene, skip rigid solver initialization.")

    def insert_rigid_object(self):
        for rigid_body in self.rigid_bodies:
            self.init_rigid_body(rigid_body)

        for rigid_block in self.rigid_blocks:
            self.init_rigid_block(rigid_block)

    def create_boundary(self, thickness: float = 0.01):
        eps = self.container.padding + self.container.particle_diameter + self.container.domain_box_thickness 
        domain_start = self.container.domain_start
        domain_end = self.container.domain_end
        domain_start = np.array(domain_start) + eps
        domain_end = np.array(domain_end) - eps
        domain_size = self.container.domain_size
        domain_center = (domain_start + domain_end) / 2

        # Creating each wall
        self.create_wall(position=[domain_center[0], domain_start[1] - thickness / 1.9, domain_center[2]], half_extents=[domain_size[0] / 2, thickness / 2, domain_size[2] / 2])
        self.create_wall(position=[domain_center[0], domain_end[1] + thickness / 1.9, domain_center[2]], half_extents=[domain_size[0] / 2, thickness / 2, domain_size[2] / 2])
        self.create_wall(position=[domain_start[0] - thickness / 1.9, domain_center[1], domain_center[2]], half_extents=[thickness / 2, domain_size[1] / 2, domain_size[2] / 2])
        self.create_wall(position=[domain_end[0] + thickness / 1.9, domain_center[1], domain_center[2]], half_extents=[thickness / 2, domain_size[1] / 2, domain_size[2] / 2])
        self.create_wall(position=[domain_center[0], domain_center[1], domain_start[2] - thickness / 1.9], half_extents=[domain_size[0] / 2, domain_size[1] / 2, thickness / 2])
        self.create_wall(position=[domain_center[0], domain_center[1], domain_end[2] + thickness / 1.9], half_extents=[domain_size[0] / 2, domain_size[1] / 2, thickness / 2])

    def create_wall(self, position, half_extents):
        # This function is a placeholder for creating walls
        pass

    def init_rigid_body(self, rigid_body):
        container_idx = rigid_body["objectId"]

        if container_idx in self.present_rigid_object:
            return
        if rigid_body["entryTime"] > self.total_time:
            return

        is_dynamic = rigid_body["isDynamic"]
        if is_dynamic:
            velocity = np.array(rigid_body["velocity"], dtype=np.float32)
        else:
            velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        translation = np.array(rigid_body["translation"])
        angle = rigid_body["rotationAngle"] / 360 * (2 * math.pi)
        direction = rigid_body["rotationAxis"]
        rotation_euler = np.array([direction[0] * angle, direction[1] * angle, direction[2] * angle])
        rotation_matrix = self.euler_to_rotation_matrix(rotation_euler)

        self.container_idx_to_rigid_idx[container_idx] = container_idx
        self.rigid_idx_to_container_idx[container_idx] = container_idx

        if is_dynamic:
            self.container.rigid_body_original_centers_of_mass[container_idx] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.container.rigid_body_centers_of_mass[container_idx] = translation
            self.container.rigid_body_rotations[container_idx] = rotation_matrix
            self.container.rigid_body_velocities[container_idx] = velocity
            self.container.rigid_body_angular_velocities[container_idx] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.present_rigid_object.append(container_idx)

    def init_rigid_block(self, rigid_block):
        raise NotImplementedError

    def apply_force(self, container_idx, force: Tuple[float, float, float]):
        rigid_idx = self.container_idx_to_rigid_idx[container_idx]
        com_pos = self.container.rigid_body_centers_of_mass[rigid_idx]
        self.container.rigid_body_forces[rigid_idx] += np.array(force)

    def apply_torque(self, container_idx, torque: Tuple[float, float, float]):
        rigid_idx = self.container_idx_to_rigid_idx[container_idx]
        self.container.rigid_body_torques[rigid_idx] += np.array(torque)

    def step(self):
        for container_id in range(self.container.object_num[None]):
            if self.container.rigid_body_is_dynamic[container_id] and self.container.object_materials[container_id] == self.container.material_rigid:
                force_i = self.container.rigid_body_forces[container_id]
                torque_i = self.container.rigid_body_torques[container_id]
                self.apply_force(container_id, force_i)
                self.apply_torque(container_id, torque_i)
                self.container.rigid_body_forces[container_id] = np.array([0.0, 0.0, 0.0])
                self.container.rigid_body_torques[container_id] = np.array([0.0, 0.0, 0.0])

        self.update_rigid_body_states()

    def update_rigid_body_states(self):
        for container_id in range(self.container.object_num[None]):
            if self.container.rigid_body_is_dynamic[container_id] and self.container.object_materials[container_id] == self.container.material_rigid:
                state_i = self.get_rigid_body_states(container_id)
                self.container.rigid_body_centers_of_mass[container_id] = state_i["position"]
                self.container.rigid_body_rotations[container_id] = state_i["rotation_matrix"]
                self.container.rigid_body_velocities[container_id] = state_i["linear_velocity"]
                self.container.rigid_body_angular_velocities[container_id] = state_i["angular_velocity"]

    def get_rigid_body_states(self, container_idx):
        rigid_idx = self.container_idx_to_rigid_idx[container_idx]
        linear_velocity = self.container.rigid_body_velocities[rigid_idx]
        angular_velocity = self.container.rigid_body_angular_velocities[rigid_idx]
        position = self.container.rigid_body_centers_of_mass[rigid_idx]
        rotation_matrix = self.container.rigid_body_rotations[rigid_idx]

        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "position": position,
            "rotation_matrix": rotation_matrix
        }

    def euler_to_rotation_matrix(self, euler):
        cx, cy, cz = np.cos(euler)
        sx, sy, sz = np.sin(euler)

        rotation_matrix = np.array([
            [cy * cz, -cy * sz, sy],
            [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
            [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy]
        ])

        return rotation_matrix