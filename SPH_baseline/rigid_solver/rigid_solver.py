import numpy as np
import math
from typing import Tuple
from ..containers import BaseContainerBaseline

class RigidSolverBaseline():
    def __init__(self, container: BaseContainerBaseline, gravity: Tuple[float, float, float] = (0, -9.8, 0), dt: float = 1e-3):
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
        for rigid_body in self.rigid_bodies:
            self.init_rigid_body(rigid_body)

        for rigid_block in self.rigid_blocks:
            self.init_rigid_block(rigid_block)

    def create_boundary(self, thickness: float = 0.01):
        eps = self.container.diameter + self.container.boundary_thickness 
        domain_start = self.container.domain_start
        domain_end = self.container.domain_end

        self.start = np.array(domain_start) + eps
        self.end = np.array(domain_end) - eps

    def init_rigid_body(self, rigid_body):
        index = rigid_body["objectId"]

        # dealing with entry time
        if index in self.present_rigid_object:
            return
        if rigid_body["entryTime"] > self.total_time:
            return

        is_dynamic = rigid_body["isDynamic"]
        if is_dynamic:
            velocity = np.array(rigid_body["velocity"], dtype=np.float32)
        else:
            velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        is_dynamic = rigid_body["isDynamic"]
        if not is_dynamic:
            return
        else:
            angle = rigid_body["rotationAngle"] / 360 * (2 * math.pi)
            direction = rigid_body["rotationAxis"]
            euler = np.array([direction[0] * angle, direction[1] * angle, direction[2] * angle])
            cx, cy, cz = np.cos(euler)
            sx, sy, sz = np.sin(euler)

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