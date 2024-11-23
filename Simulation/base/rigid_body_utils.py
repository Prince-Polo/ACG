import taichi as ti
import numpy as np
import trimesh as tm
from tqdm import tqdm
from functools import reduce

@ti.data_oriented
class RigidBodyUtils:
    def __init__(self, dim, V0, particle_num, particle_object_ids, particle_is_dynamic,
                 particle_densities, particle_positions, material_rigid, particle_materials):
        self.dim = dim
        self.V0 = V0
        self.particle_num = particle_num
        self.particle_object_ids = particle_object_ids
        self.particle_is_dynamic = particle_is_dynamic
        self.particle_densities = particle_densities
        self.particle_positions = particle_positions
        self.material_rigid = material_rigid
        self.particle_materials = particle_materials

    @ti.kernel
    def compute_rigid_body_mass(self, object_id: int) -> ti.f32:
        total_mass = 0.0
        for p_i in range(self.particle_num[None]):
            if self.particle_object_ids[p_i] == object_id and self.particle_is_dynamic[p_i]:
                total_mass += self.particle_densities[p_i] * self.V0
        return total_mass

    @ti.kernel
    def compute_rigid_body_center_of_mass(self, object_id: int) -> ti.types.vector(3, float):
        mass_times_position = ti.Vector([0.0 for _ in range(self.dim)])
        total_mass = 0.0
        for p_i in range(self.particle_num[None]):
            if self.particle_object_ids[p_i] == object_id and self.particle_is_dynamic[p_i]:
                mass_times_position += self.particle_positions[p_i] * self.particle_densities[p_i] * self.V0
                total_mass += self.particle_densities[p_i] * self.V0
        return mass_times_position / total_mass

    @ti.func
    def is_static_rigid_body(self, p):
        return self.particle_materials[p] == self.material_rigid and not self.particle_is_dynamic[p]

    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.particle_materials[p] == self.material_rigid and self.particle_is_dynamic[p]

    def load_rigid_body(self, rigid_body_config, pitch=None):
        if pitch is None:
            pitch = self.V0 ** (1 / self.dim)
        obj_id = rigid_body_config["objectId"]
        mesh = tm.load(rigid_body_config["geometryFile"])
        mesh.apply_scale(rigid_body_config["scale"])
        offset = np.array(rigid_body_config["translation"])

        angle = rigid_body_config["rotationAngle"] / 360 * 2 * np.pi
        axis = rigid_body_config["rotationAxis"]
        rotation_matrix = tm.transformations.rotation_matrix(angle, axis, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rotation_matrix)
        mesh.vertices += offset

        mesh_backup = mesh.copy()
        rigid_body_config["mesh"] = mesh_backup
        rigid_body_config["restPosition"] = mesh_backup.vertices
        rigid_body_config["restCenterOfMass"] = np.array([0.0, 0.0, 0.0])

        tm.repair.fill_holes(mesh)

        voxelized_mesh = mesh.voxelized(pitch=pitch).fill()
        voxelized_points = voxelized_mesh.points
        print(f"Rigid body {obj_id} particle count: {voxelized_points.shape[0]}")

        return voxelized_points