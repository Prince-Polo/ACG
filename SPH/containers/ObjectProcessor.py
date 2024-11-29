import numpy as np
import trimesh as tm
from tqdm import tqdm
from functools import reduce
from ..utils import SimConfig

PI = 3.1415926

def fluid_body_processor(dim, config: SimConfig, diameter):
    fluid_bodies = config.get_fluid_bodies()
    fluid_body_num = 0
    for fluid_body in fluid_bodies:
        voxelized_points_np = load_fluid_body(dim, fluid_body, pitch=diameter)
        fluid_body["particleNum"] = voxelized_points_np.shape[0]
        fluid_body["voxelizedPoints"] = voxelized_points_np
        fluid_body_num += voxelized_points_np.shape[0]
    return fluid_body_num


def load_fluid_body(dim, rigid_body, pitch):
        mesh = tm.load(rigid_body["geometryFile"])
        mesh.apply_scale(rigid_body["scale"])
        offset = np.array(rigid_body["translation"])

        angle = rigid_body["rotationAngle"] / 360 * 2 * PI
        direction = rigid_body["rotationAxis"]
        rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)
        mesh.vertices += offset

        min_point, max_point = mesh.bounding_box.bounds
        num_dim = []
        for i in range(dim):
            num_dim.append(
                np.arange(min_point[i], max_point[i], pitch))
        
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        
        print(f"processing {len(new_positions)} points to decide whether they are inside the mesh. This might take a while.")
        inside = [False for _ in range(len(new_positions))]

        # decide whether the points are inside the mesh or not
        # TODO: make it parallel or precompute and store
        pbar = tqdm(total=len(new_positions))
        for i in range(len(new_positions)):
            if mesh.contains([new_positions[i]])[0]:
                inside[i] = True
            pbar.update(1)

        pbar.close()

        new_positions = new_positions[inside]
        return new_positions

def rigid_body_processor(config: SimConfig,diameter):
    rigid_bodies = config.get_rigid_bodies()
    rigid_body_num = 0
    for rigid_body in rigid_bodies:
        voxelized_points_np = load_rigid_body(rigid_body, pitch=diameter)
        rigid_body["particleNum"] = voxelized_points_np.shape[0]
        rigid_body["voxelizedPoints"] = voxelized_points_np
        rigid_body_num += voxelized_points_np.shape[0]
    return rigid_body_num

def load_rigid_body(rigid_body, pitch):
        obj_id = rigid_body["objectId"]
        mesh = tm.load(rigid_body["geometryFile"])
        mesh.apply_scale(rigid_body["scale"])

        if rigid_body["isDynamic"] == False:
            offset = np.array(rigid_body["translation"])
            angle = rigid_body["rotationAngle"] / 360 * 2 * PI
            direction = rigid_body["rotationAxis"]
            rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
            mesh.apply_transform(rot_matrix)
            mesh.vertices += offset
        
        # Backup the original mesh for exporting obj
        mesh_backup = mesh.copy()
        rigid_body["mesh"] = mesh_backup
        rigid_body["restPosition"] = mesh_backup.vertices
        rigid_body["restCenterOfMass"] = np.array([0.0, 0.0, 0.0]) 

        voxelized_mesh = mesh.voxelized(pitch=pitch)
        voxelized_mesh = mesh.voxelized(pitch=pitch).fill()
        voxelized_points_np = voxelized_mesh.points
        print(f"rigid body {obj_id} num: {voxelized_points_np.shape[0]}")
        
        return voxelized_points_np

def fluid_block_processor(dim, config: SimConfig,diameter):
    fluid_blocks = config.get_fluid_blocks()
    fluid_block_num = 0
    for fluid in fluid_blocks:
        particle_num = compute_cube_particle_num(dim, fluid["start"], fluid["end"], space=diameter)
        fluid["particleNum"] = particle_num
        fluid_block_num += particle_num
    return fluid_block_num

def compute_cube_particle_num(dim, domain_start, domain_end, space):
        num_dim = []
        for i in range(dim):
            num_dim.append(
                np.arange(domain_start[i], domain_end[i], space))
        return reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])

def compute_box_particle_num(dim, domain_start, domain_end, diameter, thickness):
        num_dim = []
        for i in range(dim):
            num_dim.append(
                np.arange(domain_start[i], domain_end[i], diameter))
        
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        
        mask = np.zeros(new_positions.shape[0], dtype=bool)
        for i in range(dim):
            mask = mask | ((new_positions[:, i] <= domain_start[i] + thickness) | (new_positions[:, i] >= domain_end[i] - thickness))
        new_positions = new_positions[mask]
        return new_positions.shape[0]
        