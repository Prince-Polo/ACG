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


def load_fluid_body(dim, fluid_body, pitch):
    mesh = tm.load(fluid_body["geometryFile"])
    mesh.apply_scale(fluid_body["scale"])
    offset = np.array(fluid_body["translation"])

    angle = fluid_body["rotationAngle"] / 360 * 2 * np.pi
    direction = fluid_body["rotationAxis"]
    rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
    mesh.apply_transform(rot_matrix)
    mesh.vertices += offset

    min_point, max_point = mesh.bounding_box.bounds
    num_dim = []
    for i in range(dim):
        num_dim.append(np.arange(min_point[i], max_point[i], pitch))
    
    new_positions = np.array(np.meshgrid(*num_dim, sparse=False, indexing='ij'), dtype=np.float32)
    new_positions = new_positions.reshape(-1, reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
    
    print(f"processing {len(new_positions)} points to decide whether they are inside the mesh. This might take a while.")
    inside = [False for _ in range(len(new_positions))]

    pbar = tqdm(total=len(new_positions))
    for i in range(len(new_positions)):
        if mesh.contains([new_positions[i]])[0]:
            inside[i] = True
        pbar.update(1)

    pbar.close()

    new_positions = new_positions[inside]
    return new_positions

    points = mesh.voxelized(pitch=pitch).fill().points
    print(f"刚体 {body_config['objectId']} 粒子数: {len(points)}")
    return points

def fluid_block_processor(dim, config: SimConfig, diameter):
    """流体块处理"""
    total_particles = 0
    for fluid in config.get_fluid_blocks():
        num = compute_particle_num(dim, fluid["start"], fluid["end"], diameter)
        fluid["particleNum"] = num
        total_particles += num
    return total_particles

def compute_particle_num(dim, start, end, spacing):
    """计算区域内粒子数"""
    return reduce(lambda x, y: x * y, 
                 [len(np.arange(start[i], end[i], spacing)) for i in range(dim)])

def compute_box_particle_num(dim, domain_start, domain_end, diameter, thickness):
    """计算边界盒粒子数"""
    points = create_grid_points(dim, (domain_start, domain_end), diameter)
    
    # 边界条件合并
    mask = np.zeros(len(points), dtype=bool)
    for i in range(dim):
        mask |= ((points[:, i] <= domain_start[i] + thickness) | 
                (points[:, i] >= domain_end[i] - thickness))
    
    return np.sum(mask)
        