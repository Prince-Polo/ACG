import numpy as np
import trimesh as tm
from tqdm import tqdm
from functools import reduce
from ..utils import SimConfig

def process_mesh(mesh, transform_params):
    """统一的网格变换处理"""
    if transform_params:
        offset = np.array(transform_params["translation"])
        angle = transform_params["rotationAngle"] / 360 * 2 * np.pi
        direction = transform_params["rotationAxis"]
        center = mesh.vertices.mean(axis=0)
        rot_matrix = tm.transformations.rotation_matrix(angle, direction, center)
        mesh.apply_transform(rot_matrix)
        mesh.vertices += offset
    return mesh

def create_grid_points(dim, bounds, spacing):
    """统一的网格点生成"""
    min_point, max_point = bounds
    grid_ranges = [np.arange(min_point[i], max_point[i], spacing) for i in range(dim)]
    return np.array(np.meshgrid(*grid_ranges, indexing='ij')).reshape(dim, -1).T

def fluid_body_processor(dim, config: SimConfig, diameter):
    """流体物体处理"""
    total_particles = 0
    for fluid_body in config.get_fluid_bodies():
        points = load_fluid_body(dim, fluid_body, diameter)
        fluid_body.update({
            "particleNum": len(points),
            "voxelizedPoints": points
        })
        total_particles += len(points)
    return total_particles

def load_fluid_body(dim, body_config, pitch):
    """流体物体加载"""
    mesh = tm.load(body_config["geometryFile"])
    mesh.apply_scale(body_config["scale"])
    mesh = process_mesh(mesh, body_config)
    
    points = create_grid_points(dim, mesh.bounding_box.bounds, pitch)
    print(f"处理 {len(points)} 个点...")
    return points[filter_points_inside_mesh(points, mesh)]

def filter_points_inside_mesh(points, mesh):
    """网格内部点过滤"""
    inside = [False] * len(points)
    with tqdm(total=len(points)) as pbar:
        for i, point in enumerate(points):
            inside[i] = mesh.contains([point])[0]
            pbar.update(1)
    return inside

def rigid_body_processor(config: SimConfig, diameter):
    """刚体处理"""
    total_particles = 0
    for rigid_body in config.get_rigid_bodies():
        points = load_rigid_body(rigid_body, diameter)
        rigid_body.update({
            "particleNum": len(points),
            "voxelizedPoints": points
        })
        total_particles += len(points)
    return total_particles

def load_rigid_body(body_config, pitch):
    """刚体加载"""
    mesh = tm.load(body_config["geometryFile"])
    mesh.apply_scale(body_config["scale"])
    
    # 只处理静态物体的变换
    if not body_config["isDynamic"]:
        mesh = process_mesh(mesh, body_config)
    
    # 保存原始网格
    body_config.update({
        "mesh": mesh.copy(),
        "restPosition": mesh.vertices,
        "restCenterOfMass": np.zeros(3)
    })

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
        