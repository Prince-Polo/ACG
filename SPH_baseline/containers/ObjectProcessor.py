import numpy as np
import trimesh as tm
from tqdm import tqdm
from functools import reduce
from ..utils import SimConfig
from typing import Dict, List, Tuple, Any

class MeshVoxelizer:
    """网格体素化处理器"""
    def __init__(self, mesh: tm.Trimesh, pitch: float):
        self.mesh = mesh
        self.pitch = pitch
        
    def process(self) -> np.ndarray:
        """执行体素化"""
        voxel_mesh = self.mesh.voxelized(pitch=self.pitch)
        return voxel_mesh.fill().points

class GeometryTransformer:
    """几何变换处理器"""
    PI = 3.1415926
    
    @staticmethod
    def transform_static_body(mesh: tm.Trimesh, 
                            translation: List[float],
                            rotation_angle: float,
                            rotation_axis: List[float]) -> None:
        """变换静态刚体"""
        # 旋转处理
        angle = rotation_angle / 360 * 2 * GeometryTransformer.PI
        rot_matrix = tm.transformations.rotation_matrix(
            angle, rotation_axis, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)
        
        # 平移处理
        mesh.vertices += np.array(translation)

def process_rigid_body(rigid_body: Dict[str, Any], pitch: float) -> np.ndarray:
    """处理单个刚体"""
    # 加载网格
    mesh = tm.load(rigid_body["geometryFile"])
    mesh.apply_scale(rigid_body["scale"])
    
    # 处理静态物体的变换
    if not rigid_body["isDynamic"]:
        GeometryTransformer.transform_static_body(
            mesh,
            rigid_body["translation"],
            rigid_body["rotationAngle"],
            rigid_body["rotationAxis"]
        )
    
    # 保存原始网格数据
    rigid_body["mesh"] = mesh.copy()
    rigid_body["restPosition"] = mesh.vertices
    rigid_body["restCenterOfMass"] = np.array([0.0, 0.0, 0.0])
    
    # 体素化处理
    voxelizer = MeshVoxelizer(mesh, pitch)
    points = voxelizer.process()
    print(f"rigid body {rigid_body['objectId']} num: {points.shape[0]}")
    
    return points

def compute_particle_distribution(dim: int, start: List[float], 
                                end: List[float], space: float) -> int:
    """计算粒子分布"""
    dimensions = [np.arange(start[i], end[i], space) for i in range(dim)]
    return reduce(lambda x, y: x * y, [len(n) for n in dimensions])

def rigid_body_processor(config: SimConfig, diameter: float) -> int:
    """处理所有刚体"""
    total_particles = 0
    for body in config.get_rigid_bodies():
        points = process_rigid_body(body, diameter)
        body["particleNum"] = points.shape[0]
        body["voxelizedPoints"] = points
        total_particles += points.shape[0]
    return total_particles

def fluid_block_processor(dim: int, config: SimConfig, diameter: float) -> int:
    """处理所有流体块"""
    total_particles = 0
    for fluid in config.get_fluid_blocks():
        num = compute_particle_distribution(
            dim, fluid["start"], fluid["end"], diameter)
        fluid["particleNum"] = num
        total_particles += num
    return total_particles

def compute_box_particle_num(dim: int, domain_start: List[float],
                           domain_end: List[float], diameter: float,
                           thickness: float) -> int:
    """计算边界盒粒子数"""
    # 生成网格点
    dimensions = [np.arange(domain_start[i], domain_end[i], diameter) 
                 for i in range(dim)]
    positions = np.array(np.meshgrid(*dimensions, indexing='ij'))
    positions = positions.reshape(-1, reduce(lambda x, y: x * y, 
                                          list(positions.shape[1:]))).transpose()
    
    # 创建边界mask
    mask = np.zeros(positions.shape[0], dtype=bool)
    for i in range(dim):
        mask = mask | (
            (positions[:, i] <= domain_start[i] + thickness) | 
            (positions[:, i] >= domain_end[i] - thickness)
        )
    return np.sum(mask)
