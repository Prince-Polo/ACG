from numba import float32, int32
import numpy as np
from numba import cuda
from . import ObjectProcessor as op
from functools import reduce
from ..utils import SimConfig

class BaseContainer:
    def __init__(self, config: SimConfig, GGUI = False):
        # 基础属性
        base_attrs = {
            'dim': 3,
            'GGUI': GGUI,
            'cfg': config,
            'total_time': 0.0,
            'fluid_object_num': 0,
            'rigid_object_num': 0,
            'rigid_body_num': 0,
            'max_object_num': 10,
        }

        # 物理参数
        physics_attrs = {
            'gravity': np.array([0.0, -9.81, 0.0]),
            'domain_start': np.array(config.get_cfg("domainStart")),
            'domain_end': np.array(config.get_cfg("domainEnd")),
            'material_rigid': 2,
            'material_fluid': 1,
        }
        physics_attrs['domain_size'] = physics_attrs['domain_end'] - physics_attrs['domain_start']

        # 粒子参数  
        self.radius = config.get_cfg("particleRadius")
        particle_attrs = {
            'radius': self.radius,
            'diameter': 2 * self.radius,
            'dh': 4 * self.radius,
            'max_particles_per_cell': 500,
            'max_neighbors': 60,
            'padding': 4 * self.radius,
            'boundary_thickness': 0.0,
        }
        particle_attrs['V0'] = 0.8 * particle_attrs['diameter'] ** base_attrs['dim']

        # 设置属性
        for attrs in [base_attrs, physics_attrs, particle_attrs]:
            for key, value in attrs.items():
                setattr(self, key, value)

        # 边界处理
        self.add_boundary = self.cfg.get_cfg("addDomainBox")
        if self.add_boundary:
            self.domain_start = np.array([self.domain_start[i] + self.padding for i in range(self.dim)])
            self.domain_end = np.array([self.domain_end[i] - self.padding for i in range(self.dim)])
            self.boundary_thickness = 0.03

        # 初始化对象集合
        self.object_collection = dict()
        self.object_id_rigid_body = set()
        self.object_id_fluid_body = set() 
        self.present_object = []

        # 计算粒子数量
        self.fluid_bodies = self.cfg.get_fluid_bodies()
        self.fluid_blocks = self.cfg.get_fluid_blocks()
        self.rigid_bodies = self.cfg.get_rigid_bodies()

        self.fluid_particle_num = op.fluid_block_processor(self.dim, self.cfg, self.diameter)
        self.rigid_particle_num = op.rigid_body_processor(self.cfg, self.diameter)      
        self.particle_max_num = (self.fluid_particle_num + self.rigid_particle_num 
                               + (op.compute_box_particle_num(self.dim, self.domain_start, self.domain_end, 
                                                           diameter=self.diameter, 
                                                           thickness=self.boundary_thickness) if self.add_boundary else 0))

        # 初始化数组
        self._init_arrays()

    def _init_arrays(self):
        """初始化CUDA数组"""
        # 粒子基本属性
        self.particle_num = 0
        self.object_num = 0
        self.fluid_object_num = 0
        self.rigid_object_num = 0
        self.rigid_particle_num = 0
        
        # 粒子属性数组
        self.particle_positions = cuda.to_device(np.zeros((self.particle_max_num, self.dim), dtype=np.float32))
        self.particle_velocities = cuda.to_device(np.zeros((self.particle_max_num, self.dim), dtype=np.float32))
        self.particle_densities = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_rest_densities = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_pressures = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_masses = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_rest_volumes = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_materials = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.int32))
        self.particle_colors = cuda.to_device(np.zeros((self.particle_max_num, 3), dtype=np.int32))
        
        # 粒子动力学属性
        self.particle_forces = cuda.to_device(np.zeros((self.particle_max_num, self.dim), dtype=np.float32))
        self.particle_accelerations = cuda.to_device(np.zeros((self.particle_max_num, self.dim), dtype=np.float32))
        
        # 粒子对象信息
        self.particle_object_ids = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.int32))
        self.particle_is_dynamic = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.int32))
        
        # 对象属性
        self.object_materials = np.zeros(self.max_object_num, dtype=np.int32)
        self.object_particle_num = np.zeros(self.max_object_num, dtype=np.int32)
        self.object_particle_offset = np.zeros(self.max_object_num, dtype=np.int32)
        self.object_visibility = np.ones(self.max_object_num, dtype=np.int32)
        
        # 刚体属性
        self.rigid_body_particle_num = cuda.to_device(np.zeros(self.max_object_num, dtype=np.int32))
        self.rigid_body_masses = cuda.to_device(np.zeros(self.max_object_num, dtype=np.float32))
        self.rigid_body_com = cuda.to_device(np.zeros((self.max_object_num, self.dim), dtype=np.float32))
        self.rigid_body_velocities = cuda.to_device(np.zeros((self.max_object_num, self.dim), dtype=np.float32))
        self.rigid_body_angular_velocities = cuda.to_device(np.zeros((self.max_object_num, self.dim), dtype=np.float32))
        self.rigid_body_forces = cuda.to_device(np.zeros((self.max_object_num, self.dim), dtype=np.float32))
        self.rigid_body_torques = cuda.to_device(np.zeros((self.max_object_num, self.dim), dtype=np.float32))
        self.rigid_body_is_dynamic = cuda.to_device(np.zeros(self.max_object_num, dtype=np.int32))
        self.rigid_body_rotations = cuda.to_device(np.zeros((self.max_object_num, self.dim, self.dim), dtype=np.float32))
        self.rigid_body_original_com = cuda.to_device(np.zeros((self.max_object_num, self.dim), dtype=np.float32))
        
        # 刚体粒子原始位置
        self.rigid_particle_original_positions = cuda.to_device(np.zeros((self.particle_max_num, self.dim), dtype=np.float32))
        
        # 邻居搜索数组
        self.particle_neighbors = cuda.to_device(np.zeros((self.particle_max_num, 500), dtype=np.int32))
        self.particle_neighbor_num = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.int32))
        
        # 对象计数器
        self.fluid_object_num = cuda.to_device(np.array([0], dtype=np.int32))
        self.rigid_object_num = cuda.to_device(np.array([0], dtype=np.int32))

    def _calculate_max_particles(self):
        """计算最大粒子数"""
        fluid_particle_num = op.fluid_block_processor(self.dim, self.cfg, self.diameter)
        rigid_particle_num = op.rigid_body_processor(self.cfg, self.diameter)
        
        if self.add_boundary:
            boundary_particle_num = op.compute_box_particle_num(
                self.dim, 
                self.domain_start, 
                self.domain_end, 
                diameter=self.diameter,
                thickness=self.boundary_thickness
            )
        else:
            boundary_particle_num = 0
            
        return fluid_particle_num + rigid_particle_num + boundary_particle_num
    
    def prepare_neighbor_search(self):
        """执行空间邻居搜索"""
        threads_per_block = 256
        blocks = (self.particle_num + threads_per_block - 1) // threads_per_block
        
        # 重置邻居统计
        self.particle_neighbor_num.copy_to_device(np.zeros(self.particle_max_num, dtype=np.int32))
        
        # 执行空间搜索
        self._spatial_neighbor_search[blocks, threads_per_block](
            self.particle_positions,
            self.particle_neighbors,
            self.particle_neighbor_num,
            self.particle_num,
            self.dh
        )

    @staticmethod
    @cuda.jit
    def _spatial_neighbor_search(positions, neighbors, neighbor_num, particle_num, search_radius):
        """全空间邻居搜索的CUDA kernel
        
        采用直接空间映射策略进行邻居搜索
        """
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
            
        pos_i = positions[idx]
        count = 0
        
        # 全空间遍历
        for j in range(particle_num):
            if j == idx:
                continue
                
            # 计算空间距离
            dist_sq = 0.0
            for d in range(3):
                diff = pos_i[d] - positions[j][d]
                dist_sq += diff * diff
                
            # 基于搜索半径的空间映射
            if dist_sq <= search_radius * search_radius and count < 500:
                neighbors[idx, count] = j
                count += 1
                
        neighbor_num[idx] = count

    def add_cube(self, object_id, start, end, material, velocity, density, is_dynamic, color):
        """添加立方体"""
        # 计算粒子位置
        x = np.arange(start[0], end[0], self.diameter)
        y = np.arange(start[1], end[1], self.diameter)
        z = np.arange(start[2], end[2], self.diameter)
        
        positions = np.array([[xi, yi, zi] for xi in x for yi in y for zi in z], dtype=np.float32)
        num_particles = len(positions)
        
        # 更新对象信息
        self.object_collection[object_id] = num_particles
        self.object_particle_num[object_id] = num_particles
        self.object_particle_offset[object_id] = self.particle_num
        self.object_materials[object_id] = material
        
        if material == self.material_fluid:
            self.object_id_fluid_body.add(object_id)
        else:
            self.object_id_rigid_body.add(object_id)
        
        # 更新粒子属性
        start_idx = self.particle_num
        end_idx = start_idx + num_particles
        
        # 如果是刚体，保存原始位置
        if material == self.material_rigid:
            temp_rigid_particle_original_positions = self.rigid_particle_original_positions.copy_to_host()
            temp_rigid_particle_original_positions[start_idx:end_idx] = positions
            self.rigid_particle_original_positions = cuda.to_device(temp_rigid_particle_original_positions)
            
            # 初始化刚体属性
            temp_rigid_body_is_dynamic = self.rigid_body_is_dynamic.copy_to_host()
            temp_rigid_body_velocities = self.rigid_body_velocities.copy_to_host()
            temp_rigid_body_particle_num = self.rigid_body_particle_num.copy_to_host()
            temp_rigid_body_rotations = self.rigid_body_rotations.copy_to_host()
            
            temp_rigid_body_is_dynamic[object_id] = is_dynamic
            temp_rigid_body_velocities[object_id] = velocity
            temp_rigid_body_particle_num[object_id] = num_particles
            temp_rigid_body_rotations[object_id] = np.eye(self.dim, dtype=np.float32)
            
            self.rigid_body_is_dynamic = cuda.to_device(temp_rigid_body_is_dynamic)
            self.rigid_body_velocities = cuda.to_device(temp_rigid_body_velocities)
            self.rigid_body_particle_num = cuda.to_device(temp_rigid_body_particle_num)
            self.rigid_body_rotations = cuda.to_device(temp_rigid_body_rotations)
            
            if is_dynamic:
                mass = density * self.V0 * num_particles
                com = np.mean(positions, axis=0)
                
                temp_rigid_body_masses = self.rigid_body_masses.copy_to_host()
                temp_rigid_body_com = self.rigid_body_com.copy_to_host()
                temp_rigid_body_original_com = self.rigid_body_original_com.copy_to_host()
                
                temp_rigid_body_masses[object_id] = mass
                temp_rigid_body_com[object_id] = com
                temp_rigid_body_original_com[object_id] = com.copy()
                
                self.rigid_body_masses = cuda.to_device(temp_rigid_body_masses)
                self.rigid_body_com = cuda.to_device(temp_rigid_body_com)
                self.rigid_body_original_com = cuda.to_device(temp_rigid_body_original_com)
        
        # 更新通用粒子属性
        self.particle_positions[start_idx:end_idx] = cuda.to_device(positions)
        self.particle_velocities[start_idx:end_idx] = cuda.to_device(np.full((num_particles, self.dim), velocity, dtype=np.float32))
        self.particle_densities[start_idx:end_idx] = cuda.to_device(np.full(num_particles, density, dtype=np.float32))
        self.particle_masses[start_idx:end_idx] = cuda.to_device(np.full(num_particles, density * self.V0, dtype=np.float32))
        self.particle_materials[start_idx:end_idx] = cuda.to_device(np.full(num_particles, material, dtype=np.int32))
        self.particle_object_ids[start_idx:end_idx] = cuda.to_device(np.full(num_particles, object_id, dtype=np.int32))
        self.particle_is_dynamic[start_idx:end_idx] = cuda.to_device(np.full(num_particles, is_dynamic, dtype=np.int32))
        
        self.particle_num += num_particles
        if material == self.material_rigid:
            self.rigid_particle_num += num_particles

    def add_boundary_object(self, object_id, domain_start, domain_end, thickness, material, is_dynamic, space, color):
        """添加边界对象"""
        # 计算边界盒的顶点
        x_min, y_min, z_min = domain_start - thickness
        x_max, y_max, z_max = domain_end + thickness
        
        # 创建六个面的粒子
        positions = []
        
        # 底面和顶面
        for x in np.arange(x_min, x_max, space):
            for z in np.arange(z_min, z_max, space):
                positions.append([x, y_min, z])
                positions.append([x, y_max, z])
                
        # 前面和后面
        for x in np.arange(x_min, x_max, space):
            for y in np.arange(y_min, y_max, space):
                positions.append([x, y, z_min])
                positions.append([x, y, z_max])
                
        # 左面和右面
        for y in np.arange(y_min, y_max, space):
            for z in np.arange(z_min, z_max, space):
                positions.append([x_min, y, z])
                positions.append([x_max, y, z])
                
        positions = np.array(positions, dtype=np.float32)
        num_particles = len(positions)
        
        # 更新对象信息
        self.object_collection[object_id] = num_particles
        self.object_particle_num[object_id] = num_particles
        self.object_particle_offset[object_id] = self.particle_num
        self.object_materials[object_id] = material
        
        if material == self.material_fluid:
            self.object_id_fluid_body.add(object_id)
        else:
            self.object_id_rigid_body.add(object_id)
        
        # 更新粒子属性
        start_idx = self.particle_num
        end_idx = start_idx + num_particles
        
        self.particle_positions[start_idx:end_idx] = cuda.to_device(positions)
        self.particle_velocities[start_idx:end_idx] = cuda.to_device(np.zeros((num_particles, self.dim), dtype=np.float32))
        self.particle_densities[start_idx:end_idx] = cuda.to_device(np.full(num_particles, 1000.0, dtype=np.float32))
        self.particle_masses[start_idx:end_idx] = cuda.to_device(np.full(num_particles, 1000.0 * self.V0, dtype=np.float32))
        self.particle_materials[start_idx:end_idx] = cuda.to_device(np.full(num_particles, material, dtype=np.int32))
        self.particle_object_ids[start_idx:end_idx] = cuda.to_device(np.full(num_particles, object_id, dtype=np.int32))
        self.particle_is_dynamic[start_idx:end_idx] = cuda.to_device(np.full(num_particles, is_dynamic, dtype=np.int32))
        
        self.particle_num += num_particles
        self.object_num += 1

    def add_rigid_body(self, object_id, positions, material, velocity, density, is_dynamic, color):
        """添加刚体"""
        num_particles = len(positions)
        
        # 更新对象信息
        self.object_collection[object_id] = num_particles
        self.object_particle_num[object_id] = num_particles
        self.object_particle_offset[object_id] = self.particle_num
        self.object_materials[object_id] = material
        
        if material == self.material_fluid:
            self.object_id_fluid_body.add(object_id)
        else:
            self.object_id_rigid_body.add(object_id)
        
        # 更新粒子属性
        start_idx = self.particle_num
        end_idx = start_idx + num_particles
        
        self.particle_positions[start_idx:end_idx] = cuda.to_device(positions.astype(np.float32))
        self.particle_velocities[start_idx:end_idx] = cuda.to_device(np.full((num_particles, self.dim), velocity, dtype=np.float32))
        self.particle_densities[start_idx:end_idx] = cuda.to_device(np.full(num_particles, density, dtype=np.float32))
        self.particle_masses[start_idx:end_idx] = cuda.to_device(np.full(num_particles, density * self.V0, dtype=np.float32))
        self.particle_materials[start_idx:end_idx] = cuda.to_device(np.full(num_particles, material, dtype=np.int32))
        self.particle_object_ids[start_idx:end_idx] = cuda.to_device(np.full(num_particles, object_id, dtype=np.int32))
        self.particle_is_dynamic[start_idx:end_idx] = cuda.to_device(np.full(num_particles, is_dynamic, dtype=np.int32))
        
        self.particle_num += num_particles
        self.object_num += 1

    def get_particle_positions(self):
        """获取粒子位置"""
        return self.particle_positions.copy_to_host()[:self.particle_num]

    def get_particle_velocities(self):
        """获取粒子速度"""
        return self.particle_velocities.copy_to_host()[:self.particle_num]

    def get_particle_densities(self):
        """获取粒子密度"""
        return self.particle_densities.copy_to_host()[:self.particle_num]

    def get_particle_materials(self):
        """获取粒子材质"""
        return self.particle_materials.copy_to_host()[:self.particle_num]

    def get_particle_object_ids(self):
        """获取粒子对象ID"""
        return self.particle_object_ids.copy_to_host()[:self.particle_num]

    def get_particle_is_dynamic(self):
        """获取粒子是否动态"""
        return self.particle_is_dynamic.copy_to_host()[:self.particle_num]

    def add_fluid_block(self, object_id, start, end, velocity, density, color):
        """添加流体块"""
        # 计算粒子位置
        x = np.arange(start[0], end[0], self.diameter)
        y = np.arange(start[1], end[1], self.diameter)
        z = np.arange(start[2], end[2], self.diameter)
        
        positions = np.array([[xi, yi, zi] for xi in x for yi in y for zi in z], dtype=np.float32)
        num_particles = len(positions)
        
        # 更新对象信息
        self.object_collection[object_id] = num_particles
        self.object_particle_num[object_id] = num_particles
        self.object_particle_offset[object_id] = self.particle_num
        self.object_materials[object_id] = self.material_fluid
        self.object_id_fluid_body.add(object_id)
        
        # 更新粒子属性
        start_idx = self.particle_num
        end_idx = start_idx + num_particles
        
        self.particle_positions[start_idx:end_idx] = cuda.to_device(positions)
        self.particle_velocities[start_idx:end_idx] = cuda.to_device(np.full((num_particles, self.dim), velocity, dtype=np.float32))
        self.particle_densities[start_idx:end_idx] = cuda.to_device(np.full(num_particles, density, dtype=np.float32))
        self.particle_masses[start_idx:end_idx] = cuda.to_device(np.full(num_particles, density * self.V0, dtype=np.float32))
        self.particle_materials[start_idx:end_idx] = cuda.to_device(np.full(num_particles, self.material_fluid, dtype=np.int32))
        self.particle_object_ids[start_idx:end_idx] = cuda.to_device(np.full(num_particles, object_id, dtype=np.int32))
        self.particle_is_dynamic[start_idx:end_idx] = cuda.to_device(np.full(num_particles, 1, dtype=np.int32))
        
        self.particle_num += num_particles
        self.object_num += 1
        self.fluid_object_num += 1

    def add_fluid_body(self, object_id, positions, velocity, density, color):
        """添加流体体"""
        num_particles = len(positions)
        
        # 更新对象信息
        self.object_collection[object_id] = num_particles
        self.object_particle_num[object_id] = num_particles
        self.object_particle_offset[object_id] = self.particle_num
        self.object_materials[object_id] = self.material_fluid
        self.object_id_fluid_body.add(object_id)
        
        # 更新粒子属性
        start_idx = self.particle_num
        end_idx = start_idx + num_particles
        
        self.particle_positions[start_idx:end_idx] = cuda.to_device(positions.astype(np.float32))
        self.particle_velocities[start_idx:end_idx] = cuda.to_device(np.full((num_particles, self.dim), velocity, dtype=np.float32))
        self.particle_densities[start_idx:end_idx] = cuda.to_device(np.full(num_particles, density, dtype=np.float32))
        self.particle_masses[start_idx:end_idx] = cuda.to_device(np.full(num_particles, density * self.V0, dtype=np.float32))
        self.particle_materials[start_idx:end_idx] = cuda.to_device(np.full(num_particles, self.material_fluid, dtype=np.int32))
        self.particle_object_ids[start_idx:end_idx] = cuda.to_device(np.full(num_particles, object_id, dtype=np.int32))
        self.particle_is_dynamic[start_idx:end_idx] = cuda.to_device(np.full(num_particles, 1, dtype=np.int32))
        
        self.particle_num += num_particles
        self.object_num += 1
        self.fluid_object_num += 1

    def _set_object_properties(self, obj_id, obj_data, offset=None):
        """设置物体属性的辅助方法"""
        # 设置基本属性
        attrs = {
            'velocity': obj_data["velocity"],
            'density': obj_data["density"], 
            'color': obj_data["color"]
        }
        
        # 处理位置偏移
        if offset is not None:
            offset = np.array(obj_data["translation"])
            attrs['start'] = np.array(obj_data["start"]) + offset
            attrs['end'] = np.array(obj_data["end"]) + offset
            
        # 设置可见性
        self.object_visibility[obj_id] = obj_data.get("visible", 1)
        
        # 设置材质和集合
        self.object_materials[obj_id] = self.material_fluid
        self.object_collection[obj_id] = obj_data
        self.object_id_fluid_body.add(obj_id)
        
        return attrs
    
    def _set_fluid_mesh_properties(self, fluid):
        """设置流体mesh属性"""
        # 提取基础属性
        attrs = {
            'obj_id': fluid["objectId"],
            'particle_num': fluid["particleNum"],
            'voxelized_points': fluid["voxelizedPoints"],
            'velocity': np.array(fluid["velocity"], dtype=np.float32),
            'density': fluid["density"],
            'color': np.array(fluid["color"], dtype=np.int32),
        }
        
        # 设置可见性
        self.object_visibility[attrs['obj_id']] = fluid.get("visible", 1)
        
        # 设置流体属性
        self.object_id_fluid_body.add(attrs['obj_id'])
        self.object_materials[attrs['obj_id']] = self.material_fluid
        self.object_collection[attrs['obj_id']] = fluid
        
        return attrs

    def _set_rigid_body_properties(self, rigid):
        """设置刚体属性"""
        attrs = {
            'obj_id': rigid["objectId"],
            'particle_num': rigid["particleNum"],
            'positions': np.array(rigid["voxelizedPoints"], dtype=np.float32),
            'density': rigid["density"],
            'color': np.array(rigid["color"], dtype=np.int32),
            'is_dynamic': rigid["isDynamic"],
            'velocity': (np.array(rigid["velocity"], dtype=np.float32) 
                        if rigid["isDynamic"] 
                        else np.zeros(self.dim, dtype=np.float32))
        }
        
        # 设置对象属性
        self.object_visibility[attrs['obj_id']] = rigid.get("visible", 1)
        self.object_materials[attrs['obj_id']] = self.material_rigid
        self.object_collection[attrs['obj_id']] = rigid
        self.object_id_rigid_body.add(attrs['obj_id'])
        
        return attrs
    
    def _update_rigid_body_arrays(self, obj_id, attrs):
        """更新刚体数组"""
        # 基础属性更新
        arrays_basic = {
            'rigid_body_is_dynamic': attrs['is_dynamic'],
            'rigid_body_velocities': attrs['velocity'],
            'rigid_body_particle_num': attrs['particle_num'],
            'rigid_body_rotations': np.eye(self.dim, dtype=np.float32)
        }
        
        # 复制和更新基础数组
        for name, value in arrays_basic.items():
            temp_array = getattr(self, name).copy_to_host()
            temp_array[obj_id] = value
            setattr(self, name, cuda.to_device(temp_array))
        
        # 处理动态刚体属性
        if attrs['is_dynamic']:
            mass = self.compute_rigid_mass(obj_id)
            com = self.compute_rigid_com(obj_id)
            
            arrays_dynamic = {
                'rigid_body_masses': mass,
                'rigid_body_com': com,
                'rigid_body_original_com': com.copy()
            }
            
            # 复制和更新动态数组
            for name, value in arrays_dynamic.items():
                temp_array = getattr(self, name).copy_to_host()
                temp_array[obj_id] = value
                setattr(self, name, cuda.to_device(temp_array))
        
        return mass if attrs['is_dynamic'] else None

    def _update_object_counters(self, fluid_count, rigid_count, rigid_body_count):
        """更新对象计数器"""
        counter_arrays = {
            'fluid_object_num': fluid_count,
            'rigid_object_num': rigid_count,
            'object_num': fluid_count + rigid_count,
        }
        
        for name, value in counter_arrays.items():
            setattr(self, name, cuda.to_device(np.array([value], dtype=np.int32)))
        
        self.rigid_body_num = rigid_body_count

    def insert_object(self):
        """插入对象"""
        fluid_count = 0
        rigid_count = 0
        rigid_body_count = 0
        
        # 流体
        for fluid in self.fluid_blocks:
            obj_id = fluid["objectId"]
            if obj_id in self.present_object or fluid["entryTime"] > self.total_time:
                continue
            
            attrs = self._set_object_properties(obj_id, fluid, offset=True)
            start, end = attrs['start'], attrs['end']
            velocity = attrs['velocity']
            density = attrs['density'] 
            color = attrs['color']
            
            self.add_cube(
                object_id = obj_id,
                start = start,
                end = end,
                material = self.material_fluid,
                velocity = velocity,
                density = density,
                is_dynamic = True,
                color = color
            )
            
            fluid_count += 1
            self.present_object.append(obj_id)
        
        # 流体mesh
        for fluid in self.fluid_bodies:
            attrs = self._set_fluid_mesh_properties(fluid)
            obj_id = attrs['obj_id']
            
            if obj_id in self.present_object:
                continue
            if fluid["entryTime"] > self.total_time:
                continue
                
            self.add_particles(
                obj_id,
                attrs['particle_num'],
                attrs['voxelized_points'],
                np.stack([attrs['velocity'] for _ in range(attrs['particle_num'])]),
                attrs['density'] * np.ones(attrs['particle_num'], dtype=np.float32),
                np.zeros(attrs['particle_num'], dtype=np.float32),
                np.array([self.material_fluid for _ in range(attrs['particle_num'])], dtype=np.int32),
                1 * np.ones(attrs['particle_num'], dtype=np.int32),
                np.stack([attrs['color'] for _ in range(attrs['particle_num'])])
            )
            
            fluid_count += 1
            self.present_object.append(obj_id)
        
        # 刚体
        for rigid in self.rigid_bodies:
            attrs = self._set_rigid_body_properties(rigid)
            obj_id = attrs['obj_id']
            
            if obj_id in self.present_object:
                continue
            if rigid["entryTime"] > self.total_time:
                continue
                
            # 保存原始位置
            start_idx = self.particle_num
            end_idx = start_idx + attrs['particle_num']
            temp_rigid_particle_original_positions = self.rigid_particle_original_positions.copy_to_host()
            temp_rigid_particle_original_positions[start_idx:end_idx] = attrs['positions']
            self.rigid_particle_original_positions = cuda.to_device(temp_rigid_particle_original_positions)
            
            # 添加粒子
            self.add_particles(
                obj_id,
                attrs['particle_num'],
                attrs['positions'],
                np.stack([attrs['velocity'] for _ in range(attrs['particle_num'])]),
                attrs['density'] * np.ones(attrs['particle_num'], dtype=np.float32),
                np.zeros(attrs['particle_num'], dtype=np.float32),
                np.array([self.material_rigid for _ in range(attrs['particle_num'])], dtype=np.int32),
                attrs['is_dynamic'] * np.ones(attrs['particle_num'], dtype=np.int32),
                np.stack([attrs['color'] for _ in range(attrs['particle_num'])])
            )
            
            # 更新刚体属性
            self._update_rigid_body_arrays(obj_id, attrs)
            self.rigid_particle_num += attrs['particle_num']
            rigid_count += 1
            rigid_body_count += 1
            self.present_object.append(obj_id)
        
        # 更新计数器a
        self._update_object_counters(fluid_count, rigid_count, rigid_body_count)

    def dump(self, obj_id):
        """导出对象数据"""
        # 获取对象ID数组
        np_object_id = self.particle_object_ids.copy_to_host()
        # 创建掩码，找到所有属于该对象的粒子
        mask = (np_object_id == obj_id).nonzero()[0]

        # 复制位置数据到主机内存并用掩码选择
        np_x = self.particle_positions.copy_to_host()[mask]
        # 复制速度数据到主机内存并掩码选择
        np_v = self.particle_velocities.copy_to_host()[mask]

        return {
            'position': np_x,
            'velocity': np_v
        }

    def add_particles(self, object_id, new_particles_num, new_positions, new_velocities,
                 new_densities, new_pressures, new_materials, new_is_dynamic, new_colors):
        """添加多个粒子"""
        if new_particles_num == 0:
            return
        
        threads_per_block = 256
        blocks = (new_particles_num + threads_per_block - 1) // threads_per_block
        
        add_particles_kernel[blocks, threads_per_block](
            self.particle_positions, self.particle_velocities, self.particle_densities,
            self.particle_pressures, self.particle_masses, self.particle_rest_volumes,
            self.particle_rest_densities, self.particle_materials, self.particle_colors,
            self.particle_object_ids, self.particle_is_dynamic, self.rigid_particle_original_positions,
            cuda.to_device(new_positions), cuda.to_device(new_velocities),
            cuda.to_device(new_densities), cuda.to_device(new_pressures),
            cuda.to_device(new_materials), cuda.to_device(new_is_dynamic),
            cuda.to_device(new_colors), object_id, self.particle_num, new_particles_num,
            self.V0
        )
        
        self.particle_num += new_particles_num

    def compute_rigid_mass(self, object_id):
        """计算刚体质量"""
        mass = 0.0
        particle_object_ids = self.particle_object_ids.copy_to_host()
        particle_is_dynamic = self.particle_is_dynamic.copy_to_host()
        particle_rest_densities = self.particle_rest_densities.copy_to_host()
        
        for i in range(self.particle_num):
            if (particle_object_ids[i] == object_id and particle_is_dynamic[i]):
                mass += particle_rest_densities[i] * self.V0
                
        return mass

    def compute_rigid_com(self, object_id):
        """计算刚体质心"""
        mass = self.compute_rigid_mass(object_id)
        if mass == 0:
            return np.zeros(3, dtype=np.float32)
        
        com = np.zeros(3, dtype=np.float32)
        particle_object_ids = self.particle_object_ids.copy_to_host()
        particle_is_dynamic = self.particle_is_dynamic.copy_to_host()
        particle_positions = self.particle_positions.copy_to_host()
        particle_rest_densities = self.particle_rest_densities.copy_to_host()
        
        for i in range(self.particle_num):
            if (particle_object_ids[i] == object_id and particle_is_dynamic[i]):
                particle_mass = particle_rest_densities[i] * self.V0
                com += particle_positions[i] * particle_mass
                
        return (com / mass).astype(np.float32)

    @staticmethod
    @cuda.jit
    def _fill_array_kernel(arr, value):
        """填充数组的CUDA kernel"""
        idx = cuda.grid(1)
        if idx < arr.size:
            arr[idx] = value

@cuda.jit(device=True)
def add_particle_device(particle_positions, particle_velocities, particle_densities, 
                       particle_pressures, particle_masses, particle_rest_volumes,
                       particle_rest_densities, particle_materials, particle_colors,
                       particle_object_ids, particle_is_dynamic, rigid_particle_original_positions,
                       p, obj_id, x, v, density, pressure, material, is_dynamic, color, V0):
    """设备函数：添加一个粒子"""
    # 向量属性(3D)
    vector_props = {
        'positions': (particle_positions, x),
        'velocities': (particle_velocities, v),
        'rigid_positions': (rigid_particle_original_positions, x),
        'colors': (particle_colors, color)
    }
    for prop_name, (array, value) in vector_props.items():
        for d in range(3):
            array[p, d] = value[d]
    
    # 标量属性
    scalar_props = {
        'densities': (particle_densities, density),
        'pressures': (particle_pressures, pressure),
        'rest_densities': (particle_rest_densities, density),
        'rest_volumes': (particle_rest_volumes, V0),
        'masses': (particle_masses, V0 * density),
        'materials': (particle_materials, material),
        'object_ids': (particle_object_ids, obj_id),
        'is_dynamic': (particle_is_dynamic, is_dynamic)
    }
    for prop_name, (array, value) in scalar_props.items():
        array[p] = value

@cuda.jit
def add_particles_kernel(particle_positions, particle_velocities, particle_densities,
                        particle_pressures, particle_masses, particle_rest_volumes,
                        particle_rest_densities, particle_materials, particle_colors,
                        particle_object_ids, particle_is_dynamic, rigid_particle_original_positions,
                        new_positions, new_velocities, new_densities, new_pressures,
                        new_materials, new_is_dynamic, new_colors,
                        object_id, particle_num, new_particles_num, V0):
    """CUDA kernel：批量添加粒子"""
    idx = cuda.grid(1)
    if idx < new_particles_num:
        p = particle_num + idx
        add_particle_device(particle_positions, particle_velocities, particle_densities,
                          particle_pressures, particle_masses, particle_rest_volumes,
                          particle_rest_densities, particle_materials, particle_colors,
                          particle_object_ids, particle_is_dynamic, rigid_particle_original_positions,
                          p, object_id, new_positions[idx], new_velocities[idx],
                          new_densities[idx], new_pressures[idx], new_materials[idx],
                          new_is_dynamic[idx], new_colors[idx], V0)

