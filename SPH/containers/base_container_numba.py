from numba import float32, int32
import numpy as np
import numba
from numba import cuda
from . import ObjectProcessor as op
from functools import reduce
from ..utils import SimConfig

class BaseContainer:
    def __init__(self, config: SimConfig, GGUI = False):
        self.dim = 3
        self.GGUI = GGUI
        self.cfg = config
        self.total_time = 0.0
        self.fluid_object_num = 0
        self.rigid_object_num = 0
        self.rigid_body_num = 0
        
        self.gravity = np.array([0.0, -9.81, 0.0])
        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_start = np.array(self.cfg.get_cfg("domainStart"))
        self.domain_end = np.array([1.0, 1.0, 1.0])
        self.domain_end = np.array(self.cfg.get_cfg("domainEnd"))
        
        self.domain_size = self.domain_end - self.domain_start
        
        self.material_rigid = 2
        self.material_fluid = 1
        
        self.radius = 0.01 
        self.radius = self.cfg.get_cfg("particleRadius")
        self.diameter = 2 * self.radius
        self.V0 = 0.8 * self.diameter ** self.dim
        self.dh = 4 * self.radius
        
        self.max_object_num = 10
        
        self.grid_size = np.array([self.dh, self.dh, self.dh], dtype=np.float32)
        self.padding = self.grid_size[0]
        self.boundary_thickness = 0.0
        self.add_boundary = False
        self.add_boundary = self.cfg.get_cfg("addDomainBox")
        
        if self.add_boundary:
            self.domain_start = np.array([self.domain_start[i] + self.padding for i in range(self.dim)])
            self.domain_end = np.array([self.domain_end[i] - self.padding for i in range(self.dim)])
            self.boundary_thickness = 0.03
        
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        
        # All objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid_body = set()
        self.object_id_fluid_body = set()
        self.present_object = []
        
        #========== Compute number of particles ==========#
        self.fluid_bodies = self.cfg.get_fluid_bodies()
        self.fluid_blocks = self.cfg.get_fluid_blocks()
        self.rigid_bodies = self.cfg.get_rigid_bodies()
        
        self.fluid_particle_num = op.fluid_body_processor(self.dim, self.cfg, self.diameter)
        self.fluid_particle_num += op.fluid_block_processor(self.dim, self.cfg, self.diameter)
        self.rigid_particle_num = op.rigid_body_processor(self.cfg, self.diameter)      
        self.particle_max_num = (self.fluid_particle_num + self.rigid_particle_num 
                                + (op.compute_box_particle_num(self.dim, self.domain_start, self.domain_end, diameter = self.diameter, thickness=self.boundary_thickness) if self.add_boundary else 0)
                                )    
        
        #========== Initialize arrays ==========#
        self._init_arrays()

    def _init_arrays(self):
        """初始化CUDA数组"""
        # 粒子基本属性
        self.particle_num = 0
        self.object_num = 0
        self.fluid_object_num = 0
        self.rigid_object_num = 0
        self.rigid_particle_num = 0
        
        # 粒子数组
        self.particle_positions = cuda.to_device(np.zeros((self.particle_max_num, self.dim), dtype=np.float32))
        self.particle_velocities = cuda.to_device(np.zeros((self.particle_max_num, self.dim), dtype=np.float32))
        self.particle_densities = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_rest_densities = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_pressures = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_masses = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_rest_volumes = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_materials = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.int32))
        self.particle_colors = cuda.to_device(np.zeros((self.particle_max_num, 3), dtype=np.int32))
        
        # 粒子力和加速度
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
        
        # 网格相关
        self.grid_num_particles = cuda.to_device(np.zeros(reduce(lambda x, y: x * y, self.grid_num), dtype=np.int32))
        self.grid_particle_ids = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.int32))
        self.grid_ids = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.int32))
        
        # 邻居搜索
        self.particle_neighbors = cuda.to_device(np.zeros((self.particle_max_num, 60), dtype=np.int32))
        self.particle_neighbor_num = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.int32))
        
        # 对象计数器
        self.fluid_object_num = cuda.to_device(np.array([0], dtype=np.int32))
        self.rigid_object_num = cuda.to_device(np.array([0], dtype=np.int32))
        self.object_num = cuda.to_device(np.array([0], dtype=np.int32))

    def _calculate_max_particles(self):
        """计算最大粒子数"""
        fluid_particle_num = op.fluid_body_processor(self.dim, self.cfg, self.diameter)
        fluid_particle_num += op.fluid_block_processor(self.dim, self.cfg, self.diameter)
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
        """准备邻居搜索"""
        threads_per_block = 256
        blocks = (self.particle_num + threads_per_block - 1) // threads_per_block
        
        # 1. 计算网格ID
        self._compute_grid_ids_kernel[blocks, threads_per_block](
            self.particle_positions,
            self.grid_ids,
            self.grid_size,
            self.grid_num,
            self.particle_num
        )
        
        # 2. 初始化网格计数器
        grid_blocks = (self.grid_num_particles.size + threads_per_block - 1) // threads_per_block
        self._fill_array_kernel[grid_blocks, threads_per_block](self.grid_num_particles, 0)
        
        # 3. 统计每个网格中的粒子数
        self._count_particles_per_grid_kernel[blocks, threads_per_block](
            self.grid_ids,
            self.grid_num_particles,
            self.particle_num
        )
        
        # 4. 计算前缀和
        prefix_sum = np.cumsum(self.grid_num_particles.copy_to_host())
        self.grid_particle_ids = cuda.to_device(np.zeros_like(prefix_sum))  # 重置计数器
        self.grid_num_particles = cuda.to_device(prefix_sum)
        
        # 5. 创建临时数组 - 只创建一次
        temp_positions = cuda.device_array_like(self.particle_positions)
        temp_velocities = cuda.device_array_like(self.particle_velocities)
        temp_densities = cuda.device_array_like(self.particle_densities)
        temp_masses = cuda.device_array_like(self.particle_masses)
        temp_materials = cuda.device_array_like(self.particle_materials)
        temp_object_ids = cuda.device_array_like(self.particle_object_ids)
        temp_is_dynamic = cuda.device_array_like(self.particle_is_dynamic)
        temp_pressures = cuda.device_array_like(self.particle_pressures)
        temp_rest_densities = cuda.device_array_like(self.particle_rest_densities)
        temp_rest_volumes = cuda.device_array_like(self.particle_rest_volumes)
        temp_colors = cuda.device_array_like(self.particle_colors)
        temp_rigid_original_positions = cuda.device_array_like(self.rigid_particle_original_positions)
        
        # 6. 第一步排序
        self._sort_particles_step1_kernel[blocks, threads_per_block](
            self.particle_positions, self.particle_velocities, self.particle_densities,
            self.particle_masses, self.particle_materials, self.particle_object_ids,
            self.particle_is_dynamic, self.particle_pressures, self.particle_rest_densities,
            self.particle_rest_volumes, self.particle_colors, self.rigid_particle_original_positions,
            self.grid_ids, self.grid_particle_ids, self.grid_num_particles,
            temp_positions, temp_velocities, temp_densities,
            temp_masses, temp_materials, temp_object_ids,
            temp_is_dynamic, temp_pressures, temp_rest_densities,
            temp_rest_volumes, temp_colors, temp_rigid_original_positions,
            self.particle_num
        )
        
        # 7. 第二步排序
        self._sort_particles_step2_kernel[blocks, threads_per_block](
            self.particle_positions, self.particle_velocities, self.particle_densities,
            self.particle_masses, self.particle_materials, self.particle_object_ids,
            self.particle_is_dynamic, self.particle_pressures, self.particle_rest_densities,
            self.particle_rest_volumes, self.particle_colors, self.rigid_particle_original_positions,
            temp_positions, temp_velocities, temp_densities,
            temp_masses, temp_materials, temp_object_ids,
            temp_is_dynamic, temp_pressures, temp_rest_densities,
            temp_rest_volumes, temp_colors, temp_rigid_original_positions,
            self.particle_num
        )
        
        # 8. 同步确保数据复制完成
        cuda.synchronize()

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

    def find_neighbors(self):
        """执行邻居搜索"""
        threads_per_block = 256
        blocks = (self.particle_num + threads_per_block - 1) // threads_per_block
        
        self._find_neighbors_kernel[blocks, threads_per_block](
            self.particle_positions,
            self.grid_particle_ids,
            self.grid_num_particles,
            self.grid_num,
            self.particle_neighbors,
            self.particle_neighbor_num,
            self.dh,
            self.particle_num
        )

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

    def insert_object(self):
        """插入对象"""
        fluid_count = 0
        rigid_count = 0
        rigid_body_count = 0
        
        #fluid block
        for fluid in self.fluid_blocks:
            obj_id = fluid["objectId"]
            if obj_id in self.present_object:
                continue
            if fluid["entryTime"] > self.total_time:
                continue
            
            offset = np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            velocity = fluid["velocity"]
            density = fluid["density"]
            color = fluid["color"]
            self.object_id_fluid_body.add(obj_id)
            
            if "visible" in fluid:
                self.object_visibility[obj_id] = fluid["visible"]
            else:
                self.object_visibility[obj_id] = 1
            
            self.object_materials[obj_id] = self.material_fluid
            self.object_collection[obj_id] = fluid
            
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
        
        #fluid body 
        for fluid in self.fluid_bodies:
            obj_id = fluid["objectId"]
            if obj_id in self.present_object:
                continue
            if fluid["entryTime"] > self.total_time:
                continue
            
            particle_num = fluid["particleNum"]
            voxelized_points_np = fluid["voxelizedPoints"]
            velocity = np.array(fluid["velocity"], dtype=np.float32)
            
            density = fluid["density"]
            color = np.array(fluid["color"], dtype=np.int32)
            
            if "visible" in fluid:
                self.object_visibility[obj_id] = fluid["visible"]
            else:
                self.object_visibility[obj_id] = 1
            
            self.object_id_fluid_body.add(obj_id)
            self.object_materials[obj_id] = self.material_fluid
            self.object_collection[obj_id] = fluid
            
            self.add_particles(
                obj_id,
                particle_num,
                np.array(voxelized_points_np, dtype=np.float32), # position
                np.stack([velocity for _ in range(particle_num)]), # velocity
                density * np.ones(particle_num, dtype=np.float32), # density
                np.zeros(particle_num, dtype=np.float32), # pressure
                np.array([self.material_fluid for _ in range(particle_num)], dtype=np.int32), 
                1 * np.ones(particle_num, dtype=np.int32), # dynamic
                np.stack([color for _ in range(particle_num)])
            )
            
            fluid_count += 1
            self.present_object.append(obj_id)
        
        #rigid body
        for rigid in self.rigid_bodies:
            obj_id = rigid["objectId"]
            if obj_id in self.present_object:
                continue
            if rigid["entryTime"] > self.total_time:
                continue
            
            self.object_id_rigid_body.add(obj_id)
            particle_num = rigid["particleNum"]
            voxelized_points_np = rigid["voxelizedPoints"]
            positions = np.array(voxelized_points_np, dtype=np.float32)
            
            density = rigid["density"]
            color = np.array(rigid["color"], dtype=np.int32)
            is_dynamic = rigid["isDynamic"]
            
            if "visible" in rigid:
                self.object_visibility[obj_id] = rigid["visible"]
            else:
                self.object_visibility[obj_id] = 1
            
            self.object_materials[obj_id] = self.material_rigid
            self.object_collection[obj_id] = rigid
            
            if is_dynamic:
                velocity = np.array(rigid["velocity"], dtype=np.float32)
            else:
                velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
            
            # 保存原始位置
            start_idx = self.particle_num
            end_idx = start_idx + particle_num
            temp_rigid_particle_original_positions = self.rigid_particle_original_positions.copy_to_host()
            temp_rigid_particle_original_positions[start_idx:end_idx] = positions
            self.rigid_particle_original_positions = cuda.to_device(temp_rigid_particle_original_positions)
            
            self.add_particles(
                obj_id,
                particle_num,
                positions,
                np.stack([velocity for _ in range(particle_num)]),
                density * np.ones(particle_num, dtype=np.float32),
                np.zeros(particle_num, dtype=np.float32),
                np.array([self.material_rigid for _ in range(particle_num)], dtype=np.int32), 
                is_dynamic * np.ones(particle_num, dtype=np.int32),
                np.stack([color for _ in range(particle_num)])
            )
            
            # 更新刚体属性
            temp_rigid_body_is_dynamic = self.rigid_body_is_dynamic.copy_to_host()
            temp_rigid_body_velocities = self.rigid_body_velocities.copy_to_host()
            temp_rigid_body_particle_num = self.rigid_body_particle_num.copy_to_host()
            temp_rigid_body_rotations = self.rigid_body_rotations.copy_to_host()
            
            temp_rigid_body_is_dynamic[obj_id] = is_dynamic
            temp_rigid_body_velocities[obj_id] = velocity
            temp_rigid_body_particle_num[obj_id] = particle_num
            # 初始化为单位矩阵
            temp_rigid_body_rotations[obj_id] = np.eye(self.dim, dtype=np.float32)
            
            self.rigid_body_is_dynamic = cuda.to_device(temp_rigid_body_is_dynamic)
            self.rigid_body_velocities = cuda.to_device(temp_rigid_body_velocities)
            self.rigid_body_particle_num = cuda.to_device(temp_rigid_body_particle_num)
            self.rigid_body_rotations = cuda.to_device(temp_rigid_body_rotations)
            
            self.rigid_particle_num += particle_num
        
            if is_dynamic:
                mass = self.compute_rigid_mass(obj_id)
                com = self.compute_rigid_com(obj_id)
                
                temp_rigid_body_masses = self.rigid_body_masses.copy_to_host()
                temp_rigid_body_com = self.rigid_body_com.copy_to_host()
                temp_rigid_body_original_com = self.rigid_body_original_com.copy_to_host()
                
                temp_rigid_body_masses[obj_id] = mass
                temp_rigid_body_com[obj_id] = com
                temp_rigid_body_original_com[obj_id] = com.copy()  # 保存原始质心位置
                
                self.rigid_body_masses = cuda.to_device(temp_rigid_body_masses)
                self.rigid_body_com = cuda.to_device(temp_rigid_body_com)
                self.rigid_body_original_com = cuda.to_device(temp_rigid_body_original_com)
            
            rigid_count += 1
            rigid_body_count += 1
            self.present_object.append(obj_id)
        
        # 更新计数器a
        self.fluid_object_num = cuda.to_device(np.array([fluid_count], dtype=np.int32))
        self.rigid_object_num = cuda.to_device(np.array([rigid_count], dtype=np.int32))
        self.object_num = cuda.to_device(np.array([fluid_count + rigid_count], dtype=np.int32))
        self.rigid_body_num = rigid_body_count

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
    def _compute_grid_ids_kernel(positions, grid_ids, grid_size, grid_num, particle_num):
        """计算每个粒子所在的网格ID"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        # 计算粒子所在的网格索引
        cell_x = int(positions[idx, 0] / grid_size[0])
        cell_y = int(positions[idx, 1] / grid_size[1])
        cell_z = int(positions[idx, 2] / grid_size[2])
        
        # 确保在网格范围内
        if cell_x < 0:
            cell_x = 0
        elif cell_x >= grid_num[0]:
            cell_x = grid_num[0] - 1
        
        if cell_y < 0:
            cell_y = 0
        elif cell_y >= grid_num[1]:
            cell_y = grid_num[1] - 1
        
        if cell_z < 0:
            cell_z = 0
        elif cell_z >= grid_num[2]:
            cell_z = grid_num[2] - 1
            
        # 计算展平后的网格ID
        grid_ids[idx] = cell_x + cell_y * grid_num[0] + cell_z * grid_num[0] * grid_num[1]

    @staticmethod
    @cuda.jit
    def _count_particles_per_grid_kernel(grid_ids, grid_num_particles, particle_num):
        """计算每个格子中的粒子数"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        grid_id = grid_ids[idx]
        cuda.atomic.add(grid_num_particles, grid_id, 1)

    @staticmethod
    @cuda.jit
    def _sort_particles_step1_kernel(positions, velocities, densities, masses, materials, 
                                   object_ids, is_dynamic, pressures, rest_densities, 
                                   rest_volumes, colors, rigid_original_positions,
                                   grid_ids, grid_particle_ids, grid_num_particles,
                                   temp_positions, temp_velocities, temp_densities, 
                                   temp_masses, temp_materials, temp_object_ids, 
                                   temp_is_dynamic, temp_pressures, temp_rest_densities,
                                   temp_rest_volumes, temp_colors, temp_rigid_original_positions,
                                   particle_num):
        """排序第一步：复制到临时数组"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        grid_id = grid_ids[idx]
        sorted_idx = grid_num_particles[grid_id - 1] if grid_id > 0 else 0
        actual_idx = cuda.atomic.add(grid_particle_ids, sorted_idx, 1)
        
        # 复制所有粒子数据到临时数组
        for i in range(3):
            temp_positions[actual_idx, i] = positions[idx, i]
            temp_velocities[actual_idx, i] = velocities[idx, i]
            temp_colors[actual_idx, i] = colors[idx, i]
            temp_rigid_original_positions[actual_idx, i] = rigid_original_positions[idx, i]
        
        temp_densities[actual_idx] = densities[idx]
        temp_masses[actual_idx] = masses[idx]
        temp_materials[actual_idx] = materials[idx]
        temp_object_ids[actual_idx] = object_ids[idx]
        temp_is_dynamic[actual_idx] = is_dynamic[idx]
        temp_pressures[actual_idx] = pressures[idx]
        temp_rest_densities[actual_idx] = rest_densities[idx]
        temp_rest_volumes[actual_idx] = rest_volumes[idx]

    @staticmethod
    @cuda.jit
    def _sort_particles_step2_kernel(positions, velocities, densities, masses, materials,
                                   object_ids, is_dynamic, pressures, rest_densities,
                                   rest_volumes, colors, rigid_original_positions,
                                   temp_positions, temp_velocities, temp_densities,
                                   temp_masses, temp_materials, temp_object_ids,
                                   temp_is_dynamic, temp_pressures, temp_rest_densities,
                                   temp_rest_volumes, temp_colors, temp_rigid_original_positions,
                                   particle_num):
        """排序第二步：从临时数组复制回原数组"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        # 从临时数组复制所有数据回原数组
        for i in range(3):
            positions[idx, i] = temp_positions[idx, i]
            velocities[idx, i] = temp_velocities[idx, i]
            colors[idx, i] = temp_colors[idx, i]
            rigid_original_positions[idx, i] = temp_rigid_original_positions[idx, i]
        
        densities[idx] = temp_densities[idx]
        masses[idx] = temp_masses[idx]
        materials[idx] = temp_materials[idx]
        object_ids[idx] = temp_object_ids[idx]
        is_dynamic[idx] = temp_is_dynamic[idx]
        pressures[idx] = temp_pressures[idx]
        rest_densities[idx] = temp_rest_densities[idx]
        rest_volumes[idx] = temp_rest_volumes[idx]

    @staticmethod
    @cuda.jit
    def _find_neighbors_kernel(positions, grid_ids, grid_num_particles, grid_num,
                         particle_neighbors, particle_neighbor_num, h, particle_num):
        """查找邻居粒子的CUDA kernel"""
        idx = cuda.grid(1)
        if idx >= particle_num:
            return
        
        pos_i = positions[idx]
        neighbor_count = 0
        
        # 获取网格位置
        grid_idx = grid_ids[idx]
        grid_pos = cuda.local.array(3, dtype=int32)
        grid_pos[0] = grid_idx % grid_num[0]
        grid_pos[1] = (grid_idx // grid_num[0]) % grid_num[1]
        grid_pos[2] = grid_idx // (grid_num[0] * grid_num[1])
        
        # 遍历相邻网格
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    nx = grid_pos[0] + i
                    ny = grid_pos[1] + j
                    nz = grid_pos[2] + k
                    
                    if (nx < 0 or nx >= grid_num[0] or
                        ny < 0 or ny >= grid_num[1] or
                        nz < 0 or nz >= grid_num[2]):
                        continue
                        
                    neighbor_idx = nx + ny * grid_num[0] + nz * grid_num[0] * grid_num[1]
                    start_idx = 0 if neighbor_idx == 0 else grid_num_particles[neighbor_idx - 1]
                    end_idx = grid_num_particles[neighbor_idx]
                    
                    for p in range(start_idx, end_idx):
                        if p == idx:
                            continue
                            
                        r = cuda.local.array(3, dtype=float32)
                        r_norm_sq = 0.0
                        for d in range(3):
                            r[d] = pos_i[d] - positions[p][d]
                            r_norm_sq += r[d] * r[d]
                        
                        if r_norm_sq < h * h:
                            if neighbor_count < 60:  # 最大邻居数限制
                                particle_neighbors[idx, neighbor_count] = p
                                neighbor_count += 1
        
        particle_neighbor_num[idx] = neighbor_count

    @staticmethod
    @cuda.jit
    def _fill_array_kernel(arr, value):
        """填充数组的CUDA kernel"""
        idx = cuda.grid(1)
        if idx < arr.size:
            arr[idx] = value

@cuda.jit(device=True)
def flatten_grid_index(grid_index, grid_num):
    """将3D网格索引转换为1D索引"""
    ret = 0
    for i in range(3):
        ret_p = grid_index[i]
        for j in range(i+1, 3):
            ret_p *= grid_num[j]
        ret += ret_p
    return ret

@cuda.jit(device=True)
def pos_to_index(pos, grid_size, grid_num, out_index):
    """将位置转换为网格索引"""
    for i in range(3):
        out_index[i] = int(pos[i] / grid_size[i])
        # 确保索引在有效范围内
        if out_index[i] < 0:
            out_index[i] = 0
        if out_index[i] >= grid_num[i]:
            out_index[i] = grid_num[i] - 1

@cuda.jit(device=True)
def get_flatten_grid_index(pos, grid_size, grid_num):
    """获取位置对应的扁平化网格索引"""
    grid_index = pos_to_index(pos, grid_size)
    return flatten_grid_index(grid_index, grid_num)

@cuda.jit(device=True)
def add_particle_device(particle_positions, particle_velocities, particle_densities, 
                       particle_pressures, particle_masses, particle_rest_volumes,
                       particle_rest_densities, particle_materials, particle_colors,
                       particle_object_ids, particle_is_dynamic, rigid_particle_original_positions,
                       p, obj_id, x, v, density, pressure, material, is_dynamic, color, V0):
    """设备函数：添加一个粒子"""
    for d in range(3):
        particle_positions[p, d] = x[d]
        particle_velocities[p, d] = v[d]
        rigid_particle_original_positions[p, d] = x[d]
        particle_colors[p, d] = color[d]
    
    particle_densities[p] = density
    particle_pressures[p] = pressure
    particle_rest_densities[p] = density
    particle_rest_volumes[p] = V0
    particle_masses[p] = V0 * density
    particle_materials[p] = material
    particle_object_ids[p] = obj_id
    particle_is_dynamic[p] = is_dynamic

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

