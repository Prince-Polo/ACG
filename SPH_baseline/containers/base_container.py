import numpy as np
from . import ObjectProcessor as op
from ..utils import SimConfig

class BaseContainerBaseline:
    def __init__(self, config: SimConfig, GGUI = False):
        # 基础配置
        basic_attrs = {
            'dim': 3,
            'GGUI': GGUI,
            'cfg': config,
            'total_time': 0,
            'gravity': np.array([0.0, -9.81, 0.0]),
            'domain_start': np.array(config.get_cfg("domainStart")),
            'domain_end': np.array(config.get_cfg("domainEnd")),
            'material_rigid': 2,
            'material_fluid': 1,
            'max_object_num': 10
        }
        for name, value in basic_attrs.items():
            setattr(self, name, value)
            
        self.domain_size = self.domain_end - self.domain_start

        # 粒子参数
        self.radius = self.cfg.get_cfg("particleRadius")
        derived_params = {
            'diameter': 2 * self.radius,
            'V0': 0.8 * (2 * self.radius) ** self.dim,
            'dh': 4 * self.radius,
            'grid_size': 4 * self.radius,
            'padding': 4 * self.radius,
            'boundary_thickness': 0.0,
            'add_boundary': self.cfg.get_cfg("addDomainBox")
        }
        for name, value in derived_params.items():
            setattr(self, name, value)

        # 处理边界
        if self.add_boundary:
            self._process_boundary()
            
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)

        # 初始化集合
        collections = {
            'object_collection': dict(),
            'object_id_rigid_body': set(),
            'object_id_fluid_body': set(),
            'present_object': [],
            'fluid_bodies': self.cfg.get_fluid_bodies(),
            'fluid_blocks': self.cfg.get_fluid_blocks(),
            'rigid_bodies': self.cfg.get_rigid_bodies()
        }
        for name, value in collections.items():
            setattr(self, name, value)

        # 计算粒子数量
        particle_counts = {
            'fluid_particle_num': op.fluid_block_processor(self.dim, self.cfg, self.diameter),
            'rigid_particle_num': op.rigid_body_processor(self.cfg, self.diameter)
        }
        for name, value in particle_counts.items():
            setattr(self, name, value)

        boundary_particles = (op.compute_box_particle_num(self.dim, self.domain_start, self.domain_end, 
                            diameter=self.diameter, thickness=self.boundary_thickness) if self.add_boundary else 0)
        self.particle_max_num = self.fluid_particle_num + self.rigid_particle_num + boundary_particles

        # 分配内存
        num_grid = np.prod(self.grid_num)
        self._allocate_grid_storage(num_grid)
        self._allocate_particle_arrays()
        
        fluid_object_num = len(self.fluid_blocks) + len(self.fluid_bodies)
        rigid_object_num = len(self.rigid_bodies)
        self._allocate_object_arrays(fluid_object_num, rigid_object_num)

        if self.add_boundary:
            self._setup_boundary_object()

    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        """添加单个粒子"""
        # 设置基本属性
        particle_props = {
            'particle_object_ids': obj_id,
            'particle_positions': x,
            'particle_velocities': v,
            'particle_densities': density,
            'particle_pressures': pressure,
            'particle_materials': material,
            'particle_is_dynamic': is_dynamic,
            'particle_colors': color
        }
        for name, value in particle_props.items():
            getattr(self, name)[p] = value
            
        # 设置派生属性
        self._set_derived_properties(p, x, density)

    def _set_derived_properties(self, p, x, density):
        """设置粒子的派生属性"""
        derived_props = {
            'rigid_particle_original_positions': x,
            'particle_rest_volumes': self.V0,
            'particle_rest_densities': density,
            'particle_masses': self.V0 * density
        }
        for name, value in derived_props.items():
            getattr(self, name)[p] = value

    def add_particles(self, object_id, new_particles_num, new_particles_positions,
                     new_particles_velocity, new_particle_density, new_particle_pressure,
                     new_particles_material, new_particles_is_dynamic, new_particles_color):
        """批量添加粒子"""
        start_idx = self.particle_num
        self._add_particle_batch(
            object_id, new_particles_num, start_idx,
            new_particles_positions, new_particles_velocity,
            new_particle_density, new_particle_pressure,
            new_particles_material, new_particles_is_dynamic,
            new_particles_color
        )
        self.particle_num += new_particles_num

    def _add_particle_batch(self, obj_id, num_particles, start_idx,
                          positions, velocities, density, pressure,
                          material, is_dynamic, color):
        """处理批量添加粒子的核心逻辑"""
        for i in range(num_particles):
            p = start_idx + i
            self.add_particle(
                p, obj_id,
                positions[i],
                velocities[i],
                density[i],
                pressure[i],
                material[i],
                is_dynamic[i],
                color[i]
            )

    def add_cube(self, object_id, start, end, scale, material,
                is_dynamic=True, color=(0,0,0), density=None,
                velocity=None, pressure=None, diameter=None):
        """添加立方体形状的粒子集合"""
        # 初始化参数
        params = self._init_cube_params(diameter, density, velocity, pressure)
        
        # 生成粒子位置
        new_positions = self._generate_cube_positions(start, end, scale, params['diameter'])
        num_new_particles = new_positions.shape[0]
        
        # 准备粒子属性
        particle_data = self._prepare_cube_particles(
            num_new_particles, new_positions, 
            params['velocity'], params['density'], 
            params['pressure'], material, is_dynamic, color
        )
        
        # 添加粒子
        self.add_particles(
            object_id, num_new_particles, 
            particle_data['positions'], particle_data['velocities'],
            particle_data['densities'], particle_data['pressures'],
            particle_data['materials'], particle_data['is_dynamic'],
            particle_data['colors']
        )
        
        # 更新流体粒子计数
        if material == self.material_fluid:
            self.fluid_particle_num += num_new_particles

    def _init_cube_params(self, diameter, density, velocity, pressure):
        """初始化立方体参数"""
        return {
            'diameter': self.diameter if diameter is None else diameter,
            'density': 1000.0 if density is None else density,
            'velocity': np.zeros(self.dim) if velocity is None else velocity,
            'pressure': 0.0 if pressure is None else pressure
        }

    def _generate_cube_positions(self, start, end, scale, diameter):
        """生成立方体的粒子位置"""
        # 生成网格点
        num_dim = [np.arange(start[i] * scale[i], 
                           end[i] * scale[i], 
                           diameter) for i in range(self.dim)]
        # 创建粒子位置
        return np.array(np.meshgrid(*num_dim, indexing='ij')).reshape(self.dim, -1).T

    def _prepare_cube_particles(self, num_particles, positions, velocity, 
                              density, pressure, material, is_dynamic, color):
        """准备立方体粒子的属性"""
        return {
            'positions': positions,
            'velocities': np.tile(velocity, (num_particles, 1)),
            'densities': np.ones(num_particles) * density,
            'pressures': np.ones(num_particles) * pressure,
            'materials': np.ones(num_particles, dtype=np.int32) * material,
            'is_dynamic': np.ones(num_particles, dtype=np.int32) * is_dynamic,
            'colors': np.tile(color, (num_particles, 1))
        }

    def add_boundary_object(self, object_id, domain_start, domain_end, thickness,
                          material=0, is_dynamic=False, color=(0,0,0),
                          density=None, pressure=None, velocity=None, space=None):
        """添加边界对象"""
        params = self._init_boundary_params(space, density, velocity, pressure)
        positions = self._generate_boundary_positions(domain_start, domain_end, params['space'])
        mask = self._create_boundary_mask(positions, domain_start, domain_end, thickness)
        
        particle_data = self._prepare_boundary_particles(
            positions[mask], params, material, is_dynamic, color
        )
        
        self.add_particles(
            object_id, particle_data['num_particles'],
            particle_data['positions'], particle_data['velocities'],
            particle_data['densities'], particle_data['pressures'],
            particle_data['materials'], particle_data['is_dynamic'],
            particle_data['colors']
        )

    def _init_boundary_params(self, space, density, velocity, pressure):
        """初始化边界参数"""
        return {
            'space': self.diameter if space is None else space,
            'density': 1000.0 if density is None else density,
            'velocity': None if velocity is None else velocity,
            'pressure': 0.0 if pressure is None else pressure
        }

    def _generate_boundary_positions(self, domain_start, domain_end, space):
        """生成边界位置"""
        num_dim = [np.arange(domain_start[i], domain_end[i], space) 
                  for i in range(self.dim)]
        return np.array(np.meshgrid(*num_dim, indexing='ij')).reshape(self.dim, -1).T

    def _create_boundary_mask(self, positions, domain_start, domain_end, thickness):
        """创建边界掩码"""
        mask = np.zeros(positions.shape[0], dtype=bool)
        for i in range(self.dim):
            mask |= ((positions[:, i] <= domain_start[i] + thickness) | 
                    (positions[:, i] >= domain_end[i] - thickness))
        return mask

    def _prepare_boundary_particles(self, positions, params, material, is_dynamic, color):
        """准备边界粒子属性"""
        num_particles = positions.shape[0]
        return {
            'num_particles': num_particles,
            'positions': positions,
            'velocities': (np.zeros_like(positions) if params['velocity'] is None 
                         else np.tile(params['velocity'], (num_particles, 1))),
            'densities': np.ones(num_particles) * params['density'],
            'pressures': np.ones(num_particles) * params['pressure'],
            'materials': np.ones(num_particles, dtype=np.int32) * material,
            'is_dynamic': np.ones(num_particles, dtype=np.int32) * is_dynamic,
            'colors': np.tile(color, (num_particles, 1))
        }

    def insert_object(self):
        """插入对象"""
        self._insert_fluid_blocks()
        self._insert_rigid_bodies()

    def _insert_fluid_blocks(self):
        """插入流体块"""
        for fluid in self.fluid_blocks:
            if not self._should_insert_fluid(fluid):
                continue
            self._process_fluid_block(fluid)

    def _insert_rigid_bodies(self):
        """插入刚体"""
        for rigid in self.rigid_bodies:
            if not self._should_insert_rigid(rigid):
                continue
            self._process_rigid_body(rigid)

    def _should_insert_fluid(self, fluid):
        """检查是否应该插入流体"""
        obj_id = fluid["objectId"]
        return (obj_id not in self.present_object and 
                fluid["entryTime"] <= self.total_time)

    def _should_insert_rigid(self, rigid):
        """检查是否应该插入刚体"""
        obj_id = rigid["objectId"]
        return (obj_id not in self.present_object and 
                rigid["entryTime"] <= self.total_time)

    def _process_fluid_block(self, fluid):
        """处理流体块"""
        obj_id = fluid["objectId"]
        self._setup_fluid_object(fluid)
        
        offset = np.array(fluid["translation"])
        self.add_cube(
            object_id = obj_id,
            start = np.array(fluid["start"]) + offset,
            end = np.array(fluid["end"]) + offset,
            scale = np.array(fluid["scale"]),
            material = self.material_fluid,
            is_dynamic = True,
            color = fluid["color"],
            velocity = fluid["velocity"],
            density = fluid["density"],
            diameter = self.diameter
        )
        
        self.present_object.append(obj_id)

    def _setup_fluid_object(self, fluid):
        """设置流体对象属性"""
        obj_id = fluid["objectId"]
        self.object_id_fluid_body.add(obj_id)
        self.object_visibility[obj_id] = fluid.get("visible", 1)
        self.object_materials[obj_id] = self.material_fluid
        self.object_densities[obj_id] = fluid["density"]
        self.object_collection[obj_id] = fluid

    def _process_rigid_body(self, rigid):
        """处理刚体"""
        obj_id = rigid["objectId"]
        self._setup_rigid_object(rigid)
        
        self.add_particles(
            obj_id,
            rigid["particleNum"],
            np.array(rigid["voxelizedPoints"], dtype=np.float32),
            np.tile(rigid["velocity"], (rigid["particleNum"], 1)),
            rigid["density"] * np.ones(rigid["particleNum"], dtype=np.float32),
            np.zeros(rigid["particleNum"], dtype=np.float32),
            self.material_rigid * np.ones(rigid["particleNum"], dtype=np.int32),
            rigid["isDynamic"] * np.ones(rigid["particleNum"], dtype=np.int32),
            np.tile(rigid["color"], (rigid["particleNum"], 1))
        )
        
        self.present_object.append(obj_id)

    def _setup_rigid_object(self, rigid):
        """设置刚体对象属性"""
        obj_id = rigid["objectId"]
        self.object_id_rigid_body.add(obj_id)
        self.object_visibility[obj_id] = rigid.get("visible", 1)
        self.object_materials[obj_id] = self.material_rigid
        self.object_densities[obj_id] = rigid["density"]
        self.object_collection[obj_id] = rigid

    def prepare_neighbor_search(self):
        """准备邻居搜索"""
        # 重置计数器
        self.grid_num_particles.fill(0)
        
        # 遍历所有粒子
        for p_i in range(self.particle_num):
            pos = self.particle_positions[p_i]
            grid_index = self.get_flatten_grid_index(pos)
            
            # 确保索引在有效范围内
            if 0 <= grid_index < len(self.grid_num_particles):
                self.grid_ids[p_i] = grid_index
                self.grid_num_particles[grid_index] += 1
            else:
                # 处理越界情况
                print(f"Warning: Grid index {grid_index} out of bounds")
                # 将粒子分配到边界网格
                self.grid_ids[p_i] = 0

    def get_flatten_grid_index(self, pos):
        """获取展平的网格索引"""
        # 获取网格索引
        grid_index = self.pos_to_index(pos)
        
        # 确保索引在有效范围内
        for i in range(self.dim):
            if grid_index[i] < 0:
                grid_index[i] = 0
            elif grid_index[i] >= self.grid_num[i]:
                grid_index[i] = self.grid_num[i] - 1
                
        return self.flatten_grid_index(grid_index)

    def pos_to_index(self, pos):
        """将位置转换为网格索引"""
        # 计算相对位置
        rel_pos = pos - self.domain_start
        # 转换为网格索引
        return (rel_pos / self.grid_size).astype(np.int32)

    def flatten_grid_index(self, grid_index):
        """将多维网格索引展平为一维索引"""
        ret = 0
        for i in range(self.dim):
            ret_p = grid_index[i]
            for j in range(i+1, self.dim):
                ret_p *= self.grid_num[j]
            ret += ret_p
        return ret

    def for_all_neighbors(self, i, task, ret=None):
        """遍历所有邻居"""
        pos_i = self.particle_positions[i]
        for j in range(self.particle_num):
            if i != j:
                pos_j = self.particle_positions[j]
                dist = np.linalg.norm(pos_i - pos_j)
                if dist < self.dh:
                    task(j, ret)

    def dump(self, obj_id):
        """导出指定对象的粒子数据"""
        mask = self.particle_object_ids[:self.particle_num] == obj_id
        return {
            'position': self.particle_positions[:self.particle_num][mask].copy(),
            'velocity': self.particle_velocities[:self.particle_num][mask].copy()
        }
    
    def _process_boundary(self):
        """处理边界条件"""
        if self.add_boundary:
            # 更新域的边界，考虑padding
            self.domain_start = np.array([coord + self.padding for coord in self.domain_start])
            self.domain_end = np.array([coord - self.padding for coord in self.domain_end])
            
            # 更新域的大小
            self.domain_size = self.domain_end - self.domain_start
            
            # 设置边界厚度
            self.boundary_thickness = 0.03
            
            # 更新网格数量
            self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
    def _allocate_grid_storage(self, total_cells):
        """分配网格存储空间"""
        self.particle_num = 0
        self.grid_num_particles = np.zeros(total_cells, dtype=np.int32)
        self.grid_num_particles_temp = np.zeros(total_cells, dtype=np.int32)
        self.grid_ids = np.zeros(self.particle_max_num, dtype=np.int32)

    def _allocate_particle_arrays(self):
        """分配粒子属性数组"""
        max_p = self.particle_max_num
        dim = self.dim
        
        # 向量字段
        vector_fields = {
            'particle_positions': (max_p, dim),
            'particle_velocities': (max_p, dim),
            'particle_accelerations': (max_p, dim),
            'particle_colors': (max_p, 3),
            'rigid_particle_original_positions': (max_p, dim)
        }
        for name, shape in vector_fields.items():
            setattr(self, name, np.zeros(shape, dtype=np.float32))
        
        # 标量字段
        scalar_fields = {
            'particle_object_ids': np.int32,
            'particle_materials': np.int32,
            'particle_is_dynamic': np.int32,
            'particle_rest_volumes': np.float32,
            'particle_rest_densities': np.float32,
            'particle_masses': np.float32,
            'particle_densities': np.float32,
            'particle_pressures': np.float32
        }
        for name, dtype in scalar_fields.items():
            setattr(self, name, np.zeros(max_p, dtype=dtype))

    def _allocate_object_arrays(self, fluid_count, rigid_count):
        """分配对象数组"""
        max_obj = self.max_object_num
        dim = self.dim
        
        # 标量字段
        scalar_fields = {
            'object_materials': np.int32,
            'object_densities': np.float32,
            'object_visibility': np.int32,
            'rigid_body_is_dynamic': np.int32,
            'rigid_body_masses': np.float32,
            'rigid_body_particle_num': np.int32
        }
        for name, dtype in scalar_fields.items():
            setattr(self, name, np.zeros(max_obj, dtype=dtype))
        
        # 向量字段
        vector_fields = {
            'rigid_body_original_com': (max_obj, dim),
            'rigid_body_com': (max_obj, dim),
            'rigid_body_torques': (max_obj, dim),
            'rigid_body_forces': (max_obj, dim),
            'rigid_body_velocities': (max_obj, dim),
            'rigid_body_angular_velocities': (max_obj, dim)
        }
        for name, shape in vector_fields.items():
            setattr(self, name, np.zeros(shape, dtype=np.float32))
        
        # 矩阵字段
        self.rigid_body_rotations = np.zeros((max_obj, dim, dim), dtype=np.float32)
        
        # 设置计数
        self.object_num = fluid_count + rigid_count + (1 if self.add_boundary else 0)
        self.fluid_object_num = fluid_count
        self.rigid_object_num = rigid_count

    def _setup_boundary_object(self):
        """设置边界对象"""
        obj_id = self.object_num - 1
        
        self.add_boundary_object(
            object_id=obj_id,
            domain_start=self.domain_start,
            domain_end=self.domain_end,
            thickness=self.boundary_thickness,
            material=self.material_rigid,
            is_dynamic=False,
            color=(128, 128, 128),
            density=1000.0
        )
        
        # 设置边界对象属性
        boundary_props = {
            'object_visibility': 0,
            'object_materials': self.material_rigid,
            'object_densities': 1000.0,
            'rigid_body_is_dynamic': 0
        }
        for name, value in boundary_props.items():
            getattr(self, name)[obj_id] = value
            
        self.rigid_body_velocities[obj_id] = np.zeros(self.dim)
        self.object_collection[obj_id] = 0