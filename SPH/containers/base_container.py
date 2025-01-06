import taichi as ti 
import numpy as np
from . import ObjectProcessor as op
from functools import reduce
from ..utils import SimConfig

cnt = 0

@ti.data_oriented
class BaseContainer:
    def __init__(self, config: SimConfig, GGUI = False):
        # 基础配置
        basic_attrs = {
            'dim': 3,
            'GGUI': GGUI,
            'cfg': config,
            'total_time': 0,
            'gravity': ti.Vector([0.0, -9.81, 0.0]),
            'domain_start': np.array(config.get_cfg("domainStart")),
            'domain_end': np.array(config.get_cfg("domainEnd")),
            'material_rigid': 2,
            'material_fluid': 1,
            'max_object_num': 10
        }
        for name, value in basic_attrs.items():
            setattr(self, name, value)

        self.domain_size = self.domain_end - self.domain_start
        self.cnt = ti.field(dtype=int, shape=())

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

        if self.add_boundary:
            self._process_boundary()
        
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)

        # 初始化集合和数组
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
        num_grid = reduce(lambda x, y: x * y, self.grid_num)
        self._allocate_grid_storage(num_grid)
        self._allocate_particle_arrays()
        
        fluid_object_num = len(self.fluid_blocks) + len(self.fluid_bodies)
        rigid_object_num = len(self.rigid_bodies)
        self._allocate_object_arrays(fluid_object_num, rigid_object_num)

        if self.add_boundary:
            self._setup_boundary_object()

    def _process_boundary(self):
        """处理边界条件"""
        updates = {
            'domain_start': np.array([coord + self.padding for coord in self.domain_start]),
            'domain_end': np.array([coord - self.padding for coord in self.domain_end]),
            'boundary_thickness': 0.03
        }
        for name, value in updates.items():
            setattr(self, name, value)

    def _allocate_grid_storage(self, total_cells):
        """分配网格存储空间"""
        grid_fields = {
            'particle_num': (),
            'grid_num_particles': total_cells,
            'grid_num_particles_temp': total_cells
        }
        for name, shape in grid_fields.items():
            setattr(self, name, ti.field(dtype=int, shape=shape))
            
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_num_particles.shape[0])

    def _allocate_particle_arrays(self):
        """分配粒子属性数组"""
        max_p = self.particle_max_num
        dim = self.dim
        
        # 向量字段
        vector_fields = {
            'particle_positions': (dim, max_p),
            'particle_velocities': (dim, max_p),
            'particle_accelerations': (dim, max_p),
            'particle_colors': (3, max_p),
            'rigid_particle_original_positions': (dim, max_p),
            'x_vis_buffer': (dim, max_p) if self.GGUI else None,
            'color_vis_buffer': (3, max_p) if self.GGUI else None,
            # 添加缓冲区向量字段
            'particle_positions_buffer': (dim, max_p),
            'rigid_particle_original_positions_buffer': (dim, max_p),
            'particle_velocities_buffer': (dim, max_p),
            'particle_colors_buffer': (3, max_p)
        }
        for name, params in vector_fields.items():
            if params is not None:
                vec_dim, size = params
                dtype = int if 'colors' in name else float
                setattr(self, name, ti.Vector.field(n=vec_dim, dtype=dtype, shape=size))

        # 标量字段
        scalar_fields = {
            'particle_object_ids': (int, max_p),
            'particle_materials': (int, max_p),
            'particle_is_dynamic': (int, max_p),
            'particle_rest_volumes': (float, max_p),
            'particle_rest_densities': (float, max_p),
            'particle_masses': (float, max_p),
            'particle_densities': (float, max_p),
            'particle_pressures': (float, max_p),
            'grid_ids': (int, max_p),
            'grid_ids_buffer': (int, max_p),
            'grid_ids_new': (int, max_p),
            'particle_object_ids_buffer': (int, max_p),
            'particle_rest_volumes_buffer': (float, max_p),
            'particle_rest_densities_buffer': (float, max_p),
            'particle_masses_buffer': (float, max_p),
            'particle_densities_buffer': (float, max_p),
            'particle_materials_buffer': (int, max_p),
            'is_dynamic_buffer': (int, max_p),
            'fluid_particle_num': (int, ()),
            'rigid_particle_num': (int, ())
        }
        for name, (dtype, shape) in scalar_fields.items():
            setattr(self, name, ti.field(dtype=dtype, shape=shape))
            

    def _allocate_object_arrays(self, fluid_count, rigid_count):
        """分配对象数组"""
        max_obj = self.max_object_num
        dim = self.dim
        
        # 标量字段
        scalar_fields = {
            'object_materials': (int, max_obj),
            'object_densities': (float, max_obj),
            'object_visibility': (int, max_obj),
            'rigid_body_is_dynamic': (int, max_obj),
            'rigid_body_masses': (float, max_obj),
            'rigid_body_particle_num': (int, max_obj),
            'object_num': (int, ()),
            'fluid_object_num': (int, ()),
            'rigid_object_num': (int, ())
        }
        for name, (dtype, shape) in scalar_fields.items():
            setattr(self, name, ti.field(dtype=dtype, shape=shape))

        # 向量字段
        vector_fields = {
            'rigid_body_original_com': (dim, max_obj),
            'rigid_body_com': (dim, max_obj),
            'rigid_body_torques': (dim, max_obj),
            'rigid_body_forces': (dim, max_obj),
            'rigid_body_velocities': (dim, max_obj),
            'rigid_body_angular_velocities': (dim, max_obj)
        }
        for name, (vec_dim, size) in vector_fields.items():
            setattr(self, name, ti.Vector.field(vec_dim, dtype=float, shape=size))

        # 矩阵字段
        self.rigid_body_rotations = ti.Matrix.field(dim, dim, dtype=float, shape=max_obj)

        # 设置计数
        total = fluid_count + rigid_count + (1 if self.add_boundary else 0)
        counts = {
            'object_num': total,
            'fluid_object_num': fluid_count,
            'rigid_object_num': rigid_count
        }
        for name, value in counts.items():
            field = getattr(self, name)
            field[None] = value

    def _setup_boundary_object(self):
        """设置边界对象"""
        obj_id = self.object_num[None] - 1
        
        # 添加边界粒子
        self.add_boundary_object(
            object_id=obj_id,
            domain_start=self.domain_start,
            domain_end=self.domain_end,
            thickness=self.boundary_thickness,
            material=self.material_rigid,
            is_dynamic=False,
            color=(128, 128, 128),  # 灰色边界
            density=1000.0
        )
        
        self.object_collection[obj_id] = 0
    
    ######## add all kinds of particles ########    
    def insert_object(self):
        cnt = 0
        # 处理流体块
        for fluid in self.fluid_blocks:
            obj_id = fluid["objectId"]
            
            if obj_id in self.present_object:
                continue
            if fluid["entryTime"] > self.total_time:
                continue
    
            offset = np.array(fluid["translation"])
            cube_params = {
                'start': np.array(fluid["start"]) + offset,
                'end': np.array(fluid["end"]) + offset,
                'scale': np.array(fluid["scale"]),
                'velocity': fluid["velocity"],
                'density': fluid["density"],
                'color': fluid["color"]
            }
            
            self.object_id_fluid_body.add(obj_id)
            self.object_visibility[obj_id] = fluid.get("visible", 1)
            self.object_materials[obj_id] = self.material_fluid
            self.object_densities[obj_id] = cube_params['density']
            self.object_collection[obj_id] = fluid
            
            self.add_cube(
                object_id = obj_id,
                start = cube_params['start'],
                end = cube_params['end'],
                scale = cube_params['scale'],
                material = self.material_fluid,
                is_dynamic = True,
                color = cube_params['color'],
                velocity = cube_params['velocity'],
                density = cube_params['density'],
            )
            
            self.present_object.append(obj_id)
            cnt += 1
        
        # 处理刚体
        for rigid in self.rigid_bodies:
            obj_id = rigid["objectId"]
            
            if obj_id in self.present_object or rigid["entryTime"] > self.total_time:
                continue
            
            self.object_id_rigid_body.add(obj_id)
            particle_num = rigid["particleNum"]
            
            # 基础参数
            rigid_params = {
                'points': np.array(rigid["voxelizedPoints"], dtype=np.float32),
                'velocity': np.array(rigid["velocity"], dtype=np.float32),
                'density': rigid["density"],
                'color': np.array(rigid["color"], dtype=np.int32),
                'is_dynamic': rigid["isDynamic"]
            }
            
            # 设置对象属性
            self.object_visibility[obj_id] = rigid.get("visible", 1)
            self.object_materials[obj_id] = self.material_rigid
            self.object_densities[obj_id] = rigid_params['density']
            self.object_collection[obj_id] = rigid
            
            # 添加粒子
            self.add_particles(
                obj_id,
                particle_num,
                rigid_params['points'],
                np.stack([rigid_params['velocity'] for _ in range(particle_num)]),
                rigid_params['density'] * np.ones(particle_num, dtype=np.float32),
                np.zeros(particle_num, dtype=np.float32),
                np.array([self.material_rigid for _ in range(particle_num)], dtype=np.int32),
                rigid_params['is_dynamic'] * np.ones(particle_num, dtype=np.int32),
                np.stack([rigid_params['color'] for _ in range(particle_num)])
            )
            
            self.rigid_body_is_dynamic[obj_id] = rigid_params['is_dynamic']
            self.rigid_body_velocities[obj_id] = rigid_params['velocity']
            self.rigid_particle_num[None] += particle_num
            self.rigid_body_particle_num[obj_id] = particle_num
        
            if rigid_params['is_dynamic']:
                self.rigid_body_masses[obj_id] = self.compute_rigid_mass(obj_id)
                self.rigid_body_com[obj_id] = self.compute_rigid_com(obj_id)
                self.rigid_body_is_dynamic[obj_id] = 1
        
            self.present_object.append(obj_id)
        
    @ti.kernel
    def compute_rigid_mass(self, objId: int) -> ti.f32:
        mass = 0.0
        for i in range(self.particle_num[None]):
            if self._is_target_dynamic_particle(i, objId):
                mass += self.particle_rest_densities[i] * self.V0
        return mass
    
    @ti.kernel
    def compute_rigid_com(self, objId: int) -> ti.types.vector(3, float):
        com = ti.Vector([0.0 for _ in range(self.dim)])
        for i in range(self.particle_num[None]):
            if self._is_target_dynamic_particle(i, objId):
                com += self.particle_positions[i] * self.particle_rest_densities[i] * self.V0
        return com / self.rigid_body_masses[objId]
            
    @ti.func
    def _is_target_dynamic_particle(self, p_idx: int, obj_id: int) -> ti.i32:
        """检查粒子是否属于指定的动态物体"""
        return self.particle_object_ids[p_idx] == obj_id and self.particle_is_dynamic[p_idx]
    
    @ti.kernel
    def add_particles(self, oid: int, pnum: int, pos: ti.types.ndarray(), 
                    vel: ti.types.ndarray(), dens: ti.types.ndarray(),
                    press: ti.types.ndarray(), mat: ti.types.ndarray(), 
                    dyn: ti.types.ndarray(), col: ti.types.ndarray()):
        """添加多个粒子到系统中"""
        for p in range(self.particle_num[None], self.particle_num[None] + pnum):
            idx = p - self.particle_num[None]
            
            # 构建向量
            pos_vec = ti.Vector([pos[idx, d] for d in range(self.dim)])
            vel_vec = ti.Vector([vel[idx, d] for d in range(self.dim)])
            col_vec = ti.Vector([col[idx, i] for i in range(3)])
            
            # 直接设置属性
            self.particle_object_ids[p] = oid
            self.particle_positions[p] = pos_vec
            self.particle_velocities[p] = vel_vec
            self.particle_densities[p] = dens[idx]
            self.particle_rest_volumes[p] = self.V0
            self.particle_rest_densities[p] = dens[idx]
            self.particle_masses[p] = self.V0 * dens[idx]
            self.particle_pressures[p] = press[idx]
            self.particle_materials[p] = mat[idx]
            self.particle_is_dynamic[p] = dyn[idx]
            self.particle_colors[p] = col_vec
            
            # 更新刚体原始位置
            self.rigid_particle_original_positions[p] = pos_vec
            
        self.particle_num[None] += pnum
        
    def add_cube( self, object_id, start, end, scale, material, is_dynamic=True, color=(0,0,0), density=None, velocity=None, pressure=None,):      
        num_dim = [np.arange(start[i] * scale[i], end[i] * scale[i], self.diameter) for i in range(self.dim)]
        new_positions = self._create_mesh_grid(num_dim)
        
        # 准备粒子属性
        num_particles = new_positions.shape[0]
        
        # 创建粒子属性数组
        velocities = np.zeros_like(new_positions) if velocity is None else np.tile(velocity, (num_particles, 1))
        densities = np.ones(num_particles) * (density if density is not None else 1000.)
        pressures = np.zeros(num_particles) if pressure is None else np.ones(num_particles) * pressure
        materials = np.ones(num_particles, dtype=np.int32) * material
        is_dynamics = np.ones(num_particles, dtype=np.int32) * is_dynamic
        colors = np.tile(color, (num_particles, 1))

        # 添加粒子
        self.add_particles(object_id,num_particles,new_positions,velocities,densities,pressures, materials,is_dynamics,colors)
        
        if material == self.material_fluid:
            self.fluid_particle_num[None] += num_particles
    
    def add_boundary_object( self, object_id,domain_start,domain_end, thickness, material=0, is_dynamic=False,color=(128, 128, 128), density=None, pressure=None, velocity=None):
        num_dim = [np.arange(domain_start[i], domain_end[i], self.diameter) for i in range(self.dim)]
        positions = self._create_mesh_grid(num_dim)
        mask = self._create_boundary_mask(positions, domain_start, domain_end, thickness)
        positions = positions[mask]
        new_velocities = np.zeros_like(positions, dtype=np.float32) if velocity is None else np.tile(velocity, (positions.shape[0], 1))
        new_densities = np.ones(positions.shape[0]) * (density if density is not None else 1000.)
        new_pressures = np.zeros(positions.shape[0]) if pressure is None else np.ones(positions.shape[0]) * pressure
        new_materials = np.ones(positions.shape[0], dtype=np.int32) * material
        new_is_dynamic = np.ones(positions.shape[0], dtype=np.int32) * is_dynamic
        new_colors = np.tile(color, (positions.shape[0], 1))

        self.add_particles(object_id, positions.shape[0], positions, new_velocities, new_densities, new_pressures, new_materials, new_is_dynamic, new_colors)
        
    @ti.func
    def pos_to_index(self, pos):
        average = pos / self.grid_size
        return ti.cast(average, ti.i32)

    @ti.func
    def flatten_grid_index(self, grid_index):
        ret = 0
        for i in ti.static(range(self.dim)):
            for j in ti.static(range(i+1, self.dim)):
                grid_index[i] *= self.grid_num[j]
            ret += grid_index[i]
        return ret
    
    ###### initial grid ######
    @ti.kernel
    def init_grid(self):
        """初始化网格结构"""
        self.grid_num_particles.fill(0)
        self._count_particles_per_grid()
        self._copy_grid_counts()
    
    @ti.func
    def _count_particles_per_grid(self):
        """统计每个网格中的粒子数量"""
        for i in range(self.particle_num[None]):
            grid_id = self.flatten_grid_index(self.pos_to_index(self.particle_positions[i]))
            self.grid_ids[i] = grid_id
            ti.atomic_add(self.grid_num_particles[grid_id], 1)
            
    @ti.func
    def _copy_grid_counts(self):
        """复制网格粒子计数到临时数组"""
        for i in ti.grouped(self.grid_num_particles):
            self._update_temp_count(i)
            
    @ti.func
    def _update_temp_count(self, grid_idx: ti.template()):
        """更新单个网格的临时计数"""
        self.grid_num_particles_temp[grid_idx] = self.grid_num_particles[grid_idx]
    
    def compute_particle_in_grid(self):
        for i in range(self.particle_num[None]):
            grid_id = self.flatten_grid_index(self.pos_to_index(self.particle_positions[i]))
            self.grid_ids[i] = grid_id
            self.grid_num_particles[grid_id] += 1
    
    def update_buffer_count(self):
        for i in ti.grouped(self.grid_num_particles):
            self.grid_num_particles_temp[i] = self.grid_num_particles[i]
    
    @ti.kernel
    def particles_sort(self):
        """对粒子进行排序"""
        self._compute_new_indices()
        self._copy_to_buffers()
        self._copy_from_buffers()

    @ti.func
    def _compute_new_indices(self):
        """计算粒子的新索引位置"""
        for i in range(self.particle_num[None]):
            j = self.particle_num[None] - 1 - i
            base_offset = self._get_base_offset(j)
            self.grid_ids_new[j] = self._compute_particle_index(j, base_offset)
            
    @ti.func
    def _get_base_offset(self, idx: int) -> int:
        """获取基础偏移量"""
        offset = 0
        if self.grid_ids[idx] >= 1:
            offset = self.grid_num_particles[self.grid_ids[idx] - 1]
        return offset
    
    @ti.func
    def _compute_particle_index(self, idx: int, base_offset: int) -> int:
        """计算单个粒子的新索引"""
        return (ti.atomic_sub(self.grid_num_particles_temp[self.grid_ids[idx]], 1)  - 1 + base_offset)

    @ti.func
    def _copy_to_buffers(self):
        """将粒子数据复制到缓冲区"""
        for i in range(self.particle_num[None]):
            self._copy_particle_data(i, self.grid_ids_new[i], True)

    @ti.func
    def _copy_from_buffers(self):
        """从缓冲区复制回粒子数据"""
        for i in range(self.particle_num[None]):
            self._copy_particle_data(i, i, False)
            
    @ti.func
    def _copy_particle_data(self, src_idx: int, dst_idx: int, to_buffer: ti.i32):
        if to_buffer:
            self._copy_to_buffer(src_idx, dst_idx)
        else:
            self._copy_from_buffer(src_idx, dst_idx)
            
    @ti.func
    def _copy_to_buffer(self, src_idx: int, dst_idx: int):
        """复制数据到缓冲区"""
        self._copy_single_field(self.grid_ids_buffer, self.grid_ids, dst_idx, src_idx)
        self._copy_single_field(self.particle_object_ids_buffer, self.particle_object_ids, dst_idx, src_idx)
        self._copy_single_field(self.rigid_particle_original_positions_buffer, self.rigid_particle_original_positions, dst_idx, src_idx)
        self._copy_single_field(self.particle_positions_buffer, self.particle_positions, dst_idx, src_idx)
        self._copy_single_field(self.particle_velocities_buffer, self.particle_velocities, dst_idx, src_idx)
        self._copy_single_field(self.particle_rest_volumes_buffer, self.particle_rest_volumes, dst_idx, src_idx)
        self._copy_single_field(self.particle_rest_densities_buffer, self.particle_rest_densities, dst_idx, src_idx)
        self._copy_single_field(self.particle_masses_buffer, self.particle_masses, dst_idx, src_idx)
        self._copy_single_field(self.particle_densities_buffer, self.particle_densities, dst_idx, src_idx)
        self._copy_single_field(self.particle_materials_buffer, self.particle_materials, dst_idx, src_idx)
        self._copy_single_field(self.particle_colors_buffer, self.particle_colors, dst_idx, src_idx)
        self._copy_single_field(self.is_dynamic_buffer, self.particle_is_dynamic, dst_idx, src_idx)
        
    @ti.func
    def _copy_from_buffer(self, dst_idx: int, src_idx: int):
        """从缓冲区复制数据"""
        self._copy_single_field(self.grid_ids, self.grid_ids_buffer, dst_idx, src_idx)
        self._copy_single_field(self.particle_object_ids, self.particle_object_ids_buffer, dst_idx, src_idx)
        self._copy_single_field(self.rigid_particle_original_positions, self.rigid_particle_original_positions_buffer, dst_idx, src_idx)
        self._copy_single_field(self.particle_positions, self.particle_positions_buffer, dst_idx, src_idx)
        self._copy_single_field(self.particle_velocities, self.particle_velocities_buffer, dst_idx, src_idx)
        self._copy_single_field(self.particle_rest_volumes, self.particle_rest_volumes_buffer, dst_idx, src_idx)
        self._copy_single_field(self.particle_rest_densities, self.particle_rest_densities_buffer, dst_idx, src_idx)
        self._copy_single_field(self.particle_masses, self.particle_masses_buffer, dst_idx, src_idx)
        self._copy_single_field(self.particle_densities, self.particle_densities_buffer, dst_idx, src_idx)
        self._copy_single_field(self.particle_materials, self.particle_materials_buffer, dst_idx, src_idx)
        self._copy_single_field(self.particle_colors, self.particle_colors_buffer, dst_idx, src_idx)
        self._copy_single_field(self.particle_is_dynamic, self.is_dynamic_buffer, dst_idx, src_idx)

    @ti.func
    def _copy_single_field(self, dst_field: ti.template(), src_field: ti.template(), 
                          dst_idx: int, src_idx: int):
        """复制单个字段的数据"""
        dst_field[dst_idx] = src_field[src_idx]
    
    def prepare_neighbor_search(self):
        global cnt
        if cnt == 0:
            print("初始化网格")
        self.init_grid()
        if cnt == 0:
            print("排序粒子")
        self.prefix_sum_executor.run(self.grid_num_particles)
        if cnt == 0:
            print("粒子排序完成")
        self.particles_sort()
        cnt += 1
    
    @ti.func
    def for_all_neighbors(self, i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.particle_positions[i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            start_idx = 0
            grid_index = self.flatten_grid_index(center_cell + offset)
            end_idx = self.grid_num_particles[grid_index]
            if grid_index >= 1:
                start_idx = self.grid_num_particles[grid_index - 1]
            for j in range(start_idx, end_idx):
                if i != j and (self.particle_positions[i] - self.particle_positions[j]).norm() < self.dh:
                    task(i, j, ret)

    @ti.kernel
    def flush_vis_buffer(self):
        self.color_vis_buffer.fill(0.0)
        self.x_vis_buffer.fill(0.0)
    
    def copy_to_vis_buffer(self):
        self.flush_vis_buffer()
        for obj_id in self.object_collection:
            if self.object_visibility[obj_id] == 1:
                self._copy_to_vis_buffer(obj_id)

    @ti.kernel
    def _copy_to_vis_buffer(self, obj_id: int):
        for i in range(self.particle_max_num):
            if self.particle_object_ids[i] == obj_id:
                self.color_vis_buffer[i] = self.particle_colors[i] / 255.0
                self.x_vis_buffer[i] = self.particle_positions[i]
    
    def dump(self, obj_id):
        mask = np.where(self.particle_object_ids.to_numpy() == obj_id)

        return {'position': self.particle_positions.to_numpy()[mask],'velocity': self.particle_velocities.to_numpy()[mask]}
    
    def _create_mesh_grid(self, axes):
        mesh = np.array(np.meshgrid(*axes, sparse=False, indexing='ij'), 
                       dtype=np.float32)
        total_points = reduce(lambda x, y: x * y, list(mesh.shape[1:]))
        return mesh.reshape(-1, total_points).transpose()

    def _create_boundary_mask(self, positions, domain_start, domain_end, thickness):
        mask = np.zeros(positions.shape[0], dtype=bool)
        for i in range(self.dim):
            lower_bound = positions[:, i] <= domain_start[i] + thickness
            upper_bound = positions[:, i] >= domain_end[i] - thickness
            mask = mask | lower_bound | upper_bound
        return mask
    