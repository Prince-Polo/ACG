import numpy as np
from . import ObjectProcessor as op
from ..utils import SimConfig

class BaseContainerBaseline:
    def __init__(self, config: SimConfig, GGUI = False):
        self.dim = 3
        self.GGUI = GGUI
        self.cfg = config
        self.total_time = 0

        self.gravity = np.array([0.0, -9.81, 0.0])
        self.domain_start = np.array(self.cfg.get_cfg("domainStart"))
        self.domain_end = np.array(self.cfg.get_cfg("domainEnd"))
        
        self.domain_size = self.domain_end - self.domain_start

        self.material_rigid = 2
        self.material_fluid = 1

        self.radius = self.cfg.get_cfg("particleRadius")
        self.diameter = 2 * self.radius
        self.V0 = 0.8 * self.diameter ** self.dim
        self.dh = 4 * self.radius
        
        self.max_object_num = 10

        self.grid_size = self.dh
        self.padding = self.grid_size
        self.boundary_thickness = 0.0
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

        self.fluid_particle_num = op.fluid_block_processor(self.dim, self.cfg, self.diameter)
        self.rigid_particle_num = op.rigid_body_processor(self.cfg, self.diameter)      
        self.particle_max_num = (self.fluid_particle_num + self.rigid_particle_num 
                               + (op.compute_box_particle_num(self.dim, self.domain_start, self.domain_end, 
                                                           diameter=self.diameter, thickness=self.boundary_thickness) 
                                  if self.add_boundary else 0))
                       
        fluid_object_num = len(self.fluid_blocks) + len(self.fluid_bodies)
        rigid_object_num = len(self.rigid_bodies)
        
        #========== Initialize arrays ==========#
        num_grid = np.prod(self.grid_num)
        self.particle_num = 0
        self.grid_num_particles = np.zeros(num_grid, dtype=np.int32)
        self.grid_num_particles_temp = np.zeros(num_grid, dtype=np.int32)

        # Particle related properties
        self.particle_object_ids = np.zeros(self.particle_max_num, dtype=np.int32)
        self.particle_positions = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.particle_velocities = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.particle_accelerations = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.particle_rest_volumes = np.zeros(self.particle_max_num, dtype=np.float32)
        self.particle_rest_densities = np.zeros(self.particle_max_num, dtype=np.float32)
        self.particle_masses = np.zeros(self.particle_max_num, dtype=np.float32)
        self.particle_densities = np.zeros(self.particle_max_num, dtype=np.float32)
        self.particle_pressures = np.zeros(self.particle_max_num, dtype=np.float32)
        self.particle_materials = np.zeros(self.particle_max_num, dtype=np.int32)
        self.particle_colors = np.zeros((self.particle_max_num, 3), dtype=np.int32)
        self.particle_is_dynamic = np.zeros(self.particle_max_num, dtype=np.int32)
        self.rigid_particle_original_positions = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)

        # Object properties
        self.object_materials = np.zeros(self.max_object_num, dtype=np.int32)
        self.object_densities = np.zeros(self.max_object_num, dtype=np.float32)
        self.object_num = fluid_object_num + rigid_object_num + (1 if self.add_boundary else 0)
        self.fluid_object_num = fluid_object_num
        self.rigid_object_num = rigid_object_num

        # Rigid body properties
        self.rigid_body_is_dynamic = np.zeros(self.max_object_num, dtype=np.int32)
        self.rigid_body_original_com = np.zeros((self.max_object_num, self.dim), dtype=np.float32)
        self.rigid_body_masses = np.zeros(self.max_object_num, dtype=np.float32)
        self.rigid_body_com = np.zeros((self.max_object_num, self.dim), dtype=np.float32)
        self.rigid_body_rotations = np.zeros((self.max_object_num, self.dim, self.dim), dtype=np.float32)
        self.rigid_body_torques = np.zeros((self.max_object_num, self.dim), dtype=np.float32)
        self.rigid_body_forces = np.zeros((self.max_object_num, self.dim), dtype=np.float32)
        self.rigid_body_velocities = np.zeros((self.max_object_num, self.dim), dtype=np.float32)
        self.rigid_body_angular_velocities = np.zeros((self.max_object_num, self.dim), dtype=np.float32)
        self.rigid_body_particle_num = np.zeros(self.max_object_num, dtype=np.int32)

        # Visibility
        self.object_visibility = np.zeros(self.max_object_num, dtype=np.int32)

        # Grid ids
        self.grid_ids = np.zeros(self.particle_max_num, dtype=np.int32)

        if self.add_boundary:
            self.add_boundary_object(
                object_id = self.object_num - 1,
                domain_start = self.domain_start,
                domain_end = self.domain_end,
                thickness = self.boundary_thickness,
                material = self.material_rigid,
                is_dynamic = False,
                space = self.diameter,
                color = (127,127,127)
            )
            
            self.object_visibility[self.object_num-1] = 0
            self.object_materials[self.object_num-1] = self.material_rigid
            self.object_densities[self.object_num-1] = 1000.0
            self.rigid_body_is_dynamic[self.object_num-1] = 0
            self.rigid_body_velocities[self.object_num-1] = np.zeros(self.dim)
            self.object_collection[self.object_num-1] = 0

    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        self.particle_object_ids[p] = obj_id
        self.particle_positions[p] = x
        self.rigid_particle_original_positions[p] = x
        self.particle_velocities[p] = v
        self.particle_densities[p] = density
        self.particle_rest_volumes[p] = self.V0
        self.particle_rest_densities[p] = density
        self.particle_masses[p] = self.V0 * density
        self.particle_pressures[p] = pressure
        self.particle_materials[p] = material
        self.particle_is_dynamic[p] = is_dynamic
        self.particle_colors[p] = color

    def add_particles(self, object_id, new_particles_num, new_particles_positions,
                     new_particles_velocity, new_particle_density, new_particle_pressure,
                     new_particles_material, new_particles_is_dynamic, new_particles_color):
        start_idx = self.particle_num
        end_idx = start_idx + new_particles_num
        
        for i in range(new_particles_num):
            p = start_idx + i
            self.add_particle(p, object_id,
                            new_particles_positions[i],
                            new_particles_velocity[i],
                            new_particle_density[i],
                            new_particle_pressure[i],
                            new_particles_material[i],
                            new_particles_is_dynamic[i],
                            new_particles_color[i])
        
        self.particle_num += new_particles_num

    def add_cube(self, object_id, start, end, scale, material,
                is_dynamic=True, color=(0,0,0), density=None,
                velocity=None, pressure=None, diameter=None):
        if diameter is None:
            diameter = self.diameter
            
        num_dim = [np.arange(start[i] * scale[i], end[i] * scale[i], diameter) for i in range(self.dim)]
        new_positions = np.array(np.meshgrid(*num_dim, indexing='ij')).reshape(self.dim, -1).T
        
        num_new_particles = new_positions.shape[0]
        new_velocities = np.zeros_like(new_positions) if velocity is None else np.tile(velocity, (num_new_particles, 1))
        new_densities = np.ones(num_new_particles) * (density if density is not None else 1000.)
        new_pressures = np.zeros(num_new_particles) if pressure is None else np.ones(num_new_particles) * pressure
        new_materials = np.ones(num_new_particles, dtype=np.int32) * material
        new_is_dynamic = np.ones(num_new_particles, dtype=np.int32) * is_dynamic
        new_colors = np.tile(color, (num_new_particles, 1))

        self.add_particles(object_id, num_new_particles, new_positions, new_velocities,
                         new_densities, new_pressures, new_materials, new_is_dynamic, new_colors)
        
        if material == self.material_fluid:
            self.fluid_particle_num += num_new_particles

    def add_boundary_object(self, object_id, domain_start, domain_end, thickness,
                          material=0, is_dynamic=False, color=(0,0,0),
                          density=None, pressure=None, velocity=None, space=None):
        if space is None:
            space = self.diameter
            
        num_dim = [np.arange(domain_start[i], domain_end[i], space) for i in range(self.dim)]
        new_positions = np.array(np.meshgrid(*num_dim, indexing='ij')).reshape(self.dim, -1).T
        
        # Create boundary mask
        mask = np.zeros(new_positions.shape[0], dtype=bool)
        for i in range(self.dim):
            mask |= ((new_positions[:, i] <= domain_start[i] + thickness) | 
                    (new_positions[:, i] >= domain_end[i] - thickness))
            
        new_positions = new_positions[mask]
        num_new_particles = new_positions.shape[0]
        
        new_velocities = np.zeros_like(new_positions) if velocity is None else np.tile(velocity, (num_new_particles, 1))
        new_densities = np.ones(num_new_particles) * (density if density is not None else 1000.)
        new_pressures = np.zeros(num_new_particles) if pressure is None else np.ones(num_new_particles) * pressure
        new_materials = np.ones(num_new_particles, dtype=np.int32) * material
        new_is_dynamic = np.ones(num_new_particles, dtype=np.int32) * is_dynamic
        new_colors = np.tile(color, (num_new_particles, 1))

        self.add_particles(object_id, num_new_particles, new_positions, new_velocities,
                         new_densities, new_pressures, new_materials, new_is_dynamic, new_colors)

    def insert_object(self):
        # Fluid blocks
        for fluid in self.fluid_blocks:
            obj_id = fluid["objectId"]
            
            if obj_id in self.present_object or fluid["entryTime"] > self.total_time:
                continue
            
            offset = np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            scale = np.array(fluid["scale"])
            velocity = fluid["velocity"]
            density = fluid["density"]
            color = fluid["color"]
            self.object_id_fluid_body.add(obj_id)
            self.object_visibility[obj_id] = fluid.get("visible", 1)
            self.object_materials[obj_id] = self.material_fluid
            self.object_densities[obj_id] = density
            self.object_collection[obj_id] = fluid
            
            self.add_cube(
                object_id = obj_id,
                start = start,
                end = end,
                scale = scale,
                material = self.material_fluid,
                is_dynamic = True,
                color = color,
                velocity = velocity,
                density = density,
                diameter = self.diameter
            )
            
            self.present_object.append(obj_id)
            
        # Rigid bodies
        for rigid in self.rigid_bodies:
            obj_id = rigid["objectId"]
            
            if obj_id in self.present_object:
                continue
            if rigid["entryTime"] > self.total_time:
                continue
            
            self.object_id_rigid_body.add(obj_id)
            particle_num = rigid["particleNum"]
            voxelized_points_np = rigid["voxelizedPoints"]
            
            velocity = np.array(rigid["velocity"], dtype=np.float32)
            density = rigid["density"]
            color = np.array(rigid["color"], dtype=np.int32)
            is_dynamic = rigid["isDynamic"]
            
            self.object_visibility[obj_id] = rigid.get("visible", 1)
            self.object_materials[obj_id] = self.material_rigid
            self.object_densities[obj_id] = density
            self.object_collection[obj_id] = rigid
            
            self.add_particles(
                obj_id,
                particle_num,
                np.array(voxelized_points_np, dtype=np.float32),
                np.tile(velocity, (particle_num, 1)),
                density * np.ones(particle_num, dtype=np.float32),
                np.zeros(particle_num, dtype=np.float32),
                self.material_rigid * np.ones(particle_num, dtype=np.int32),
                is_dynamic * np.ones(particle_num, dtype=np.int32),
                np.tile(color, (particle_num, 1))
            )
            
            self.present_object.append(obj_id)

    def prepare_neighbor_search(self):
        """准备邻居搜索"""
        self.grid_num_particles.fill(0)
        for p_i in range(self.particle_num):
            grid_index = self.get_flatten_grid_index(self.particle_positions[p_i])
            self.grid_ids[p_i] = grid_index
            self.grid_num_particles[grid_index] += 1

    def get_flatten_grid_index(self, pos):
        """获取展平的网格索引"""
        return self.flatten_grid_index(self.pos_to_index(pos))

    def pos_to_index(self, pos):
        """将位置转换为网格索引"""
        return (pos / self.grid_size).astype(np.int32)

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
    