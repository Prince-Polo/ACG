import taichi as ti 
import numpy as np
from . import ObjectProcessor as op
from functools import reduce
from ..utils import SimConfig

GRAVITY = ti.Vector([0.0, -9.81, 0.0])

@ti.data_oriented
class BaseContainer:
    def __init__(self, config: SimConfig, GGUI = False):
        self.dim = 3
        self.GGUI = GGUI
        self.cfg = config
        self.total_time = 0

        self.gravity = GRAVITY
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

        self.grid_size = self.dh
        self.padding = self.grid_size
        self.boundary_thickness = 0.0
        self.add_boundary = False
        self.add_boundary = self.cfg.get_cfg("addDomainBox")
        
        if self.add_boundary:
            self.domain_start = [self.domain_start[i] + self.padding for i in range(self.dim)]
            self.domain_end = [self.domain_end[i] - self.padding for i in range(self.dim)]
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
                       
        fluid_object_num = len(self.fluid_blocks) + len(self.fluid_bodies)
        rigid_object_num = len(self.rigid_bodies)
        
        #========== Allocate memory ==========#
        # Particle num of each grid
        num_grid = reduce(lambda x, y: x * y, self.grid_num)
        self.particle_num = ti.field(int, shape=())
        self.grid_num_particles = ti.field(int, shape=int(num_grid))
        self.grid_num_particles_temp = ti.field(int, shape=int(num_grid))

        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_num_particles.shape[0])

        # Particle related properties
        self.particle_object_ids = ti.field(dtype=int, shape=self.particle_max_num)
        self.particle_positions = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_velocities = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_accelerations = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_rest_volumes = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_rest_densities = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_masses = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_densities = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_num_densities = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_pressures = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_materials = ti.field(dtype=int, shape=self.particle_max_num)
        self.particle_colors = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.particle_is_dynamic = ti.field(dtype=int, shape=self.particle_max_num)
        self.rigid_particle_num = ti.field(int, shape=())
        self.fluid_particle_num = ti.field(int, shape=())
        self.rigid_particle_original_positions = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        
        self.object_materials = ti.field(dtype=int, shape=self.max_object_num)
        self.object_num = ti.field(dtype=int, shape=())
        self.fluid_object_num = ti.field(dtype=int, shape=())
        self.rigid_object_num = ti.field(dtype=int, shape=())
        self.object_num[None] = fluid_object_num + rigid_object_num + (1 if self.add_boundary else 0) # add 1 for domain box object
        self.fluid_object_num[None] = fluid_object_num
        self.rigid_object_num[None] = rigid_object_num

        self.rigid_body_is_dynamic = ti.field(dtype=int, shape=self.max_object_num)
        self.rigid_body_original_com = ti.Vector.field(self.dim, dtype=float, shape=self.max_object_num)
        self.rigid_body_masses = ti.field(dtype=float, shape=self.max_object_num)
        self.rigid_body_com = ti.Vector.field(self.dim, dtype=float, shape=self.max_object_num)
        self.rigid_body_rotations = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=self.max_object_num)
        self.rigid_body_torques = ti.Vector.field(self.dim, dtype=float, shape=self.max_object_num)
        self.rigid_body_forces = ti.Vector.field(self.dim, dtype=float, shape=self.max_object_num)
        self.rigid_body_velocities = ti.Vector.field(self.dim, dtype=float, shape=self.max_object_num)
        self.rigid_body_angular_velocities = ti.Vector.field(self.dim, dtype=float, shape=self.max_object_num)
        self.rigid_body_particle_num = ti.field(dtype=int, shape=self.max_object_num)
        
        # Buffer for sort
        self.particle_object_ids_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.particle_positions_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.rigid_particle_original_positions_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_velocities_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_rest_volumes_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_rest_densities_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_masses_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_densities_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_num_densities_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_materials_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.particle_colors_buffer = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.is_dynamic_buffer = ti.field(dtype=int, shape=self.particle_max_num)

        # Visibility of object
        self.object_visibility = ti.field(dtype=int, shape=self.max_object_num)

        # Grid id for each particle
        self.grid_ids = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_buffer = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_new = ti.field(int, shape=self.particle_max_num)
        
        self.x_vis_buffer = None
        if self.GGUI:
            self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
            self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        
        if self.add_boundary:
            self.add_boundary_object(
                object_id = self.object_num[None] - 1,
                domain_start = self.domain_start,
                domain_end = self.domain_end,
                thickness = self.boundary_thickness,
                material = self.material_rigid,
                is_dynamic = False,
                space = self.diameter,
                color = (127,127,127)
            )
            
            self.object_visibility[self.object_num[None]-1] = 0
            self.object_materials[self.object_num[None]-1] = self.material_rigid
            self.rigid_body_is_dynamic[self.object_num[None]-1] = 0
            self.rigid_body_velocities[self.object_num[None]-1] = ti.Vector([0.0 for _ in range(self.dim)])
            self.object_collection[self.object_num[None]-1] = 0
    
    ######## add all kinds of particles ########    
    def insert_object(self):
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
            scale = np.array(fluid["scale"])
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
                scale = scale,
                material= self.material_fluid,
                is_dynamic = True,
                color = color,
                velocity = velocity,
                density = density,
                diameter = self.diameter,
                )
            
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
            
            self.present_object.append(obj_id)
            self.fluid_particle_num += particle_num
        
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
            
            density = rigid["density"]
            color = np.array(rigid["color"], dtype=np.int32)
            is_dynamic = rigid["isDynamic"]
            
            if "visible" in rigid:
                self.object_visibility[obj_id] = rigid["visible"]
            else:
                self.object_visibility[obj_id] = 1
            
            self.object_materials[obj_id] = self.material_rigid
            self.object_collection[obj_id] = rigid
            
            self.add_particles(
                obj_id,
                particle_num,
                np.array(voxelized_points_np, dtype=np.float32), # position
                np.stack([velocity for _ in range(particle_num)]), # velocity
                density * np.ones(particle_num, dtype=np.float32), # density
                np.zeros(particle_num, dtype=np.float32), # pressure
                np.array([self.material_rigid for _ in range(particle_num)], dtype=np.int32), 
                is_dynamic * np.ones(particle_num, dtype=np.int32), # is_dynamic
                np.stack([color for _ in range(particle_num)])
            )
            
            self.rigid_body_is_dynamic[obj_id] = is_dynamic
            self.rigid_body_velocities[obj_id] = velocity
            self.rigid_particle_num[None] += particle_num
            self.rigid_body_particle_num[obj_id] = particle_num
        
            if is_dynamic:
                velocity = np.array(rigid["velocity"], dtype=np.float32)
                self.rigid_body_masses[obj_id] = self.compute_rigid_mass(obj_id)
                self.rigid_body_com[obj_id] = self.compute_rigid_com(obj_id)
                self.rigid_body_is_dynamic[obj_id] = 1
            else:
                velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
        
            self.present_object.append(obj_id)
        
    @ti.kernel
    def compute_rigid_mass(self, objId: int) -> ti.f32:
        sum_mass = 0.0
        for i in range(self.particle_num[None]):
            if self.particle_object_ids[i] == objId and self.particle_is_dynamic[i]:
                sum_mass += self.particle_rest_densities[i] * self.V0
        return sum_mass
    
    @ti.kernel
    def compute_rigid_com(self, objId: int) -> ti.types.vector(3, float):
        sum_com = ti.Vector([0.0 for _ in range(self.dim)])
        for i in range(self.particle_num[None]):
            if self.particle_object_ids[i] == objId and self.particle_is_dynamic[i]:
                sum_com += self.particle_positions[i] * self.particle_rest_densities[i] * self.V0

        return sum_com / self.rigid_body_masses[objId]
            
    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        self.particle_object_ids[p] = obj_id
        self.particle_positions[p] = x
        self.rigid_particle_original_positions[p] = x
        self.particle_velocities[p] = v
        self.particle_densities[p] = density
        self.particle_num_densities[p] = density
        self.particle_rest_volumes[p] = self.V0
        self.particle_rest_densities[p] = density
        self.particle_masses[p] = self.V0 * density
        self.particle_pressures[p] = pressure
        self.particle_materials[p] = material
        self.particle_is_dynamic[p] = is_dynamic
        self.particle_colors[p] = color
    
    def add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()
                      ):
        
        self._add_particles(object_id,
                      new_particles_num,
                      new_particles_positions,
                      new_particles_velocity,
                      new_particle_density,
                      new_particle_pressure,
                      new_particles_material,
                      new_particles_is_dynamic,
                      new_particles_color
                      )
    
    @ti.kernel
    def _add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, object_id, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_is_dynamic[p - self.particle_num[None]],
                              ti.Vector([new_particles_color[p - self.particle_num[None], i] for i in range(3)])
                              )
        self.particle_num[None] += new_particles_num
        
    def add_cube(
        self,
        object_id,
        start,
        end,
        scale,
        material,
        is_dynamic=True,
        color=(0,0,0),
        density=None,
        velocity=None,
        pressure=None,
        diameter=None,
    ):
        if diameter is None:
            diameter = self.diameter
        num_dim = [np.arange(start[i] * scale[i], end[i] * scale[i], diameter) for i in range(self.dim)]
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        num_new_particles = new_positions.shape[0]
        new_velocities = np.zeros_like(new_positions, dtype=np.float32) if velocity is None else np.tile(velocity, (num_new_particles, 1))
        new_densities = np.ones(num_new_particles) * (density if density is not None else 1000.)
        new_pressures = np.zeros(num_new_particles) if pressure is None else np.ones(num_new_particles) * pressure
        new_materials = np.ones(num_new_particles, dtype=np.int32) * material
        new_is_dynamic = np.ones(num_new_particles, dtype=np.int32) * is_dynamic
        new_colors = np.tile(color, (num_new_particles, 1))

        self.add_particles(object_id, num_new_particles, new_positions, new_velocities, new_densities, new_pressures, new_materials, new_is_dynamic, new_colors)
        
        if material == self.material_fluid:
            self.fluid_particle_num[None] += num_new_particles
    
    def add_boundary_object(
        self,
        object_id,
        domain_start,
        domain_end,
        thickness,
        material=0,
        is_dynamic=False,
        color=(0,0,0),
        density=None,
        pressure=None,
        velocity=None,
        space=None,
    ):
        if space is None:
            space = self.diameter
        num_dim = [np.arange(domain_start[i], domain_end[i], space) for i in range(self.dim)]
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        mask =np.zeros(new_positions.shape[0], dtype=bool)
        for i in range(self.dim):
            mask = mask | ((new_positions[:, i] <= domain_start[i] + thickness) | (new_positions[:, i] >= domain_end[i] - thickness))
        new_positions = new_positions[mask]
        new_velocities = np.zeros_like(new_positions, dtype=np.float32) if velocity is None else np.tile(velocity, (new_positions.shape[0], 1))
        new_densities = np.ones(new_positions.shape[0]) * (density if density is not None else 1000.)
        new_pressures = np.zeros(new_positions.shape[0]) if pressure is None else np.ones(new_positions.shape[0]) * pressure
        new_materials = np.ones(new_positions.shape[0], dtype=np.int32) * material
        new_is_dynamic = np.ones(new_positions.shape[0], dtype=np.int32) * is_dynamic
        new_colors = np.tile(color, (new_positions.shape[0], 1))

        self.add_particles(object_id, new_positions.shape[0], new_positions, new_velocities, new_densities, new_pressures, new_materials, new_is_dynamic, new_colors)
        
    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)


    @ti.func
    def flatten_grid_index(self, grid_index):
        ret = 0
        for i in ti.static(range(self.dim)):
            ret_p = grid_index[i]
            for j in ti.static(range(i+1, self.dim)):
                ret_p *= self.grid_num[j]
            ret += ret_p
        
        return ret
    
    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))
    
    @ti.func
    def is_static_rigid_body(self, p):
        return self.particle_materials[p] == self.material_rigid and (not self.particle_is_dynamic[p])

    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.particle_materials[p] == self.material_rigid and self.particle_is_dynamic[p]
    
    ###### initial grid ######
    @ti.kernel
    def init_grid(self):
        self.grid_num_particles.fill(0)
        for i in range(self.particle_num[None]):
            grid_id = self.get_flatten_grid_index(self.particle_positions[i])
            self.grid_ids[i] = grid_id
            ti.atomic_add(self.grid_num_particles[grid_id], 1)
        for i in ti.grouped(self.grid_num_particles):
            self.grid_num_particles_temp[i] = self.grid_num_particles[i]
    
    @ti.kernel
    def particles_sort(self):
        for i in range(self.particle_num[None]):
            j = self.particle_num[None] - 1 - i
            base_offset = 0
            if self.grid_ids[j] -1 >= 0:
                base_offset = self.grid_num_particles[self.grid_ids[j] - 1]
            self.grid_ids_new[j] = ti.atomic_sub(self.grid_num_particles_temp[self.grid_ids[j]], 1) - 1 + base_offset

        for i in range(self.particle_num[None]):
            new_index = self.grid_ids_new[i]
            self.grid_ids_buffer[new_index] = self.grid_ids[i]
            self.particle_object_ids_buffer[new_index] = self.particle_object_ids[i]
            self.rigid_particle_original_positions_buffer[new_index] = self.rigid_particle_original_positions[i]
            self.particle_positions_buffer[new_index] = self.particle_positions[i]
            self.particle_velocities_buffer[new_index] = self.particle_velocities[i]
            self.particle_rest_volumes_buffer[new_index] = self.particle_rest_volumes[i]
            self.particle_rest_densities_buffer[new_index] = self.particle_rest_densities[i]
            self.particle_masses_buffer[new_index] = self.particle_masses[i]
            self.particle_densities_buffer[new_index] = self.particle_densities[i]
            self.particle_materials_buffer[new_index] = self.particle_materials[i]
            self.particle_num_densities_buffer[new_index] = self.particle_num_densities[i]
            self.particle_colors_buffer[new_index] = self.particle_colors[i]
            self.is_dynamic_buffer[new_index] = self.particle_is_dynamic[i]
            
        for i in range(self.particle_num[None]):
            self.grid_ids[i] = self.grid_ids_buffer[i]
            self.particle_object_ids[i] = self.particle_object_ids_buffer[i]
            self.rigid_particle_original_positions[i] = self.rigid_particle_original_positions_buffer[i]
            self.particle_positions[i] = self.particle_positions_buffer[i]
            self.particle_velocities[i] = self.particle_velocities_buffer[i]
            self.particle_rest_volumes[i] = self.particle_rest_volumes_buffer[i]
            self.particle_rest_densities[i] = self.particle_rest_densities_buffer[i]
            self.particle_masses[i] = self.particle_masses_buffer[i]
            self.particle_densities[i] = self.particle_densities_buffer[i]
            self.particle_num_densities[i] = self.particle_num_densities_buffer[i]
            self.particle_materials[i] = self.particle_materials_buffer[i]
            self.particle_colors[i] = self.particle_colors_buffer[i]
            self.particle_is_dynamic[i] = self.is_dynamic_buffer[i]
            
    def prepare_neighbor_search(self):
        self.init_grid()
        self.prefix_sum_executor.run(self.grid_num_particles)
        self.particles_sort()
    
    @ti.func
    def for_all_neighbors(self, i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.particle_positions[i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            neighbor_cell = center_cell + offset
            grid_index = self.flatten_grid_index(neighbor_cell)
            start_idx = 0
            if grid_index > 0:
                start_idx = self.grid_num_particles[grid_index - 1]
            end_idx = self.grid_num_particles[grid_index]
            for j in range(start_idx, end_idx):
                if i != j and (self.particle_positions[i] - self.particle_positions[j]).norm() < self.dh:
                    task(i, j, ret)
    
    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]
    
    def copy_to_vis_buffer(self, invisible_objects=[], dim=3):
        self.flush_vis_buffer()
        for obj_id in self.object_collection:
            if self.object_visibility[obj_id] == 1:
                self._copy_to_vis_buffer(obj_id)
                    
    @ti.kernel
    def flush_vis_buffer(self):
        self.x_vis_buffer.fill(0.0)
        self.color_vis_buffer.fill(0.0)

    @ti.kernel
    def _copy_to_vis_buffer(self, obj_id: int):
        assert self.GGUI
        for i in range(self.particle_max_num):
            if self.particle_object_ids[i] == obj_id:
                self.x_vis_buffer[i] = self.particle_positions[i]
                self.color_vis_buffer[i] = self.particle_colors[i] / 255.0
    
    def dump(self, obj_id):
        np_object_id = self.particle_object_ids.to_numpy()
        mask = (np_object_id == obj_id).nonzero()

        np_x = self.particle_positions.to_numpy()[mask]
        np_v = self.particle_velocities.to_numpy()[mask]

        return {
            'position': np_x,
            'velocity': np_v
        }
    