import taichi as ti
from functools import reduce
import numpy as np

@ti.data_oriented
class ParticleUtils:
    def __init__(self, dim, particle_diameter, V0, particle_num, particle_positions, particle_velocities,
                 particle_densities, particle_pressures, particle_materials, particle_is_dynamic,
                 particle_colors, particle_object_ids, rigid_particle_original_positions,
                 particle_masses, particle_rest_volumes):
        self.dim = dim
        self.particle_diameter = particle_diameter
        self.V0 = V0
        self.particle_num = particle_num
        self.particle_positions = particle_positions
        self.particle_velocities = particle_velocities
        self.particle_densities = particle_densities
        self.particle_pressures = particle_pressures
        self.particle_materials = particle_materials
        self.particle_is_dynamic = particle_is_dynamic
        self.particle_colors = particle_colors
        self.particle_object_ids = particle_object_ids
        self.rigid_particle_original_positions = rigid_particle_original_positions
        self.particle_masses = particle_masses
        self.particle_rest_volumes = particle_rest_volumes

    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        self.particle_object_ids[p] = obj_id
        self.particle_positions[p] = x
        self.rigid_particle_original_positions[p] = x
        self.particle_velocities[p] = v
        self.particle_densities[p] = density
        self.particle_rest_volumes[p] = self.V0
        self.particle_masses[p] = self.V0 * density
        self.particle_pressures[p] = pressure
        self.particle_materials[p] = material
        self.particle_is_dynamic[p] = is_dynamic
        self.particle_colors[p] = color

    def add_particles(self, object_id, new_particles_num, new_particles_positions,
                      new_particles_velocity, new_particle_density, new_particle_pressure,
                      new_particles_material, new_particles_is_dynamic, new_particles_color):
        self._add_particles(object_id, new_particles_num, new_particles_positions,
                            new_particles_velocity, new_particle_density, new_particle_pressure,
                            new_particles_material, new_particles_is_dynamic, new_particles_color)

    @ti.kernel
    def _add_particles(self, object_id: int, new_particles_num: int,
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
                              ti.Vector([new_particles_color[p - self.particle_num[None], i] for i in range(3)]))
        self.particle_num[None] += new_particles_num

    def compute_cube_particle_num(self, start, end, space=None):
        if space is None:
            space = self.particle_diameter
        num_dim = []
        for i in range(self.dim):
            num_dim.append(np.arange(start[i], end[i], space))
        return reduce(lambda x, y: x * y, [len(n) for n in num_dim])

    def compute_box_particle_num(self, lower_corner, cube_size, thickness, space=None):
        if space is None:
            space = self.particle_diameter
        num_dim = []
        for i in range(self.dim):
            num_dim.append(np.arange(lower_corner[i], lower_corner[i] + cube_size[i], space))

        new_positions = np.array(np.meshgrid(*num_dim, sparse=False, indexing='ij'), dtype=np.float32)
        new_positions = new_positions.reshape(-1, reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        mask = np.zeros(new_positions.shape[0], dtype=bool)
        for i in range(self.dim):
            mask |= ((new_positions[:, i] <= lower_corner[i] + thickness) |
                     (new_positions[:, i] >= lower_corner[i] + cube_size[i] - thickness))
        new_positions = new_positions[mask]
        return new_positions.shape[0]

    def add_cube(self, object_id, lower_corner, cube_size, material, is_dynamic,
                 density=None, pressure=None, velocity=None, space=None, color=(0, 0, 0)):
        if space is None:
            space = self.particle_diameter
        num_dim = []
        for i in range(self.dim):
            num_dim.append(np.arange(lower_corner[i], lower_corner[i] + cube_size[i], space))
        new_positions = np.array(np.meshgrid(*num_dim, sparse=False, indexing='ij'), dtype=np.float32)
        new_positions = new_positions.reshape(-1, reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        num_new_particles = new_positions.shape[0]
        if velocity is None:
            velocity_arr = np.zeros_like(new_positions, dtype=np.float32)
        else:
            velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)
        material_arr = np.full(num_new_particles, material, dtype=np.int32)
        is_dynamic_arr = np.full(num_new_particles, is_dynamic, dtype=np.int32)
        color_arr = np.stack([np.full(num_new_particles, c, dtype=np.int32) for c in color], axis=1)
        density_arr = np.full(num_new_particles, density if density is not None else 1000., dtype=np.float32)
        pressure_arr = np.full(num_new_particles, pressure if pressure is not None else 0., dtype=np.float32)
        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr,
                           density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)

    def add_box(self, object_id, lower_corner, cube_size, thickness, material, is_dynamic,
                density=None, pressure=None, velocity=None, space=None, color=(0, 0, 0)):
        if space is None:
            space = self.particle_diameter
        num_dim = []
        for i in range(self.dim):
            num_dim.append(np.arange(lower_corner[i], lower_corner[i] + cube_size[i], space))
        new_positions = np.array(np.meshgrid(*num_dim, sparse=False, indexing='ij'), dtype=np.float32)
        new_positions = new_positions.reshape(-1, reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        mask = np.zeros(new_positions.shape[0], dtype=bool)
        for i in range(self.dim):
            mask |= ((new_positions[:, i] <= lower_corner[i] + thickness) |
                     (new_positions[:, i] >= lower_corner[i] + cube_size[i] - thickness))
        new_positions = new_positions[mask]
        num_new_particles = new_positions.shape[0]
        if velocity is None:
            velocity_arr = np.zeros_like(new_positions, dtype=np.float32)
        else:
            velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)
        material_arr = np.full(num_new_particles, material, dtype=np.int32)
        is_dynamic_arr = np.full(num_new_particles, is_dynamic, dtype=np.int32)
        color_arr = np.stack([np.full(num_new_particles, c, dtype=np.int32) for c in color], axis=1)
        density_arr = np.full(num_new_particles, density if density is not None else 1000., dtype=np.float32)
        pressure_arr = np.full(num_new_particles, pressure if pressure is not None else 0., dtype=np.float32)
        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr,
                           density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)