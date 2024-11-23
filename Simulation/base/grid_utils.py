import taichi as ti
import numpy as np

@ti.data_oriented
class GridUtils:
    def __init__(self, dim, grid_size, grid_num, particle_max_num, particle_positions, particle_num, dh):
        self.dim = dim
        self.grid_size = grid_size
        self.grid_num = grid_num
        self.particle_max_num = particle_max_num
        self.particle_positions = particle_positions
        self.particle_num = particle_num
        self.dh = dh

        num_grid = np.prod(self.grid_num)
        self.grid_num_particles = ti.field(int, shape=int(num_grid))
        self.grid_num_particles_temp = ti.field(int, shape=int(num_grid))
        self.grid_ids = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_buffer = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_new = ti.field(int, shape=self.particle_max_num)
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_num_particles.shape[0])

    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)

    @ti.func
    def flatten_grid_index(self, grid_index):
        ret = 0
        for i in ti.static(range(self.dim)):
            ret_p = grid_index[i]
            for j in ti.static(range(i + 1, self.dim)):
                ret_p *= self.grid_num[j]
            ret += ret_p
        return ret

    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))

    @ti.kernel
    def init_grid(self):
        self.grid_num_particles.fill(0)
        for p_i in range(self.particle_num[None]):
            grid_index = self.get_flatten_grid_index(self.particle_positions[p_i])
            self.grid_ids[p_i] = grid_index
            ti.atomic_add(self.grid_num_particles[grid_index], 1)
        for p_i in ti.grouped(self.grid_num_particles):
            self.grid_num_particles_temp[p_i] = self.grid_num_particles[p_i]

    @ti.kernel
    def reorder_particles(self,
                          particle_object_ids: ti.template(),
                          rigid_particle_original_positions: ti.template(),
                          particle_velocities: ti.template(),
                          particle_rest_volumes: ti.template(),
                          particle_masses: ti.template(),
                          particle_densities: ti.template(),
                          particle_materials: ti.template(),
                          particle_colors: ti.template(),
                          particle_is_dynamic: ti.template(),
                          particle_object_ids_buffer: ti.template(),
                          rigid_particle_original_positions_buffer: ti.template(),
                          particle_positions_buffer: ti.template(),
                          particle_velocities_buffer: ti.template(),
                          particle_rest_volumes_buffer: ti.template(),
                          particle_masses_buffer: ti.template(),
                          particle_densities_buffer: ti.template(),
                          particle_materials_buffer: ti.template(),
                          particle_colors_buffer: ti.template(),
                          is_dynamic_buffer: ti.template()):
        for i in range(self.particle_num[None]):
            p_i = self.particle_num[None] - 1 - i
            base_offset = 0
            if self.grid_ids[p_i] - 1 >= 0:
                base_offset = self.grid_num_particles[self.grid_ids[p_i] - 1]
            self.grid_ids_new[p_i] = ti.atomic_sub(self.grid_num_particles_temp[self.grid_ids[p_i]], 1) - 1 + base_offset

        for p_i in range(self.particle_num[None]):
            new_index = self.grid_ids_new[p_i]
            self.grid_ids_buffer[new_index] = self.grid_ids[p_i]
            particle_object_ids_buffer[new_index] = particle_object_ids[p_i]
            rigid_particle_original_positions_buffer[new_index] = rigid_particle_original_positions[p_i]
            particle_positions_buffer[new_index] = self.particle_positions[p_i]
            particle_velocities_buffer[new_index] = particle_velocities[p_i]
            particle_rest_volumes_buffer[new_index] = particle_rest_volumes[p_i]
            particle_masses_buffer[new_index] = particle_masses[p_i]
            particle_densities_buffer[new_index] = particle_densities[p_i]
            particle_materials_buffer[new_index] = particle_materials[p_i]
            particle_colors_buffer[new_index] = particle_colors[p_i]
            is_dynamic_buffer[new_index] = particle_is_dynamic[p_i]

        for p_i in range(self.particle_num[None]):
            self.grid_ids[p_i] = self.grid_ids_buffer[p_i]
            particle_object_ids[p_i] = particle_object_ids_buffer[p_i]
            rigid_particle_original_positions[p_i] = rigid_particle_original_positions_buffer[p_i]
            self.particle_positions[p_i] = particle_positions_buffer[p_i]
            particle_velocities[p_i] = particle_velocities_buffer[p_i]
            particle_rest_volumes[p_i] = particle_rest_volumes_buffer[p_i]
            particle_masses[p_i] = particle_masses_buffer[p_i]
            particle_densities[p_i] = particle_densities_buffer[p_i]
            particle_materials[p_i] = particle_materials_buffer[p_i]
            particle_colors[p_i] = particle_colors_buffer[p_i]
            particle_is_dynamic[p_i] = is_dynamic_buffer[p_i]

    def prepare_neighborhood_search(self):
        self.init_grid()
        self.prefix_sum_executor.run(self.grid_num_particles)

    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.particle_positions[p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            grid_index = self.flatten_grid_index(center_cell + offset)
            start_idx = 0
            end_idx = self.grid_num_particles[grid_index]
            if grid_index - 1 >= 0:
                start_idx = self.grid_num_particles[grid_index - 1]
            for p_j in range(start_idx, end_idx):
                if p_i != p_j and (self.particle_positions[p_i] - self.particle_positions[p_j]).norm() < self.dh:
                    task(p_i, p_j, ret)