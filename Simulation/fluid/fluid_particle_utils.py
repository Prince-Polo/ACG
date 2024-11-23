import taichi as ti

@ti.data_oriented
class FluidParticleUtils:
    def __init__(self, container):
        self.container = container

    @ti.kernel
    def update_fluid_velocity(self):
        """
        update velocity for each particle from acceleration
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_velocities[p_i] += self.container.dt[None] * self.container.particle_accelerations[p_i]

    @ti.kernel
    def update_fluid_position(self):
        """
        update position for each particle from velocity
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_positions[p_i] += self.container.dt[None] * self.container.particle_velocities[p_i]

            elif self.container.particle_positions[p_i][1] > self.container.g_upper:
                # the emitter part
                obj_id = self.container.particle_object_ids[p_i]
                if self.container.object_materials[obj_id] == self.container.material_fluid:
                    self.container.particle_positions[p_i] += self.container.dt[None] * self.container.particle_velocities[p_i]
                    if self.container.particle_positions[p_i][1] <= self.container.g_upper:
                        self.container.particle_materials[p_i] = self.container.material_fluid

    @ti.kernel
    def prepare_emitter(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                if self.container.particle_positions[p_i][1] > self.container.g_upper:
                    # an awful hack to realize emitter
                    # not elegant but works
                    # feel free to implement your own emitter
                    self.container.particle_materials[p_i] = self.container.material_rigid