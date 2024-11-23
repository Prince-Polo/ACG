import taichi as ti
import numpy as np

@ti.data_oriented
class Utils:
    def __init__(self, particle_num, particle_positions, particle_velocities, particle_object_ids):
        self.particle_num = particle_num
        self.particle_positions = particle_positions
        self.particle_velocities = particle_velocities
        self.particle_object_ids = particle_object_ids

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]

    def dump(self, obj_id):
        np_object_id = self.particle_object_ids.to_numpy()
        mask = (np_object_id == obj_id).nonzero()

        np_x = self.particle_positions.to_numpy()[mask]
        np_v = self.particle_velocities.to_numpy()[mask]

        return {
            'position': np_x,
            'velocity': np_v
        }