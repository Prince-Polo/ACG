import taichi as ti

@ti.data_oriented
class VisualizationUtils:
    def __init__(self, particle_max_num, dim, domain_size, GGUI):
        self.particle_max_num = particle_max_num
        self.dim = dim
        self.domain_size = domain_size
        self.GGUI = GGUI

        if self.GGUI:
            self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
            self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)

    def copy_to_vis_buffer(self, particle_positions, particle_colors, particle_object_ids, object_visibility, object_collection, dim=3):
        self.flush_vis_buffer()
        for obj_id in object_collection:
            if object_visibility[obj_id] == 1:
                if dim == 3:
                    self._copy_to_vis_buffer_3d(obj_id, particle_positions, particle_colors, particle_object_ids)
                elif dim == 2:
                    self._copy_to_vis_buffer_2d(obj_id, particle_positions, particle_colors, particle_object_ids)

    @ti.kernel
    def flush_vis_buffer(self):
        self.x_vis_buffer.fill(0.0)
        self.color_vis_buffer.fill(0.0)

    @ti.kernel
    def _copy_to_vis_buffer_2d(self, obj_id: int, particle_positions: ti.template(), particle_colors: ti.template(), particle_object_ids: ti.template()):
        assert self.GGUI
        domain_size = ti.Vector([self.domain_size[0], self.domain_size[1]])
        for i in range(self.particle_max_num):
            if particle_object_ids[i] == obj_id:
                self.x_vis_buffer[i] = particle_positions[i] / domain_size
                self.color_vis_buffer[i] = particle_colors[i] / 255.0

    @ti.kernel
    def _copy_to_vis_buffer_3d(self, obj_id: int, particle_positions: ti.template(), particle_colors: ti.template(), particle_object_ids: ti.template()):
        assert self.GGUI
        for i in range(self.particle_max_num):
            if particle_object_ids[i] == obj_id:
                self.x_vis_buffer[i] = particle_positions[i]
                self.color_vis_buffer[i] = particle_colors[i] / 255.0